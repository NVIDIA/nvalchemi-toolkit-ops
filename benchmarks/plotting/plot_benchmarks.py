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
"""Benchmark Plotting — generates 3-panel figures (time, throughput, memory).

Each figure = 1 row of 3 panels for one (module, system, scaling_mode) combination.
Reads CSV files produced by the benchmark scripts.

Note: Interactive Plotly plots are deferred to a future release.

Usage:
    python plot_benchmarks.py /path/to/results/
    python plot_benchmarks.py /path/to/results/ --output-dir /path/to/plots/
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from benchmarks.plotting.styles import (
    ACCURACY_COLORS,
    AXIS_LABEL_SIZE,
    BACKEND_STYLES,
    CUTOFF_COLORS,
    D3_CUTOFF_COLORS,
    D3_CUTOFF_STYLES,
    EL_BATCH_PALETTE,
    GPU_VRAM_REFS,
    GRAY,
    GRID_STYLE,
    MEMORY_MERGE_TOLERANCE,
    NL_BATCH_PALETTE,
    NVIDIA_GREEN,
    SINGLE_PANEL_SIZE,
    THREE_PANEL_SIZE,
    TITLE_SIZE,
    X_AXIS_LIMITS,
    X_AXIS_TICKS_MEDIUM,
    create_table_legend,
    format_accuracy,
    format_legend_label,
    format_num,
    format_system_name,
    log_scale_formatter,
    memory_formatter,
    setup_log2_xaxis,
    setup_plot_style,
)

__all__ = [
    # Public API — used by generate_plots.py and benchmark_suite.py
    "add_vram_reference_lines",
    "detect_and_plot",
    "detect_module_mode",
    "generate_comparison_panels",
    "load_csv",
    "plot_comparison_panel",
    "plot_module",
    "plot_single_panel",
    "render_d3_panel",
    "render_el_panel",
    "render_nl_panel",
    # Constants
    "DEFAULT_CUTOFF",
    "DEFAULT_TOTAL_ATOMS",
    "SECONDARY_LINESTYLE",
]

# Line alpha for all data lines. 55% lets overlapping lines remain visible
# while keeping each line distinct. Applies to all ax.plot() calls.
LINE_ALPHA = 0.55

# Shared magic values extracted as named constants
# Dashed linestyle used to mark a secondary method in per-backend panels
# (naive in NL, ewald in EL, 25A "large" in D3 batch). In comparison
# panels the same constant distinguishes the secondary method between
# two overlaid backend lines. Single constant, four context-dependent
# uses — rename from NAIVE_LINESTYLE makes the polysemy explicit.
SECONDARY_LINESTYLE = (0, (4, 2))
DEFAULT_CUTOFF = 15.0  # default cutoff when CSV field is missing
DEFAULT_TOTAL_ATOMS = 131_072  # default total atoms for constant-workload mode


def _plot_data_line(
    ax: Axes,
    x: list[float],
    y: list[float | None],
    *,
    color: str,
    linestyle: str | tuple[int, tuple[int, int]],
    marker: str,
    label: str,
    linewidth: float = 2,
    **kwargs: Any,
) -> None:
    """Plot a single benchmark data line with standard styling.

    ``**kwargs`` forwards to ``ax.plot``; callers use it to override
    marker face (``markerfacecolor="white"`` for hollow markers) or
    edge weight when they need extra visual contrast.
    """
    plot_kwargs = {
        "markeredgecolor": "black",
        "markeredgewidth": 0.5,
    }
    plot_kwargs.update(kwargs)
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=linewidth,
        markersize=5,
        alpha=LINE_ALPHA,
        label=label,
        **plot_kwargs,
    )


def add_vram_reference_lines(
    ax: Axes, unit: str = "MB", gpu_vram_gb: int | None = None
) -> None:
    """Add horizontal GPU VRAM reference line with label on the right edge.

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
        ax.text(
            0.99,
            val,
            f" {vram_gb} GB {gpu_name}",
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="right",
            fontsize=8,
            color="black",
            alpha=0.7,
        )
    if max_vram_val > 0:
        ax.set_ylim(top=max_vram_val * 1.5)
        saved_formatter = ax.yaxis.get_major_formatter()

        def _vram_aware_formatter(val, pos, _orig=saved_formatter, _cap=max_vram_val):
            if val > _cap:
                return ""
            return _orig(val, pos)

        import matplotlib.ticker as _vtick

        ax.yaxis.set_major_formatter(_vtick.FuncFormatter(_vram_aware_formatter))


# =============================================================================
# CSV Loading
# =============================================================================


def load_csv(filepath: str | Path) -> list[dict[str, Any]]:
    """Load benchmark CSV with automatic type conversion.

    Filters out failed rows (success=False) since they have zero/empty values
    that would break plotting (e.g. empty string time fields).
    """
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
            # Skip failed rows — they have zero/empty timing values
            if row.get("success") is False:
                continue
            results.append(row)
    return results


def load_failures(csv_path: str | Path) -> list[dict[str, Any]]:
    """Load the OOM/failure sidecar for a CSV, if present.

    Expected sidecar name: ``{csv-stem}-failures.csv``. Columns:
    system, scaling_mode, method, backend, atoms_per_system, batch_size,
    cutoff, accuracy, failure_reason. Types coerced like load_csv.
    Returns [] if the sidecar is missing so callers can just iterate.
    """
    csv_path = Path(csv_path)
    sidecar = csv_path.parent / f"{csv_path.stem}-failures.csv"
    if not sidecar.exists():
        return []
    results = []
    with open(sidecar) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in list(row.items()):
                try:
                    if val == "":
                        continue
                    if "." in val or "e" in val.lower():
                        row[key] = float(val)
                    elif val.lstrip("-").isdigit():
                        row[key] = int(val)
                except (ValueError, AttributeError):
                    pass
            results.append(row)
    return results


def _failure_x(r: dict[str, Any], x_key: str) -> float | None:
    """Extract the x coordinate for a failure row, matching the renderer's
    choice of x_key ("total_atoms" or "atoms_per_system")."""
    if x_key == "total_atoms":
        aps = r.get("atoms_per_system")
        if aps in (None, ""):
            return None
        bs = r.get("batch_size", 1) or 1
        try:
            return float(aps) * float(bs)
        except (TypeError, ValueError):
            return None
    val = r.get(x_key)
    if val in (None, ""):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def draw_oom_markers(
    ax,
    failures: list[dict[str, Any]],
    *,
    x_key: str,
    color_resolver: Callable[[dict[str, Any]], str] | None = None,
) -> None:
    """Render an "×" marker near the top of the plot at each OOMed x
    position, colour-matched to the line that owned the failed series.

    The marker is drawn with alpha=0.55 (subtle) and a short dotted
    vertical connector runs from just below the marker down into the
    plot area at alpha=0.2, visually tying the × back to where the
    missing line would have continued.

    Parameters
    ----------
    ax
        The axes receiving the markers.
    failures
        Rows from :func:`load_failures`.
    x_key
        Which field carries the x coordinate — mirrors the main
        renderer's choice ("total_atoms" for system_size / batch,
        "atoms_per_system" for constant_workload).
    color_resolver
        Optional callable that takes a failure row and returns the
        line colour for that row's series. When absent, a neutral
        grey is used for all markers.
    """
    if not failures:
        return

    default_color = "#888888"
    seen: set[tuple[float, str]] = set()
    legend_added = False
    for r in failures:
        x = _failure_x(r, x_key)
        if x is None:
            continue
        color = color_resolver(r) if color_resolver else default_color
        key = (x, color)
        if key in seen:
            continue
        seen.add(key)
        ax.plot(
            [x],
            [0.92],
            transform=ax.get_xaxis_transform(),
            marker="x",
            color=color,
            alpha=0.55,
            markersize=9,
            markeredgewidth=2.0,
            linestyle="None",
            label="OOM" if not legend_added else None,
            zorder=10,
        )
        ax.plot(
            [x, x],
            [0.72, 0.905],
            transform=ax.get_xaxis_transform(),
            color=color,
            alpha=0.22,
            linestyle=":",
            linewidth=1.1,
            zorder=9,
        )
        legend_added = True


# =============================================================================
# NL Plotting
# =============================================================================


def _plot_3panel(
    data: list[dict[str, Any]],
    system_name: str,
    mode: str,
    module: str,
    output_dir: str | Path,
    fname: str,
) -> Path:
    """Generic 3-panel figure: delegates each panel to the single-panel renderer.

    This is the ONLY place where 3-panel figures are assembled. All plotting
    logic lives in render_nl_panel / render_d3_panel / render_el_panel.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=THREE_PANEL_SIZE)

    renderer = {"nl": render_nl_panel, "d3": render_d3_panel, "el": render_el_panel}[
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
    mode_str = _build_mode_title(mode, data=data, module=module)
    fig.suptitle(
        _build_panel_title(module, system_name, mode_str),
        fontsize=TITLE_SIZE + 1,
        y=1.02,
    )

    plt.tight_layout()
    output_path = Path(output_dir) / fname
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


MODE_FNAMES = {
    "system_size": "system-size-scaling",
    "constant_workload": "constant-workload-scaling",
    "batch_scaling": "batch-scaling",
}


def plot_module(
    data: list[dict[str, Any]],
    system_name: str,
    mode_name: str,
    module: str,
    output_dir: str | Path,
) -> Path:
    """Generic 3-panel plot for any module (nl, d3, el)."""
    fname = f"{module}-{system_name}-{MODE_FNAMES.get(mode_name, mode_name)}.png"
    return _plot_3panel(data, system_name, mode_name, module, output_dir, fname)


# =============================================================================
# Single-Panel API — for docs/generate_plots.py and Sphinx integration
# =============================================================================


def plot_single_panel(
    csv_path: str | Path, panel: str, output_path: str | Path
) -> None:
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
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    data = load_csv(csv_path)
    if not data:
        return

    # Filter to torch backend for static plots
    backends = {r.get("backend", "torch") for r in data}
    if len(backends) > 1:
        data = [r for r in data if r.get("backend", "torch") == "torch"]

    setup_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=SINGLE_PANEL_SIZE)

    system = data[0].get("system", "unknown")
    name = csv_path.stem

    # Detect module + mode from filename
    module, mode = detect_module_mode(name)

    # Dispatch to the right plotting logic. `failures=` is accepted by
    # each renderer but unused — OOMed configs surface as a missing
    # point on the right-hand end of their line, and the policy is
    # explained in the "Missing data points" admonition in
    # docs/benchmarks/index.md. The sidecar {stem}-failures.csv files
    # still ship so tooling can distinguish "didn't run (OOM)" from
    # "wasn't in the grid".
    if module == "nl":
        render_nl_panel(ax, data, system, mode, panel)
    elif module == "d3":
        render_d3_panel(ax, data, system, mode, panel)
    elif module == "el":
        render_el_panel(ax, data, system, mode, panel)
    else:
        print(f"  Unknown module for {name}")
        plt.close()
        return

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def detect_module_mode(name: str) -> tuple[str | None, str | None]:
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


def _setup_panel_axes(
    ax: Axes,
    panel: str,
    x_ticks: list[int] | None = None,
    x_limits: tuple[int, int] | None = None,
    x_label: str | None = None,
) -> None:
    """Common y-axis and grid setup for single panels.

    NOTE: does NOT call setup_log2_xaxis -- renderers handle x-axis
    themselves (with mode-specific show_batch_size etc).
    """
    import numpy as np
    import matplotlib.ticker as ticker_sp

    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(ticker_sp.NullFormatter())
    ax.grid(True, which="both", **GRID_STYLE)

    # LogLocator with subs=(1,2,5) leaves narrow y-ranges (< ~0.5 decade)
    # with zero ticks inside the visible window — e.g. D3 JAX const_workload
    # values sit between 0.27 and 0.35 μs/atom, and 2×10⁻¹=0.2 / 5×10⁻¹=0.5
    # both fall outside. Use every 10^N × {1..9} sub so at least one tick
    # lands inside any reasonable window, and label the (1,2,3,5) subset
    # (formatter handles label suppression for the rest).
    _LOG_SUBS = tuple(np.arange(1, 10) * 1.0)

    if panel == "time":
        ax.set_ylabel("Time per atom [\u03bcs]", fontsize=AXIS_LABEL_SIZE)
        ax.yaxis.set_major_locator(
            ticker_sp.LogLocator(base=10, subs=_LOG_SUBS, numticks=20)
        )
        ax.yaxis.set_major_formatter(ticker_sp.FuncFormatter(log_scale_formatter))
    elif panel == "throughput":
        ax.set_ylabel("Throughput [10\u2076 atoms/s]", fontsize=AXIS_LABEL_SIZE)
        ax.yaxis.set_major_locator(
            ticker_sp.LogLocator(base=10, subs=_LOG_SUBS, numticks=20)
        )
        ax.yaxis.set_major_formatter(ticker_sp.FuncFormatter(log_scale_formatter))
    elif panel == "memory":
        ax.set_ylabel("Peak Memory", fontsize=AXIS_LABEL_SIZE)
        ax.yaxis.set_major_formatter(ticker_sp.FuncFormatter(memory_formatter))
        add_vram_reference_lines(ax, unit="MB")


MODULE_NAMES = {"nl": "Neighbor List", "d3": "DFT-D3", "el": "Electrostatics"}


def _build_mode_title(
    mode: str,
    target: int | None = None,
    data: list[dict[str, Any]] | None = None,
    module: str | None = None,
) -> str:
    """Build human-readable mode string for panel titles."""
    if mode == "constant_workload":
        if target is None:
            target = (
                data[0].get("total_atoms", DEFAULT_TOTAL_ATOMS)
                if data
                else DEFAULT_TOTAL_ATOMS
            )
        return f"Constant {format_num(target)} Atoms"
    elif mode == "batch_scaling":
        return "Batch Scaling (15\u00c5)" if module == "nl" else "Batch Scaling"
    else:
        return "System Size Scaling"


def _build_panel_title(
    module: str, system: str, mode_str: str, suffix: str | None = None
) -> str:
    """Build panel title: 'Module | System | Mode [| Suffix]'."""
    sys_label = format_system_name(system)
    parts = [MODULE_NAMES.get(module, module), sys_label, mode_str]
    if suffix:
        parts.append(suffix)
    return " | ".join(parts)


def _should_merge_memory(data: list[dict[str, Any]]) -> bool:
    """Check if cell and naive memory values overlap within tolerance.

    Returns True when all corresponding sorted cell/naive mem_peak_gb
    values are within MEMORY_MERGE_TOLERANCE of each other, per cutoff.
    """
    by_cutoff_method = defaultdict(lambda: defaultdict(list))
    for r in data:
        by_cutoff_method[r.get("cutoff", DEFAULT_CUTOFF)][r.get("method", "?")].append(
            r.get("mem_peak_gb", 0)
        )
    for methods in by_cutoff_method.values():
        cell_mem = sorted(methods.get("cell", []))
        naive_mem = sorted(methods.get("naive", []))
        if cell_mem and naive_mem:
            if len(cell_mem) != len(naive_mem):
                return False
            for c, n in zip(cell_mem, naive_mem):
                if c > 0 and abs(c - n) / c > MEMORY_MERGE_TOLERANCE:
                    return False
    return True


def _finalize_panel(
    ax: Axes,
    panel: str,
    mode: str,
    x_ticks: list[int],
    x_limits: tuple[int, int],
    x_label: str,
    target_atoms: int | None = None,
    legend_header: str = "cutoff",
) -> None:
    """Apply x-axis, y-axis, grid, and legend to a completed panel."""
    if mode == "constant_workload" and target_atoms is not None:
        setup_log2_xaxis(
            ax,
            ticks=x_ticks,
            limits=x_limits,
            show_batch_size=True,
            target_atoms=target_atoms,
            label=x_label,
        )
    else:
        setup_log2_xaxis(ax, ticks=x_ticks, limits=x_limits, label=x_label)
    _setup_panel_axes(ax, panel, x_ticks, x_limits, x_label)
    create_table_legend(ax, loc="best", col2_header=legend_header)


def _group_nl_by_cutoff_merged(
    data: list[dict[str, Any]],
) -> dict[float, list[dict[str, Any]]]:
    """Group NL rows by cutoff, preferring cell method with naive fallback.

    Used for memory panels where cell and naive lines are merged into a
    single "Cell/Naive" line.
    """
    grouped = defaultdict(list)
    for r in data:
        if r["method"] == "cell":
            grouped[r.get("cutoff", DEFAULT_CUTOFF)].append(r)
    if not grouped:
        for r in data:
            grouped[r.get("cutoff", DEFAULT_CUTOFF)].append(r)
    return dict(grouped)


def _group_nl_by_method_cutoff(
    data: list[dict[str, Any]],
) -> dict[tuple[str, float], list[dict[str, Any]]]:
    """Group NL rows by (method, cutoff) tuple for separate cell/naive lines."""
    grouped = defaultdict(list)
    for r in data:
        grouped[(r["method"], r.get("cutoff", DEFAULT_CUTOFF))].append(r)
    return dict(grouped)


def _render_nl_batch(
    ax: Axes,
    data: list[dict[str, Any]],
    panel: str,
    merge_methods: bool,
) -> tuple[str, str]:
    """Render NL batch-scaling mode: filter to default cutoff, color by atoms_per_system.

    Returns (x_label, mode_str) for the caller to use in title/axis setup.
    """
    data_15 = [
        r for r in data if abs(r.get("cutoff", DEFAULT_CUTOFF) - DEFAULT_CUTOFF) < 1.0
    ]
    if not data_15:
        data_15 = data

    if merge_methods:
        # Memory: group by aps only, use cell data, label "Cell/Naive"
        grouped = defaultdict(list)
        for r in data_15:
            if r["method"] == "cell":
                grouped[r["atoms_per_system"]].append(r)
        if not grouped:  # fallback to naive
            for r in data_15:
                grouped[r["atoms_per_system"]].append(r)

        aps_values = sorted(grouped.keys())
        aps_colors = {
            aps: NL_BATCH_PALETTE[i % len(NL_BATCH_PALETTE)]
            for i, aps in enumerate(aps_values)
        }
        for aps in aps_values:
            rows = sorted(grouped[aps], key=lambda r: r["batch_size"])
            x = [r["total_atoms"] for r in rows]
            y = _get_panel_y(rows, panel)
            label = f"{'Cell/Naive':<11s} N={format_num(aps)}"
            _plot_data_line(
                ax,
                x,
                y,
                color=aps_colors.get(aps, GRAY),
                linestyle="-",
                marker="o",
                label=label,
            )
    else:
        grouped = defaultdict(list)
        for r in data_15:
            grouped[(r["method"], r["atoms_per_system"])].append(r)
        for key in grouped:
            grouped[key] = sorted(grouped[key], key=lambda r: r["batch_size"])

        aps_values = sorted({r["atoms_per_system"] for r in data_15})
        aps_colors = {
            aps: NL_BATCH_PALETTE[i % len(NL_BATCH_PALETTE)]
            for i, aps in enumerate(aps_values)
        }

        for method, aps in sorted(grouped.keys(), key=lambda k: (k[1], k[0])):
            rows = grouped[(method, aps)]
            x = [r["total_atoms"] for r in rows]
            y = _get_panel_y(rows, panel)
            is_naive = method == "naive"
            method_str = "Naive" if is_naive else "Cell"
            _plot_data_line(
                ax,
                x,
                y,
                color=aps_colors.get(aps, GRAY),
                linestyle=SECONDARY_LINESTYLE if is_naive else "-",
                marker="s" if is_naive else "o",
                label=f"{method_str:<7s}  N={format_num(aps)}",
            )

    return "Total Atoms", "Batch Scaling (15\u00c5)"


def _render_nl_by_cutoff(
    ax: Axes,
    data: list[dict[str, Any]],
    panel: str,
    merge_methods: bool,
    x_key: str,
) -> None:
    """Render NL system-size or constant-workload mode: color by cutoff.

    Handles both modes — they differ only in which CSV field is the x-axis
    (``total_atoms`` for system_size, ``atoms_per_system`` for constant_workload).
    """
    if merge_methods:
        grouped = _group_nl_by_cutoff_merged(data)
        for cutoff in sorted(grouped.keys()):
            rows = sorted(grouped[cutoff], key=lambda r: r[x_key])
            x = [r[x_key] for r in rows]
            y = _get_panel_y(rows, panel)
            _plot_data_line(
                ax,
                x,
                y,
                color=CUTOFF_COLORS.get(cutoff, GRAY),
                linestyle="-",
                marker="o",
                label=format_legend_label("Cell/Naive", cutoff),
            )
    else:
        grouped = _group_nl_by_method_cutoff(data)
        for key in grouped:
            grouped[key] = sorted(grouped[key], key=lambda r: r[x_key])

        for method, cutoff in sorted(grouped.keys(), key=lambda k: (k[1], k[0])):
            rows = grouped[(method, cutoff)]
            x = [r[x_key] for r in rows]
            y = _get_panel_y(rows, panel)
            is_naive = method == "naive"
            _plot_data_line(
                ax,
                x,
                y,
                color=CUTOFF_COLORS.get(cutoff, GRAY),
                linestyle=SECONDARY_LINESTYLE if is_naive else "-",
                marker="s" if is_naive else "o",
                label=format_legend_label(method.capitalize(), cutoff),
            )


def render_nl_panel(
    ax: Axes, data: list[dict[str, Any]], system: str, mode: str, panel: str,
    *,
    failures: list[dict[str, Any]] | None = None,
) -> None:
    """Render a single NL panel onto *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : list[dict]
        Rows from load_csv(), pre-filtered to one backend.
    system : str
        System name ('cscl' or 'nh3').
    mode : str
        'system_size', 'constant_workload', or 'batch_scaling'.
    panel : str
        'time', 'throughput', or 'memory'.

    Notes
    -----
    For memory panels, cell and naive lines are merged into "Cell/Naive"
    when their values are within MEMORY_MERGE_TOLERANCE (5%).
    """
    merge = _should_merge_memory(data) if panel == "memory" else False
    target = None

    if mode == "batch_scaling":
        x_label, mode_str = _render_nl_batch(ax, data, panel, merge)
    elif mode == "constant_workload":
        target = (
            data[0].get("total_atoms", DEFAULT_TOTAL_ATOMS)
            if data
            else DEFAULT_TOTAL_ATOMS
        )
        _render_nl_by_cutoff(ax, data, panel, merge, "atoms_per_system")
        x_label = "Atoms per System [\u00d7batch]"
        mode_str = f"Constant {format_num(target)} Atoms"
    else:
        _render_nl_by_cutoff(ax, data, panel, merge, "total_atoms")
        x_label = "System Size (atoms)"
        mode_str = "System Size Scaling"

    ax.set_title(_build_panel_title("nl", system, mode_str), fontsize=TITLE_SIZE)

    _finalize_panel(
        ax,
        panel,
        mode,
        X_AXIS_TICKS_MEDIUM,
        X_AXIS_LIMITS["medium"],
        x_label,
        target_atoms=target if mode == "constant_workload" else None,
        legend_header="batch" if mode == "batch_scaling" else "cutoff",
    )


def render_d3_panel(
    ax: Axes, data: list[dict[str, Any]], system: str, mode: str, panel: str,
    *,
    failures: list[dict[str, Any]] | None = None,
) -> None:
    """Render a single D3 panel onto *ax*.

    Uses ``time_d3_us_per_atom`` (D3-only time, excluding NL) when available,
    falling back to ``time_us_per_atom`` (total) for older CSVs.

    Parameters
    ----------
    ax, data, system, mode, panel : same as :func:`render_nl_panel`
    """
    is_batch = mode == "batch_scaling"
    target = None
    x_ticks = X_AXIS_TICKS_MEDIUM
    x_limits = X_AXIS_LIMITS["medium"]

    grouped = defaultdict(list)
    for r in data:
        if is_batch:
            key = (r.get("cutoff", DEFAULT_CUTOFF), r["atoms_per_system"])
        else:
            key = r.get("cutoff", DEFAULT_CUTOFF)
        grouped[key].append(r)

    if mode == "constant_workload":
        x_key, x_label = "atoms_per_system", "Atoms per System [\u00d7batch]"
        target = (
            data[0].get("total_atoms", DEFAULT_TOTAL_ATOMS)
            if data
            else DEFAULT_TOTAL_ATOMS
        )
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
            # Color = cutoff (primary), linestyle + marker = system size (secondary)
            linestyle = SECONDARY_LINESTYLE if is_large else "-"
            marker = "^" if is_large else "o"
            label = f"D3 {int(cutoff)}\u00c5 N={format_num(aps)}"
            _plot_data_line(
                ax, x, y, color=color, linestyle=linestyle, marker=marker, label=label
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
            style = D3_CUTOFF_STYLES.get(cutoff, {"marker": "o", "linestyle": "-"})
            _plot_data_line(ax, x, y, color=color, label=label, **style)

    mode_str = _build_mode_title(mode, target=target, module="d3")
    ax.set_title(_build_panel_title("d3", system, mode_str), fontsize=TITLE_SIZE)

    _finalize_panel(
        ax,
        panel,
        mode,
        x_ticks,
        x_limits,
        x_label,
        target_atoms=target if mode == "constant_workload" else None,
        legend_header="cutoff",
    )


def render_el_panel(
    ax: Axes, data: list[dict[str, Any]], system: str, mode: str, panel: str,
    *,
    failures: list[dict[str, Any]] | None = None,
) -> None:
    """Render a single Electrostatics panel onto *ax*.

    Drops ``_cg`` (charge-gradient) method variants to keep torch vs
    JAX apples-to-apples (JAX ``ewald_summation`` doesn't support
    ``compute_charge_gradients``, so its ``ewald_cg`` rows are
    ``success=False``). Colors by accuracy (green = 1e-4, blue = 1e-6),
    linestyle by method (PME = solid, Ewald = dashed).

    Parameters
    ----------
    ax, data, system, mode, panel : same as :func:`render_nl_panel`
    """
    target = None
    # Always plot the base methods (pme, ewald) — not the _cg (charge-
    # gradient) variants. JAX ewald_summation doesn't support
    # compute_charge_gradients so ewald_cg is written as success=False;
    # dropping _cg everywhere keeps torch vs JAX apples-to-apples. The
    # overhead of computing charge gradients on our data is quoted in
    # docs/benchmarks/electrostatics.md for users who need that cost.
    data = [r for r in data if not r.get("method", "").endswith("_cg")]

    is_batch = mode == "batch_scaling"
    grouped = defaultdict(list)
    for r in data:
        if is_batch:
            key = (r["method"], r.get("accuracy", 1e-4), r["atoms_per_system"])
        else:
            key = (r["method"], r.get("accuracy", 1e-4))
        grouped[key].append(r)

    if mode == "constant_workload":
        x_key, x_label = "atoms_per_system", "Atoms per System [\u00d7batch]"
        target = (
            data[0].get("total_atoms", DEFAULT_TOTAL_ATOMS)
            if data
            else DEFAULT_TOTAL_ATOMS
        )
    elif is_batch:
        x_key, x_label = "total_atoms", "Total Atoms"
    else:
        x_key, x_label = "total_atoms", "System Size (atoms)"

    for k in grouped:
        grouped[k] = sorted(grouped[k], key=lambda r: r.get(x_key, 0))

    x_ticks = X_AXIS_TICKS_MEDIUM
    x_limits = X_AXIS_LIMITS["medium"]

    # Color by aps for batch (green=small, orange=large), by accuracy for non-batch

    if is_batch:
        aps_values = sorted({r["atoms_per_system"] for r in data})
        acc_values = sorted({r.get("accuracy", 1e-4) for r in data})
        combo_colors = {}
        ci = 0
        for aps in aps_values:
            for acc in acc_values:
                combo_colors[(aps, acc)] = EL_BATCH_PALETTE[ci % len(EL_BATCH_PALETTE)]
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
            linestyle = SECONDARY_LINESTYLE if method_base == "ewald" else "-"
            aps_str = format_num(aps).rjust(3)
            label = f"{method_clean:<7s}  [{aps_str}, {format_accuracy(accuracy)}]"
        else:
            method, accuracy = key
            color = ACCURACY_COLORS.get(accuracy, GRAY)
            method_clean = "Ewald" if "ewald" in method.lower() else "PME"
            is_ewald = "ewald" in method.lower()
            marker = "^" if is_ewald else "o"
            linestyle = SECONDARY_LINESTYLE if is_ewald else "-"
            label = format_legend_label(method_clean, accuracy, is_cutoff=False)

        _plot_data_line(
            ax, x, y, color=color, marker=marker, linestyle=linestyle, label=label
        )

    if mode == "constant_workload":
        target = (
            data[0].get("total_atoms", DEFAULT_TOTAL_ATOMS)
            if data
            else DEFAULT_TOTAL_ATOMS
        )
    mode_str = _build_mode_title(mode, target=target, module="el")
    ax.set_title(_build_panel_title("el", system, mode_str), fontsize=TITLE_SIZE)

    _finalize_panel(
        ax,
        panel,
        mode,
        x_ticks,
        x_limits,
        x_label,
        target_atoms=target if mode == "constant_workload" else None,
        legend_header="batch" if is_batch else "accuracy",
    )


def _get_memory_y(rows: list[dict[str, Any]]) -> list[float | None]:
    """Extract memory y-values, handling JAX XLA pool unreliability.

    JAX mem_peak_gb is unreliable (XLA pre-allocates a pool and reuses it),
    so we use mem_delta_mb for JAX and mem_peak_gb (converted to MB) for torch.
    Zero/negative JAX deltas are replaced with None so matplotlib skips them.
    """
    backend = rows[0].get("backend", "torch") if rows else "torch"
    if backend == "jax":
        vals = [r["mem_delta_mb"] for r in rows]
        return [v if v > 0 else None for v in vals]
    return [r["mem_peak_gb"] * 1024 for r in rows]  # GB -> MB


def _get_panel_y(rows: list[dict[str, Any]], panel: str) -> list[float | None]:
    """Extract y-values for a given panel type.

    Parameters
    ----------
    rows : list[dict]
        CSV rows for one data series.
    panel : str
        Panel type: "time" (us/atom), "throughput" (Matoms/s), or "memory" (MB).
        For JAX memory, uses mem_delta_mb (None for zeros) since XLA pool
        makes mem_peak_gb unreliable.
    """
    if panel == "time":
        return [r["time_us_per_atom"] for r in rows]
    elif panel == "throughput":
        return [r["throughput_atoms_per_sec"] / 1e6 for r in rows]
    elif panel == "memory":
        return _get_memory_y(rows)
    return []


def _get_panel_y_from_raw(
    rows: list[dict[str, Any]], panel: str, time_values: list[float]
) -> list[float | None]:
    """Extract y-values for D3 panels using pre-extracted D3-only time.

    Parameters
    ----------
    rows : list[dict]
        CSV rows (used for memory extraction).
    panel : str
        Panel type: "time", "throughput", or "memory".
    time_values : list[float]
        Pre-extracted time_d3_us_per_atom values (D3-only, excluding NL overhead).
        Throughput = 1/time_us gives Matoms/s.
    """
    if panel == "time":
        return time_values
    elif panel == "throughput":
        return [1.0 / t if t > 0 else 0 for t in time_values]
    elif panel == "memory":
        return _get_memory_y(rows)
    return []


# =============================================================================
# Backend Comparison Panels
# =============================================================================


def plot_comparison_panel(
    csv_path: str | Path,
    panel: str,
    output_path: str | Path,
    module: str,
    fixed_param: float | None = None,
) -> None:
    """Render one panel with both backends overlaid at a fixed parameter.

    Generates backend comparison plots: torch (solid) vs jax (dashed),
    same color per method. Filters to a single representative parameter
    value to keep the plot clean (2-4 lines max).

    Parameters
    ----------
    csv_path : str or Path
        Path to benchmark CSV with both backends.
    panel : str
        'time', 'throughput', or 'memory'.
    output_path : str or Path
        Where to save the PNG.
    module : str
        'nl', 'd3', or 'el' — determines filtering and grouping.
    fixed_param : float or None
        Fixed cutoff (NL/D3) or accuracy (EL) to filter to.
        Defaults: NL=15.0 A, D3=15.0 A, EL=1e-6.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    all_data = load_csv(csv_path)
    if not all_data:
        return

    # Defaults per module
    if fixed_param is None:
        fixed_param = {"nl": DEFAULT_CUTOFF, "d3": DEFAULT_CUTOFF, "el": 1e-6}.get(
            module, DEFAULT_CUTOFF
        )

    # Filter to fixed parameter
    if module in ("nl", "d3"):
        data = [
            r
            for r in all_data
            if abs(r.get("cutoff", DEFAULT_CUTOFF) - fixed_param) < 1.0
        ]
    elif module == "el":
        data = [
            r
            for r in all_data
            if abs(r.get("accuracy", 1e-6) - fixed_param) < fixed_param * 0.5
        ]
        # Backend Comparison shows the base methods (pme, ewald). The _cg
        # variants live on their own per-backend panels — including them
        # here would either double the legend or (if jax ewald_cg rows are
        # dropped as success=False) produce an asymmetric comparison.
        data = [r for r in data if not r.get("method", "").endswith("_cg")]
    else:
        data = all_data

    if not data:
        print(f"  SKIP comparison: no data after filtering to {fixed_param}")
        return

    # Must have both backends
    backends = {r.get("backend", "torch") for r in data}
    if len(backends) < 2:
        print(f"  SKIP comparison: only {backends} present (need both torch+jax)")
        return

    setup_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=SINGLE_PANEL_SIZE)

    system = data[0].get("system", "unknown")
    mode = data[0].get("scaling_mode", "system_size")

    # Detect mode from filename if not in data
    name = csv_path.stem
    if "system-size" in name:
        mode = "system_size"
    elif "constant-workload" in name:
        mode = "constant_workload"
    elif "batch-scaling" in name:
        mode = "batch_scaling"

    if mode == "constant_workload":
        x_key = "atoms_per_system"
    else:
        x_key = "total_atoms"

    # For NL memory panels, merge methods if they're within 5%
    merge_nl_memory = (
        _should_merge_memory(data) if module == "nl" and panel == "memory" else False
    )

    # Group by (method, backend) — for batch mode, also by atoms_per_system
    is_batch = mode == "batch_scaling"
    grouped = defaultdict(list)
    for r in data:
        if merge_nl_memory:
            # Only keep cell data, group by backend (+ aps for batch)
            if r.get("method") != "cell":
                continue
            method_key = "cell/naive"
        else:
            method_key = r.get("method", "unknown")

        if is_batch:
            key = (method_key, r.get("backend", "torch"), r.get("atoms_per_system", 0))
        else:
            key = (method_key, r.get("backend", "torch"), None)
        grouped[key].append(r)

    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda r: r.get(x_key, 0))

    # Comparison-panel encoding (Tier 2 rewrite):
    #
    # Non-batch modes:
    #   color = backend    (green=torch, blue=jax)  — constant across modules
    #   linestyle = method (solid=primary, dashed=secondary)
    #   marker = method    (circle=primary, square=naive, triangle=ewald)
    #
    # Batch mode keeps the old encoding (color=aps) because three
    # dimensions (backend × method × aps) need three visual channels and
    # aps distinction is more valuable to a viewer than a redundant
    # backend-color when the backend is already captured by linestyle.
    backend_colors = {
        "torch": NVIDIA_GREEN,
        "jax": "#31688E",
    }
    # Which method in each module is the "secondary" line (dashed).
    secondary_methods = {
        "nl": {"naive"},
        "el": {"ewald", "ewald_cg"},
        "d3": set(),  # single method
    }

    for key in sorted(grouped.keys()):
        method, backend, aps = key
        rows = grouped[key]
        x = [r[x_key] for r in rows]

        # Get y values
        if module == "d3":
            y_raw = [r.get("time_d3_us_per_atom", r["time_us_per_atom"]) for r in rows]
            y = _get_panel_y_from_raw(rows, panel, y_raw)
        else:
            y = _get_panel_y(rows, panel)

        # Marker by method: circle for primary, square for naive, triangle for ewald
        if method in ("naive",):
            marker = "s"
        elif method in ("ewald", "ewald_cg"):
            marker = "^"
        else:
            marker = "o"

        # Linestyle by method (non-batch) or by backend (batch)
        is_secondary = method in secondary_methods.get(module, set())
        extra_kwargs: dict[str, Any] = {}
        if is_batch and aps is not None:
            # Batch comparison needs three visual channels
            # (backend × method × aps). Encoding:
            #   color        = aps (viridis-like palette)
            #   linestyle    = backend (solid=torch, dotted=jax)
            #   marker fill  = backend (filled=torch, hollow=jax)
            #   linewidth    = backend (thicker=torch)
            # Backend distinction is duplicated across linestyle + marker
            # fill because lines can overlap tightly on log axes — filled
            # vs. hollow markers stay visible even when lines coincide.
            aps_values = sorted({k[2] for k in grouped.keys() if k[2] is not None})
            aps_palette = (
                NL_BATCH_PALETTE
                if module == "nl"
                else [NVIDIA_GREEN, "#31688E", "#440154"]
            )
            aps_colors = {
                a: aps_palette[i % len(aps_palette)] for i, a in enumerate(aps_values)
            }
            color = aps_colors.get(aps, GRAY)
            if backend == "torch":
                linestyle = "-"
                linewidth = 2.5
                # default marker face (filled)
            else:  # jax
                linestyle = (0, (1, 2))  # tight dotted; distinct from "-" on log axes
                linewidth = 1.8
                extra_kwargs["markerfacecolor"] = "white"
                extra_kwargs["markeredgewidth"] = 1.8
        else:
            color = backend_colors.get(backend, GRAY)
            linestyle = SECONDARY_LINESTYLE if is_secondary else "-"
            linewidth = 2.0

        # Clean method name for legend
        if method == "cell/naive":
            method_clean = "Cell/Naive"
        else:
            method_clean = method.replace("_cg", "").capitalize()
            if method_clean == "Dftd3":
                method_clean = "D3"
        if is_batch and aps is not None:
            label = f"{method_clean}  {backend} N={format_num(aps)}"
        else:
            label = f"{method_clean}  {backend}"

        _plot_data_line(
            ax,
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            label=label,
            **extra_kwargs,
        )

    # Title
    param_label = (
        f"{int(fixed_param)}\u00c5"
        if fixed_param >= 1
        else format_accuracy(fixed_param)
    )
    mode_str = _build_mode_title(mode, data=data, module=module)
    ax.set_title(
        _build_panel_title(module, system, mode_str, suffix=param_label),
        fontsize=TITLE_SIZE,
    )

    # Axes
    x_ticks = X_AXIS_TICKS_MEDIUM
    x_limits = X_AXIS_LIMITS["medium"]
    x_label = (
        "Atoms per System [×batch]"
        if mode == "constant_workload"
        else "System Size (atoms)"
    )

    target = (
        data[0].get("total_atoms", DEFAULT_TOTAL_ATOMS)
        if mode == "constant_workload" and data
        else None
    )
    _finalize_panel(
        ax,
        panel,
        mode,
        x_ticks,
        x_limits,
        x_label,
        target_atoms=target,
        legend_header="backend",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison: {output_path}")


def generate_comparison_panels(csv_dir: str | Path, output_dir: str | Path) -> None:
    """Generate all backend comparison PNGs from CSVs with both backends."""
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)

    for csv_path in sorted(csv_dir.glob("*.csv")):
        name = csv_path.stem
        # Detect module
        if name.startswith(("nl-", "nl_")):
            module = "nl"
        elif name.startswith(("d3-", "d3_")):
            module = "d3"
        elif name.startswith(("el-", "el_")):
            module = "el"
        else:
            continue

        for panel in ("time", "throughput", "memory"):
            out_name = name.replace("-scaling", f"-comparison-{panel}")
            if panel not in out_name:
                out_name = f"{name}-comparison-{panel}"
            out_path = output_dir / f"{out_name}.png"
            try:
                plot_comparison_panel(csv_path, panel, out_path, module)
            except Exception as e:
                print(f"  ERROR comparison {csv_path.name}/{panel}: {e}")


# =============================================================================
# CLI — auto-detects all CSVs and plots them
# =============================================================================


def detect_and_plot(csv_path: Path, output_dir: str | Path) -> None:
    """Detect CSV type from filename and plot accordingly.

    Supports both old naming (nl_cscl_system_size_gpu.csv) and
    new naming (nl-cscl-system-size-scaling.csv).
    """
    data = load_csv(csv_path)
    if not data:
        return

    # Matplotlib static plots show torch backend only (JAX shown in Plotly interactive).
    # If both backends present, filter to torch for cleaner static plots.
    backends = {r.get("backend", "torch") for r in data}
    if len(backends) > 1:
        data = [r for r in data if r.get("backend", "torch") == "torch"]

    system = data[0].get("system", "unknown")
    name = csv_path.stem

    module, mode = detect_module_mode(name)
    if module and mode:
        plot_module(data, system, mode, module, output_dir)


def main() -> None:
    """CLI entry point for benchmark plotting."""
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
            detect_and_plot(csv_path, output_dir)
        except Exception as e:
            print(f"  ERROR plotting {csv_path.name}: {e}")


if __name__ == "__main__":
    main()
