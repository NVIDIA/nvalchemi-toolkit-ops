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
import math
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from benchmarks.plotting.styles import (
    AXIS_LABEL_SIZE,
    BACKEND_LINESTYLES,
    BATCH_SYSTEM_STYLES,
    D3_CUTOFF_COLORS,
    D3_CUTOFF_STYLES,
    DATA_LINE_WIDTH,
    DATA_MARKER_SIZE,
    DEFAULT_MARKER_FILLSTYLE,
    EL_ACCURACY_STYLES,
    EL_METHOD_STYLES,
    GPU_VRAM_REFS,
    GRAY,
    GRID_STYLE,
    LEGEND_SIZE,
    LINE_ALPHA,
    NL_CUTOFF_STYLES,
    NL_METHOD_STYLES,
    PNG_EXPORT_DPI,
    SECONDARY_LINESTYLE,
    SINGLE_PANEL_SIZE,
    THREE_PANEL_SIZE,
    TITLE_SIZE,
    X_AXIS_LIMITS,
    X_AXIS_TICKS_MEDIUM,
    format_accuracy,
    format_num,
    format_system_name,
    log_scale_formatter,
    memory_formatter,
    setup_log2_xaxis,
    setup_plot_style,
)


def _savefig_atomic(
    fig: Any,
    output_path: str | Path,
    *,
    dpi: int = PNG_EXPORT_DPI,
) -> None:
    """Write a figure beside its destination and atomically publish it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_suffix = output_path.suffix or ".png"
    temp_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp{output_suffix}"
    )
    try:
        fig.savefig(temp_path, dpi=dpi, bbox_inches="tight")
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


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

DEFAULT_CUTOFF = 15.0  # default cutoff when CSV field is missing
DEFAULT_TOTAL_ATOMS = 131_072  # default total atoms for constant-workload mode
NL_BACKEND_COMPARABLE_FAMILIES = frozenset({"naive_scalar"})


def _plot_data_line(
    ax: Axes,
    x: list[float],
    y: list[float | None],
    *,
    color: str,
    linestyle: str | tuple[int, tuple[int, int]],
    marker: str,
    label: str,
    linewidth: float = DATA_LINE_WIDTH,
    timing_methods: list[Any] | None = None,
    gid: str | None = None,
    **kwargs: Any,
) -> bool:
    """Plot a single benchmark data line with standard styling.

    ``**kwargs`` forwards to ``ax.plot`` for panel-specific visual overrides.
    """
    if not _has_plottable_y(y):
        return False
    x, y = _insert_timing_breaks(x, y, timing_methods)
    plot_kwargs = {
        "fillstyle": DEFAULT_MARKER_FILLSTYLE,
        "dash_capstyle": "round",
        "markeredgecolor": color,
        "markeredgewidth": 0.9,
        "markerfacecolor": "none",
        "markersize": DATA_MARKER_SIZE,
    }
    plot_kwargs.update(kwargs)
    (line,) = ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=linewidth,
        alpha=LINE_ALPHA,
        label=label,
        **plot_kwargs,
    )
    if gid is not None:
        line.set_gid(gid)
    return True


def _insert_timing_breaks(
    x: list[float],
    y: list[float | None],
    timing_methods: list[Any] | None,
) -> tuple[list[float], list[float | None]]:
    """Break a line when adjacent points use different timing boundaries."""
    if not timing_methods or len(timing_methods) != len(x) or len(y) != len(x):
        return x, y

    split_x: list[float] = []
    split_y: list[float | None] = []
    previous = None
    for x_value, y_value, timing_method in zip(x, y, timing_methods):
        current = str(timing_method or "")
        if previous is not None and current and previous and current != previous:
            split_x.append(math.nan)
            split_y.append(math.nan)
        split_x.append(x_value)
        split_y.append(y_value)
        previous = current
    return split_x, split_y


def _has_plottable_y(y: list[float | None]) -> bool:
    """Return True if at least one y-value will produce visible plot data."""
    for value in y:
        if value is None:
            continue
        try:
            if math.isfinite(float(value)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def add_vram_reference_lines(
    ax: Axes, unit: str = "MB", gpu_vram_gb: int | None = None
) -> None:
    """Add horizontal GPU VRAM reference line with label on the right edge.

    Parameters
    ----------
    gpu_vram_gb : int, optional
        Explicit VRAM in GB from artifact metadata or caller override. If
        None, no hardware-specific reference line is drawn.
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

_PLOT_MEASUREMENT_FIELDS = frozenset(
    {
        "time_us_per_atom",
        "throughput_atoms_per_sec",
        "mem_delta_mb",
        "mem_peak_gb",
        "time_d3_us_per_atom",
        "time_real_us_per_atom",
        "time_reciprocal_us_per_atom",
    }
)


def load_csv(filepath: str | Path) -> list[dict[str, Any]]:
    """Load benchmark CSV with automatic type conversion.

    Failed rows are retained with their NaN measurements so Matplotlib breaks
    lines at failed coordinates instead of connecting across them.
    """
    results = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                try:
                    val = row[key]
                    if val == "" and key in _PLOT_MEASUREMENT_FIELDS:
                        row[key] = math.nan
                    elif val in ("True", "False"):
                        row[key] = val == "True"
                    elif val.lower() in {"nan", "inf", "+inf", "-inf"}:
                        row[key] = float(val)
                    elif "." in val or "e" in val.lower():
                        row[key] = float(val)
                    elif val.lstrip("-").isdigit():
                        row[key] = int(val)
                except (ValueError, AttributeError):
                    pass
            results.append(row)
    return results


def _truthy_csv_value(value: Any) -> bool:
    """Return whether a parsed CSV field represents true."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _is_backend_comparison_row(row: dict[str, Any]) -> bool:
    """Return whether a successful row should enter backend comparison plots."""
    if row.get("success") is False:
        return False
    if row.get("timing_method") == "jax_wall_block_each":
        return False
    if "backend_comparable" in row and not _truthy_csv_value(
        row.get("backend_comparable")
    ):
        return False
    if "timing_scope" in row and str(row.get("timing_scope")) != "backend_comparison":
        return False
    return True


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

    renderer = {
        "nl": render_nl_panel,
        "d3": render_d3_panel,
        "el": render_el_panel,
    }[module]

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
    _savefig_atomic(fig, output_path)
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


def _filter_plot_rows(
    data: list[dict[str, Any]], filters: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply exact row filters, with tolerant float comparison."""
    filtered = data
    for key, expected in filters.items():
        if isinstance(expected, float):
            filtered = [
                row
                for row in filtered
                if math.isclose(
                    float(row.get(key, math.nan)),
                    expected,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ]
        elif isinstance(expected, (list, tuple, set, frozenset)):
            expected_values = tuple(expected)
            if all(isinstance(value, float) for value in expected_values):
                filtered = [
                    row
                    for row in filtered
                    if any(
                        math.isclose(
                            float(row.get(key, math.nan)),
                            value,
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        )
                        for value in expected_values
                    )
                ]
            else:
                filtered = [row for row in filtered if row.get(key) in expected_values]
        else:
            filtered = [row for row in filtered if row.get(key) == expected]
    return filtered


# =============================================================================
# Single-Panel API — for docs/generate_plots.py and Sphinx integration
# =============================================================================


def plot_single_panel(
    csv_path: str | Path,
    panel: str,
    output_path: str | Path,
    *,
    filters: dict[str, Any] | None = None,
    title_suffix: str | None = None,
) -> bool:
    """Render one panel (time/throughput/memory) from a CSV to an image.

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
        Where to save the image. PNG and SVG are supported.
    filters : dict, optional
        Exact row filters to apply before plotting. Float filters use
        :func:`math.isclose`, useful for cutoff-specific docs figures.
    title_suffix : str, optional
        Extra text appended to the panel title.

    Returns
    -------
    bool
        True when a plot image was written; False when the CSV contained no
        successful data or the module name was not recognized.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    data = load_csv(csv_path)
    if not data:
        return False

    explicit_backend = bool(filters and "backend" in filters)

    # Filter to torch backend for static plots unless the docs generator asks
    # for a specific backend panel such as the JAX tab images.
    backends = {r.get("backend", "torch") for r in data}
    if len(backends) > 1 and not explicit_backend:
        data = [r for r in data if r.get("backend", "torch") == "torch"]
    if filters:
        data = _filter_plot_rows(data, filters)
    if not data or not any(row.get("success") is not False for row in data):
        return False

    setup_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=SINGLE_PANEL_SIZE)

    system = data[0].get("system", "unknown")
    name = csv_path.stem

    # Detect module + mode from filename
    module, mode = detect_module_mode(name)

    # Failed/OOM rows remain as NaN coordinates so lines break at the failed
    # case instead of bridging it as if the intermediate point had succeeded.
    if module == "nl":
        render_nl_panel(ax, data, system, mode, panel, title_suffix=title_suffix)
    elif module == "d3":
        render_d3_panel(ax, data, system, mode, panel)
    elif module == "el":
        render_el_panel(ax, data, system, mode, panel)
    else:
        print(f"  Unknown module for {name}")
        plt.close()
        return False

    if title_suffix and module != "nl":
        ax.set_title(f"{ax.get_title()} - {title_suffix}")

    plt.tight_layout()
    _savefig_atomic(fig, output_path)
    plt.close()
    print(f"Saved: {output_path}")
    return True


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
    import matplotlib.ticker as ticker_sp
    import numpy as np

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


MODULE_NAMES = {
    "nl": "Neighbor List",
    "d3": "DFT-D3",
    "el": "Electrostatics",
}


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
        return "Batch Scaling"
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


def _nl_method_family(method: str) -> str:
    """Map compatibility aliases and canonical NL names to a strategy family."""
    method = method.replace("-", "_")
    if method.startswith("batch_"):
        method = method[len("batch_") :]
    if method in {"naive", "naive_neighbor_list"}:
        return "naive"
    if method in {"naive_scalar", "naive_tile"}:
        return method
    if method in {"cell", "cell_list"}:
        return "cell_list"
    if method in {"cell_list_atom_centric", "cell_list_pair_centric"}:
        return method
    if method == "cluster_tile" or "cluster" in method:
        return "cluster_tile"
    return method


def _nl_method_kind(method: str) -> str:
    """Return the coarse visual kind for marker and linestyle selection."""
    family = _nl_method_family(method)
    if family.startswith("naive"):
        return "naive"
    if family.startswith("cell"):
        return "cell"
    if family == "cluster_tile":
        return "cluster_tile"
    return family


def _nl_method_label(method: str) -> str:
    """Return a compact display label for an NL method."""
    family = _nl_method_family(method)
    exact_style = NL_METHOD_STYLES.get(family)
    if exact_style is not None:
        return str(exact_style["label"])
    labels = {
        "naive": "Naive",
        "naive_scalar": "Naive scalar",
        "naive_tile": "Naive tile",
        "cell_list": "Cell list",
        "cell_list_atom_centric": "Cell atom",
        "cell_list_pair_centric": "Cell pair",
        "cluster_tile": "Cluster tile",
    }
    if family in labels:
        return labels[family]
    return method.replace("_neighbor_list", "").replace("_", " ").title()


def _nl_line_style(method: str) -> str | tuple[int, tuple[int, int]]:
    """Return the method-specific line style used in NL plots."""
    kind = _nl_method_kind(method)
    family = _nl_method_family(method)
    exact_style = NL_METHOD_STYLES.get(family)
    if exact_style is not None:
        return exact_style["linestyle"]
    if kind == "naive":
        return SECONDARY_LINESTYLE if family != "naive_tile" else (0, (2, 1))
    if kind == "cluster_tile":
        return (0, (1, 2))
    if family == "cell_list_pair_centric":
        return (0, (5, 2))
    return "-"


def _nl_marker(method: str) -> str:
    """Return the method-specific marker used in NL plots."""
    family = _nl_method_family(method)
    exact_style = NL_METHOD_STYLES.get(family)
    if exact_style is not None:
        return str(exact_style["marker"])
    if family.startswith("naive"):
        return "s" if family != "naive_tile" else "D"
    if family == "cluster_tile":
        return "^"
    if family == "cell_list_pair_centric":
        return "P"
    return "o"


def _nl_method_color(method: str) -> str:
    """Return the stable color for an NL algorithm family."""
    exact_style = NL_METHOD_STYLES.get(_nl_method_family(method))
    if exact_style is not None:
        return str(exact_style["color"])
    return GRAY


def _nl_method_zorder(method: str) -> int:
    """Keep accelerated NL strategies visible above baseline lines."""
    exact_style = NL_METHOD_STYLES.get(_nl_method_family(method))
    return int(exact_style.get("zorder", 3)) if exact_style is not None else 3


def _nl_cutoff_style(cutoff: float) -> dict[str, Any]:
    """Return weight and marker-fill styling for one cutoff layer."""
    return NL_CUTOFF_STYLES.get(
        float(cutoff),
        {"fillstyle": DEFAULT_MARKER_FILLSTYLE, "linewidth_scale": 1.0},
    )


def _nl_cutoff_gid(cutoff: float, *parts: Any) -> str:
    """Build a stable SVG group id for a cutoff-specific data line."""
    cutoff_token = f"{float(cutoff):g}".replace(".", "p") + "A"
    suffix = "-".join(
        "".join(char if str(char).isalnum() else "-" for char in str(part)).strip("-")
        for part in parts
    )
    return f"nl-cutoff-{cutoff_token}-{suffix}"


def _nl_batch_series_rows(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select batch APIs for modes whose x-axis varies batch structure."""
    batch_rows = [
        row
        for row in data
        if str(row.get("method", "")).replace("-", "_").startswith("batch_")
    ]
    return batch_rows or data


def _finalize_panel(
    ax: Axes,
    panel: str,
    mode: str,
    x_ticks: list[int],
    x_limits: tuple[int, int],
    x_label: str,
    target_atoms: int | None = None,
) -> None:
    """Apply shared x-axis, y-axis, and grid styling to a panel."""
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


def _group_nl_by_method_cutoff(
    data: list[dict[str, Any]],
) -> dict[tuple[str, float], list[dict[str, Any]]]:
    """Group NL rows by (method, cutoff) tuple."""
    grouped = defaultdict(list)
    for r in data:
        grouped[(r["method"], r.get("cutoff", DEFAULT_CUTOFF))].append(r)
    return dict(grouped)


def _render_nl_batch(
    ax: Axes,
    data: list[dict[str, Any]],
    panel: str,
) -> tuple[str, str]:
    """Render NL batch scaling without joining distinct cutoff series.

    Returns the x label and mode title.
    """
    data = _nl_batch_series_rows(data)
    grouped = defaultdict(list)
    for r in data:
        grouped[
            (
                r["method"],
                r.get("cutoff", DEFAULT_CUTOFF),
                r["atoms_per_system"],
            )
        ].append(r)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda r: r["batch_size"])

    methods = sorted({r["method"] for r in data}, key=_nl_method_label)
    cutoffs = sorted({r.get("cutoff", DEFAULT_CUTOFF) for r in data})
    aps_values = sorted({r["atoms_per_system"] for r in data})

    for method, cutoff, aps in sorted(
        grouped.keys(), key=lambda key: (key[1], key[2], key[0])
    ):
        rows = grouped[(method, cutoff, aps)]
        x = [r["total_atoms"] for r in rows]
        y = _get_panel_y(rows, panel)
        color = _nl_method_color(method)
        cutoff_style = _nl_cutoff_style(cutoff)
        if len(aps_values) > 1:
            aps_scale = 1.14 if aps == max(aps_values) else 0.86
        else:
            aps_scale = 1.0
        _plot_data_line(
            ax,
            x,
            y,
            color=color,
            linestyle=_nl_line_style(method),
            marker=_nl_marker(method),
            label="_nolegend_",
            linewidth=DATA_LINE_WIDTH * cutoff_style["linewidth_scale"],
            markersize=DATA_MARKER_SIZE * aps_scale,
            markerfacecolor=color,
            markerfacecoloralt="white",
            fillstyle=cutoff_style["fillstyle"],
            zorder=_nl_method_zorder(method),
            timing_methods=[r.get("timing_method") for r in rows],
            gid=_nl_cutoff_gid(cutoff, method, f"N{aps}", panel),
        )

    _create_nl_dimension_legend(ax, methods, cutoffs, aps_values)

    return "Total Atoms", "Batch Scaling"


def _create_dimension_legend(
    ax: Axes,
    columns: list[tuple[str, list[tuple[Any, str]]]],
) -> None:
    """Render the shared columnar legend used by all reportable plots."""
    from matplotlib.lines import Line2D

    columns = [(header, items) for header, items in columns if items]
    if not columns:
        return

    invisible = Line2D([], [], color="none", marker="None", linestyle="None")
    rows_per_column = max(len(items) for _, items in columns) + 1
    handles = []
    labels = []
    header_indices = []
    for header, items in columns:
        header_indices.append(len(labels))
        handles.append(invisible)
        labels.append(header)
        for handle, label in items:
            handles.append(handle)
            labels.append(label)
        while len(labels) % rows_per_column:
            handles.append(invisible)
            labels.append("")

    legend = ax.legend(
        handles,
        labels,
        ncol=len(columns),
        loc="best",
        prop={"family": "DejaVu Sans Mono", "size": LEGEND_SIZE - 0.5},
        handlelength=2.0,
        handletextpad=0.55,
        columnspacing=0.9,
        labelspacing=0.3,
        borderpad=0.5,
        frameon=True,
        fancybox=True,
        framealpha=0.88,
        edgecolor="#D8D8D8",
        facecolor="white",
    )
    for index in header_indices:
        legend.get_texts()[index].set_color("#666666")
        legend.get_texts()[index].set_weight("bold")


def _backend_legend_items(backends: list[str] | None) -> list[tuple[Any, str]]:
    """Return backend line-pattern keys for a dimension legend."""
    from matplotlib.lines import Line2D

    return [
        (
            Line2D(
                [],
                [],
                color=GRAY,
                linestyle=BACKEND_LINESTYLES.get(backend, "-"),
                marker="None",
                linewidth=DATA_LINE_WIDTH,
                dash_capstyle="round",
                alpha=LINE_ALPHA,
            ),
            backend,
        )
        for backend in backends or []
    ]


def _system_size_legend_items(
    aps_values: list[int] | None,
) -> list[tuple[Any, str]]:
    """Return marker-size keys for batch system sizes."""
    from matplotlib.lines import Line2D

    items = []
    for aps in aps_values or []:
        if len(aps_values or []) > 1:
            marker_scale = 1.14 if aps == max(aps_values or []) else 0.86
        else:
            marker_scale = 1.0
        items.append(
            (
                Line2D(
                    [],
                    [],
                    color=GRAY,
                    marker="o",
                    linestyle="None",
                    markersize=DATA_MARKER_SIZE * marker_scale,
                    markerfacecolor="none",
                    markeredgecolor=GRAY,
                    markeredgewidth=0.9,
                    fillstyle=DEFAULT_MARKER_FILLSTYLE,
                    alpha=LINE_ALPHA,
                ),
                f"N={format_num(aps)}",
            )
        )
    return items


def _batch_system_style(
    atoms_per_system: int,
    aps_values: list[int],
) -> dict[str, Any]:
    """Return an explicit line-and-marker style for one batch system size."""
    ordered = sorted(set(aps_values))
    try:
        index = ordered.index(atoms_per_system)
    except ValueError:
        index = 0
    return BATCH_SYSTEM_STYLES[index % len(BATCH_SYSTEM_STYLES)]


def _batch_system_legend_items(
    aps_values: list[int] | None,
    *,
    include_line: bool,
) -> list[tuple[Any, str]]:
    """Return readable batch-system keys using line and marker shape."""
    from matplotlib.lines import Line2D

    items = []
    values = sorted(set(aps_values or []))
    for aps in values:
        style = _batch_system_style(aps, values)
        items.append(
            (
                Line2D(
                    [],
                    [],
                    color=GRAY,
                    linestyle=style["linestyle"] if include_line else "None",
                    marker=style["marker"],
                    linewidth=DATA_LINE_WIDTH,
                    markersize=DATA_MARKER_SIZE,
                    markerfacecolor="none",
                    markeredgecolor=GRAY,
                    markeredgewidth=0.9,
                    fillstyle=DEFAULT_MARKER_FILLSTYLE,
                    dash_capstyle="round",
                    alpha=LINE_ALPHA,
                ),
                f"N={format_num(aps)}",
            )
        )
    return items


def _create_nl_dimension_legend(
    ax: Axes,
    methods: list[str],
    cutoffs: list[float],
    aps_values: list[int] | None = None,
    backends: list[str] | None = None,
) -> None:
    """Add compact keys for NL method, cutoff, backend, and system size."""
    from matplotlib.lines import Line2D

    method_items = [
        (
            Line2D(
                [],
                [],
                color=_nl_method_color(method),
                linestyle="-" if backends else _nl_line_style(method),
                marker=_nl_marker(method),
                linewidth=DATA_LINE_WIDTH,
                markersize=DATA_MARKER_SIZE,
                markerfacecolor=_nl_method_color(method),
                markeredgecolor=_nl_method_color(method),
                markeredgewidth=0.9,
                fillstyle=DEFAULT_MARKER_FILLSTYLE,
                dash_capstyle="round",
                alpha=LINE_ALPHA,
            ),
            _nl_method_label(method),
        )
        for method in methods
    ]
    cutoff_items = []
    for cutoff in cutoffs:
        cutoff_style = _nl_cutoff_style(cutoff)
        cutoff_items.append(
            (
                Line2D(
                    [],
                    [],
                    color=GRAY,
                    linestyle="-",
                    marker="o",
                    linewidth=DATA_LINE_WIDTH * cutoff_style["linewidth_scale"],
                    markersize=DATA_MARKER_SIZE,
                    markerfacecolor=GRAY,
                    markerfacecoloralt="white",
                    markeredgecolor=GRAY,
                    markeredgewidth=0.9,
                    fillstyle=cutoff_style["fillstyle"],
                    dash_capstyle="round",
                    alpha=LINE_ALPHA,
                ),
                f"{cutoff:g}Å",
            )
        )
    _create_dimension_legend(
        ax,
        [
            ("Method", method_items),
            ("Backend", _backend_legend_items(backends)),
            ("Cutoff", cutoff_items),
            ("System", _system_size_legend_items(aps_values)),
        ],
    )


def _el_method_family(method: str) -> str:
    """Return the stable electrostatics method family for plotting."""
    normalized = method.lower().replace("-", "_")
    if "pme" in normalized or "particle_mesh" in normalized:
        return "pme"
    if "ewald" in normalized:
        return "ewald"
    return normalized


def _el_method_style(method: str) -> dict[str, Any]:
    """Return color, line, marker, and label for an EL method."""
    family = _el_method_family(method)
    return EL_METHOD_STYLES.get(
        family,
        {
            "label": family.replace("_", " ").title(),
            "color": GRAY,
            "linestyle": "-",
            "marker": "o",
        },
    )


def _el_accuracy_style(accuracy: float) -> dict[str, Any]:
    """Return the secondary marker-fill and line-weight accuracy style."""
    for configured, style in EL_ACCURACY_STYLES.items():
        if math.isclose(float(accuracy), configured, rel_tol=1e-9, abs_tol=1e-12):
            return style
    return {"fillstyle": DEFAULT_MARKER_FILLSTYLE, "linewidth_scale": 1.0}


def _create_el_dimension_legend(
    ax: Axes,
    methods: list[str],
    accuracies: list[float] | None = None,
    aps_values: list[int] | None = None,
    backends: list[str] | None = None,
) -> None:
    """Add method, accuracy, backend, and system-size keys for EL."""
    from matplotlib.lines import Line2D

    method_items = []
    for method in methods:
        style = _el_method_style(method)
        method_items.append(
            (
                Line2D(
                    [],
                    [],
                    color=style["color"],
                    linestyle="-" if backends else style["linestyle"],
                    marker="None" if aps_values else style["marker"],
                    linewidth=DATA_LINE_WIDTH,
                    markersize=DATA_MARKER_SIZE,
                    markerfacecolor=style["color"],
                    markeredgecolor=style["color"],
                    markeredgewidth=0.9,
                    fillstyle=DEFAULT_MARKER_FILLSTYLE,
                    dash_capstyle="round",
                    alpha=LINE_ALPHA,
                ),
                str(style["label"]),
            )
        )

    accuracy_items = []
    for accuracy in accuracies or []:
        layer_style = _el_accuracy_style(accuracy)
        accuracy_items.append(
            (
                Line2D(
                    [],
                    [],
                    color=GRAY,
                    linestyle="-",
                    marker="o",
                    linewidth=DATA_LINE_WIDTH * layer_style["linewidth_scale"],
                    markersize=DATA_MARKER_SIZE,
                    markerfacecolor=GRAY,
                    markerfacecoloralt="white",
                    markeredgecolor=GRAY,
                    markeredgewidth=0.9,
                    fillstyle=layer_style["fillstyle"],
                    dash_capstyle="round",
                    alpha=LINE_ALPHA,
                ),
                format_accuracy(accuracy),
            )
        )

    _create_dimension_legend(
        ax,
        [
            ("Method", method_items),
            ("Backend", _backend_legend_items(backends)),
            ("Accuracy", accuracy_items),
            (
                "System",
                _batch_system_legend_items(
                    aps_values,
                    include_line=not bool(backends),
                ),
            ),
        ],
    )


def _create_d3_dimension_legend(
    ax: Axes,
    cutoffs: list[float],
    aps_values: list[int] | None = None,
    backends: list[str] | None = None,
) -> None:
    """Add cutoff, backend, and system-size keys for D3."""
    from matplotlib.lines import Line2D

    cutoff_items = []
    for cutoff in cutoffs:
        color = D3_CUTOFF_COLORS.get(cutoff, GRAY)
        style = D3_CUTOFF_STYLES.get(
            cutoff,
            {
                "marker": "o",
                "linestyle": "-",
                "fillstyle": DEFAULT_MARKER_FILLSTYLE,
            },
        )
        cutoff_items.append(
            (
                Line2D(
                    [],
                    [],
                    color=color,
                    linestyle="-" if backends or aps_values else style["linestyle"],
                    marker="o" if aps_values else style["marker"],
                    linewidth=DATA_LINE_WIDTH,
                    markersize=DATA_MARKER_SIZE,
                    markerfacecolor=color,
                    markerfacecoloralt="white",
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                    fillstyle=style["fillstyle"],
                    dash_capstyle="round",
                    alpha=LINE_ALPHA,
                ),
                "D3" if backends and len(cutoffs) == 1 else f"{cutoff:g}Å",
            )
        )

    _create_dimension_legend(
        ax,
        [
            ("Method" if backends and len(cutoffs) == 1 else "Cutoff", cutoff_items),
            ("Backend", _backend_legend_items(backends)),
            (
                "System",
                _batch_system_legend_items(
                    aps_values,
                    include_line=not bool(backends),
                ),
            ),
        ],
    )


def _render_nl_by_cutoff(
    ax: Axes,
    data: list[dict[str, Any]],
    panel: str,
    x_key: str,
) -> None:
    """Render NL system-size or constant-workload mode.

    Handles both modes — they differ only in which CSV field is the x-axis
    (``total_atoms`` for system_size, ``atoms_per_system`` for constant_workload).
    """
    grouped = _group_nl_by_method_cutoff(data)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda r: r[x_key])

    for method, cutoff in sorted(grouped.keys(), key=lambda k: (k[1], k[0])):
        rows = grouped[(method, cutoff)]
        x = [r[x_key] for r in rows]
        y = _get_panel_y(rows, panel)
        color = _nl_method_color(method)
        cutoff_style = _nl_cutoff_style(cutoff)
        _plot_data_line(
            ax,
            x,
            y,
            color=color,
            linestyle=_nl_line_style(method),
            marker=_nl_marker(method),
            label="_nolegend_",
            linewidth=DATA_LINE_WIDTH * cutoff_style["linewidth_scale"],
            markerfacecolor=color,
            markerfacecoloralt="white",
            fillstyle=cutoff_style["fillstyle"],
            zorder=_nl_method_zorder(method),
            timing_methods=[r.get("timing_method") for r in rows],
            gid=_nl_cutoff_gid(cutoff, method, panel),
        )

    methods = sorted({method for method, _ in grouped}, key=_nl_method_label)
    cutoffs = sorted({cutoff for _, cutoff in grouped})
    _create_nl_dimension_legend(ax, methods, cutoffs)


def render_nl_panel(
    ax: Axes,
    data: list[dict[str, Any]],
    system: str,
    mode: str,
    panel: str,
    title_suffix: str | None = None,
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
    title_suffix : str, optional
        Extra text appended to the panel title.

    Notes
    -----
    NL methods are kept separate in every panel because scalar/tile and
    atom-/pair-centric paths are user-facing benchmark methods.
    """
    target = None

    if mode == "batch_scaling":
        x_label, mode_str = _render_nl_batch(ax, data, panel)
    elif mode == "constant_workload":
        target = (
            data[0].get("total_atoms", DEFAULT_TOTAL_ATOMS)
            if data
            else DEFAULT_TOTAL_ATOMS
        )
        _render_nl_by_cutoff(
            ax,
            _nl_batch_series_rows(data),
            panel,
            "atoms_per_system",
        )
        x_label = "Atoms per System [\u00d7batch]"
        mode_str = f"Constant {format_num(target)} Atoms"
    else:
        _render_nl_by_cutoff(ax, data, panel, "total_atoms")
        x_label = "System Size (atoms)"
        mode_str = "System Size Scaling"

    ax.set_title(
        _build_panel_title("nl", system, mode_str, suffix=title_suffix),
        fontsize=TITLE_SIZE,
    )

    _finalize_panel(
        ax,
        panel,
        mode,
        X_AXIS_TICKS_MEDIUM,
        X_AXIS_LIMITS["medium"],
        x_label,
        target_atoms=target if mode == "constant_workload" else None,
    )


def render_d3_panel(
    ax: Axes,
    data: list[dict[str, Any]],
    system: str,
    mode: str,
    panel: str,
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
    aps_values: list[int] = []
    plotted_cutoffs: set[float] = set()

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
        # Match NL's hierarchy: cutoff keeps its color/marker while system size
        # changes only marker scale.
        aps_values = sorted({r["atoms_per_system"] for r in data})

        for key in sorted(grouped.keys()):
            rows = grouped[key]
            cutoff, aps = key
            x = [r[x_key] for r in rows]
            y_raw = [r.get("time_d3_us_per_atom", r["time_us_per_atom"]) for r in rows]
            y = _get_panel_y_from_raw(rows, panel, y_raw)
            color = D3_CUTOFF_COLORS.get(cutoff, GRAY)
            style = D3_CUTOFF_STYLES.get(
                cutoff,
                {
                    "marker": "o",
                    "linestyle": "-",
                    "fillstyle": DEFAULT_MARKER_FILLSTYLE,
                },
            )
            system_style = _batch_system_style(aps, aps_values)
            if _plot_data_line(
                ax,
                x,
                y,
                color=color,
                label="_nolegend_",
                linestyle=system_style["linestyle"],
                marker=system_style["marker"],
                markerfacecolor=color,
                markerfacecoloralt="white",
                fillstyle=style["fillstyle"],
                timing_methods=[r.get("timing_method") for r in rows],
            ):
                plotted_cutoffs.add(float(cutoff))
    else:
        # System size / constant workload: color by cutoff (D3 palette)
        for key in sorted(grouped.keys()):
            rows = grouped[key]
            cutoff = key
            x = [r[x_key] for r in rows]
            y_raw = [r.get("time_d3_us_per_atom", r["time_us_per_atom"]) for r in rows]
            y = _get_panel_y_from_raw(rows, panel, y_raw)
            color = D3_CUTOFF_COLORS.get(cutoff, GRAY)
            style = D3_CUTOFF_STYLES.get(
                cutoff,
                {
                    "marker": "o",
                    "linestyle": "-",
                    "fillstyle": DEFAULT_MARKER_FILLSTYLE,
                },
            )
            if _plot_data_line(
                ax,
                x,
                y,
                color=color,
                label="_nolegend_",
                markerfacecolor=color,
                timing_methods=[r.get("timing_method") for r in rows],
                **style,
            ):
                plotted_cutoffs.add(float(cutoff))

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
    )
    _create_d3_dimension_legend(
        ax,
        sorted(plotted_cutoffs),
        aps_values if is_batch else None,
    )


def render_el_panel(
    ax: Axes,
    data: list[dict[str, Any]],
    system: str,
    mode: str,
    panel: str,
) -> None:
    """Render a single Electrostatics panel onto *ax*.

    Method identity is stable across panels: PME and Ewald keep their color,
    line pattern, and marker at every accuracy. Accuracy is a secondary layer
    expressed by marker fill and restrained line weight. Every row uses the
    same energy, force, and charge-gradient workload.

    Parameters
    ----------
    ax, data, system, mode, panel : same as :func:`render_nl_panel`
    """
    target = None

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

    # Match NL's hierarchy in every mode: method identity is primary, accuracy
    # is a secondary layer, and batch system size changes marker scale only.
    aps_values = sorted({r["atoms_per_system"] for r in data}) if is_batch else []
    plotted_keys: list[tuple[str, float, int | None]] = []

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
            system_style = _batch_system_style(aps, aps_values)
            marker = system_style["marker"]
            linestyle = system_style["linestyle"]
        else:
            method, accuracy = key
            aps = None

        method_style = _el_method_style(method)
        if not is_batch:
            marker = method_style["marker"]
            linestyle = method_style["linestyle"]
        accuracy_style = _el_accuracy_style(float(accuracy))
        if _plot_data_line(
            ax,
            x,
            y,
            color=method_style["color"],
            marker=marker,
            linestyle=linestyle,
            linewidth=DATA_LINE_WIDTH * accuracy_style["linewidth_scale"],
            label="_nolegend_",
            markerfacecolor=method_style["color"],
            markerfacecoloralt="white",
            fillstyle=accuracy_style["fillstyle"],
            timing_methods=[r.get("timing_method") for r in rows],
        ):
            plotted_keys.append((method, float(accuracy), aps))

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
    )
    methods = sorted(
        {_el_method_family(method) for method, _, _ in plotted_keys},
        key=lambda method: str(_el_method_style(method)["label"]),
    )
    accuracies = sorted({accuracy for _, accuracy, _ in plotted_keys})
    plotted_aps = sorted({aps for _, _, aps in plotted_keys if aps is not None})
    _create_el_dimension_legend(
        ax,
        methods,
        accuracies,
        plotted_aps if is_batch else None,
    )


def _get_memory_y(rows: list[dict[str, Any]]) -> list[float | None]:
    """Extract memory y-values, handling JAX XLA pool unreliability.

    JAX memory fields are intentionally reported as NaN because XLA
    pre-allocates a pool and reuses it.  Return ``None`` for JAX rows so
    matplotlib skips them instead of plotting allocator noise.
    """
    backend = rows[0].get("backend", "torch") if rows else "torch"
    if backend == "jax":
        return [None for _ in rows]
    values = []
    for r in rows:
        v = r["mem_peak_gb"]
        values.append(v * 1024 if math.isfinite(v) and v > 0 else None)
    return values


def _get_panel_y(rows: list[dict[str, Any]], panel: str) -> list[float | None]:
    """Extract y-values for a given panel type.

    Parameters
    ----------
    rows : list[dict]
        CSV rows for one data series.
    panel : str
        Panel type: "time" (us/atom), "throughput" (Matoms/s), or "memory" (MB).
        For JAX memory, returns ``None`` values because XLA pool allocation
        makes per-call memory attribution misleading.
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


def _filter_grouped_to_matched_backend_x(
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]],
    x_key: str,
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """Keep only exact x-values where both Torch and JAX succeeded."""
    x_by_comparison: dict[tuple[Any, ...], dict[str, set[Any]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for key, rows in grouped.items():
        method = key[0]
        backend = key[1]
        rest = key[2:]
        comparison_key = (method, *rest)
        x_by_comparison[comparison_key][backend].update(row.get(x_key) for row in rows)

    matched: dict[tuple[Any, ...], set[Any]] = {}
    for comparison_key, backend_x in x_by_comparison.items():
        if not {"torch", "jax"}.issubset(backend_x):
            continue
        shared = backend_x["torch"] & backend_x["jax"]
        if shared:
            matched[comparison_key] = shared

    filtered: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for key, rows in grouped.items():
        method = key[0]
        backend = key[1]
        rest = key[2:]
        comparison_key = (method, *rest)
        if backend not in {"torch", "jax"} or comparison_key not in matched:
            continue
        kept_rows = [row for row in rows if row.get(x_key) in matched[comparison_key]]
        if kept_rows:
            filtered[key] = kept_rows
    return filtered


def plot_comparison_panel(
    csv_path: str | Path,
    panel: str,
    output_path: str | Path,
    module: str,
    fixed_param: float | None = None,
    *,
    layer_all_params: bool = False,
) -> bool:
    """Render one panel with both backends overlaid.

    Scientific method or parameter color remains primary while backend line
    pattern is secondary. NL can retain every cutoff as independently
    switchable SVG layers; D3 and EL use one fixed parameter per comparison.

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
    layer_all_params : bool
        Keep every NL cutoff and assign cutoff-specific SVG group IDs. This is
        used by the interactive documentation comparison panels.

    Returns
    -------
    bool
        True when an image was written, otherwise False.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    all_data = [r for r in load_csv(csv_path) if _is_backend_comparison_row(r)]
    if module == "nl":
        all_data = [
            row
            for row in all_data
            if _nl_method_family(str(row.get("method", "")))
            in NL_BACKEND_COMPARABLE_FAMILIES
        ]
    if not all_data:
        return False

    if layer_all_params and module != "nl":
        raise ValueError("layer_all_params is supported only for NL comparisons")

    # Defaults per module
    if fixed_param is None and not layer_all_params:
        fixed_param = {"nl": DEFAULT_CUTOFF, "d3": DEFAULT_CUTOFF, "el": 1e-6}.get(
            module, DEFAULT_CUTOFF
        )
    if not layer_all_params and fixed_param is None:
        raise ValueError("A fixed comparison parameter is required")

    # Filter to a fixed parameter unless the NL page needs interactive layers.
    if layer_all_params:
        data = all_data
    elif module in ("nl", "d3"):
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
    else:
        data = all_data

    if not data:
        print(f"  SKIP comparison: no data after filtering to {fixed_param}")
        return False

    # Must have both backends
    backends = {r.get("backend", "torch") for r in data}
    if len(backends) < 2:
        print(f"  SKIP comparison: only {backends} present (need both torch+jax)")
        return False

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

    if module == "nl" and mode in {"constant_workload", "batch_scaling"}:
        data = _nl_batch_series_rows(data)

    # Keep the scientific parameter in the comparison key so backend matching
    # never crosses cutoff or accuracy layers.
    is_batch = mode == "batch_scaling"
    grouped = defaultdict(list)
    for r in data:
        method_key = r.get("method", "unknown")
        parameter = (
            r.get("accuracy", 1e-6)
            if module == "el"
            else r.get("cutoff", DEFAULT_CUTOFF)
        )
        atoms_per_system = r.get("atoms_per_system", 0) if is_batch else None
        key = (
            method_key,
            r.get("backend", "torch"),
            parameter,
            atoms_per_system,
        )
        grouped[key].append(r)

    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda r: r.get(x_key, 0))
    grouped = _filter_grouped_to_matched_backend_x(grouped, x_key)
    if not grouped:
        print("  SKIP comparison: no matched successful torch/jax x-values")
        plt.close(fig)
        return False

    # Use the NL hierarchy across every comparison panel: scientific color is
    # primary, backend line pattern is secondary, and marker shape identifies
    # the concrete method. Batch system size only changes marker scale.
    aps_values = sorted({key[3] for key in grouped if key[3] is not None})

    plotted_keys: list[tuple[str, str, float, int | None]] = []
    for key in sorted(grouped.keys()):
        method, backend, parameter, aps = key
        rows = grouped[key]
        x = [r[x_key] for r in rows]

        # Get y values
        if module == "d3":
            y_raw = [r.get("time_d3_us_per_atom", r["time_us_per_atom"]) for r in rows]
            y = _get_panel_y_from_raw(rows, panel, y_raw)
        else:
            y = _get_panel_y(rows, panel)

        extra_kwargs: dict[str, Any] = {}
        gid = None
        if module == "nl":
            color = _nl_method_color(method)
            marker = _nl_marker(method)
            cutoff_style = _nl_cutoff_style(parameter)
            linewidth = DATA_LINE_WIDTH * cutoff_style["linewidth_scale"]
            extra_kwargs.update(
                {
                    "fillstyle": cutoff_style["fillstyle"],
                    "markerfacecolor": color,
                    "markerfacecoloralt": "white",
                    "zorder": _nl_method_zorder(method),
                }
            )
            gid = _nl_cutoff_gid(parameter, method, backend, aps or "single", panel)
        elif module == "el":
            method_style = _el_method_style(method)
            accuracy_style = _el_accuracy_style(float(parameter))
            color = method_style["color"]
            marker = method_style["marker"]
            linewidth = DATA_LINE_WIDTH
            extra_kwargs.update(
                {
                    "fillstyle": accuracy_style["fillstyle"],
                    "markerfacecolor": color,
                    "markerfacecoloralt": "white",
                }
            )
        elif module == "d3":
            color = D3_CUTOFF_COLORS.get(parameter, GRAY)
            cutoff_style = D3_CUTOFF_STYLES.get(
                parameter,
                {
                    "marker": "o",
                    "linestyle": "-",
                    "fillstyle": DEFAULT_MARKER_FILLSTYLE,
                },
            )
            marker = cutoff_style["marker"]
            linewidth = DATA_LINE_WIDTH
            extra_kwargs.update(
                {
                    "fillstyle": cutoff_style["fillstyle"],
                    "markerfacecolor": color,
                    "markerfacecoloralt": "white",
                }
            )
        else:
            color = GRAY
            marker = "o"
            linewidth = DATA_LINE_WIDTH

        linestyle = BACKEND_LINESTYLES.get(backend, "-")
        if is_batch and aps is not None and module in {"d3", "el"}:
            marker = _batch_system_style(aps, aps_values)["marker"]
        elif is_batch and aps is not None:
            marker_scale = (
                1.14
                if len(aps_values) > 1 and aps == max(aps_values)
                else 0.86
                if len(aps_values) > 1
                else 1.0
            )
            extra_kwargs["markersize"] = DATA_MARKER_SIZE * marker_scale

        if _plot_data_line(
            ax,
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            label="_nolegend_",
            timing_methods=[r.get("timing_method") for r in rows],
            gid=gid,
            **extra_kwargs,
        ):
            plotted_keys.append((method, str(backend), float(parameter), aps))

    if not plotted_keys:
        print(f"  SKIP comparison: no plottable {panel} data")
        plt.close(fig)
        return False

    # Title
    param_label = None
    if not layer_all_params:
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
        else "Total Atoms"
        if mode == "batch_scaling"
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
    )
    plotted_aps = sorted({aps for _, _, _, aps in plotted_keys if aps is not None})
    backend_order = {"torch": 0, "jax": 1, "warp": 2}
    plotted_backends = sorted(
        {backend for _, backend, _, _ in plotted_keys},
        key=lambda backend: (backend_order.get(backend, len(backend_order)), backend),
    )
    if module == "nl":
        methods = sorted({key[0] for key in plotted_keys}, key=_nl_method_label)
        cutoffs = sorted({key[2] for key in plotted_keys})
        _create_nl_dimension_legend(
            ax,
            methods,
            cutoffs,
            plotted_aps,
            plotted_backends,
        )
    elif module == "el":
        methods = sorted(
            {_el_method_family(key[0]) for key in plotted_keys},
            key=lambda method: str(_el_method_style(method)["label"]),
        )
        _create_el_dimension_legend(
            ax,
            methods,
            aps_values=plotted_aps,
            backends=plotted_backends,
        )
    elif module == "d3":
        _create_d3_dimension_legend(
            ax,
            sorted({key[2] for key in plotted_keys}),
            plotted_aps,
            plotted_backends,
        )

    plt.tight_layout()
    _savefig_atomic(fig, output_path)
    plt.close(fig)
    print(f"  Comparison: {output_path}")
    return True


def generate_comparison_panels(csv_dir: str | Path, output_dir: str | Path) -> None:
    """Generate all backend comparison PNGs from CSVs with both backends."""
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)

    for csv_path in sorted(csv_dir.glob("*.csv")):
        # Skip separate failure sidecars if a non-suite results directory
        # contains them. Current runs keep failures in the main CSV rows.
        if csv_path.stem.endswith("-failures"):
            continue
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


def detect_and_plot(csv_path: Path, output_dir: str | Path) -> bool:
    """Detect CSV type from filename and plot accordingly.

    Supports both old naming (nl_cscl_system_size_gpu.csv) and
    new naming (nl-cscl-system-size-scaling.csv).
    """
    data = load_csv(csv_path)
    if not data:
        return False

    # Matplotlib static plots show torch backend only (JAX shown in Plotly interactive).
    # If both backends present, filter to torch for cleaner static plots.
    backends = {r.get("backend", "torch") for r in data}
    if len(backends) > 1:
        data = [r for r in data if r.get("backend", "torch") == "torch"]
    if not any(row.get("success") is not False for row in data):
        return False

    system = data[0].get("system", "unknown")
    name = csv_path.stem

    module, mode = detect_module_mode(name)
    if module and mode:
        plot_module(data, system, mode, module, output_dir)
        return True
    return False


def main() -> None:
    """CLI entry point for benchmark plotting."""
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("input_dir", type=Path, help="Directory with CSV files")
    parser.add_argument("--output-dir", "-o", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
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
