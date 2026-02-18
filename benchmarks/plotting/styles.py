# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot styling constants and utilities for NVIDIA branding.

Carries over all styling from benchmarks-original/plotting/styles.py:
- NVIDIA green color scheme
- Log base 2 x-axis that looks linear (evenly spaced power-of-2 ticks)
- Publication-quality font sizes and figure dimensions
- O(N)/O(N²) scaling reference lines
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# NVIDIA Color Scheme
# =============================================================================

NVIDIA_GREEN = "#76B900"
DARK_GREEN = "#4A7A00"
DARKEST_GREEN = "#2D4A00"
LIGHT_GREEN = "#A3D550"

GRAY = "#555555"
LIGHT_GRAY = "#888888"
DARK_GRAY = "#333333"

# Method colors (NL)
METHOD_COLORS = {
    "naive": NVIDIA_GREEN,
    "cell": DARK_GREEN,
    "cell_list": DARK_GREEN,
    "batch_naive": NVIDIA_GREEN,
    "batch_cell_list": DARK_GREEN,
    # Electrostatics
    "pme": NVIDIA_GREEN,
    "ewald": DARK_GREEN,
    "pme_cg": LIGHT_GREEN,
    "ewald_cg": DARKEST_GREEN,
    # D3
    "dftd3": NVIDIA_GREEN,
}

# Method line styles
METHOD_STYLES = {
    "naive": {"marker": "o", "linestyle": "-", "linewidth": 2},
    "cell": {"marker": "^", "linestyle": "-", "linewidth": 2},
    "cell_list": {"marker": "^", "linestyle": "-", "linewidth": 2},
    "batch_naive": {"marker": "s", "linestyle": "--", "linewidth": 2},
    "batch_cell_list": {"marker": "D", "linestyle": "--", "linewidth": 2},
    "pme": {"marker": "o", "linestyle": "-", "linewidth": 2},
    "ewald": {"marker": "^", "linestyle": "-", "linewidth": 2},
    "pme_cg": {"marker": "s", "linestyle": "--", "linewidth": 2},
    "ewald_cg": {"marker": "D", "linestyle": "--", "linewidth": 2},
    "dftd3": {"marker": "o", "linestyle": "-", "linewidth": 2},
}

# Cutoff-specific colors: NVIDIA green for smallest cutoff in each module
# NL uses 6/15/25Å, D3 uses 15/25Å — 15Å gets green when 6Å absent
CUTOFF_COLORS = {
    6.0: NVIDIA_GREEN,  # NVIDIA green baseline
    15.0: "#31688E",  # Viridis mid-blue (or green when 6Å absent)
    25.0: "#440154",  # Viridis deep purple
}

# D3-specific colors: 15Å gets green since no 6Å
# With only 2 lines, use green + blue; purple reserved for 3+ lines
D3_CUTOFF_COLORS = {
    15.0: NVIDIA_GREEN,  # NVIDIA green for smallest D3 cutoff
    25.0: "#31688E",  # Blue (purple reserved for 3+ lines)
}

# GPU VRAM reference lines for memory panels (keep minimal)
GPU_VRAM_REFS = {
    "H100": 80,  # GB
}

CUTOFF_STYLES = {
    6.0: {"marker": "o", "linestyle": "-"},
    15.0: {"marker": "^", "linestyle": "-"},
    25.0: {"marker": "s", "linestyle": "--"},
}

# Grid style matching benchmarks-temp/
GRID_STYLE = dict(color="#A0A0A0", linestyle="--", linewidth=0.3)

# Batch size markers
BATCH_MARKERS = ["o", "^", "s", "D", "v", "<", ">", "p", "*", "h"]
BATCH_LINESTYLES = ["-", "--", "-.", ":"]

# =============================================================================
# Font Sizes
# =============================================================================

TITLE_SIZE = 14
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 10
LEGEND_SIZE = 9
ANNOTATION_SIZE = 8

# =============================================================================
# Figure Sizes
# =============================================================================

SINGLE_PLOT_SIZE = (8, 6)
TWO_COLUMN_SIZE = (12, 5)
THREE_PANEL_SIZE = (14, 4.5)  # Standard: 3 panels in a row
FOUR_PANEL_SIZE = (14, 10)

# =============================================================================
# X-Axis Configuration (Powers of 2, log base 2)
# =============================================================================

X_AXIS_TICKS_SMALL = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
X_AXIS_TICKS_MEDIUM = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
]
X_AXIS_TICKS_LARGE = [1024, 4096, 16384, 65536, 262144, 1048576]

X_AXIS_LIMITS = {
    "small": (50, 25000),
    "medium": (90, 180000),
    "large": (700, 1500000),
}


# =============================================================================
# Utilities
# =============================================================================


def format_num(n):
    """Format atom count: 1024->'1k', 131072->'128k', 1048576->'1M'."""
    if isinstance(n, float):
        n = int(n)
    if n >= 1048576:
        return f"{n // 1048576}M"
    elif n >= 1024:
        return f"{n // 1024}k"
    return str(n)


def format_system_name(name):
    """Format system name with chemical conventions (subscripts via mathtext).

    'nh3' -> 'NH$_3$', 'cscl' -> 'CsCl'
    """
    chem_names = {
        "nh3": r"NH$_3$",
        "cscl": "CsCl",
    }
    return chem_names.get(name.lower(), name.upper())


def setup_plot_style():
    """Configure matplotlib for publication-quality plots with NVIDIA branding."""
    plt.style.use("seaborn-v0_8-whitegrid")

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            # DejaVu Sans ships with matplotlib, always available.
            # NVIDIA Sans only used if installed on the system.
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": TICK_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "legend.framealpha": 0.9,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )


def get_cutoff_style(cutoff):
    """Get color + line style for a cutoff value."""
    color = CUTOFF_COLORS.get(cutoff, GRAY)
    style = CUTOFF_STYLES.get(cutoff, {"marker": "o", "linestyle": "-"})
    return {
        "color": color,
        **style,
        "linewidth": 2,
        "markeredgecolor": "black",
        "markeredgewidth": 0.5,
    }


def get_method_style(method):
    """Get color + line style for a method name."""
    color = METHOD_COLORS.get(method, GRAY)
    style = METHOD_STYLES.get(method, {"marker": "o", "linestyle": "-", "linewidth": 2})
    return {"color": color, **style, "markeredgecolor": "black", "markeredgewidth": 0.5}


def format_accuracy(val):
    """Format accuracy as compact exponent, e.g. 1e-4 → 'e-4'."""
    import math

    exp = int(round(math.log10(val)))
    return f"e{exp}"


def format_legend_label(method, value, is_cutoff=True):
    """Format legend label with fixed-width columns for table-like alignment.

    Uses fixed widths: method=7 chars, gap=2 chars, value=6-8 chars.
    Requires monospace font in the legend to align properly.
    """
    name = method.ljust(7)
    if is_cutoff:
        c = int(value)
        val_str = f"{c}\u00c5".rjust(6)  # Angstrom symbol
    else:
        val_str = format_accuracy(value).rjust(8)
    return f"{name}  {val_str}"


def create_table_legend(ax, loc="best", col2_header="cutoff"):
    """Create a table-like legend with header row and separator.

    Adds invisible handle entries for header and separator so they align
    exactly with the data entries. Uses DejaVu Sans Mono for alignment.
    """
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    invisible = Line2D([0], [0], color="none", marker="None", linestyle="None")

    if col2_header == "cutoff":
        header_label = "Method   Cutoff"
        sep_label = "\u00b7\u00b7\u00b7\u00b7\u00b7\u00b7\u00b7  \u00b7\u00b7\u00b7\u00b7\u00b7\u00b7"
    elif col2_header == "batch":
        header_label = "Method   Batch"
        sep_label = "\u00b7" * 7 + "  " + "\u00b7" * 11
    else:
        header_label = "Method   Accuracy"
        sep_label = "\u00b7\u00b7\u00b7\u00b7\u00b7\u00b7\u00b7  \u00b7\u00b7\u00b7\u00b7\u00b7\u00b7\u00b7\u00b7"

    all_handles = [invisible, invisible] + handles
    all_labels = [header_label, sep_label] + labels

    legend = ax.legend(
        all_handles,
        all_labels,
        loc=loc,
        prop={"size": LEGEND_SIZE - 1, "family": "DejaVu Sans Mono"},
        handlelength=3.5,
        labelspacing=0.3,
        handletextpad=0.5,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        facecolor="white",
    )

    # Color header rows gray
    texts = legend.get_texts()
    texts[0].set_color("#666666")
    texts[1].set_color("#999999")

    return legend


def setup_log2_xaxis(
    ax,
    ticks=None,
    limits=None,
    show_every_n=2,
    show_batch_size=False,
    target_atoms=None,
    label="System Size (atoms)",
):
    """Set up log base 2 x-axis with evenly-spaced power-of-2 ticks.

    This makes the x-axis look linear while actually being logarithmic,
    because powers of 2 are evenly spaced on a log-2 scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    ticks : list, optional
        Tick positions (powers of 2). Default: X_AXIS_TICKS_MEDIUM.
    limits : tuple, optional
        (min, max) limits. Default: 'medium'.
    show_every_n : int, default=2
        Show label for every Nth tick.
    show_batch_size : bool
        If True, show [x batch] below atom count.
    target_atoms : int, optional
        Total atoms for batch size calculation.
    label : str
        X-axis label.
    """
    if ticks is None:
        ticks = X_AXIS_TICKS_MEDIUM
    if limits is None:
        limits = X_AXIS_LIMITS["medium"]

    ax.set_xscale("log", base=2)
    # Add small horizontal margins (10% padding on each side in log space)
    lo, hi = limits
    margin_factor = 0.85  # shrink limits slightly for padding
    ax.set_xlim(lo * margin_factor, hi / margin_factor)
    ax.set_xticks(ticks)

    n = len(ticks)
    labels = []
    for i, atoms in enumerate(ticks):
        # Always show first and last tick; skip others by show_every_n
        is_first = i == 0
        is_last = i == n - 1
        is_nth = (n - 1 - i) % show_every_n == 0
        if is_first or is_last or is_nth:
            atom_str = format_num(atoms)
            if show_batch_size and target_atoms:
                batch = target_atoms // atoms if atoms > 0 else 1
                labels.append(f"{atom_str}\n[x{batch}]")
            else:
                labels.append(atom_str)
        else:
            labels.append("")

    ax.set_xticklabels(labels, fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel(label, fontsize=AXIS_LABEL_SIZE)


def add_scaling_reference(
    ax, x_data, y_start, scaling="linear", label=None, color="gray", alpha=0.5
):
    """Add O(N) or O(N^2) reference line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x_data : array-like
    y_start : float
        Y value at first x point.
    scaling : str
        'linear' for O(N), 'quadratic' for O(N^2).
    """
    x = np.array(x_data)
    x0 = x[0]

    if scaling == "linear":
        y = y_start * (x / x0)
        default_label = "O(N)"
    elif scaling == "quadratic":
        y = y_start * (x / x0) ** 2
        default_label = "O(N^2)"
    else:
        raise ValueError(f"Unknown scaling: {scaling}")

    ax.plot(
        x,
        y,
        "--",
        color=color,
        alpha=alpha,
        linewidth=1.5,
        label=label or default_label,
        zorder=1,
    )


# =============================================================================
# Shared Y-Axis Formatters
# =============================================================================


def throughput_formatter(val, pos):
    """Format throughput axis: integers when >=1, one decimal when >=0.1."""
    if val >= 1 and val == int(val):
        return f"{int(val)}"
    elif val >= 0.1:
        return f"{val:.1f}"
    else:
        return f"{val:.2f}"


def memory_formatter(val, pos):
    """Format memory axis in human-readable binary units (KB/MB/GB)."""
    if val >= 1024:
        return f"{val / 1024:.0f} GB"
    elif val >= 1:
        return f"{val:.0f} MB"
    else:
        return f"{val * 1024:.0f} KB"


# =============================================================================
# Single-Panel Figure Size
# =============================================================================

SINGLE_PANEL_SIZE = (6, 4.5)
