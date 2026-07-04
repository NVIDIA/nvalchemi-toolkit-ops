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
"""Plot styling constants and utilities for NVIDIA branding.

- NVIDIA green color scheme
- Log base 2 x-axis that looks linear (evenly spaced power-of-2 ticks)
- Publication-quality font sizes and figure dimensions
"""

from __future__ import annotations

from typing import TypedDict

import matplotlib as mpl
import matplotlib.pyplot as plt


class CutoffStyle(TypedDict):
    """Style parameters for a cutoff-specific line."""

    marker: str
    linestyle: str
    fillstyle: str


class LayerStyle(TypedDict):
    """Secondary styling for one parameter layer."""

    fillstyle: str
    linewidth_scale: float


__all__ = [
    # TypedDicts
    "CutoffStyle",
    "LayerStyle",
    # Colors
    "BACKEND_COLORS",
    "BACKEND_LINESTYLES",
    "D3_CUTOFF_COLORS",
    "DEFAULT_MARKER_FILLSTYLE",
    "EL_ACCURACY_STYLES",
    "EL_METHOD_STYLES",
    "GRAY",
    "LINE_ALPHA",
    "NAIVE_ORANGE",
    "NL_CUTOFF_STYLES",
    "NL_METHOD_COLORS",
    "NL_METHOD_STYLES",
    "NVIDIA_BLUE",
    "NVIDIA_GREEN",
    # Styles
    "BATCH_SYSTEM_STYLES",
    "DATA_LINE_WIDTH",
    "DATA_MARKER_SIZE",
    "D3_CUTOFF_STYLES",
    "GRID_STYLE",
    "PNG_EXPORT_DPI",
    "SECONDARY_LINESTYLE",
    # Tolerances
    "MEMORY_MERGE_TOLERANCE",
    # References
    "GPU_VRAM_REFS",
    # Dimensions
    "AXIS_LABEL_SIZE",
    "SINGLE_PANEL_SIZE",
    "SINGLE_PANEL_MIN_PIXEL_SIZE",
    "THREE_PANEL_SIZE",
    "TITLE_SIZE",
    "X_AXIS_LIMITS",
    "X_AXIS_TICKS_MEDIUM",
    # Functions
    "format_accuracy",
    "format_num",
    "format_system_name",
    "log_scale_formatter",
    "memory_formatter",
    "setup_log2_xaxis",
    "setup_plot_style",
]

# =============================================================================
# NVIDIA Color Scheme
# =============================================================================

NVIDIA_GREEN = "#76B900"
NVIDIA_BLUE = "#2E6F9E"
GRAY = "#555555"
LIGHT_GRAY = "#A0A0A0"
NAIVE_ORANGE = "#C56A2D"

LINE_ALPHA = 0.88
DATA_LINE_WIDTH = 0.95
DATA_MARKER_SIZE = 5.4
SECONDARY_LINESTYLE = (0, (4.2, 2.0))
DEFAULT_MARKER_FILLSTYLE = "none"
# Tight-cropped single panels remain at least 2x a 900 px documentation column.
PNG_EXPORT_DPI = 360

# Batch system size is a first-class series dimension for D3 and EL. Avoid
# subtle marker-size differences: line pattern and marker shape carry the key.
BATCH_SYSTEM_STYLES = (
    {"linestyle": "-", "marker": "o"},
    {"linestyle": SECONDARY_LINESTYLE, "marker": "D"},
    {"linestyle": (0, (1.2, 1.8)), "marker": "P"},
    {"linestyle": (0, (6.0, 2.0, 1.2, 2.0)), "marker": "X"},
)

# D3-specific colors: 15Å gets green since no 6Å
# With only 2 lines, use green + blue; purple reserved for 3+ lines
D3_CUTOFF_COLORS = {
    15.0: NVIDIA_GREEN,  # NVIDIA green for smallest D3 cutoff
    25.0: NVIDIA_BLUE,
}

# Neighbor-list methods form a small hierarchy. Color is the primary strategy
# cue, with related leaves kept in the same tonal neighborhood; line and marker
# styling reinforce the concrete strategy without changing across cutoffs.
NL_METHOD_COLORS = {
    "cluster_tile": NVIDIA_GREEN,
    "cell_list_pair_centric": "#2E8A73",
    "cell_list_atom_centric": NVIDIA_BLUE,
    "naive_scalar": "#666666",
    "naive_tile": "#2B2B2B",
}

BACKEND_COLORS = {
    "torch": NVIDIA_GREEN,
    "jax": NVIDIA_BLUE,
    "warp": NAIVE_ORANGE,
}

# Backend is a secondary comparison dimension. Keep the scientific method or
# parameter color intact and distinguish frameworks with line pattern.
BACKEND_LINESTYLES = {
    "torch": "-",
    "jax": SECONDARY_LINESTYLE,
    "warp": (0, (1.2, 1.8)),
}

NL_METHOD_STYLES = {
    "naive_scalar": {
        "label": "Naive scalar",
        "color": NL_METHOD_COLORS["naive_scalar"],
        "linestyle": SECONDARY_LINESTYLE,
        "marker": "s",
        "zorder": 2,
    },
    "naive_tile": {
        "label": "Naive tile",
        "color": NL_METHOD_COLORS["naive_tile"],
        "linestyle": "-",
        "marker": "D",
        "zorder": 2,
    },
    "cell_list_atom_centric": {
        "label": "Cell atom",
        "color": NL_METHOD_COLORS["cell_list_atom_centric"],
        "linestyle": "-",
        "marker": "o",
        "zorder": 3,
    },
    "cell_list_pair_centric": {
        "label": "Cell pair",
        "color": NL_METHOD_COLORS["cell_list_pair_centric"],
        "linestyle": SECONDARY_LINESTYLE,
        "marker": "P",
        "zorder": 3,
    },
    "cluster_tile": {
        "label": "Cluster tile",
        "color": NL_METHOD_COLORS["cluster_tile"],
        "linestyle": "-",
        "marker": "^",
        "zorder": 5,
    },
}

# Cutoff is a layer on top of the method hierarchy. Keep method color, dash,
# and marker shape stable while cutoff changes; use only weight and marker fill
# to distinguish simultaneously visible cutoff layers.
NL_CUTOFF_STYLES: dict[float, LayerStyle] = {
    6.0: {"fillstyle": "left", "linewidth_scale": 0.9},
    15.0: {"fillstyle": DEFAULT_MARKER_FILLSTYLE, "linewidth_scale": 1.0},
    25.0: {"fillstyle": "full", "linewidth_scale": 1.1},
}

# Electrostatics follows the same hierarchy as NL: the method keeps its color,
# line pattern, and marker at every accuracy, while accuracy changes only the
# marker fill and a small amount of line weight.
EL_METHOD_STYLES = {
    "pme": {
        "label": "PME",
        "color": NVIDIA_GREEN,
        "linestyle": "-",
        "marker": "o",
    },
    "ewald": {
        "label": "Ewald",
        "color": NVIDIA_BLUE,
        "linestyle": SECONDARY_LINESTYLE,
        "marker": "^",
    },
}

EL_ACCURACY_STYLES: dict[float, LayerStyle] = {
    1e-4: {"fillstyle": "full", "linewidth_scale": 0.9},
    1e-6: {"fillstyle": DEFAULT_MARKER_FILLSTYLE, "linewidth_scale": 1.1},
}

# Tolerances
MEMORY_MERGE_TOLERANCE = 0.05  # Merge NL cell/naive memory if within 5%

# GPU VRAM reference lines for memory panels. Filled only from explicit
# artifact metadata or caller overrides; never assume a fixed hardware limit.
GPU_VRAM_REFS: dict[str, int] = {}

# D3 cutoff styles follow the shared scientific-series hierarchy: color first,
# then line pattern, then marker shape.
D3_CUTOFF_STYLES: dict[float, CutoffStyle] = {
    15.0: {
        "marker": "^",
        "linestyle": "-",
        "fillstyle": DEFAULT_MARKER_FILLSTYLE,
    },
    25.0: {
        "marker": "s",
        "linestyle": SECONDARY_LINESTYLE,
        "fillstyle": "full",
    },
}

# Grid style matching benchmarks-temp/
GRID_STYLE = dict(color=LIGHT_GRAY, linestyle="--", linewidth=0.3)

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

THREE_PANEL_SIZE = (14, 4.5)  # Standard: 3 panels in a row

# =============================================================================
# X-Axis Configuration (Powers of 2, log base 2)
# =============================================================================

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
            "savefig.dpi": PNG_EXPORT_DPI,
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
            "lines.linewidth": DATA_LINE_WIDTH,
            "lines.markersize": DATA_MARKER_SIZE,
        }
    )


def format_accuracy(val):
    """Format accuracy as compact exponent, e.g. 1e-4 → 'e-4'."""
    import math

    exp = int(round(math.log10(val)))
    return f"e{exp}"


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


# =============================================================================
# Shared Y-Axis Formatters
# =============================================================================


def log_scale_formatter(val, pos):
    """Format log-scale axis ticks with labels at 1-2-3-5 subdivisions
    per decade.

    Used for both time (μs/atom) and throughput (Matoms/s) axes. The
    panel locator emits ticks at every integer sub (1..9); this
    formatter keeps labels readable by only labelling a subset.

    Adding 3 to the labelled set rescues narrow y-windows — e.g. a
    panel whose data sits entirely between 2×10⁻¹ and 5×10⁻¹ would
    otherwise have only two labels; showing 3×10⁻¹ gives the reader a
    reference inside the actual data range.

    Parameters
    ----------
    val : float
        Tick value.
    pos : int
        Tick position (unused, required by matplotlib).
    """
    if val <= 0:
        return ""
    import math

    exp = math.floor(math.log10(val))
    coeff = val / 10**exp
    if not any(abs(coeff - c) < 0.01 for c in (1, 2, 3, 5)):
        return ""
    if val >= 1 and val == int(val):
        return f"{int(val)}"
    return f"{val:g}"


def memory_formatter(val, pos):
    """Format memory axis in human-readable units (KB/MB/GB).

    Only labels 1-2-5 positions per decade (in MB) to avoid clutter.

    Parameters
    ----------
    val : float
        Memory value in MB.
    pos : int
        Tick position (unused, required by matplotlib).
    """
    if val <= 0:
        return ""
    import math

    exp = math.floor(math.log10(val))
    coeff = val / 10**exp
    if not any(abs(coeff - c) < 0.01 for c in (1, 2, 5)):
        return ""
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
SINGLE_PANEL_MIN_PIXEL_SIZE = (1800, 1400)
