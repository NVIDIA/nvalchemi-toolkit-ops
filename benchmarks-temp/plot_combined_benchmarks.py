#!/usr/bin/env python3
"""
Combined Benchmark Plotting Script
===================================
Plots all benchmark results in a single figure:
- Panel A (top-left): Neighbor List scaling (constant 128k atoms)
- Panel B (top-right): DFT-D3 system size scaling
- Panel C (bottom-left): Electrostatics system size scaling
- Panel D (bottom-right): Electrostatics 128k batched

Output: 13.333 x 10 inches (4:3 aspect, taller for 4 panels)

Usage:
    python plot_combined_benchmarks.py /path/to/benchmark_results/
    python plot_combined_benchmarks.py --input-dir ./benchmark-results/benchmark_2026-01-27_12-00-00
    python plot_combined_benchmarks.py --input-dir ./results --output-dir ./plots
"""

import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np

# ============== DEFAULT PATHS ==============
SCRIPT_DIR = Path(__file__).parent

# Figure size - 2x2 panels, each panel 16:9-ish
FIG_WIDTH = 13.333
FIG_HEIGHT = 11  # Extra height to accommodate spacing between rows

# Font sizes - consistent across all plots
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 10
TITLE_SIZE = 14

# Plot options
INCLUDE_CHARGE_GRADIENTS = False  # Set to True to include +cg lines in electrostatics
SHOW_TITLES = True  # Global flag to control panel titles (A, B, C, D)
# ====================================

# Style constants - NVIDIA green color scheme
NVIDIA_GREEN = '#76B900'
DARK_GREEN = '#4A7A00'
DARKEST_GREEN = '#2D4A00'
GRAY = '#555555'

# NL cutoff colors and styles
CUTOFF_COLORS = {
    6.0: NVIDIA_GREEN,
    15.0: DARK_GREEN,
    25.0: DARKEST_GREEN,
}
CUTOFF_STYLES = {
    6.0: {'marker': 'o', 'linestyle': '-'},
    15.0: {'marker': '^', 'linestyle': '-'},
    25.0: {'marker': 's', 'linestyle': '--'},
}

# Electrostatics accuracy colors
ACCURACY_COLORS = {
    1e-4: NVIDIA_GREEN,
    1e-6: DARK_GREEN,
}

# Common baseline settings
BASELINE_COLOR = '#AED6F1'
MLIP_BASELINE_US = (4.0, 6.0)
TARGET_ATOMS = 131072

# Unified x-axis range (all panels show same atom range, with padding)
X_AXIS_MIN = 90       # Padding before first tick (128)
X_AXIS_MAX = 180000   # Padding after last tick (131072)

# Standard x-axis ticks (powers of 2, from 128 to 128k)
# These are the actual system sizes in the benchmark data
X_AXIS_TICKS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

# MACE-MP0 baseline data (H100)
MACE_BASELINE = [
    (192, 129.53), (504, 51.41), (1032, 25.04), (2088, 11.52),
    (4104, 6.99), (8232, 4.85), (16382, 3.94), (32796, 3.80),
    (65568, 3.65), (95304, 3.69),
]

try:
    plt.rcParams['font.family'] = 'NVIDIA Sans'
except:
    pass


def format_num(n):
    """Format atom count using binary prefix."""
    if n >= 1024:
        return f'{n // 1024}k'
    return str(n)


def format_legend_label(method, value, is_cutoff=True):
    """Format legend label with fixed-width columns for table-like alignment.
    
    For NL/D3: method + cutoff in Å
    For Electrostatics: method + accuracy
    
    Uses fixed widths: method=7 chars, gap=2 chars, value=6-8 chars
    """
    name = method.ljust(7)
    if is_cutoff:
        c = int(value)
        val_str = f'{c}Å'.rjust(6)
    else:
        val_str = f'{value:.0e}'.rjust(8)
    return f'{name}  {val_str}'  # 2 spaces between columns


def create_table_legend(ax, loc='best', col2_header='cutoff'):
    """Create a table-like legend with header row as fake entry (perfect alignment).
    
    Adds invisible handle entries for header and separator so they align
    exactly with the data entries.
    """
    from matplotlib.lines import Line2D
    
    # Get existing legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    
    # Create invisible handle for header/separator rows
    invisible = Line2D([0], [0], color='none', marker='None', linestyle='None')
    
    # Header and separator labels - must match format_legend_label exactly:
    # method.ljust(7) + '  ' + value.rjust(6 or 8)
    if col2_header == 'cutoff':
        # 7 + 2 + 6 = 15 chars total
        header_label = 'Method   Cutoff'  # method(7) + gap(2) + cutoff(6)
        sep_label =    '·······  ······'
    else:
        # 7 + 2 + 8 = 17 chars total
        header_label = 'Method   Accuracy'  # method(7) + gap(2) + accuracy(8)
        sep_label =    '·······  ········'
    
    # Prepend header + separator to handles/labels
    all_handles = [invisible, invisible] + handles
    all_labels = [header_label, sep_label] + labels
    
    legend = ax.legend(all_handles, all_labels, loc=loc,
                       prop={'size': LEGEND_SIZE - 1, 'family': 'DejaVu Sans Mono'},
                       handlelength=2.5, labelspacing=0.3, handletextpad=0.5)
    
    # Color the header rows gray
    texts = legend.get_texts()
    texts[0].set_color('#666666')  # header
    texts[1].set_color('#999999')  # separator
    
    return legend


def load_csv(path):
    """Load benchmark results from CSV with flexible type conversion."""
    if not path.exists():
        return None
    
    results = []
    with open(path) as f:
        for row in csv.DictReader(f):
            e = {}
            for k, v in row.items():
                try:
                    if '.' in v or 'e' in v.lower():
                        e[k] = float(v)
                    else:
                        e[k] = int(v)
                except (ValueError, AttributeError):
                    e[k] = v
            results.append(e)
    return results


def plot_empty_panel(ax, message):
    """Display a message in an empty panel."""
    ax.text(0.5, 0.5, message, ha='center', va='center',
            fontsize=TITLE_SIZE, color='gray', transform=ax.transAxes,
            style='italic')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_nl_scaling(results, ax):
    """Panel A: NL scaling (constant 128k atoms)."""
    if not results:
        plot_empty_panel(ax, 'No NL data\n(benchmark_nl_results.csv)')
        if SHOW_TITLES:
            ax.set_title('A. Neighbor List', fontsize=TITLE_SIZE, fontweight='bold', loc='left')
        return
    
    target_atoms = results[0].get('target_atoms', TARGET_ATOMS)
    
    # MLIP baseline band
    ax.axhspan(MLIP_BASELINE_US[0], MLIP_BASELINE_US[1], 
               color=BASELINE_COLOR, alpha=0.3, zorder=0)
    mlip_center = (MLIP_BASELINE_US[0] * MLIP_BASELINE_US[1]) ** 0.5
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.98, mlip_center, 'MLIP baseline', transform=trans,
            fontsize=LEGEND_SIZE, color='#1565C0', ha='right', va='center', clip_on=True)
    
    # Plot data by cutoff and method
    for cutoff in sorted(set(r['cutoff'] for r in results)):
        style = CUTOFF_STYLES.get(cutoff, {'marker': 'o', 'linestyle': '-'})
        color = CUTOFF_COLORS.get(cutoff, NVIDIA_GREEN)
        
        for method in ['naive', 'cell']:
            sub = sorted([r for r in results if r['cutoff']==cutoff and r['method']==method], 
                        key=lambda x: x['atoms_per_system'])
            if sub:
                linestyle = '--' if method == 'naive' else '-'
                ax.plot([r['atoms_per_system'] for r in sub], [r['time_us'] for r in sub], 
                       color=color, linewidth=1.5 if method=='naive' else 2, alpha=0.85,
                       marker=style['marker'], linestyle=linestyle, markersize=6,
                       label=format_legend_label(method.capitalize(), cutoff, is_cutoff=True))
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlim(X_AXIS_MIN, X_AXIS_MAX)
    ax.set_ylim(0.005, 20)
    
    # X-axis labels (unified ticks across all panels)
    ax.set_xticks(X_AXIS_TICKS)
    n = len(X_AXIS_TICKS)
    labels = []
    for i, atoms in enumerate(X_AXIS_TICKS):
        batch = target_atoms // atoms if atoms > 0 else 1
        if (n - 1 - i) % 2 == 0:
            labels.append(f'{format_num(atoms)}\n[x{batch}]')
        else:
            labels.append('')
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_SIZE - 2)
    
    ax.set_xlabel('System size [batch]', fontsize=AXIS_LABEL_SIZE - 2)
    ax.set_ylabel('Time per atom [μs]', fontsize=AXIS_LABEL_SIZE - 2)
    ax.tick_params(axis='y', which='major', labelsize=TICK_LABEL_SIZE - 2)
    create_table_legend(ax, loc='best', col2_header='cutoff')
    ax.grid(True, which='both', color='#A0A0A0', linestyle='--', linewidth=0.3)
    if SHOW_TITLES:
        ax.set_title('A. Neighbor List | Constant 128k atoms, variable system and batch sizes', fontsize=TITLE_SIZE - 2, fontweight='bold', loc='left')


def plot_d3_scaling(results, ax):
    """Panel B: D3 system size scaling (batch=1)."""
    if not results:
        plot_empty_panel(ax, 'No D3 data\n(benchmark_d3_scaling_results.csv)')
        if SHOW_TITLES:
            ax.set_title('B. DFT-D3', fontsize=TITLE_SIZE, fontweight='bold', loc='left')
        return
    
    # MACE baseline line
    mace_x = [m[0] for m in MACE_BASELINE]
    mace_y = [m[1] for m in MACE_BASELINE]
    ax.plot(mace_x, mace_y, color=BASELINE_COLOR, linewidth=10, alpha=0.5,
           linestyle='-', zorder=1)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    mlip_center = (mace_y[-1] + mace_y[-2]) / 2
    ax.text(0.98, mlip_center, 'MLIP baseline', transform=trans,
            fontsize=LEGEND_SIZE, color='#1565C0', ha='right', va='center', clip_on=True)
    
    # Plot D3 by cutoff (skip 6Å - too short for D3)
    for cutoff in sorted(set(r['cutoff'] for r in results)):
        if cutoff == 6.0:
            continue  # Skip 6Å cutoff for D3
        style = CUTOFF_STYLES.get(cutoff, {'marker': 'o', 'linestyle': '-'})
        color = CUTOFF_COLORS.get(cutoff, NVIDIA_GREEN)
        
        sub = sorted([r for r in results if r['cutoff']==cutoff], 
                    key=lambda x: x['atoms_per_system'])
        if sub:
            x = [r['atoms_per_system'] for r in sub]
            y = [r['time_d3_us'] for r in sub]
            ax.plot(x, y, color=color, linewidth=2, alpha=0.85,
                   marker=style['marker'], linestyle='-', markersize=6,
                   label=format_legend_label('D3', cutoff, is_cutoff=True), zorder=2)
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlim(X_AXIS_MIN, X_AXIS_MAX)
    
    # Y-axis from data (excluding 6Å)
    all_y = mace_y + [r['time_d3_us'] for r in results if r['cutoff'] != 6.0]
    ax.set_ylim(min(all_y) * 0.5, max(all_y) * 1.5)
    
    # X-axis labels (unified ticks across all panels)
    ax.set_xticks(X_AXIS_TICKS)
    n = len(X_AXIS_TICKS)
    labels = []
    for i, atoms in enumerate(X_AXIS_TICKS):
        if (n - 1 - i) % 2 == 0:
            labels.append(f'{format_num(atoms)}\n[x1]')
        else:
            labels.append('')
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_SIZE - 2)
    
    ax.set_xlabel('System size [batch]', fontsize=AXIS_LABEL_SIZE - 2)
    ax.set_ylabel('Time per atom [μs]', fontsize=AXIS_LABEL_SIZE - 2)
    ax.tick_params(axis='y', which='major', labelsize=TICK_LABEL_SIZE - 2)
    create_table_legend(ax, loc='best', col2_header='cutoff')
    ax.grid(True, which='both', color='#A0A0A0', linestyle='--', linewidth=0.3)
    if SHOW_TITLES:
        ax.set_title('B. DFT-D3 | Variable system size, batch = 1', fontsize=TITLE_SIZE - 2, fontweight='bold', loc='left')


def plot_electrostatics_scaling(results, ax):
    """Panel C: Electrostatics system size scaling (batch=1)."""
    if not results:
        plot_empty_panel(ax, 'No Electrostatics scaling data\n(benchmark_electrostatics_results.csv)')
        if SHOW_TITLES:
            ax.set_title('D. Electrostatics', fontsize=TITLE_SIZE, fontweight='bold', loc='left')
        return
    
    # MACE baseline line
    mace_x = [m[0] for m in MACE_BASELINE]
    mace_y = [m[1] for m in MACE_BASELINE]
    ax.plot(mace_x, mace_y, color=BASELINE_COLOR, linewidth=10, alpha=0.5,
           linestyle='-', zorder=1)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    mlip_center = (mace_y[-1] + mace_y[-2]) / 2
    ax.text(0.98, mlip_center, 'MLIP baseline', transform=trans,
            fontsize=LEGEND_SIZE, color='#1565C0', ha='right', va='center', clip_on=True)
    
    # Group by accuracy
    accuracies = sorted(set(r['accuracy'] for r in results))
    
    for accuracy in accuracies:
        color = ACCURACY_COLORS.get(accuracy, NVIDIA_GREEN)
        sub = sorted([r for r in results if r['accuracy'] == accuracy], 
                    key=lambda x: x['n_atoms'])
        if not sub:
            continue
        
        x = [r['n_atoms'] for r in sub]
        
        # PME (solid, circles)
        y_pme = [r['pme_time_us'] for r in sub]
        ax.plot(x, y_pme, color=color, linewidth=2, alpha=0.85, 
                marker='o', linestyle='-', markersize=6,
                label=format_legend_label('PME', accuracy, is_cutoff=False), zorder=2)
        
        # Ewald (dashed, triangles - thinner line, hollow marker for visibility)
        y_ewald = [r['ewald_time_us'] for r in sub]
        ax.plot(x, y_ewald, color=color, linewidth=1.5, alpha=0.85, 
                marker='^', linestyle='--', markersize=6,                 label=format_legend_label('Ewald', accuracy, is_cutoff=False), zorder=2)
        
        # Optional: +cg lines
        if INCLUDE_CHARGE_GRADIENTS and 'pme_cg_time_us' in sub[0]:
            y_pme_cg = [r['pme_cg_time_us'] for r in sub]
            ax.plot(x, y_pme_cg, color=color, linewidth=1.5, alpha=0.5, 
                    marker='s', linestyle='-', markersize=6,
                    label=format_legend_label('PME+cg', accuracy, is_cutoff=False), zorder=2)
            
            y_ewald_cg = [r['ewald_cg_time_us'] for r in sub]
            ax.plot(x, y_ewald_cg, color=color, linewidth=1.5, alpha=0.5, 
                    marker='D', linestyle='--', markersize=6,
                    label=format_legend_label('Ewald+cg', accuracy, is_cutoff=False), zorder=2)
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlim(X_AXIS_MIN, X_AXIS_MAX)
    
    # X-axis labels (unified ticks across all panels)
    ax.set_xticks(X_AXIS_TICKS)
    n = len(X_AXIS_TICKS)
    labels = []
    for i, atoms in enumerate(X_AXIS_TICKS):
        if (n - 1 - i) % 2 == 0:
            labels.append(f'{format_num(atoms)}\n[x1]')
        else:
            labels.append('')
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_SIZE - 2)
    
    # Secondary x-axis for cutoffs (only when showing titles)
    if SHOW_TITLES:
        ax_top = ax.twiny()
        ax_top.set_xscale('log', base=2)
        ax_top.set_xlim(X_AXIS_MIN, X_AXIS_MAX)
        ax_top.set_xticks(X_AXIS_TICKS)
        
        cutoff_1e4 = {r['n_atoms']: r['real_space_cutoff'] 
                      for r in results if r['accuracy'] == 1e-4}
        cutoff_1e6 = {r['n_atoms']: r['real_space_cutoff'] 
                      for r in results if r['accuracy'] == 1e-6}
        
        cutoff_labels = []
        for i, atoms in enumerate(X_AXIS_TICKS):
            if (n - 1 - i) % 2 == 0:
                c4 = cutoff_1e4.get(atoms)
                c6 = cutoff_1e6.get(atoms)
                if c4 and c6:
                    cutoff_labels.append(f'{c4:.1f}\n{c6:.1f}')
                elif c4:
                    cutoff_labels.append(f'{c4:.1f}\n-')
                elif c6:
                    cutoff_labels.append(f'-\n{c6:.1f}')
                else:
                    cutoff_labels.append('')
            else:
                cutoff_labels.append('')
        ax_top.set_xticklabels(cutoff_labels, fontsize=TICK_LABEL_SIZE - 4)
        ax_top.set_xlabel('r_cut [Å] (1e-4/1e-6)', fontsize=AXIS_LABEL_SIZE - 4)
    
    ax.set_xlabel('System size [batch]', fontsize=AXIS_LABEL_SIZE - 2)
    ax.set_ylabel('Time per atom [μs]', fontsize=AXIS_LABEL_SIZE - 2)
    ax.tick_params(axis='y', which='major', labelsize=TICK_LABEL_SIZE - 2)
    create_table_legend(ax, loc='upper right', col2_header='accuracy')
    ax.grid(True, which='both', color='#A0A0A0', linestyle='--', linewidth=0.3)
    if SHOW_TITLES:
        ax.set_title('D. Electrostatics | Variable system size, batch = 1', fontsize=TITLE_SIZE - 2, fontweight='bold', loc='left')


def plot_electrostatics_batched(results, ax):
    """Panel D: Electrostatics 128k batched."""
    if not results:
        plot_empty_panel(ax, 'No Electrostatics 128k data\n(benchmark_electrostatics_128k_results.csv)')
        if SHOW_TITLES:
            ax.set_title('C. Electrostatics', fontsize=TITLE_SIZE, fontweight='bold', loc='left')
        return
    
    # MLIP baseline band
    ax.axhspan(MLIP_BASELINE_US[0], MLIP_BASELINE_US[1], 
               color=BASELINE_COLOR, alpha=0.3, zorder=0)
    mlip_center = (MLIP_BASELINE_US[0] * MLIP_BASELINE_US[1]) ** 0.5
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.98, mlip_center, 'MLIP baseline', transform=trans,
            fontsize=LEGEND_SIZE, color='#1565C0', ha='right', va='center', clip_on=True)
    
    # Group by accuracy
    accuracies = sorted(set(r.get('accuracy', 1e-4) for r in results))
    
    for accuracy in accuracies:
        color = ACCURACY_COLORS.get(accuracy, NVIDIA_GREEN)
        sub = sorted([r for r in results if r.get('accuracy', 1e-4) == accuracy], 
                    key=lambda x: x['atoms_per_system'])
        if not sub:
            continue
        
        x = [r['atoms_per_system'] for r in sub]
        
        # PME (solid, circles)
        y_pme = [r['pme_time_us'] for r in sub]
        ax.plot(x, y_pme, color=color, linewidth=2, alpha=0.85,
                marker='o', linestyle='-', markersize=6, 
                label=format_legend_label('PME', accuracy, is_cutoff=False), zorder=2)
        
        # Ewald (dashed, triangles - thinner line, hollow marker for visibility)
        y_ewald = [r['ewald_time_us'] for r in sub]
        ax.plot(x, y_ewald, color=color, linewidth=1.5, alpha=0.85,
                marker='^', linestyle='--', markersize=6,                 label=format_legend_label('Ewald', accuracy, is_cutoff=False), zorder=2)
        
        # Optional: +cg lines
        if INCLUDE_CHARGE_GRADIENTS and 'pme_cg_time_us' in sub[0]:
            y_pme_cg = [r['pme_cg_time_us'] for r in sub]
            ax.plot(x, y_pme_cg, color=color, linewidth=1.5, alpha=0.5,
                    marker='s', linestyle='-', markersize=6, 
                    label=format_legend_label('PME+cg', accuracy, is_cutoff=False), zorder=2)
            
            y_ewald_cg = [r['ewald_cg_time_us'] for r in sub]
            ax.plot(x, y_ewald_cg, color=color, linewidth=1.5, alpha=0.5,
                    marker='D', linestyle='--', markersize=6, 
                    label=format_legend_label('Ewald+cg', accuracy, is_cutoff=False), zorder=2)
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlim(X_AXIS_MIN, X_AXIS_MAX)
    
    # X-axis labels (unified ticks across all panels)
    ax.set_xticks(X_AXIS_TICKS)
    n = len(X_AXIS_TICKS)
    labels = []
    for i, atoms in enumerate(X_AXIS_TICKS):
        batch = TARGET_ATOMS // atoms if atoms > 0 else 1
        if (n - 1 - i) % 2 == 0:
            labels.append(f'{format_num(atoms)}\n[x{batch}]')
        else:
            labels.append('')
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_SIZE - 2)
    
    # Secondary x-axis for cutoffs (only when showing titles)
    if SHOW_TITLES:
        ax_top = ax.twiny()
        ax_top.set_xscale('log', base=2)
        ax_top.set_xlim(X_AXIS_MIN, X_AXIS_MAX)
        ax_top.set_xticks(X_AXIS_TICKS)
        
        cutoff_1e4 = {r['atoms_per_system']: r['real_space_cutoff'] 
                      for r in results if r.get('accuracy', 1e-4) == 1e-4}
        cutoff_1e6 = {r['atoms_per_system']: r['real_space_cutoff'] 
                      for r in results if r.get('accuracy', 1e-4) == 1e-6}
        
        cutoff_labels = []
        for i, atoms in enumerate(X_AXIS_TICKS):
            if (n - 1 - i) % 2 == 0:
                c4 = cutoff_1e4.get(atoms)
                c6 = cutoff_1e6.get(atoms)
                if c4 and c6:
                    cutoff_labels.append(f'{c4:.1f}\n{c6:.1f}')
                elif c4:
                    cutoff_labels.append(f'{c4:.1f}\n-')
                elif c6:
                    cutoff_labels.append(f'-\n{c6:.1f}')
                else:
                    cutoff_labels.append('')
            else:
                cutoff_labels.append('')
        ax_top.set_xticklabels(cutoff_labels, fontsize=TICK_LABEL_SIZE - 4)
        ax_top.set_xlabel('r_cut [Å] (1e-4/1e-6)', fontsize=AXIS_LABEL_SIZE - 4)
    
    ax.set_xlabel('System size [batch]', fontsize=AXIS_LABEL_SIZE - 2)
    ax.set_ylabel('Time per atom [μs]', fontsize=AXIS_LABEL_SIZE - 2)
    ax.tick_params(axis='y', which='major', labelsize=TICK_LABEL_SIZE - 2)
    create_table_legend(ax, loc='upper left', col2_header='accuracy')
    ax.grid(True, which='both', color='#A0A0A0', linestyle='--', linewidth=0.3)
    if SHOW_TITLES:
        ax.set_title('C. Electrostatics | Constant 128k atoms, variable system and batch sizes', fontsize=TITLE_SIZE - 2, fontweight='bold', loc='left')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot combined benchmark results from CSV files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python plot_combined_benchmarks.py ./benchmark-results/benchmark_2026-01-27_12-00-00
    python plot_combined_benchmarks.py --input-dir ./results --output-dir ./plots
    python plot_combined_benchmarks.py -i ./results -o ./plots --no-titles
        """
    )
    
    parser.add_argument(
        'input_dir',
        nargs='?',
        type=Path,
        default=None,
        help='Directory containing benchmark CSV files (required)'
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=Path,
        default=None,
        dest='input_dir_opt',
        help='Directory containing benchmark CSV files (alternative to positional)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=None,
        help='Output directory for plots (default: same as input)'
    )
    
    parser.add_argument(
        '--no-titles',
        action='store_true',
        help='Generate plots without panel titles (for slides)'
    )
    
    parser.add_argument(
        '--include-charge-gradients',
        action='store_true',
        help='Include charge gradient lines in electrostatics plots'
    )
    
    args = parser.parse_args()
    
    # Resolve input directory (positional takes precedence over --input-dir)
    if args.input_dir is not None:
        args.resolved_input_dir = args.input_dir
    elif args.input_dir_opt is not None:
        args.resolved_input_dir = args.input_dir_opt
    else:
        parser.error('Input directory is required. Specify as positional argument or with --input-dir')
    
    # Resolve output directory
    if args.output_dir is not None:
        args.resolved_output_dir = args.output_dir
    else:
        args.resolved_output_dir = args.resolved_input_dir
    
    return args


def main():
    args = parse_args()
    
    input_dir = args.resolved_input_dir
    output_dir = args.resolved_output_dir
    
    # Update global flags based on args
    global SHOW_TITLES, INCLUDE_CHARGE_GRADIENTS
    if args.no_titles:
        SHOW_TITLES = False
    if args.include_charge_gradients:
        INCLUDE_CHARGE_GRADIENTS = True
    
    print('='*70)
    print('COMBINED BENCHMARK PLOTTING')
    print('='*70)
    print(f'Input directory: {input_dir}')
    print(f'Output directory: {output_dir}')
    
    if not input_dir.exists():
        print(f'ERROR: Input directory does not exist: {input_dir}')
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all CSVs with warnings for missing files
    nl_csv = input_dir / 'benchmark_nl_results.csv'
    d3_csv = input_dir / 'benchmark_d3_scaling_results.csv'
    electro_csv = input_dir / 'benchmark_electrostatics_results.csv'
    electro_128k_csv = input_dir / 'benchmark_electrostatics_128k_results.csv'
    
    print()
    nl_results = load_csv(nl_csv)
    if nl_results:
        print(f'Loaded: {nl_csv.name} ({len(nl_results)} rows)')
    else:
        print(f'WARNING: {nl_csv.name} not found - Panel A will be empty')
    
    d3_results = load_csv(d3_csv)
    if d3_results:
        print(f'Loaded: {d3_csv.name} ({len(d3_results)} rows)')
    else:
        print(f'WARNING: {d3_csv.name} not found - Panel B will be empty')
    
    electro_results = load_csv(electro_csv)
    if electro_results:
        print(f'Loaded: {electro_csv.name} ({len(electro_results)} rows)')
    else:
        print(f'WARNING: {electro_csv.name} not found - Panel C will be empty')
    
    electro_128k_results = load_csv(electro_128k_csv)
    if electro_128k_results:
        print(f'Loaded: {electro_128k_csv.name} ({len(electro_128k_results)} rows)')
    else:
        print(f'WARNING: {electro_128k_csv.name} not found - Panel D will be empty')
    
    # Create figure - 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # Panel A: NL scaling (top-left)
    plot_nl_scaling(nl_results, axes[0, 0])
    
    # Panel B: D3 scaling (top-right)
    plot_d3_scaling(d3_results, axes[0, 1])
    
    # Panel C: Electrostatics 128k batched (bottom-left)
    plot_electrostatics_batched(electro_128k_results, axes[1, 0])
    
    # Panel D: Electrostatics scaling (bottom-right)
    plot_electrostatics_scaling(electro_results, axes[1, 1])
    
    # Main title
    fig.suptitle('NVIDIA Alchemi Toolkit-Ops Benchmarks (H100)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.55, wspace=0.25, top=0.92)
    
    # Save with appropriate filename based on --no-titles flag
    if args.no_titles:
        output_path = output_dir / 'combined_benchmarks_notitle.png'
    else:
        output_path = output_dir / 'combined_benchmarks.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nSaved: {output_path}')
    plt.close(fig)
    
    print('\nDone!')
    return 0


if __name__ == '__main__':
    exit(main())
