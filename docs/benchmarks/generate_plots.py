#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate single-panel benchmark plots for Sphinx documentation.

This script is called during the Sphinx build via the ``builder-inited`` hook
in ``docs/conf.py``. It reads CSV benchmark results and produces individual
PNG plots (one per metric) using the shared plotting infrastructure from
``benchmarks.plotting``.

The CSVs in ``benchmark_results/`` are committed to the repo so that
``make docs`` never needs to re-run benchmarks. Users can reproduce results
on their own hardware and replace the CSVs.

Note on imports: during a Sphinx build, ``docs/benchmarks/`` shadows the
project-root ``benchmarks/`` package in ``sys.modules``.  We use
``importlib.util`` to load from explicit file paths, fully bypassing
the normal package resolution.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_from_file(module_name: str, file_path: Path):
    """Load a Python file as a module with an arbitrary dotted name.

    Registers all ancestor packages in sys.modules so that intra-package
    imports (e.g. ``from benchmarks.plotting.styles import ...`` inside
    ``plot_benchmarks.py``) resolve correctly to the project-root tree.
    """
    parts = module_name.split(".")
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        if parent_name not in sys.modules:
            parent_pkg = types.ModuleType(parent_name)
            parent_pkg.__path__ = [
                str(_PROJECT_ROOT / "/".join(parts[:i]))
            ]
            parent_pkg.__package__ = parent_name
            sys.modules[parent_name] = parent_pkg

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_plotting_modules():
    """Load benchmarks.plotting.styles and benchmarks.plotting.plot_benchmarks."""
    bench_root = _PROJECT_ROOT / "benchmarks"
    plot_root = bench_root / "plotting"

    # Temporarily override sys.modules entries for the benchmarks tree
    # so that internal imports within plot_benchmarks.py resolve correctly.
    saved = {}
    override_keys = [
        "benchmarks",
        "benchmarks.plotting",
        "benchmarks.plotting.styles",
        "benchmarks.plotting.plot_benchmarks",
    ]
    for key in override_keys:
        if key in sys.modules:
            saved[key] = sys.modules.pop(key)

    try:
        styles = _load_from_file(
            "benchmarks.plotting.styles",
            plot_root / "styles.py",
        )
        plot_mod = _load_from_file(
            "benchmarks.plotting.plot_benchmarks",
            plot_root / "plot_benchmarks.py",
        )
    finally:
        # Restore originals so Sphinx can keep resolving docs/benchmarks/*
        for key in override_keys:
            if key in sys.modules and key not in saved:
                del sys.modules[key]
        sys.modules.update(saved)

    return styles, plot_mod


def main():
    """Generate all single-panel PNGs from benchmark CSVs."""
    _styles, _plot = _load_plotting_modules()

    results_dir = Path(__file__).parent / "benchmark_results"
    output_dir = Path(__file__).parent / "_static"
    output_dir.mkdir(exist_ok=True)

    _styles.setup_plot_style()

    csvs = sorted(results_dir.glob("*.csv"))
    if not csvs:
        print("  [generate_plots] No CSVs found, skipping plot generation")
        return

    print(f"  [generate_plots] Generating plots from {len(csvs)} CSVs")

    panels = ("time", "throughput", "memory")
    count = 0
    for csv_path in csvs:
        for panel in panels:
            out_path = output_dir / f"{csv_path.stem}-{panel}.png"
            try:
                _plot.plot_single_panel(csv_path, panel, out_path)
                count += 1
            except Exception as e:
                print(f"  [generate_plots] ERROR {csv_path.stem}-{panel}: {e}")

    print(f"  [generate_plots] Generated {count} plots in {output_dir}")


if __name__ == "__main__":
    main()
