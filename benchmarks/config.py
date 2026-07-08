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

"""Shared CLI argument definitions, YAML loader, and override merging.

Each per-module benchmark runner reuses these helpers so the common flags
(``--system``, ``--mode``, ``--timing-runs``, ``--warmup-runs``,
``--output-dir``, ``--backend``, ``--dry-run``, ``--list``, ``--count``,
``--max-total-atoms``)
stay in sync. Module-specific flags (``--cutoffs``, ``--accuracies``) are added
by each runner's own ``parse_args`` on top of :func:`add_common_cli_args`.
The shared method selector accepts both ``--method`` and the compatibility
``--methods`` spelling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

__all__ = [
    "add_common_cli_args",
    "enabled_method_names",
    "load_yaml_config",
    "merge_common_cli_overrides",
    "normalize_method_name",
]

_METHOD_ALIASES = {
    "naive": "naive_neighbor_list",
    "naive-scalar": "naive_scalar",
    "naive-tile": "naive_tile",
    "batch-naive-scalar": "batch_naive_scalar",
    "batch-naive-tile": "batch_naive_tile",
    "batch_naive": "batch_naive_neighbor_list",
    "cell": "cell_list",
    "cell-atom": "cell_list_atom_centric",
    "cell-list-atom": "cell_list_atom_centric",
    "cell-list-atom-centric": "cell_list_atom_centric",
    "cell-pair": "cell_list_pair_centric",
    "cell-list-pair": "cell_list_pair_centric",
    "cell-list-pair-centric": "cell_list_pair_centric",
    "batch-cell-atom": "batch_cell_list_atom_centric",
    "batch-cell-list-atom": "batch_cell_list_atom_centric",
    "batch-cell-list-atom-centric": "batch_cell_list_atom_centric",
    "batch-cell-pair": "batch_cell_list_pair_centric",
    "batch-cell-list-pair": "batch_cell_list_pair_centric",
    "batch-cell-list-pair-centric": "batch_cell_list_pair_centric",
    "batch-cell-list": "batch_cell_list",
    "batch-naive": "batch_naive_neighbor_list",
    "cluster": "cluster_tile",
    "tile": "cluster_tile",
    "cluster-tile": "cluster_tile",
    "batch_cluster": "batch_cluster_tile",
    "batch-cluster": "batch_cluster_tile",
    "batch-cluster-tile": "batch_cluster_tile",
    "d3": "dftd3",
}


def normalize_method_name(method: str) -> str:
    """Return the canonical benchmark method name for a CLI/config token."""
    return _METHOD_ALIASES.get(method, method)


def enabled_method_names(config: dict) -> list[str]:
    """Return canonical names for enabled methods in a benchmark config."""
    selected = config.get("runtime", {}).get("selected_methods")
    if selected:
        return [normalize_method_name(m) for m in selected]
    return [
        normalize_method_name(m["name"])
        for m in config.get("methods", [])
        if m.get("enabled", True)
    ]


def load_yaml_config(config_path: str | Path) -> dict:
    """Load benchmark configuration from a YAML file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def add_common_cli_args(
    parser: argparse.ArgumentParser,
    *,
    backends: tuple[str, ...] = ("torch", "jax", "warp"),
    include_d3_params_path: bool = False,
) -> None:
    """Register the flags shared across every benchmark runner and the suite.

    Does NOT add ``--config`` — runners declare that themselves
    (``required=True``) while the suite resolves per-module configs from its
    own ``RUNNERS`` map. ``backends`` and ``include_d3_params_path`` keep each
    runner's help limited to options it can actually execute.
    """
    parser.add_argument(
        "--system",
        "-s",
        nargs="+",
        default=None,
        help="Filter systems (subset of config['systems'] keys, or 'all')",
    )
    parser.add_argument(
        "--mode",
        "-m",
        nargs="+",
        default=None,
        help="Filter scaling modes (system_size, constant_workload, batch_scaling, or all)",
    )
    parser.add_argument(
        "--timing-runs",
        "-n",
        type=int,
        default=None,
        help="Override timing iterations",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=None,
        help="Override warmup iterations",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=backends,
        help="Framework backend (default: torch)",
    )
    parser.add_argument(
        "--method",
        "--methods",
        dest="methods",
        nargs="+",
        default=None,
        help="Restrict benchmark methods/APIs. Accepted values are module-specific.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded benchmark plan and exit without GPU allocation.",
    )
    parser.add_argument(
        "--list",
        dest="list_plan",
        action="store_true",
        help="Alias for --dry-run: print the expanded benchmark plan and exit.",
    )
    parser.add_argument(
        "--count",
        dest="count_plan",
        action="store_true",
        help="Print only the expanded benchmark row counts and exit.",
    )
    parser.add_argument(
        "--max-total-atoms",
        type=int,
        default=None,
        help=(
            "Opt-in planning filter: write SkippedByPolicy rows for concrete "
            "cases above this total atom count before allocation."
        ),
    )
    if include_d3_params_path:
        parser.add_argument(
            "--d3-params-path",
            type=Path,
            default=None,
            help=(
                "Override the DFT-D3 reference parameter cache path. Useful for "
                "scratch-only cluster runs without outbound network access."
            ),
        )


def merge_common_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply the shared CLI flags on top of a YAML config.

    Only non-None CLI values override. Mutates and returns ``config``.
    Module-specific flags (cutoffs, methods, accuracies) are handled by
    each runner's own ``merge_cli_overrides`` on top of this.
    """
    if args.timing_runs is not None:
        config["parameters"]["timing_runs"] = args.timing_runs
    if args.warmup_runs is not None:
        config["parameters"]["warmup_runs"] = args.warmup_runs

    if args.system is not None and "all" not in args.system:
        for sys_name in list(config["systems"].keys()):
            config["systems"][sys_name]["enabled"] = sys_name in args.system

    if args.mode is not None and "all" not in args.mode:
        for mode_name in list(config["scaling"].keys()):
            if isinstance(config["scaling"][mode_name], dict):
                config["scaling"][mode_name]["enabled"] = mode_name in args.mode

    if args.output_dir is not None:
        config["output"]["base_dir"] = str(args.output_dir)
    if getattr(args, "backend", None) is not None:
        config.setdefault("runtime", {})["backend"] = args.backend
    plan_mode = None
    if getattr(args, "count_plan", False):
        plan_mode = "count"
    elif getattr(args, "list_plan", False):
        plan_mode = "list"
    elif getattr(args, "dry_run", False):
        plan_mode = "dry_run"
    if plan_mode is not None:
        config.setdefault("runtime", {})["dry_run"] = True
        config.setdefault("runtime", {})["plan_output"] = plan_mode
    if getattr(args, "max_total_atoms", None) is not None:
        max_total_atoms = args.max_total_atoms
        config.setdefault("parameters", {})["max_total_atoms"] = max_total_atoms
        for mode_config in config.get("scaling", {}).values():
            if isinstance(mode_config, dict) and "max_total_atoms" in mode_config:
                mode_config["max_total_atoms"] = max_total_atoms
    if getattr(args, "d3_params_path", None) is not None and "params_path" in config:
        config["params_path"] = str(args.d3_params_path)
    if getattr(args, "methods", None) is not None and "methods" in config:
        selected_list = [normalize_method_name(m) for m in args.methods]
        selected = set(selected_list)
        config.setdefault("runtime", {})["explicit_methods"] = True
        config.setdefault("runtime", {})["selected_methods"] = selected_list
        for method in config["methods"]:
            method["enabled"] = normalize_method_name(method["name"]) in selected

    return config
