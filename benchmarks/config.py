# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared CLI argument definitions, YAML loader, and override merging.

Each per-module benchmark runner reuses these helpers so the common flags
(``--system``, ``--mode``, ``--timing-runs``, ``--timing-mode``,
``--warmup-runs``, ``--output-dir``, ``--backend``) stay in sync. Module-
specific flags (``--cutoffs``, ``--methods``, ``--accuracies``) are added
by each runner's own ``parse_args`` on top of :func:`add_common_cli_args`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

__all__ = [
    "add_common_cli_args",
    "load_yaml_config",
    "merge_common_cli_overrides",
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


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register the flags shared across every benchmark runner and the suite.

    Does NOT add ``--config`` — runners declare that themselves
    (``required=True``) while the suite resolves per-module configs from its
    own ``RUNNERS`` map.
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
        choices=["torch", "jax"],
        help="Framework backend (default: torch)",
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

    return config
