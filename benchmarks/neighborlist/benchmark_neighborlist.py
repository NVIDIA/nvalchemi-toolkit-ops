#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neighbor List Benchmark.

Benchmarks naive O(N²) and cell-list O(N) neighbor list construction
across two chemical systems (CsCl, NH3) and three scaling modes.
Configuration is loaded from a per-module YAML file.

Usage:
    cd benchmarks/neighborlist
    python benchmark_neighborlist.py --config benchmark_config.yaml
    python benchmark_neighborlist.py --config benchmark_config.yaml --system cscl --mode system_size
    python benchmark_neighborlist.py --config benchmark_config.yaml --output-dir ../../docs/benchmarks/benchmark_results
"""

import argparse
from pathlib import Path

import torch
import warp as wp
import yaml

from benchmarks.systems import (
    create_system,
    get_constant_atoms_configs,
    get_constant_total_configs,
    get_system_size_configs,
)
from benchmarks.utils import (
    build_result,
    clean_gpu,
    create_run_directory,
    cuda_timed_runs,
    format_num,
    get_gpu_memory_info,
    make_csv_name,
    save_results,
)

# Official nvalchemiops APIs (tme branch)
from nvalchemiops.neighbors import estimate_max_neighbors
from nvalchemiops.torch.neighbors import (
    batch_cell_list,
    batch_naive_neighbor_list,
)

# =============================================================================
# Config Loading
# =============================================================================


def load_config(config_path):
    """Load benchmark configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to benchmark_config.yaml.

    Returns
    -------
    dict
        Parsed YAML configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(config, args):
    """Apply CLI overrides on top of YAML config.

    CLI arguments take precedence over YAML values.
    Only non-None CLI values override.

    Parameters
    ----------
    config : dict
        Parsed YAML config.
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    dict
        Config with CLI overrides applied.
    """
    # Timing overrides
    if args.timing_runs is not None:
        config["parameters"]["timing_runs"] = args.timing_runs
    if args.timing_mode is not None:
        config["parameters"]["timing_mode"] = args.timing_mode
    if args.warmup_runs is not None:
        config["parameters"]["warmup_runs"] = args.warmup_runs
    if args.cutoffs is not None:
        config["parameters"]["cutoffs"] = args.cutoffs

    # System filter: disable systems not requested
    if args.system is not None and "all" not in args.system:
        for sys_name in list(config["systems"].keys()):
            config["systems"][sys_name]["enabled"] = sys_name in args.system

    # Mode filter: disable modes not requested
    if args.mode is not None and "all" not in args.mode:
        for mode_name in list(config["scaling"].keys()):
            if isinstance(config["scaling"][mode_name], dict):
                config["scaling"][mode_name]["enabled"] = mode_name in args.mode

    # Method filter
    if args.methods is not None:
        for method in config.get("methods", []):
            method["enabled"] = method["name"] in args.methods

    # Output overrides
    if args.output_dir is not None:
        config["output"]["base_dir"] = str(args.output_dir)
    if args.gpu_sku is not None:
        config["output"]["gpu_sku_override"] = args.gpu_sku

    return config


# =============================================================================
# Core Benchmark Function
# =============================================================================


def benchmark_nl(data, cutoff, method, num_runs, timing_mode="batch", warmup_runs=3):
    """Benchmark a single NL configuration.

    Optimized: no redundant clean_gpu or extra kernel calls.
    Memory is measured from a single warmup run. Neighbor count
    captured from warmup result. clean_gpu() is the caller's
    responsibility (once per atom-size group, not per config).

    Parameters
    ----------
    data : dict
        System data from create_system().
    cutoff : float
        Cutoff distance in Angstroms.
    method : str
        'naive' or 'cell'.
    num_runs : int
        Number of timing iterations.
    timing_mode : str
        'batch' or 'per_run'.
    warmup_runs : int
        Number of warmup iterations.

    Returns
    -------
    dict
        Timing and memory results with NL-specific extras.
    """
    positions = data["positions"]
    cell = data["cell"]
    pbc = data["pbc"]
    batch_idx = data.get("batch_idx")
    total_atoms = data.get("total_atoms", data["atoms_per_system"])

    maxnb = estimate_max_neighbors(cutoff, atomic_density=0.2, safety_factor=1.0)
    nl_func = batch_cell_list if method == "cell" else batch_naive_neighbor_list

    # Ensure batch_idx exists even for single systems
    if batch_idx is None:
        batch_idx = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )

    def run_nl():
        return nl_func(
            positions=positions,
            cell=cell,
            pbc=pbc,
            cutoff=cutoff,
            batch_idx=batch_idx,
            max_neighbors=maxnb,
        )

    # Single warmup run: captures neighbor count + peak memory
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    result = run_nl()
    torch.cuda.synchronize()
    wp.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    mem_delta = mem_peak - mem_before
    gpu_info = get_gpu_memory_info()

    n_neighbors = int(result[0].shape[1]) if hasattr(result[0], "shape") else 0

    mem_info = {
        "mem_delta_bytes": mem_delta,
        "mem_peak_bytes": mem_peak,
        "mem_delta_mb": mem_delta / 1024**2,
        "mem_peak_gb": mem_peak / 1024**3,
        "mem_gpu_percent": 100.0 * mem_peak / gpu_info["total"],
    }

    # Timing (warmup inside cuda_timed_runs handles GPU pipeline warmup)
    time_sec = cuda_timed_runs(
        run_nl, num_runs, mode=timing_mode, warmup_runs=warmup_runs
    )

    return {
        "time_seconds": time_sec,
        "mem_info": mem_info,
        "max_neighbors": n_neighbors,
        "total_neighbor_pairs": total_atoms * n_neighbors,
    }


# =============================================================================
# Config-Driven Runner
# =============================================================================


def run_from_config(config, output_dir=None):
    """Run NL benchmarks driven entirely by YAML config.

    This is the main entry point, used both standalone and from benchmark_suite.py.

    Parameters
    ----------
    config : dict
        Merged config (YAML + CLI overrides).
    output_dir : Path, optional
        Override output directory. If None, uses config['output']['base_dir'].

    Returns
    -------
    list[dict]
        All benchmark results.
    """
    params = config["parameters"]
    num_runs = params["timing_runs"]
    timing_mode = params["timing_mode"]
    warmup_runs = params["warmup_runs"]
    cutoffs = params["cutoffs"]
    cutoff_limits = params.get("cutoff_limits", {})

    # Resolve output directory
    if output_dir is None:
        output_dir = create_run_directory(config["output"]["base_dir"], prefix="nl")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect enabled methods
    methods = [m["name"] for m in config.get("methods", []) if m.get("enabled", True)]

    print("NL Benchmark Suite")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Cutoffs: {cutoffs} Å | Methods: {methods}")
    print(f"Timing: {num_runs} runs")
    print(f"Output: {output_dir}")

    all_results = []

    # Iterate: systems × scaling modes
    for sys_name, sys_config in config["systems"].items():
        if not sys_config.get("enabled", True):
            continue

        # Resolve NH3 pdb_dir relative to benchmarks/ root
        nh3_dir = sys_config.get("pdb_dir")
        if nh3_dir:
            nh3_dir = Path(nh3_dir)
            if not nh3_dir.is_absolute():
                # Resolve relative to the config file's parent's parent (benchmarks/)
                nh3_dir = Path(__file__).parent.parent / nh3_dir.name
                if not nh3_dir.exists():
                    nh3_dir = Path(__file__).parent.parent / "nh3"

        atom_counts = sys_config.get("atom_counts", [])
        constant_atoms_sizes = sys_config.get("constant_atoms_sizes", [1024, 8192])

        for mode_name, mode_config in config["scaling"].items():
            if not isinstance(mode_config, dict) or not mode_config.get(
                "enabled", True
            ):
                continue

            print(f"\n{'=' * 70}")
            print(f"NL: {sys_name.upper()} / {mode_name}")
            print(f"{'=' * 70}")

            # Get configs for this (system, mode) combination
            if mode_name == "system_size":
                configs = list(get_system_size_configs(sys_name, atom_counts, nh3_dir))
            elif mode_name == "constant_workload":
                target = mode_config.get("target_atoms", 131072)
                configs = list(get_constant_total_configs(sys_name, target, nh3_dir))
            elif mode_name == "batch_scaling":
                max_total = mode_config.get("max_total_atoms", 131072)
                configs = list(
                    get_constant_atoms_configs(
                        sys_name, constant_atoms_sizes, max_total, nh3_dir
                    )
                )
            else:
                continue

            results = []

            for cfg in configs:
                n, bs = cfg["num_atoms"], cfg["batch_size"]

                clean_gpu()

                try:
                    data = create_system(
                        sys_name,
                        num_atoms=n,
                        pdb_path=cfg.get("pdb_path"),
                        batch_size=bs,
                    )
                except Exception as e:
                    print(f"    SKIP: {e}")
                    continue

                actual_total = data.get("total_atoms", data["atoms_per_system"])
                actual_n = data["atoms_per_system"]
                alloc_gb = torch.cuda.memory_allocated() / 1024**3
                print(
                    f"\n  {format_num(actual_n)} atoms × {bs} batch = {format_num(actual_total)} total"
                )
                print(f"  [GPU: {alloc_gb:.1f} GB allocated]")

                for cutoff in cutoffs:
                    # Check cutoff limits
                    limit = cutoff_limits.get(cutoff) or cutoff_limits.get(str(cutoff))
                    if limit and actual_total > limit:
                        print(f"    {cutoff}Å: SKIP (>{format_num(limit)} limit)")
                        continue

                    # Note: cell_size < 2*cutoff violates minimum image convention
                    # but we still benchmark for completeness. Document in sphinx docs.
                    if data["cell_size"] < 2 * cutoff:
                        print(
                            f"    {cutoff}Å: WARNING cell {data['cell_size']:.1f}Å < 2×cutoff (benchmarking anyway)"
                        )

                    for method in methods:
                        try:
                            r = benchmark_nl(
                                data,
                                cutoff,
                                method,
                                num_runs,
                                timing_mode,
                                warmup_runs,
                            )
                            result = build_result(
                                system=sys_name,
                                scaling_mode=mode_name,
                                method=method,
                                atoms_per_system=data["atoms_per_system"],
                                batch_size=data.get("batch_size", 1),
                                total_atoms=actual_total,
                                time_seconds=r["time_seconds"],
                                mem_info=r["mem_info"],
                                cutoff=cutoff,
                                max_neighbors=r["max_neighbors"],
                                throughput_pairs_per_sec=(
                                    r["total_neighbor_pairs"] / r["time_seconds"]
                                    if r["time_seconds"] > 0
                                    else 0.0
                                ),
                            )
                            results.append(result)
                            print(
                                f"    {cutoff}Å {method:5s}: "
                                f"{result['time_us_per_atom']:.3f} μs/atom | "
                                f"{result['throughput_matoms_per_sec']:.1f} Matom/s | "
                                f"{result['mem_delta_mb']:.1f} MB"
                            )
                        except torch.cuda.OutOfMemoryError:
                            print(f"    {cutoff}Å {method:5s}: OOM")
                            clean_gpu()
                        except Exception as e:
                            print(f"    {cutoff}Å {method:5s}: FAILED - {e}")

                # Explicit cleanup: free ALL GPU tensors before next config
                del data

            # Save per-(system, mode) CSV with standardized name
            if results:
                csv_name = make_csv_name("nl", sys_name, mode_name)
                save_results(results, output_dir / csv_name)
                all_results.extend(results)

    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {len(all_results)} results saved to {output_dir}")
    print(f"{'=' * 70}")

    return all_results


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command-line arguments for neighbor list benchmarks."""
    parser = argparse.ArgumentParser(
        description="Neighbor List Benchmark (2 systems × 3 modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_neighborlist.py --config benchmark_config.yaml
    python benchmark_neighborlist.py --config benchmark_config.yaml --system cscl --mode system_size
    python benchmark_neighborlist.py --config benchmark_config.yaml --cutoffs 6 15 --methods cell
    python benchmark_neighborlist.py --config benchmark_config.yaml --output-dir ../../docs/benchmarks/benchmark_results
        """,
    )
    # Required: config file
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to benchmark_config.yaml"
    )

    # Optional CLI overrides
    parser.add_argument(
        "--system",
        "-s",
        nargs="+",
        default=None,
        help="Systems to benchmark (cscl, nh3, or all)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        nargs="+",
        default=None,
        help="Scaling modes (system_size, constant_workload, batch_scaling, or all)",
    )
    parser.add_argument(
        "--cutoffs",
        "-c",
        type=float,
        nargs="+",
        default=None,
        help="Override cutoff radii in Angstroms",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None, help="Override NL methods (naive, cell)"
    )
    parser.add_argument(
        "--timing-runs",
        "-n",
        type=int,
        default=None,
        help="Override number of timing runs",
    )
    parser.add_argument(
        "--timing-mode",
        default=None,
        choices=["batch", "per_run"],
        help="Override timing mode",
    )
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None, help="Override output directory"
    )
    parser.add_argument(
        "--gpu-sku", default=None, help="Override GPU SKU name for filenames"
    )
    return parser.parse_args()


def main():
    """Run neighbor list benchmarks."""
    args = parse_args()

    # Load YAML config, merge CLI overrides
    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    # Run benchmarks from config
    run_from_config(config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
