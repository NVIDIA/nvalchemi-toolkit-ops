#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFT-D3 Dispersion Benchmark.

CRITICAL: D3 operates in atomic units (Bohr). All positions, cells, and
cutoffs are converted from Angstroms to Bohr before calling the D3 API.
Times NL and D3 separately.

Usage:
    cd benchmarks/interactions/dispersion
    python benchmark_dftd3.py --config benchmark_config.yaml
    python benchmark_dftd3.py --config benchmark_config.yaml --output-dir ../../../docs/benchmarks/benchmark_results
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
from nvalchemiops.neighbors import estimate_max_neighbors
from nvalchemiops.torch.interactions.dispersion import dftd3
from nvalchemiops.torch.neighbors import batch_cell_list

ANGSTROM_TO_BOHR = 1.8897259886


# =============================================================================
# Config Loading (same pattern as NL)
# =============================================================================


def load_config(config_path):
    """Load and validate YAML benchmark configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(config, args):
    """Merge CLI argument overrides into the YAML config."""
    if args.timing_runs is not None:
        config["parameters"]["timing_runs"] = args.timing_runs
    if args.timing_mode is not None:
        config["parameters"]["timing_mode"] = args.timing_mode
    if args.warmup_runs is not None:
        config["parameters"]["warmup_runs"] = args.warmup_runs
    if args.cutoffs is not None:
        config["parameters"]["cutoffs"] = args.cutoffs
    if args.system is not None and "all" not in args.system:
        for sys_name in list(config["systems"].keys()):
            config["systems"][sys_name]["enabled"] = sys_name in args.system
    if args.mode is not None and "all" not in args.mode:
        for mode_name in list(config["scaling"].keys()):
            if isinstance(config["scaling"][mode_name], dict):
                config["scaling"][mode_name]["enabled"] = mode_name in args.mode
    if args.output_dir is not None:
        config["output"]["base_dir"] = str(args.output_dir)
    if args.gpu_sku is not None:
        config["output"]["gpu_sku_override"] = args.gpu_sku
    return config


# =============================================================================
# Core Benchmark
# =============================================================================


def benchmark_d3(
    data,
    cutoff,
    d3_params,
    d3_func_params,
    num_runs,
    timing_mode="batch",
    warmup_runs=3,
):
    """Benchmark D3 for a single configuration. Times NL and D3 separately."""
    positions = data["positions"]
    cell = data["cell"]
    pbc = data["pbc"]
    batch_idx = data.get("batch_idx")
    numbers = data["atomic_numbers"]

    if batch_idx is None:
        batch_idx = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )

    # Convert to Bohr
    pos_bohr = positions * ANGSTROM_TO_BOHR
    cell_bohr = cell * ANGSTROM_TO_BOHR
    cutoff_bohr = cutoff * ANGSTROM_TO_BOHR

    maxnb = estimate_max_neighbors(cutoff, atomic_density=0.2, safety_factor=1.0)

    a1 = d3_func_params.get("a1", 0.4145)
    a2 = d3_func_params.get("a2", 4.8593)
    s8 = d3_func_params.get("s8", 1.2177)

    # Single warmup: build NL + run D3, capture memory
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    nbmat, _, nbmat_shifts = batch_cell_list(
        positions=pos_bohr,
        cell=cell_bohr,
        pbc=pbc,
        cutoff=cutoff_bohr,
        batch_idx=batch_idx,
        max_neighbors=maxnb,
    )
    dftd3(
        positions=pos_bohr,
        cell=cell_bohr,
        numbers=numbers,
        batch_idx=batch_idx,
        neighbor_matrix=nbmat,
        neighbor_matrix_shifts=nbmat_shifts,
        d3_params=d3_params,
        a1=a1,
        a2=a2,
        s8=s8,
    )
    torch.cuda.synchronize()
    wp.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    mem_delta = mem_peak - mem_before
    gpu_info = get_gpu_memory_info()
    mem_info = {
        "mem_delta_bytes": mem_delta,
        "mem_peak_bytes": mem_peak,
        "mem_delta_mb": mem_delta / 1024**2,
        "mem_peak_gb": mem_peak / 1024**3,
        "mem_gpu_percent": 100.0 * mem_peak / gpu_info["total"],
    }

    # NL timing
    def run_nl():
        nonlocal nbmat, nbmat_shifts
        nbmat, _, nbmat_shifts = batch_cell_list(
            positions=pos_bohr,
            cell=cell_bohr,
            pbc=pbc,
            cutoff=cutoff_bohr,
            batch_idx=batch_idx,
            max_neighbors=maxnb,
        )

    time_nl = cuda_timed_runs(
        run_nl, num_runs, mode=timing_mode, warmup_runs=warmup_runs
    )

    # D3 timing (uses pre-computed NL from last run_nl call)
    def run_d3():
        dftd3(
            positions=pos_bohr,
            cell=cell_bohr,
            numbers=numbers,
            batch_idx=batch_idx,
            neighbor_matrix=nbmat,
            neighbor_matrix_shifts=nbmat_shifts,
            d3_params=d3_params,
            a1=a1,
            a2=a2,
            s8=s8,
        )

    time_d3 = cuda_timed_runs(
        run_d3, num_runs, mode=timing_mode, warmup_runs=warmup_runs
    )

    return {
        "time_nl_seconds": time_nl,
        "time_d3_seconds": time_d3,
        "time_total_seconds": time_nl + time_d3,
        "mem_info": mem_info,
    }


# =============================================================================
# Config-Driven Runner
# =============================================================================


def run_from_config(config, output_dir=None):
    """Run D3 benchmarks driven by YAML config."""
    params = config["parameters"]
    num_runs = params["timing_runs"]
    timing_mode = params["timing_mode"]
    warmup_runs = params["warmup_runs"]
    cutoffs = params["cutoffs"]
    d3_func_params = config.get("dftd3_parameters", {})

    # Load D3 reference parameters
    d3_params_path = Path(
        config.get("params_path", "~/.cache/nvalchemiops/dftd3_parameters.pt")
    )
    d3_params_path = d3_params_path.expanduser()
    if not d3_params_path.exists():
        print(f"ERROR: D3 parameters not found at {d3_params_path}")
        print("Run: python examples/dispersion/01_dftd3_molecule.py (downloads params)")
        return []
    d3_params = torch.load(d3_params_path, map_location="cuda", weights_only=True)
    print(f"Loaded D3 parameters from {d3_params_path}")

    if output_dir is None:
        output_dir = create_run_directory(config["output"]["base_dir"], prefix="d3")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"D3 Benchmark Suite | GPU: {torch.cuda.get_device_name(0)}")
    print(f"Cutoffs: {cutoffs} Å | Timing: {num_runs} runs")
    print(f"Output: {output_dir}")

    all_results = []

    for sys_name, sys_config in config["systems"].items():
        if not sys_config.get("enabled", True):
            continue

        nh3_dir = sys_config.get("pdb_dir")
        if nh3_dir:
            nh3_dir = Path(nh3_dir)
            if not nh3_dir.is_absolute():
                nh3_dir = Path(__file__).parent.parent.parent / nh3_dir.name
                if not nh3_dir.exists():
                    nh3_dir = Path(__file__).parent.parent.parent / "nh3"

        atom_counts = sys_config.get("atom_counts", [])
        constant_atoms_sizes = sys_config.get("constant_atoms_sizes", [1024, 8192])

        for mode_name, mode_config in config["scaling"].items():
            if not isinstance(mode_config, dict) or not mode_config.get(
                "enabled", True
            ):
                continue

            print(f"\n{'=' * 70}")
            print(f"D3: {sys_name.upper()} / {mode_name}")
            print(f"{'=' * 70}")

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

                try:
                    actual_total = data.get("total_atoms", data["atoms_per_system"])
                    actual_n = data["atoms_per_system"]
                    alloc_gb = torch.cuda.memory_allocated() / 1024**3
                    print(
                        f"\n  {format_num(actual_n)} atoms × {bs} batch = {format_num(actual_total)} total"
                    )
                    print(f"    [GPU: {alloc_gb:.1f} GB allocated]")

                    for cutoff in cutoffs:
                        if data["cell_size"] < 2 * cutoff:
                            print(
                                f"    {cutoff}Å: WARNING cell {data['cell_size']:.1f}Å < 2×cutoff (benchmarking anyway)"
                            )

                        try:
                            r = benchmark_d3(
                                data,
                                cutoff,
                                d3_params,
                                d3_func_params,
                                num_runs,
                                timing_mode,
                                warmup_runs,
                            )
                            result = build_result(
                                system=sys_name,
                                scaling_mode=mode_name,
                                method="dftd3",
                                atoms_per_system=data["atoms_per_system"],
                                batch_size=data.get("batch_size", 1),
                                total_atoms=actual_total,
                                time_seconds=r["time_total_seconds"],
                                mem_info=r["mem_info"],
                                cutoff=cutoff,
                                time_nl_seconds=r["time_nl_seconds"],
                                time_d3_seconds=r["time_d3_seconds"],
                                time_nl_us_per_atom=(r["time_nl_seconds"] * 1e6)
                                / actual_total,
                                time_d3_us_per_atom=(r["time_d3_seconds"] * 1e6)
                                / actual_total,
                            )
                            results.append(result)
                            print(
                                f"    {cutoff}Å: NL={result['time_nl_us_per_atom']:.3f} "
                                f"D3={result['time_d3_us_per_atom']:.3f} μs/atom | "
                                f"{result['mem_delta_mb']:.1f} MB"
                            )
                        except torch.cuda.OutOfMemoryError:
                            print(f"    {cutoff}Å: OOM")
                            clean_gpu()
                        except Exception as e:
                            print(f"    {cutoff}Å: FAILED - {e}")
                finally:
                    # Free GPU tensors so gc.collect() in clean_gpu() can reclaim memory
                    del data
                    clean_gpu()

            if results:
                csv_name = make_csv_name("d3", sys_name, mode_name)
                save_results(results, output_dir / csv_name)
                all_results.extend(results)

    print(f"\nCOMPLETE: {len(all_results)} results in {output_dir}")
    return all_results


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command-line arguments for DFT-D3 benchmarks."""
    parser = argparse.ArgumentParser(
        description="DFT-D3 Benchmark (2 systems × 3 modes)"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--system", "-s", nargs="+", default=None)
    parser.add_argument("--mode", "-m", nargs="+", default=None)
    parser.add_argument("--cutoffs", "-c", type=float, nargs="+", default=None)
    parser.add_argument("--timing-runs", "-n", type=int, default=None)
    parser.add_argument("--timing-mode", default=None, choices=["batch", "per_run"])
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--output-dir", "-o", type=Path, default=None)
    parser.add_argument("--gpu-sku", default=None)
    return parser.parse_args()


def main():
    """Run DFT-D3 dispersion benchmarks."""
    args = parse_args()
    config = load_config(args.config)
    config = merge_cli_overrides(config, args)
    run_from_config(config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
