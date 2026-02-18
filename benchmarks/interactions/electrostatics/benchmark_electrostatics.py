#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Electrostatics Benchmark (Ewald + PME).

CRITICAL: Electrostatics requires float64 for positions and cells.
K-vectors are pre-computed ONCE outside the timing loop.

Usage:
    cd benchmarks/interactions/electrostatics
    python benchmark_electrostatics.py --config benchmark_config.yaml
    python benchmark_electrostatics.py --config benchmark_config.yaml --output-dir ../../../docs/benchmarks/benchmark_results
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
from nvalchemiops.torch.interactions.electrostatics import (
    estimate_ewald_parameters,
    estimate_pme_parameters,
    ewald_summation,
    generate_k_vectors_ewald_summation,
    generate_k_vectors_pme,
    particle_mesh_ewald,
)
from nvalchemiops.torch.neighbors import batch_naive_neighbor_list

# =============================================================================
# Config Loading
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
    if args.system is not None and "all" not in args.system:
        for sys_name in list(config["systems"].keys()):
            config["systems"][sys_name]["enabled"] = sys_name in args.system
    if args.mode is not None and "all" not in args.mode:
        for mode_name in list(config["scaling"].keys()):
            if isinstance(config["scaling"][mode_name], dict):
                config["scaling"][mode_name]["enabled"] = mode_name in args.mode
    if args.accuracies is not None:
        config["accuracies"] = args.accuracies
    if args.methods is not None:
        for method in config.get("methods", []):
            method["enabled"] = method["name"] in args.methods
    if args.output_dir is not None:
        config["output"]["base_dir"] = str(args.output_dir)
    if args.gpu_sku is not None:
        config["output"]["gpu_sku_override"] = args.gpu_sku
    return config


# =============================================================================
# Core Benchmarks
# =============================================================================


def benchmark_pme(
    positions,
    charges,
    cell,
    batch_idx,
    nl_data,
    nl_shifts,
    nl_ptr,
    alpha,
    mesh_dims,
    accuracy,
    compute_cg,
    num_runs,
    timing_mode,
    warmup_runs,
):
    """Benchmark Particle Mesh Ewald.

    Accepts pre-converted f64 tensors to avoid redundant GPU copies.
    """
    spline_order = 4

    # Pre-compute k-vectors ONCE (outside timing loop)
    k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dims)

    def run_pme():
        particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=spline_order,
            batch_idx=batch_idx,
            k_vectors=k_vectors,  # noqa: F821
            k_squared=k_squared,  # noqa: F821
            neighbor_list=nl_data,
            neighbor_ptr=nl_ptr,
            neighbor_shifts=nl_shifts,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
            accuracy=accuracy,
        )

    # Memory from single warmup run
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    run_pme()
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
    time_sec = cuda_timed_runs(
        run_pme, num_runs, mode=timing_mode, warmup_runs=warmup_runs
    )

    # Free k-vectors immediately after timing (they can be large)
    del k_vectors, k_squared
    return {"time_seconds": time_sec, "mem_info": mem_info}


def benchmark_ewald(
    positions,
    charges,
    cell,
    batch_idx,
    nl_data,
    nl_shifts,
    nl_ptr,
    alpha,
    k_cutoff,
    accuracy,
    compute_cg,
    num_runs,
    timing_mode,
    warmup_runs,
):
    """Benchmark Ewald summation using the unified ewald_summation() API.

    Accepts pre-converted f64 tensors to avoid redundant GPU copies.
    """
    # Pre-compute k-vectors ONCE (outside timing loop)
    k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff)
    if k_vectors.ndim == 2:
        k_vectors = k_vectors.unsqueeze(0)

    def run_ewald():
        ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            batch_idx=batch_idx,
            neighbor_list=nl_data,
            neighbor_ptr=nl_ptr,
            neighbor_shifts=nl_shifts,
            k_vectors=k_vectors,  # noqa: F821
            compute_forces=True,
            compute_charge_gradients=compute_cg,
            accuracy=accuracy,
        )

    # Memory from single warmup run
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    run_ewald()
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
    time_sec = cuda_timed_runs(
        run_ewald, num_runs, mode=timing_mode, warmup_runs=warmup_runs
    )

    # Free k-vectors immediately after timing (they can be large)
    del k_vectors
    return {"time_seconds": time_sec, "mem_info": mem_info}


# =============================================================================
# Config-Driven Runner
# =============================================================================


def run_from_config(config, output_dir=None):
    """Run electrostatics benchmarks driven by YAML config."""
    params = config["parameters"]
    num_runs = params["timing_runs"]
    timing_mode = params["timing_mode"]
    warmup_runs = params["warmup_runs"]
    accuracies = config.get("accuracies", [1e-4, 1e-6])
    max_atoms = config.get("max_atoms", 131072)
    skip_accuracy_for_large = config.get("skip_accuracy_for_large", {})
    cg_options = config.get("compute_charge_gradients", [False, True])
    methods_config = config.get("methods", [])
    method_names = [m["name"] for m in methods_config if m.get("enabled", True)]

    if output_dir is None:
        output_dir = create_run_directory(config["output"]["base_dir"], prefix="el")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Electrostatics Benchmark | GPU: {torch.cuda.get_device_name(0)}")
    print(f"Methods: {method_names} | Accuracies: {accuracies}")
    print(f"Timing: {num_runs} runs")
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
            print(f"ELECTROSTATICS: {sys_name.upper()} / {mode_name}")
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

            for accuracy in accuracies:
                print(f"\n  --- Accuracy: {accuracy:.0e} ---")

                for cfg in configs:
                    n, bs = cfg["num_atoms"], cfg["batch_size"]

                    # OOM prevention (use actual CsCl atom count for threshold)
                    from benchmarks.systems import cscl_actual_atoms

                    actual_n_est = cscl_actual_atoms(n) if sys_name == "cscl" else n
                    skip_threshold = skip_accuracy_for_large.get(
                        accuracy
                    ) or skip_accuracy_for_large.get(str(accuracy))
                    if skip_threshold and actual_n_est >= skip_threshold:
                        print(
                            f"  {format_num(actual_n_est)}: SKIP (OOM risk at {accuracy:.0e})"
                        )
                        continue
                    total_est = actual_n_est * bs
                    if total_est > max_atoms:
                        print(
                            f"  {format_num(actual_n_est)}×{bs}: SKIP (>{format_num(max_atoms)})"
                        )
                        continue

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

                    atoms_per_system = data["atoms_per_system"]
                    actual_total = data.get("total_atoms", atoms_per_system)
                    batch_size = data.get("batch_size", 1)
                    alloc_gb = torch.cuda.memory_allocated() / 1024**3
                    print(
                        f"\n  {format_num(atoms_per_system)} atoms × {batch_size} batch = {format_num(actual_total)} total  [GPU: {alloc_gb:.1f} GB allocated]"
                    )
                    pbc = data["pbc"]

                    # Convert to f64 once, then drop original f32 data dict
                    positions = data["positions"].to(torch.float64)
                    charges = data["charges"].to(torch.float64)
                    cell = data["cell"].to(torch.float64)
                    batch_idx = data.get("batch_idx")
                    del data  # Free f32 tensors immediately

                    # Estimate parameters
                    try:
                        pme_params = estimate_pme_parameters(
                            positions, cell, batch_idx=batch_idx, accuracy=accuracy
                        )
                        ewald_params = estimate_ewald_parameters(
                            positions, cell, batch_idx=batch_idx, accuracy=accuracy
                        )
                    except Exception as e:
                        print(f"    SKIP (params): {e}")
                        del positions, charges, cell, batch_idx
                        continue

                    alpha = pme_params.alpha.clone()
                    if alpha.dim() > 0:
                        alpha = alpha.mean()
                    real_cutoff = (
                        pme_params.real_space_cutoff[0].item()
                        if pme_params.real_space_cutoff.dim() > 0
                        else pme_params.real_space_cutoff.item()
                    )
                    mesh_dims = tuple(pme_params.mesh_dimensions)
                    k_cutoff = ewald_params.reciprocal_space_cutoff.max().item()
                    del pme_params, ewald_params

                    # Generate NL in LIST format (COO + ptr)
                    if batch_idx is None:  # noqa: F821
                        batch_idx = torch.zeros(
                            positions.shape[0],  # noqa: F821
                            dtype=torch.int32,
                            device=positions.device,  # noqa: F821
                        )
                    maxnb = estimate_max_neighbors(
                        real_cutoff, atomic_density=0.2, safety_factor=1.0
                    )

                    try:
                        nl_data, nl_ptr, nl_shifts = batch_naive_neighbor_list(
                            positions=positions,  # noqa: F821
                            cutoff=real_cutoff,
                            batch_idx=batch_idx,
                            pbc=pbc,
                            cell=cell,  # noqa: F821
                            max_neighbors=maxnb,
                            return_neighbor_list=True,
                        )
                    except Exception as e:
                        print(f"    SKIP (NL): {e}")
                        del positions, charges, cell, batch_idx  # noqa: F821
                        continue

                    print(
                        f"    alpha={alpha.item():.4f}, r_cut={real_cutoff:.2f}Å, NL pairs={nl_data.shape[1]:,}"
                    )

                    for method in method_names:
                        for compute_cg in cg_options:
                            cg_label = "+cg" if compute_cg else ""
                            label = f"{method.upper()}{cg_label}"

                            try:
                                if method == "pme":
                                    r = benchmark_pme(
                                        positions,  # noqa: F821
                                        charges,  # noqa: F821
                                        cell,  # noqa: F821
                                        batch_idx,  # noqa: F821
                                        nl_data,
                                        nl_shifts,
                                        nl_ptr,
                                        alpha,
                                        mesh_dims,
                                        accuracy,
                                        compute_cg,
                                        num_runs,
                                        timing_mode,
                                        warmup_runs,
                                    )
                                else:
                                    r = benchmark_ewald(
                                        positions,  # noqa: F821
                                        charges,  # noqa: F821
                                        cell,  # noqa: F821
                                        batch_idx,  # noqa: F821
                                        nl_data,
                                        nl_shifts,
                                        nl_ptr,
                                        alpha,
                                        k_cutoff,
                                        accuracy,
                                        compute_cg,
                                        num_runs,
                                        timing_mode,
                                        warmup_runs,
                                    )

                                result = build_result(
                                    system=sys_name,
                                    scaling_mode=mode_name,
                                    method=f"{method}{'_cg' if compute_cg else ''}",
                                    atoms_per_system=atoms_per_system,
                                    batch_size=batch_size,
                                    total_atoms=actual_total,
                                    time_seconds=r["time_seconds"],
                                    mem_info=r["mem_info"],
                                    accuracy=accuracy,
                                    alpha=alpha.item(),
                                    real_space_cutoff=real_cutoff,
                                    compute_charge_gradients=compute_cg,
                                )
                                results.append(result)
                                print(
                                    f"    {label:10s}: {result['time_us_per_atom']:.3f} μs/atom | {result['mem_peak_gb']:.1f} GB"
                                )
                            except torch.cuda.OutOfMemoryError:
                                print(f"    {label:10s}: OOM")
                                clean_gpu()
                            except Exception as e:
                                print(f"    {label:10s}: FAILED - {e}")

                    # --- Explicit cleanup: free ALL GPU tensors before next config ---
                    del positions, charges, cell, batch_idx  # noqa: F821
                    del nl_data, nl_ptr, nl_shifts

            if results:
                csv_name = make_csv_name("el", sys_name, mode_name)
                save_results(results, output_dir / csv_name)
                all_results.extend(results)

    print(f"\nCOMPLETE: {len(all_results)} results in {output_dir}")
    return all_results


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command-line arguments for electrostatics benchmarks."""
    parser = argparse.ArgumentParser(
        description="Electrostatics Benchmark (2 systems × 3 modes)"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--system", "-s", nargs="+", default=None)
    parser.add_argument("--mode", "-m", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None, choices=["pme", "ewald"])
    parser.add_argument("--accuracies", "-a", type=float, nargs="+", default=None)
    parser.add_argument("--timing-runs", "-n", type=int, default=None)
    parser.add_argument("--timing-mode", default=None, choices=["batch", "per_run"])
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--output-dir", "-o", type=Path, default=None)
    parser.add_argument("--gpu-sku", default=None)
    return parser.parse_args()


def main():
    """Run electrostatics benchmarks."""
    args = parse_args()
    config = load_config(args.config)
    config = merge_cli_overrides(config, args)
    run_from_config(config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
