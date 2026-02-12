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

"""
Neighbor List Scaling Benchmarks
=================================

CLI tool to benchmark neighbor list algorithms and generate CSV files
for documentation. Results are saved with GPU-specific naming:
`neighbor_list_benchmark_<method>_<gpu_sku>.csv`

Usage:
    python benchmark_neighborlist.py --config benchmark_config.yaml

The config file specifies which methods to benchmark and their parameters.
Results are saved per-method to allow selective benchmarking.
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path

import torch
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.systems import create_crystal_system
from benchmarks.utils import BenchmarkTimer
from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.batch_cell_list import estimate_batch_cell_list_sizes
from nvalchemiops.torch.neighbors.cell_list import estimate_cell_list_sizes
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
    compute_naive_num_shifts,
    estimate_max_neighbors,
)


def get_gpu_sku() -> str:
    """Get GPU SKU name for filename generation."""
    if not torch.cuda.is_available():
        return "cpu"

    try:
        gpu_name = torch.cuda.get_device_name(0)
        # Clean up GPU name for filename (remove spaces, special chars)
        sku = gpu_name.replace(" ", "-").replace("_", "-")
        # Remove common prefixes to shorten
        sku = sku.replace("NVIDIA-", "").replace("GeForce-", "")
        return sku.lower()
    except Exception:
        return "unknown_gpu"


def load_config(config_path: Path) -> dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict) -> None:
    """Validate benchmark configuration structure."""
    required_keys = ["methods", "parameters"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config missing required key: {key}")

    param_keys = ["cutoff", "warmup_iterations", "timing_iterations", "dtype"]
    for key in param_keys:
        if key not in config["parameters"]:
            raise ValueError(f"Config parameters missing required key: {key}")

    for method_config in config["methods"]:
        if (
            "name" not in method_config
            or "atom_counts" not in method_config
            or "batch_sizes" not in method_config
        ):
            raise ValueError(f"Method config missing required keys: {method_config}")


# %%
# Utility Functions
# -----------------
# Helper functions for preparing inputs and running benchmarks.


def prepare_inputs(method, atoms_per_system, batch_size, cutoff, device, dtype):
    """Prepare inputs for a specific neighbor list method with pre-allocated tensors."""
    is_batch = "batch" in method
    device_obj = torch.device(device)

    if is_batch:
        positions_list = []
        cells_list = []
        pbc_list = []
        batch_idx_list = []

        try:
            for i in range(batch_size):
                system = create_crystal_system(
                    atoms_per_system, lattice_type="fcc", device=device_obj, dtype=dtype
                )

                # Debug: Check system creation
                if (
                    "positions" not in system
                    or "cell" not in system
                    or "pbc" not in system
                ):
                    raise ValueError(
                        f"System {i} missing required keys. Has: {system.keys()}"
                    )

                # Validate positions shape
                pos = system["positions"]
                if pos.shape[0] != atoms_per_system:
                    raise ValueError(
                        f"System {i}: requested {atoms_per_system} atoms, got {pos.shape[0]}. "
                        f"FCC lattice may not support exact atom count."
                    )
                positions_list.append(pos)

                # Ensure cell has right shape before squeezing
                cell = system["cell"]
                if cell.ndim == 3:
                    cell = cell.squeeze(0)
                elif cell.ndim != 2 or cell.shape != (3, 3):
                    raise ValueError(
                        f"System {i} cell has unexpected shape: {cell.shape}. Expected (3,3) or (1,3,3)"
                    )
                cells_list.append(cell)

                # Ensure pbc has right shape
                pbc = system["pbc"]
                if pbc.ndim == 2:
                    pbc = pbc.squeeze(0)
                elif pbc.ndim != 1 or pbc.shape[0] != 3:
                    raise ValueError(
                        f"System {i} pbc has unexpected shape: {pbc.shape}. Expected (3,) or (1,3)"
                    )
                pbc_list.append(pbc)

                batch_idx_list.extend(
                    [i] * pos.shape[0]
                )  # Use actual atom count, not requested

        except Exception as e:
            raise ValueError(
                f"Error creating batch systems (atoms_per_system={atoms_per_system}, batch_size={batch_size}): {e}"
            ) from e

        # Check if we have any systems before stacking
        if not cells_list or not pbc_list:
            raise ValueError(
                f"No systems created for batching. cells_list={len(cells_list)}, pbc_list={len(pbc_list)}"
            )

        # Debug: Check list contents before stacking
        if len(cells_list) != batch_size:
            raise ValueError(f"Expected {batch_size} cells, got {len(cells_list)}")
        if len(pbc_list) != batch_size:
            raise ValueError(f"Expected {batch_size} pbc tensors, got {len(pbc_list)}")

        try:
            positions = torch.cat(positions_list)
            cells = torch.stack(cells_list)
            pbc = torch.stack(pbc_list)
            batch_idx = torch.tensor(batch_idx_list, dtype=torch.int32, device=device)
        except Exception as e:
            raise ValueError(
                f"Error stacking batch tensors: {e}. "
                f"cells_list[0].shape={cells_list[0].shape if cells_list else 'N/A'}, "
                f"pbc_list[0].shape={pbc_list[0].shape if pbc_list else 'N/A'}"
            ) from e

        # Prepare batch_ptr for batch methods
        batch_ptr = torch.arange(
            0,
            (batch_size + 1) * atoms_per_system,
            atoms_per_system,
            dtype=torch.int32,
            device=device,
        )

        total_atoms_actual = positions.shape[0]

        # Pre-allocate tensors
        max_neighbors = estimate_max_neighbors(
            cutoff, atomic_density=0.35, safety_factor=1.0
        )
        neighbor_matrix = torch.full(
            (total_atoms_actual, max_neighbors),
            total_atoms_actual,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts = torch.zeros(
            (total_atoms_actual, max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(
            total_atoms_actual, dtype=torch.int32, device=device
        )

        inputs = {
            "positions": positions,
            "cutoff": cutoff,
            "cell": cells,
            "pbc": pbc,
            "method": method,
            "batch_idx": batch_idx,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "num_neighbors": num_neighbors,
        }

        # Method-specific allocations
        if "naive" in method:
            # Pre-compute shifts for naive method
            shift_range_per_dimension, shift_offset, total_shifts = (
                compute_naive_num_shifts(cells, cutoff, pbc)
            )
            inputs["shift_range_per_dimension"] = shift_range_per_dimension
            inputs["shift_offset"] = shift_offset
            inputs["total_shifts"] = total_shifts
            inputs["batch_ptr"] = batch_ptr
        elif "cell_list" in method:
            # Pre-allocate cell list cache (use batch-specific estimator for batch methods)
            max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
                cells,
                pbc,
                cutoff,
            )
            cell_list_cache = allocate_cell_list(
                total_atoms_actual, max_total_cells, neighbor_search_radius, device
            )
            inputs["cells_per_dimension"] = cell_list_cache[0]
            inputs["neighbor_search_radius"] = cell_list_cache[1]
            inputs["atom_periodic_shifts"] = cell_list_cache[2]
            inputs["atom_to_cell_mapping"] = cell_list_cache[3]
            inputs["atoms_per_cell_count"] = cell_list_cache[4]
            inputs["cell_atom_start_indices"] = cell_list_cache[5]
            inputs["cell_atom_list"] = cell_list_cache[6]
            # Note: batch_cell_list uses batch_idx, not batch_ptr
            # batch_idx is already in inputs

        return inputs
    else:
        # Single system
        system = create_crystal_system(
            atoms_per_system, lattice_type="fcc", device=device_obj, dtype=dtype
        )

        positions = system["positions"]
        cell = system["cell"].reshape(1, 3, 3)
        pbc = system["pbc"].reshape(1, 3)
        total_atoms_actual = positions.shape[0]

        # Pre-allocate tensors
        max_neighbors = estimate_max_neighbors(
            cutoff, atomic_density=0.35, safety_factor=1.0
        )
        neighbor_matrix = torch.full(
            (total_atoms_actual, max_neighbors),
            total_atoms_actual,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts = torch.zeros(
            (total_atoms_actual, max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(
            total_atoms_actual, dtype=torch.int32, device=device
        )

        inputs = {
            "positions": positions,
            "cutoff": cutoff,
            "cell": cell,
            "pbc": pbc,
            "method": method,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "num_neighbors": num_neighbors,
        }

        # Method-specific allocations
        if "naive" in method:
            # Pre-compute shifts for naive method
            shift_range_per_dimension, shift_offset, total_shifts = (
                compute_naive_num_shifts(cell, cutoff, pbc)
            )
            inputs["shift_range_per_dimension"] = shift_range_per_dimension
            inputs["shift_offset"] = shift_offset
            inputs["total_shifts"] = total_shifts
        elif "cell_list" in method:
            # Pre-allocate cell list cache
            max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
                cell, pbc, cutoff
            )
            cell_list_cache = allocate_cell_list(
                total_atoms_actual, max_total_cells, neighbor_search_radius, device
            )
            inputs["cells_per_dimension"] = cell_list_cache[0]
            inputs["neighbor_search_radius"] = cell_list_cache[1]
            inputs["atom_periodic_shifts"] = cell_list_cache[2]
            inputs["atom_to_cell_mapping"] = cell_list_cache[3]
            inputs["atoms_per_cell_count"] = cell_list_cache[4]
            inputs["cell_atom_start_indices"] = cell_list_cache[5]
            inputs["cell_atom_list"] = cell_list_cache[6]

        return inputs


def run_single_benchmark(
    method, num_atoms_per_system, batch_size, timer, cutoff, device, dtype
):
    """Run a single benchmark configuration."""
    # Prepare inputs (includes pre-allocated tensors)
    inputs = prepare_inputs(
        method, num_atoms_per_system, batch_size, cutoff, device, dtype
    )

    # Time the neighbor list construction
    timing_results = timer.time_function(neighbor_list, **inputs)

    # Check if benchmark was successful
    if not timing_results.get("success", False):
        # Return error result with inf for median_time_us
        return {
            "method": method,
            "total_atoms": num_atoms_per_system * batch_size
            if "batch" in method
            else num_atoms_per_system,
            "atoms_per_system": num_atoms_per_system,
            "total_neighbors": 0,  # Changed from None to 0
            "batch_size": batch_size,
            "median_time_us": float("inf"),  # Changed from None to inf
            "success": False,
            "error": timing_results.get("error", "Unknown error"),
            "error_type": timing_results.get("error_type", "Unknown"),
            "peak_memory_mb": timing_results.get("peak_memory_mb"),
        }

    # Extract number of neighbors from the pre-allocated num_neighbors tensor
    # (neighbor_list was already called during timing, results are in the tensors)
    num_neighbors_total = inputs["num_neighbors"].sum().item()

    # Convert from ms to us
    median_time_ms = timing_results["median"]

    return {
        "method": method,
        "total_atoms": num_atoms_per_system * batch_size
        if "batch" in method
        else num_atoms_per_system,
        "atoms_per_system": num_atoms_per_system,
        "total_neighbors": num_neighbors_total,
        "batch_size": batch_size,
        "median_time_ms": float(median_time_ms),
        "peak_memory_mb": timing_results.get("peak_memory_mb"),
        "success": True,
    }


def run_benchmarks_for_method(
    method_config: dict,
    gpu_sku: str,
    cutoff: float,
    device: str,
    dtype: torch.dtype,
    timer: BenchmarkTimer,
    output_dir: Path,
) -> None:
    """Run benchmarks for a single method and save results."""
    method = method_config["name"]
    atom_counts = method_config["atom_counts"]
    batch_sizes = method_config["batch_sizes"]
    is_batch_method = "batch" in method

    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {method}")
    print(f"{'=' * 70}")

    all_results = []

    for atoms in atom_counts:
        for batch_size in batch_sizes:
            # Pre-validate configuration for batch methods
            if is_batch_method:
                atoms_per_system = atoms
                total_atoms = atoms_per_system * batch_size

                if atoms_per_system < 1:
                    # Skip invalid configuration silently
                    continue
            else:
                atoms_per_system = atoms
                total_atoms = atoms_per_system

            try:
                result = run_single_benchmark(
                    method, atoms_per_system, batch_size, timer, cutoff, device, dtype
                )
                result["method"] = method.replace("_", "-")
                error_type = (
                    None if "error_type" not in result else result.pop("error_type")
                )
                all_results.append(result)

                if result.get("success", True):
                    # Successful benchmark
                    atoms_str = f"{result['total_atoms']:,}"
                    time_str = f"{result['median_time_ms']:.1f}"
                    neighbors_str = f"{result['total_neighbors']:,}"

                    print(
                        f"  {atoms_str:>8} atoms, batch={batch_size:2d}: "
                        f"{time_str:>8} ms, {neighbors_str:>10} neighbors"
                    )
                else:
                    # Failed benchmark
                    atoms_str = f"{result['total_atoms']:,}"

                    print(
                        f"  {atoms_str:>8} atoms, batch={batch_size:2d}: FAILED ({error_type})"
                    )
                    if error_type in ["OOM", "Timeout"]:
                        print("    └─ Skipping larger systems for this method")
                        break

            except ValueError as e:
                # Handle configuration errors
                print(
                    f"  {total_atoms:>8} atoms, batch={batch_size:2d}: SKIPPED - {str(e)}"
                )
                continue
            except Exception as e:
                # Print full traceback for debugging
                print(
                    f"  {total_atoms:>8} atoms, batch={batch_size:2d}: EXCEPTION - {type(e).__name__}: {e}"
                )
                print("\nFULL TRACEBACK:")
                traceback.print_exc()

                # Add failed result
                all_results.append(
                    {
                        "method": method.replace("_", "-"),
                        "total_atoms": total_atoms,
                        "atoms_per_system": atoms_per_system,
                        "total_neighbors": 0,
                        "batch_size": batch_size,
                        "median_time_ms": float("inf"),
                    }
                )

                # Break on critical errors
                if isinstance(e, (IndexError, KeyError, RuntimeError)):
                    print("  Critical error - skipping remaining configurations")
                    break

    # Save results to CSV with GPU-specific name
    if all_results:
        output_file = (
            output_dir
            / f"neighbor_list_benchmark_{method.replace('_', '-')}_{gpu_sku}.csv"
        )
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✓ Results saved to: {output_file}")

        # Print summary
        successful = [r for r in all_results if r.get("success", True)]
        failed = [r for r in all_results if not r.get("success", True)]
        print(
            f"  Total: {len(all_results)} | Successful: {len(successful)} | Failed: {len(failed)}"
        )


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark neighbor list algorithms and generate CSV files for documentation"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./benchmark_results"),
        help="Output directory for CSV files (default: ./benchmark_results)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Specific methods to benchmark (default: all methods in config)",
    )
    parser.add_argument(
        "--gpu-sku",
        type=str,
        help="Override GPU SKU name for output files (default: auto-detect)",
    )

    args = parser.parse_args()

    # Load and validate config
    config = load_config(args.config)
    validate_config(config)

    # Get parameters
    params = config["parameters"]
    cutoff = float(params["cutoff"])
    warmup = int(params["warmup_iterations"])
    timing = int(params["timing_iterations"])
    dtype_str = params["dtype"]
    dtype = getattr(torch, dtype_str)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    # Get GPU SKU
    gpu_sku = args.gpu_sku if args.gpu_sku else get_gpu_sku()

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize timer
    timer = BenchmarkTimer(device_obj, warmup_runs=warmup, timing_runs=timing)

    # Print configuration
    print("=" * 70)
    print("NEIGHBOR LIST BENCHMARK")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"GPU SKU: {gpu_sku}")
    print(f"Cutoff: {cutoff} Å")
    print(f"Dtype: {dtype}")
    print(f"Warmup iterations: {warmup}")
    print(f"Timing iterations: {timing}")
    print(f"Output directory: {output_dir}")

    # Filter methods if specified
    methods_to_run = config["methods"]
    if args.methods:
        methods_to_run = [m for m in methods_to_run if m["name"] in args.methods]
        print(f"Running methods: {[m['name'] for m in methods_to_run]}")
    else:
        print(f"Running all {len(methods_to_run)} methods from config")

    # Run benchmarks for each method
    for method_config in methods_to_run:
        run_benchmarks_for_method(
            method_config,
            gpu_sku,
            cutoff,
            device,
            dtype,
            timer,
            output_dir,
        )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
