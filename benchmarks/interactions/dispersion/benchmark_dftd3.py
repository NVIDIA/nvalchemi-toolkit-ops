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
DFT-D3 Dispersion Scaling Benchmark
====================================

CLI tool to benchmark DFT-D3 dispersion corrections and generate CSV files
for documentation. Results are saved with GPU-specific naming:
`dftd3_benchmark_<system_type>_<gpu_sku>.csv`

Usage:
    python benchmark_dftd3.py --config benchmark_config.yaml --output-dir ../../docs/benchmark_results

The config file specifies system configurations and DFT-D3 parameters.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from pymatgen.core import Lattice, Structure

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.utils import BenchmarkTimer
from nvalchemiops.torch.interactions.dispersion import (
    D3Parameters,
    dftd3,
)
from nvalchemiops.torch.neighbors import neighbor_list

# Optional torch-dftd imports (only needed for torch_dftd backend)
try:
    import torch_dftd
    from torch_dftd.functions.dftd3 import edisp
    from torch_dftd.functions.distance import calc_distances

    TORCH_DFTD_AVAILABLE = True
except ImportError:
    TORCH_DFTD_AVAILABLE = False
    torch_dftd = None  # type: ignore
    edisp = None  # type: ignore
    calc_distances = None  # type: ignore

# Constants
ANGSTROM_TO_BOHR = 1.88973


def get_gpu_sku() -> str:
    """Get GPU SKU name for filename generation."""
    if not torch.cuda.is_available():
        return "cpu"

    try:
        gpu_name = torch.cuda.get_device_name(0)
        # Clean up GPU name for filename
        sku = gpu_name.replace(" ", "-").replace("_", "-")
        sku = sku.replace("NVIDIA-", "").replace("GeForce-", "")
        return sku.lower()
    except Exception:
        return "unknown_gpu"


def load_config(config_path: Path) -> dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def create_cscl_supercell(size: int) -> Structure:
    """Create CsCl supercell of given linear size (2*size³ atoms)."""
    # Create cubic lattice
    lattice = Lattice.cubic(4.14)  # ~4.14 Å cubic cell

    # Create base unit cell with Cs and Cl atoms
    species = ["Cs", "Cl"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    base_unitcell = Structure(lattice, species, coords, coords_are_cartesian=False)

    # Create supercell
    supercell_matrix = [[size, 0, 0], [0, size, 0], [0, 0, size]]
    base_unitcell.make_supercell(supercell_matrix)

    return base_unitcell


def create_d3_parameters(
    device: torch.device, dtype: torch.dtype = torch.float32
) -> D3Parameters:
    """Create simplified D3 parameters for Cs and Cl."""
    # Covalent radii (Bohr)
    rcov = torch.zeros(56, dtype=dtype, device=device)
    rcov[17] = 1.88  # Cl
    rcov[55] = 4.91  # Cs

    # r4r2 expectation values
    r4r2 = torch.zeros(56, dtype=dtype, device=device)
    r4r2[17] = 8.0  # Cl
    r4r2[55] = 18.0  # Cs

    # C6 reference values (simplified 5x5 grid)
    c6ab = torch.zeros(56, 56, 5, 5, dtype=dtype, device=device)
    c6ab[17, 17, :, :] = 50.0  # Cl-Cl
    c6ab[17, 55, :, :] = 200.0  # Cl-Cs
    c6ab[55, 17, :, :] = 200.0  # Cs-Cl
    c6ab[55, 55, :, :] = 800.0  # Cs-Cs

    # CN reference grids
    cn_ref = torch.zeros(56, 56, 5, 5, dtype=dtype, device=device)
    for i in range(5):
        for j in range(5):
            cn_ref[:, :, i, j] = i * 0.5

    return D3Parameters(rcov=rcov, r4r2=r4r2, c6ab=c6ab, cn_ref=cn_ref)


def prepare_system_and_neighborlist(
    supercell_size: int,
    cutoff: float,
    max_neighbors: int,
    device: str,
    dtype: torch.dtype,
    batch_size: int = 1,
    return_neighbor_list: bool = False,
) -> dict:
    """
    Create supercell(s), prepare tensors, and build neighbor list.

    Parameters
    ----------
    supercell_size : int
        Linear size of the supercell (creates 2*size³ atoms per system).
    cutoff : float
        Cutoff distance in Angstroms for neighbor list.
    max_neighbors : int
        Maximum number of neighbors per atom.
    device : str
        Device string ('cuda' or 'cpu').
    dtype : torch.dtype
        Data type for floating point tensors.
    batch_size : int, default=1
        Number of systems to batch together.
    return_neighbor_list : bool, default=False
        If True, return neighbor list in COO format (2, num_pairs) instead of
        neighbor matrix (N_total, max_neighbors).

    Returns
    -------
    dict
        Dictionary containing:
        - positions: Tensor of positions in Bohr (N_total, 3)
        - numbers: Tensor of atomic numbers (N_total,)
        - coord: Tensor of positions in Angstroms (N_total, 3)
        - cell: Tensor of cell vectors (batch_size, 3, 3) or (3, 3)
        - pbc: Tensor of periodic boundary conditions (batch_size, 3) or (3,)
        - neighbor_data: Neighbor list matrix (N_total, max_neighbors) if return_neighbor_list=False,
                        or COO format (2, num_pairs) if return_neighbor_list=True
        - num_neighbor_data: Number of neighbors per atom (N_total,) if return_neighbor_list=False,
                            or neighbor pointer (N_total+1,) if return_neighbor_list=True
        - batch_idx: Batch indices (N_total,) or None for single system
        - batch_ptr: Batch pointer (batch_size+1,) or None for single system
        - total_atoms: Total number of atoms across all systems
        - total_neighbors: Total number of neighbor pairs
    """
    is_batched = batch_size > 1

    if is_batched:
        # Create multiple systems
        all_structures = [
            create_cscl_supercell(supercell_size) for _ in range(batch_size)
        ]

        # Concatenate all systems
        all_positions = []
        all_numbers = []
        all_coords = []
        all_cells = []
        all_pbc = []
        ptr = [0]

        for structure in all_structures:
            all_positions.append(structure.cart_coords * ANGSTROM_TO_BOHR)
            all_numbers.append(
                np.array([site.specie.Z for site in structure], dtype=np.int32)
            )
            all_coords.append(structure.cart_coords)
            all_cells.append(structure.lattice.matrix)
            all_pbc.append(np.array([True, True, True]))
            ptr.append(ptr[-1] + len(structure))

        positions = torch.tensor(
            np.concatenate(all_positions, axis=0), dtype=dtype, device=device
        )
        numbers = torch.tensor(
            np.concatenate(all_numbers, axis=0), dtype=torch.int32, device=device
        )
        coord = torch.tensor(
            np.concatenate(all_coords, axis=0), dtype=dtype, device=device
        )
        cell = torch.tensor(np.stack(all_cells, axis=0), dtype=dtype, device=device)
        pbc = torch.tensor(np.stack(all_pbc, axis=0), dtype=torch.bool, device=device)
        ptr = torch.tensor(ptr, dtype=torch.int32, device=device)

        # Create batch_idx
        batch_idx = torch.zeros(coord.shape[0], dtype=torch.int32, device=device)
        for i in range(batch_size):
            batch_idx[ptr[i] : ptr[i + 1]] = i

        total_atoms = coord.shape[0]

        # Build neighbor list
        neighbor_data, num_neighbor_data, _ = neighbor_list(
            coord,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=ptr,
            method="batch_cell_list",
            max_neighbors=max_neighbors,
            return_neighbor_list=return_neighbor_list,
        )
    else:
        # Single system
        structure = create_cscl_supercell(supercell_size)
        total_atoms = len(structure)

        positions = torch.tensor(
            structure.cart_coords * ANGSTROM_TO_BOHR, dtype=dtype, device=device
        )
        numbers = torch.tensor(
            np.array([site.specie.Z for site in structure], dtype=np.int32),
            dtype=torch.int32,
            device=device,
        )
        coord = torch.tensor(structure.cart_coords, dtype=dtype, device=device)
        cell = torch.tensor(structure.lattice.matrix, dtype=dtype, device=device)
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)

        neighbor_data, num_neighbor_data, _ = neighbor_list(
            coord,
            cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list",
            max_neighbors=max_neighbors,
            return_neighbor_list=return_neighbor_list,
        )

        batch_idx = None
        ptr = None

    # Calculate total neighbors
    if return_neighbor_list:
        # neighbor_data is (2, num_pairs), so num_pairs is the total neighbors
        total_neighbors = neighbor_data.shape[1]
    else:
        # num_neighbor_data is (N_total,) with counts per atom
        total_neighbors = num_neighbor_data.sum().item()

    return {
        "positions": positions,
        "numbers": numbers,
        "coord": coord,
        "cell": cell,
        "pbc": pbc,
        "neighbor_data": neighbor_data,
        "num_neighbor_data": num_neighbor_data,
        "batch_idx": batch_idx,
        "batch_ptr": ptr,
        "total_atoms": total_atoms,
        "total_neighbors": total_neighbors,
    }


def load_torch_dftd_parameters(
    device: torch.device, dtype: torch.dtype = torch.float32
) -> dict:
    """Load DFT-D3 parameters from torch-dftd package."""
    if not TORCH_DFTD_AVAILABLE:
        raise ImportError(
            "torch-dftd not installed. Install via: pip install torch-dftd"
        )

    # Load parameters from torch_dftd
    d3_filepath = str(
        Path(os.path.abspath(torch_dftd.__file__)).parent
        / "nn"
        / "params"
        / "dftd3_params.npz"
    )
    d3_params_np = np.load(d3_filepath)

    c6ab = torch.tensor(d3_params_np["c6ab"], dtype=dtype, device=device)
    r0ab = torch.tensor(d3_params_np["r0ab"], dtype=dtype, device=device)
    rcov = torch.tensor(d3_params_np["rcov"], dtype=dtype, device=device)
    r2r4 = torch.tensor(d3_params_np["r2r4"], dtype=dtype, device=device)

    # Convert rcov to Bohr
    rcov = rcov * ANGSTROM_TO_BOHR

    return {
        "c6ab": c6ab,
        "r0ab": r0ab,
        "rcov": rcov,
        "r2r4": r2r4,
    }


def run_dftd3_nvalchemiops_benchmark(
    supercell_size: int,
    cutoff: float,
    d3_params: D3Parameters,
    dftd3_config: dict,
    max_neighbors: int,
    timer: BenchmarkTimer,
    device: str,
    dtype: torch.dtype,
    batch_size: int = 1,
) -> dict:
    """Run DFT-D3 benchmark using nvalchemiops backend (single or batched)."""
    try:
        # Prepare system and neighbor list (matrix format for nvalchemiops)
        system_data = prepare_system_and_neighborlist(
            supercell_size,
            cutoff,
            max_neighbors,
            device,
            dtype,
            batch_size,
            return_neighbor_list=False,
        )

        positions = system_data["positions"]
        numbers = system_data["numbers"]
        neighbor_matrix = system_data["neighbor_data"]
        total_atoms = system_data["total_atoms"]
        total_neighbors = system_data["total_neighbors"]
        batch_idx = system_data["batch_idx"]

        # Define the function to benchmark
        def dftd3_call():
            return dftd3(
                positions=positions,
                numbers=numbers,
                d3_params=d3_params,
                neighbor_matrix=neighbor_matrix,
                fill_value=total_atoms,
                a1=dftd3_config["a1"],
                a2=dftd3_config["a2"],
                s6=dftd3_config["s6"],
                s8=dftd3_config["s8"],
                k1=dftd3_config["k1"],
                k3=dftd3_config["k3"],
                batch_idx=batch_idx,
                s5_smoothing_on=dftd3_config["s5_smoothing_on"],
                s5_smoothing_off=dftd3_config["s5_smoothing_off"],
                device=device,
            )

        # Time the function
        timing_results = timer.time_function(dftd3_call)

        if not timing_results["success"]:
            return {
                "total_atoms": total_atoms,
                "batch_size": batch_size,
                "supercell_size": supercell_size,
                "total_neighbors": 0,
                "median_time_ms": float("inf"),
                "peak_memory_mb": timing_results.get("peak_memory_mb"),
                "success": False,
                "error": timing_results.get("error", "Unknown error"),
                "error_type": timing_results.get("error_type", "Unknown"),
            }

        median_time_ms = timing_results["median"]
        peak_memory_mb = timing_results.get("peak_memory_mb")

        return {
            "total_atoms": total_atoms,
            "batch_size": batch_size,
            "supercell_size": supercell_size,
            "total_neighbors": total_neighbors,
            "median_time_ms": float(median_time_ms),
            "peak_memory_mb": peak_memory_mb,
            "success": True,
        }

    except Exception as e:
        total_atoms = 2 * supercell_size**3 * batch_size
        return {
            "total_atoms": total_atoms,
            "batch_size": batch_size,
            "supercell_size": supercell_size,
            "total_neighbors": 0,
            "median_time_ms": float("inf"),
            "peak_memory_mb": None,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def run_dftd3_torch_dftd_benchmark(
    supercell_size: int,
    cutoff: float,
    torch_dftd_params: dict,
    dftd3_config: dict,
    max_neighbors: int,
    timer: BenchmarkTimer,
    device: str,
    dtype: torch.dtype,
    batch_size: int = 1,
) -> dict:
    """Run DFT-D3 benchmark using torch-dftd backend (single or batched)."""
    if not TORCH_DFTD_AVAILABLE:
        return {
            "total_atoms": 2 * supercell_size**3 * batch_size,
            "batch_size": batch_size,
            "supercell_size": supercell_size,
            "total_neighbors": 0,
            "median_time_ms": float("inf"),
            "peak_memory_mb": None,
            "success": False,
            "error": "torch-dftd not installed",
            "error_type": "ImportError",
        }

    try:
        # Prepare system and neighbor list in COO format (torch-dftd uses edge_index)
        system_data = prepare_system_and_neighborlist(
            supercell_size,
            cutoff,
            max_neighbors,
            device,
            dtype,
            batch_size,
            return_neighbor_list=True,
        )

        edge_index = system_data[
            "neighbor_data"
        ]  # Already in COO format (2, num_pairs)
        total_atoms = system_data["total_atoms"]
        total_neighbors = system_data["total_neighbors"]

        # Prepare positions with gradients for torch-dftd
        positions = system_data["positions"].clone().requires_grad_(True)
        Z = system_data["numbers"].to(torch.int64)

        # Note: torch-dftd does not require explicit batch tensors
        # It handles batched systems through concatenated positions/numbers
        # and the edge_index structure
        batch = None
        batch_edge = None

        # Prepare torch-dftd params dict
        params = {
            "s6": dftd3_config["s6"],
            "s18": dftd3_config["s8"],
            "rs6": dftd3_config["a1"],
            "rs18": dftd3_config["a2"],
            "alp": 14.0,  # Default for BJ damping
        }

        # Define the function to benchmark
        def torch_dftd_call():
            # Reset grad
            if positions.grad is not None:
                positions.grad.zero_()

            # Calculate distances
            r = calc_distances(positions, edge_index, cell=None, shift_pos=None)

            # Compute energy
            energy = edisp(
                Z=Z,
                r=r,
                edge_index=edge_index,
                c6ab=torch_dftd_params["c6ab"],
                r0ab=torch_dftd_params["r0ab"],
                rcov=torch_dftd_params["rcov"],
                r2r4=torch_dftd_params["r2r4"],
                params=params,
                cutoff=None,
                cnthr=None,
                batch=batch,
                batch_edge=batch_edge,
                shift_pos=None,
                pos=positions,
                cell=None,
                damping="bj",
                bidirectional=True,
                abc=False,
                k1=dftd3_config["k1"],
                k3=dftd3_config["k3"],
            )

            # Compute forces
            forces = -torch.autograd.grad(
                outputs=energy,
                inputs=positions,
                create_graph=False,
                retain_graph=False,
            )[0]

            return energy, forces

        # Time the function
        timing_results = timer.time_function(torch_dftd_call)

        if not timing_results["success"]:
            return {
                "total_atoms": total_atoms,
                "batch_size": batch_size,
                "supercell_size": supercell_size,
                "total_neighbors": 0,
                "median_time_ms": float("inf"),
                "peak_memory_mb": timing_results.get("peak_memory_mb"),
                "success": False,
                "error": timing_results.get("error", "Unknown error"),
                "error_type": timing_results.get("error_type", "Unknown"),
            }

        median_time_ms = timing_results["median"]
        peak_memory_mb = timing_results.get("peak_memory_mb")

        return {
            "total_atoms": total_atoms,
            "batch_size": batch_size,
            "supercell_size": supercell_size,
            "total_neighbors": total_neighbors,
            "median_time_ms": float(median_time_ms),
            "peak_memory_mb": peak_memory_mb,
            "success": True,
        }

    except Exception as e:
        return {
            "total_atoms": 2 * supercell_size**3 * batch_size,
            "batch_size": batch_size,
            "supercell_size": supercell_size,
            "total_neighbors": 0,
            "median_time_ms": float("inf"),
            "peak_memory_mb": None,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark DFT-D3 dispersion corrections and generate CSV files"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./benchmark_results"),
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["nvalchemiops", "torch_dftd"],
        default="nvalchemiops",
        help="Backend to use for benchmarking (default: nvalchemiops)",
    )
    parser.add_argument(
        "--gpu-sku",
        type=str,
        help="Override GPU SKU name for output files (default: auto-detect)",
    )

    args = parser.parse_args()

    # Check if torch_dftd is available when requested
    if args.backend == "torch_dftd" and not TORCH_DFTD_AVAILABLE:
        print("ERROR: torch-dftd backend requested but not installed.")
        print("Install via: pip install torch-dftd")
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    # Get parameters
    params = config["parameters"]
    cutoff = float(params["cutoff"])
    warmup = int(params["warmup_iterations"])
    timing = int(params["timing_iterations"])
    dtype_str = params["dtype"]
    dtype = getattr(torch, dtype_str)

    dftd3_config = config["dftd3_parameters"]

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

    # Backend-specific setup
    if args.backend == "nvalchemiops":
        d3_params = create_d3_parameters(device_obj, dtype)
        torch_dftd_params = None
    else:  # torch_dftd
        d3_params = None
        torch_dftd_params = load_torch_dftd_parameters(device_obj, dtype)

    # Print configuration
    print("=" * 70)
    print("DFT-D3 DISPERSION BENCHMARK")
    print("=" * 70)
    print(f"Backend: {args.backend}")
    print(f"Device: {device}")
    print(f"GPU SKU: {gpu_sku}")
    print(f"Cutoff: {cutoff:.2f} Å ({cutoff * ANGSTROM_TO_BOHR:.1f} Bohr)")
    print(f"Dtype: {dtype}")
    print(f"Warmup iterations: {warmup}")
    print(f"Timing iterations: {timing}")
    print(f"Output directory: {output_dir}")

    # Run benchmarks for each system configuration
    for system_config in config["systems"]:
        system_name = system_config["name"]
        system_type = system_config["system_type"]
        supercell_sizes = system_config["supercell_sizes"]
        batch_sizes = system_config.get("batch_sizes", [1])
        max_neighbors = system_config["max_neighbors"]

        # Group results by batch size
        results_by_batch = {}

        for batch_size in batch_sizes:
            is_batched = batch_size > 1

            print(f"\n{'=' * 70}")
            print(f"Benchmarking: {system_name} ({system_type})")
            if is_batched:
                print(f"Batch size: {batch_size}")
            print(f"{'=' * 70}")
            print(f"Supercell sizes: {supercell_sizes}")

            all_results = []

            for size in supercell_sizes:
                # Reset peak memory stats before each configuration
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()

                atoms_per_system = 2 * size**3
                total_atoms = atoms_per_system * batch_size

                if is_batched:
                    print(
                        f"\n  {total_atoms:6,d} atoms ({atoms_per_system:,d} x {batch_size})...",
                        end=" ",
                        flush=True,
                    )
                else:
                    print(
                        f"\n  {total_atoms:6,d} atoms (supercell {size}³)...",
                        end=" ",
                        flush=True,
                    )

                # Choose benchmark function based on backend
                if args.backend == "nvalchemiops":
                    result = run_dftd3_nvalchemiops_benchmark(
                        size,
                        cutoff,
                        d3_params,
                        dftd3_config,
                        max_neighbors,
                        timer,
                        device,
                        dtype,
                        batch_size,
                    )
                else:  # torch_dftd
                    result = run_dftd3_torch_dftd_benchmark(
                        size,
                        cutoff,
                        torch_dftd_params,
                        dftd3_config,
                        max_neighbors,
                        timer,
                        device,
                        dtype,
                        batch_size,
                    )

                # Add backend to result for CSV
                result["backend"] = args.backend
                error_type = (
                    None if "error_type" not in result else result.pop("error_type")
                )
                error = None if "error" not in result else result.pop("error")
                all_results.append(result)

                if result["success"]:
                    throughput = result["total_atoms"] / result["median_time_ms"] * 1000
                    mem_str = ""
                    if result.get("peak_memory_mb") is not None:
                        mem_str = f" | {result['peak_memory_mb']:.1f} MB"
                    print(
                        f"{result['median_time_ms']:.3f} ms "
                        f"({throughput:.1f} atoms/s){mem_str}"
                    )
                else:
                    print(f"FAILED ({error_type}): {error}")
                    if error_type in ["OOM", "Timeout"]:
                        print("  Skipping larger systems")
                        break

            # Store results for this batch size
            if all_results:
                results_by_batch[batch_size] = all_results

        # Separate batched and non-batched results
        non_batched_results = []
        batched_results = []

        for batch_size, results in results_by_batch.items():
            if batch_size == 1:
                non_batched_results.extend(results)
            else:
                batched_results.extend(results)

        # Save non-batched results
        if non_batched_results:
            output_file = output_dir / f"dftd3_benchmark_{args.backend}_{gpu_sku}.csv"
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=non_batched_results[0].keys())
                writer.writeheader()
                writer.writerows(non_batched_results)
            print(f"\n✓ Non-batched results saved to: {output_file}")

            successful = [r for r in non_batched_results if r.get("success", True)]
            failed = [r for r in non_batched_results if not r.get("success", True)]
            print(
                f"  Total: {len(non_batched_results)} | Successful: {len(successful)} | Failed: {len(failed)}"
            )

        # Save batched results (all batch sizes in one file)
        if batched_results:
            output_file = (
                output_dir / f"dftd3_benchmark_batch_{args.backend}_{gpu_sku}.csv"
            )
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=batched_results[0].keys())
                writer.writeheader()
                writer.writerows(batched_results)
            print(f"\n✓ Batched results saved to: {output_file}")

            successful = [r for r in batched_results if r.get("success", True)]
            failed = [r for r in batched_results if not r.get("success", True)]
            print(
                f"  Total: {len(batched_results)} | Successful: {len(successful)} | Failed: {len(failed)}"
            )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
