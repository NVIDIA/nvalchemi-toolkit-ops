#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Electrostatics Benchmark
========================

CLI tool to benchmark electrostatic interaction methods (Ewald summation and PME)
and generate CSV files for documentation. Results are saved with GPU-specific naming:
`electrostatics_benchmark_<method>_<backend>_<gpu_sku>.csv`

Supports two backends:
1. nvalchemiops (Warp kernels): Custom implementation using PyTorch + Warp
2. torchpme: Reference PyTorch implementation

Usage:
    python benchmark_electrostatics.py --config benchmark_config.yaml --output-dir ./results
    python benchmark_electrostatics.py --config benchmark_config.yaml --backend both --method both
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Literal

import torch
import warp as wp

# Add repo root to path for imports (4 levels up from this script)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import yaml

from benchmarks.systems import create_crystal_system
from benchmarks.utils import BenchmarkTimer
from nvalchemiops.torch.interactions.electrostatics import (
    estimate_ewald_parameters,
    estimate_pme_parameters,
    ewald_real_space,
    ewald_reciprocal_space,
    ewald_summation,
    particle_mesh_ewald,
    pme_reciprocal_space,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
    generate_k_vectors_pme,
)
from nvalchemiops.torch.neighbors import neighbor_list

# Optional torchpme imports
try:
    from torchpme import EwaldCalculator, PMECalculator
    from torchpme.potentials import CoulombPotential

    TORCHPME_AVAILABLE = True
except ImportError:
    TORCHPME_AVAILABLE = False
    EwaldCalculator = None
    PMECalculator = None
    CoulombPotential = None


# ==============================================================================
# Utilities
# ==============================================================================


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


# ==============================================================================
# System Generation
# ==============================================================================


def prepare_single_system(
    supercell_size: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """Prepare a single system for benchmarking.

    Parameters
    ----------
    supercell_size : int
        Linear dimension of the supercell. For BCC lattice (2 atoms per unit cell),
        this creates 2 * supercell_size³ atoms total.
    """
    # BCC lattice has 2 atoms per unit cell, so total atoms = 2 * size³
    target_atoms = 2 * supercell_size**3
    system = create_crystal_system(
        target_atoms,
        lattice_type="bcc",
        lattice_constant=4.14,
        device=device,
        dtype=dtype,
    )
    total_atoms = system["num_atoms"]

    positions = system["positions"]
    charges = system["atomic_charges"]
    cell = system["cell"]
    pbc = system["pbc"]

    ewald_params = estimate_ewald_parameters(positions, cell, accuracy=1e-6)
    alpha = ewald_params.alpha

    k_cutoff = ewald_params.reciprocal_space_cutoff.item()
    cutoff = ewald_params.real_space_cutoff.item()

    pme_params = estimate_pme_parameters(positions, cell, accuracy=1e-6)
    alpha = pme_params.alpha

    mesh_dimensions = pme_params.mesh_dimensions
    mesh_spacing = pme_params.mesh_spacing.tolist()

    # Build neighbor list
    neighbor_list_data, neighbor_ptr, neighbor_shifts = neighbor_list(
        positions,
        cutoff,
        cell=cell,
        pbc=pbc,
        return_neighbor_list=True,
    )

    # Precompute k-vectors for PME (avoids regenerating them every iteration)
    k_vectors_pme, k_squared_pme = generate_k_vectors_pme(cell, mesh_dimensions)

    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "pbc": pbc,
        "neighbor_list": neighbor_list_data,
        "neighbor_ptr": neighbor_ptr,
        "neighbor_shifts": neighbor_shifts,
        "total_atoms": total_atoms,
        "batch_idx": None,
        "alpha": alpha,
        "k_cutoff": k_cutoff,
        "cutoff": cutoff,
        "mesh_dimensions": mesh_dimensions,
        "mesh_spacing": mesh_spacing,
        "spline_order": 4,
        "k_vectors_pme": k_vectors_pme,
        "k_squared_pme": k_squared_pme,
    }


def prepare_batch_system(
    supercell_size: int,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """Prepare a batched system for benchmarking.

    Parameters
    ----------
    supercell_size : int
        Linear dimension of each supercell. For BCC lattice (2 atoms per unit cell),
        each system has 2 * supercell_size³ atoms.
    batch_size : int
        Number of systems to batch together.
    """
    # BCC lattice has 2 atoms per unit cell, so atoms per system = 2 * size³
    target_atoms_per_system = 2 * supercell_size**3

    all_positions = []
    all_charges = []
    all_cells = []
    all_pbc = []
    batch_idx_list = []

    for i in range(batch_size):
        system = create_crystal_system(
            target_atoms_per_system,
            lattice_type="bcc",
            lattice_constant=4.14,
            device=device,
            dtype=dtype,
        )
        n_atoms = system["num_atoms"]

        positions = system["positions"]
        charges = system["atomic_charges"]
        cell = system["cell"]
        pbc = system["pbc"]

        all_positions.append(positions)
        all_charges.append(charges)
        all_cells.append(cell)
        all_pbc.append(pbc)
        batch_idx_list.extend([i] * n_atoms)

    positions = torch.cat(all_positions, dim=0)
    charges = torch.cat(all_charges, dim=0)
    cells = torch.cat(all_cells, dim=0)
    pbc = torch.stack(all_pbc, dim=0)

    batch_idx = torch.tensor(batch_idx_list, dtype=torch.int32, device=device)
    total_atoms = positions.shape[0]
    ewald_params = estimate_ewald_parameters(positions, cells, batch_idx, accuracy=1e-6)
    alpha = ewald_params.alpha
    k_cutoff = ewald_params.reciprocal_space_cutoff[0].item()
    cutoff = ewald_params.real_space_cutoff[0].item()
    pme_params = estimate_pme_parameters(positions, cells, batch_idx, accuracy=1e-6)
    alpha = pme_params.alpha
    mesh_dimensions = pme_params.mesh_dimensions
    mesh_spacing = pme_params.mesh_spacing

    # Build neighbor list for batch
    neighbor_list_data, neighbor_ptr, neighbor_shifts = neighbor_list(
        positions,
        cutoff,
        cell=cells,
        pbc=pbc,
        batch_idx=batch_idx,
        method="batch_naive",
        return_neighbor_list=True,
    )

    # Precompute k-vectors for PME (avoids regenerating them every iteration)
    k_vectors_pme, k_squared_pme = generate_k_vectors_pme(cells, mesh_dimensions)

    return {
        "positions": positions,
        "charges": charges,
        "cell": cells,
        "pbc": pbc,
        "neighbor_list": neighbor_list_data,
        "neighbor_ptr": neighbor_ptr,
        "neighbor_shifts": neighbor_shifts,
        "total_atoms": total_atoms,
        "batch_idx": batch_idx,
        "batch_size": batch_size,
        "alpha": alpha,
        "k_cutoff": k_cutoff,
        "cutoff": cutoff,
        "mesh_dimensions": mesh_dimensions,
        "mesh_spacing": mesh_spacing,
        "spline_order": 4,
        "k_vectors_pme": k_vectors_pme,
        "k_squared_pme": k_squared_pme,
    }


# ==============================================================================
# nvalchemiops Backend
# ==============================================================================


def run_nvalchemiops_ewald(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Ewald summation using nvalchemiops backend."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha")
    k_cutoff = system_data.get("k_cutoff")
    k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff)

    neighbor_list_data = system_data.get("neighbor_list")
    neighbor_ptr = system_data.get("neighbor_ptr")
    neighbor_shifts = system_data.get("neighbor_shifts")

    if batch_idx is None:
        # Single system

        if component == "real":
            return ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        elif component == "reciprocal":
            return ewald_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                k_vectors=k_vectors,
                alpha=alpha,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        else:  # full
            return ewald_summation(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                k_cutoff=k_cutoff,
                k_vectors=k_vectors,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
    else:
        # Batch system
        if component == "real":
            return ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                batch_idx=batch_idx,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        elif component == "reciprocal":
            return ewald_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                k_vectors=k_vectors,
                alpha=alpha,
                batch_idx=batch_idx,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        else:  # full
            return ewald_summation(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                k_cutoff=k_cutoff,
                k_vectors=k_vectors,
                batch_idx=batch_idx,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )


def run_nvalchemiops_pme(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run PME using nvalchemiops backend."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha")
    mesh_dimensions = system_data.get("mesh_dimensions")
    spline_order = system_data.get("spline_order")
    k_vectors_pme = system_data.get("k_vectors_pme")
    k_squared_pme = system_data.get("k_squared_pme")

    neighbor_list_data = system_data.get("neighbor_list")
    neighbor_ptr = system_data.get("neighbor_ptr")
    neighbor_shifts = system_data.get("neighbor_shifts")

    if batch_idx is None:
        # Single system

        if component == "real":
            return ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        elif component == "reciprocal":
            return pme_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                k_vectors=k_vectors_pme,
                k_squared=k_squared_pme,
            )
        else:  # full
            return particle_mesh_ewald(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                k_vectors=k_vectors_pme,
                k_squared=k_squared_pme,
            )
    else:
        # Batch system

        if component == "real":
            return ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                batch_idx=batch_idx,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        elif component == "reciprocal":
            return pme_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                batch_idx=batch_idx,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                k_vectors=k_vectors_pme,
                k_squared=k_squared_pme,
            )
        else:  # full
            return particle_mesh_ewald(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                batch_idx=batch_idx,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                k_vectors=k_vectors_pme,
                k_squared=k_squared_pme,
            )


# ==============================================================================
# torchpme Backend
# ==============================================================================


def prepare_torchpme_neighbors(
    system_data: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare neighbor data in torchpme format."""
    positions = system_data["positions"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")

    if batch_idx is None:
        # Single system
        neighbor_list_data = system_data.get("neighbor_list")
        neighbor_shifts = system_data.get("neighbor_shifts")

        if neighbor_list_data is not None:
            neighbor_indices = neighbor_list_data.T
            cell_2d = cell.squeeze(0)
            neighbor_distances = torch.norm(
                positions[neighbor_list_data[1]]
                - positions[neighbor_list_data[0]]
                + neighbor_shifts.to(dtype=positions.dtype) @ cell_2d,
                dim=1,
            )
        else:
            neighbor_indices = torch.zeros(
                (0, 2), dtype=torch.int32, device=positions.device
            )
            neighbor_distances = torch.zeros(
                0, dtype=positions.dtype, device=positions.device
            )

        return neighbor_indices, neighbor_distances
    else:
        # For batch, we need to handle each system separately for torchpme
        # This is a limitation - torchpme doesn't natively support batched neighbors
        raise NotImplementedError("torchpme batch mode requires per-system handling")


def run_torchpme_ewald(
    system_data: dict,
    compute_forces: bool,
    compute_virial: bool = False,
    calculator: EwaldCalculator | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run Ewald summation using torchpme backend."""
    if not TORCHPME_AVAILABLE:
        raise ImportError("torchpme not available")

    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    alpha = system_data.get("alpha").item()
    k_cutoff = system_data.get("k_cutoff")
    dtype = positions.dtype
    device = positions.device
    neighbor_indices, neighbor_distances = prepare_torchpme_neighbors(
        system_data,
    )

    if calculator is None:
        lr_wavelength = 2 * torch.pi / k_cutoff
        smearing = 1.0 / alpha
        calculator = EwaldCalculator(
            potential=CoulombPotential(smearing=smearing).to(
                device=device, dtype=dtype
            ),
            lr_wavelength=lr_wavelength,
        ).to(device=device, dtype=dtype)

    charges_expanded = charges.unsqueeze(1)
    cell_2d = cell.squeeze(0)

    if not compute_forces and not compute_virial:
        energy = calculator.forward(
            charges_expanded,
            cell_2d,
            positions,
            neighbor_indices,
            neighbor_distances,
        )
        return energy, None

    # Compute forces and/or virial via autograd
    positions_grad = positions.clone().detach().requires_grad_(True)
    cell_grad = (
        cell_2d.clone().detach().requires_grad_(True) if compute_virial else cell_2d
    )
    potentials_grad = calculator.forward(
        charges_expanded,
        cell_grad,
        positions_grad,
        neighbor_indices,
        neighbor_distances,
    )
    energy_grad = (potentials_grad * charges_expanded).sum()
    energy_grad.backward()
    forces = -positions_grad.grad if compute_forces else None
    virial = cell_grad.grad if compute_virial else None

    return energy_grad, forces, virial


def run_torchpme_pme(
    system_data: dict,
    compute_forces: bool,
    compute_virial: bool = False,
    calculator: PMECalculator | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run PME using torchpme backend."""
    if not TORCHPME_AVAILABLE:
        raise ImportError("torchpme not available")

    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    alpha = system_data.get("alpha").item()
    mesh_spacing = system_data.get("mesh_spacing")[0][0]
    spline_order = system_data.get("spline_order")
    dtype = positions.dtype
    device = positions.device

    neighbor_indices, neighbor_distances = prepare_torchpme_neighbors(
        system_data,
    )
    if calculator is None:
        smearing = 1.0 / alpha
        calculator = PMECalculator(
            potential=CoulombPotential(smearing=smearing).to(
                device=device, dtype=dtype
            ),
            mesh_spacing=mesh_spacing,
            interpolation_nodes=spline_order,
            full_neighbor_list=True,
            prefactor=1.0,
        ).to(device=device, dtype=dtype)

    charges_expanded = charges.unsqueeze(1)
    cell_2d = cell.squeeze(0)

    if not compute_forces and not compute_virial:
        energy = calculator.forward(
            charges_expanded,
            cell_2d,
            positions,
            neighbor_indices,
            neighbor_distances,
        )
        return energy, None

    # Compute forces and/or virial via autograd
    positions_grad = positions.clone().detach().requires_grad_(True)
    cell_grad = (
        cell_2d.clone().detach().requires_grad_(True) if compute_virial else cell_2d
    )
    potentials_grad = calculator.forward(
        charges_expanded,
        cell_grad,
        positions_grad,
        neighbor_indices,
        neighbor_distances,
    )
    energy_grad = (potentials_grad * charges_expanded).sum()
    energy_grad.backward()
    forces = -positions_grad.grad if compute_forces else None
    virial = cell_grad.grad if compute_virial else None

    return energy_grad, forces, virial


# ==============================================================================
# Benchmark Runner
# ==============================================================================


def run_benchmark(
    method: Literal["ewald", "pme"],
    backend: Literal["nvalchemiops", "torchpme"],
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool,
    timer: BenchmarkTimer,
) -> dict:
    """Run a single benchmark configuration."""
    total_atoms = system_data["total_atoms"]
    batch_size = system_data.get("batch_size", 1)

    effective_virial = compute_virial

    try:
        # Define benchmark function based on method and backend
        if backend == "nvalchemiops":
            if method == "ewald":

                def bench_fn():
                    return run_nvalchemiops_ewald(
                        system_data,
                        component,
                        compute_forces,
                        effective_virial,
                    )
            else:  # pme

                def bench_fn():
                    return run_nvalchemiops_pme(
                        system_data,
                        component,
                        compute_forces,
                        effective_virial,
                    )
        else:  # torchpme
            if system_data.get("batch_idx") is not None:
                return {
                    "total_atoms": total_atoms,
                    "batch_size": batch_size,
                    "method": method,
                    "backend": backend,
                    "component": component,
                    "compute_forces": compute_forces,
                    "compute_virial": effective_virial,
                    "median_time_ms": float("inf"),
                    "peak_memory_mb": None,
                    "success": False,
                    "error": "torchpme does not support native batched evaluation",
                    "error_type": "NotImplemented",
                }

            if method == "ewald":

                def bench_fn():
                    return run_torchpme_ewald(
                        system_data, compute_forces, effective_virial
                    )
            else:  # pme

                def bench_fn():
                    return run_torchpme_pme(
                        system_data,
                        compute_forces,
                        effective_virial,
                    )

        # Run benchmark
        timing_results = timer.time_function(bench_fn)
        torch.cuda.empty_cache()
        if not timing_results["success"]:
            print(f"Benchmark failed: {timing_results.get('error', 'Unknown error')}")
            return {
                "total_atoms": total_atoms,
                "batch_size": batch_size,
                "method": method,
                "backend": backend,
                "component": component,
                "compute_forces": compute_forces,
                "compute_virial": effective_virial,
                "median_time_ms": float("inf"),
                "peak_memory_mb": timing_results.get("peak_memory_mb"),
                "success": False,
                "error": timing_results.get("error", "Unknown error"),
                "error_type": timing_results.get("error_type", "Unknown"),
            }

        return {
            "total_atoms": total_atoms,
            "batch_size": batch_size,
            "method": method,
            "backend": backend,
            "component": component,
            "compute_forces": compute_forces,
            "compute_virial": effective_virial,
            "median_time_ms": float(timing_results["median"]),
            "peak_memory_mb": timing_results.get("peak_memory_mb"),
            "success": True,
        }

    except Exception as e:
        print(f"Benchmark failed: {e}")
        return {
            "total_atoms": total_atoms,
            "batch_size": batch_size,
            "method": method,
            "backend": backend,
            "component": component,
            "compute_forces": compute_forces,
            "compute_virial": effective_virial,
            "median_time_ms": float("inf"),
            "peak_memory_mb": None,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


# ==============================================================================
# Main
# ==============================================================================


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark electrostatic interaction methods and generate CSV files"
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
        choices=["nvalchemiops", "torchpme", "both"],
        default="nvalchemiops",
        help="Backend to use for benchmarking (default: nvalchemiops)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ewald", "pme", "both"],
        default="both",
        help="Method to benchmark (default: both)",
    )
    parser.add_argument(
        "--gpu-sku",
        type=str,
        help="Override GPU SKU name for output files (default: auto-detect)",
    )

    args = parser.parse_args()

    # Check if torchpme is available when requested
    if args.backend in ["torchpme", "both"] and not TORCHPME_AVAILABLE:
        if args.backend == "torchpme":
            print("ERROR: torchpme backend requested but not installed.")
            print("Install via: pip install torch-pme")
            sys.exit(1)
        else:
            print("WARNING: torchpme not installed, skipping torchpme benchmarks")

    # Load config
    config = load_config(args.config)

    # Get parameters
    params = config["parameters"]
    warmup = int(params["warmup_iterations"])
    timing = int(params["timing_iterations"])
    dtype_str = params["dtype"]
    dtype = getattr(torch, dtype_str)
    device_str = params.get("device", "cuda")

    # Setup device
    device = device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu"
    device_obj = torch.device(device)

    # Get GPU SKU
    gpu_sku = args.gpu_sku if args.gpu_sku else get_gpu_sku()

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize timer
    timer = BenchmarkTimer(device_obj, warmup_runs=warmup, timing_runs=timing)

    # Initialize Warp
    wp.init()

    # Determine what to benchmark
    methods = ["ewald", "pme"] if args.method == "both" else [args.method]
    backends = []
    if args.backend in ["nvalchemiops", "both"]:
        backends.append("nvalchemiops")
    if args.backend in ["torchpme", "both"] and TORCHPME_AVAILABLE:
        backends.append("torchpme")
    if len(backends) == 0:
        backends.append(
            "nvalchemiops"
        )  # Default to nvalchemiops if no backends are specified

    components = config.get("components", ["full"])
    compute_forces = config.get("compute_forces", True)
    compute_virial = config.get("compute_virial", False)

    # Print configuration
    print("=" * 70)
    print("ELECTROSTATICS BENCHMARK")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"GPU SKU: {gpu_sku}")
    print(f"Dtype: {dtype}")
    print(f"Methods: {methods}")
    print(f"Backends: {backends}")
    print(f"Components: {components}")
    print(f"Compute forces: {compute_forces}")
    print(f"Compute virial: {compute_virial}")
    print(f"Warmup iterations: {warmup}")
    print(f"Timing iterations: {timing}")
    print(f"Output directory: {output_dir}")

    # Run benchmarks for each system configuration
    all_results = []

    for system_config in config["systems"]:
        system_name = system_config["name"]
        mode = system_config["mode"]

        print(f"\n{'=' * 70}")
        print(f"System: {system_name} ({mode})")
        print(f"{'=' * 70}")

        if mode == "single":
            supercell_sizes = system_config["supercell_sizes"]

            for size in supercell_sizes:
                expected_atoms = 2 * size**3  # BCC: 2 atoms per unit cell
                print(f"\n  ~{expected_atoms:,d} atoms (supercell {size}³)...")

                # Reset memory
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()

                # Prepare system
                try:
                    system_data = prepare_single_system(size, device, dtype)
                except Exception as e:
                    print(f"    Failed to prepare system: {e}")
                    traceback.print_exc()
                    continue

                for method in methods:
                    for backend in backends:
                        for component in components:
                            result = run_benchmark(
                                method,
                                backend,
                                system_data,
                                component,
                                compute_forces,
                                compute_virial,
                                timer,
                            )
                            result["supercell_size"] = size
                            result["mode"] = mode
                            all_results.append(result)

                            if result["success"]:
                                throughput = (
                                    result["total_atoms"]
                                    / result["median_time_ms"]
                                    * 1000
                                )
                                mem_str = ""
                                if result.get("peak_memory_mb"):
                                    mem_str = f" | {result['peak_memory_mb']:.1f} MB"
                                print(
                                    f"    {method:5s} {backend:12s} {component:10s}: "
                                    f"{result['median_time_ms']:.3f} ms "
                                    f"({throughput:.1f} atoms/s){mem_str}"
                                )
                            else:
                                print(
                                    f"    {method:5s} {backend:12s} {component:10s}: "
                                    f"FAILED ({result.get('error_type', 'Unknown')})"
                                )

        else:  # batched
            base_size = system_config["base_supercell_size"]
            batch_sizes = system_config["batch_sizes"]
            atoms_per_system = 2 * base_size**3

            for batch_size in batch_sizes:
                total_atoms = atoms_per_system * batch_size
                print(
                    f"\n  {total_atoms:,d} atoms "
                    f"({atoms_per_system:,d} x {batch_size})..."
                )

                # Reset memory
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()

                # Prepare system
                try:
                    system_data = prepare_batch_system(
                        base_size, batch_size, device, dtype
                    )
                except Exception as e:
                    print(f"    Failed to prepare system: {e}")
                    traceback.print_exc()
                    continue

                for method in methods:
                    for backend in backends:
                        for component in components:
                            result = run_benchmark(
                                method,
                                backend,
                                system_data,
                                component,
                                compute_forces,
                                compute_virial,
                                timer,
                            )
                            result["supercell_size"] = base_size
                            result["mode"] = mode
                            all_results.append(result)

                            if result["success"]:
                                throughput = (
                                    result["total_atoms"]
                                    / result["median_time_ms"]
                                    * 1000
                                )
                                mem_str = ""
                                if result.get("peak_memory_mb"):
                                    mem_str = f" | {result['peak_memory_mb']:.1f} MB"
                                print(
                                    f"    {method:5s} {backend:12s} {component:10s}: "
                                    f"{result['median_time_ms']:.3f} ms "
                                    f"({throughput:.1f} atoms/s){mem_str}"
                                )
                            else:
                                print(
                                    f"    {method:5s} {backend:12s} {component:10s}: "
                                    f"FAILED ({result.get('error_type', 'Unknown')})"
                                )

    # Save results
    if all_results:
        # Group by method and backend
        for method in methods:
            for backend in backends:
                method_results = [
                    r
                    for r in all_results
                    if r["method"] == method and r["backend"] == backend
                ]
                if method_results:
                    output_file = (
                        output_dir
                        / f"electrostatics_benchmark_{method}_{backend}_{gpu_sku}.csv"
                    )
                    # Collect all fieldnames across all results
                    all_fieldnames = []
                    seen = set()
                    for r in method_results:
                        for k in r.keys():
                            if k not in seen:
                                all_fieldnames.append(k)
                                seen.add(k)
                    with open(output_file, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=all_fieldnames, extrasaction="ignore"
                        )
                        writer.writeheader()
                        writer.writerows(method_results)
                    print(f"\n✓ Results saved to: {output_file}")

                    successful = [r for r in method_results if r.get("success", True)]
                    failed = [r for r in method_results if not r.get("success", True)]
                    print(
                        f"  Total: {len(method_results)} | "
                        f"Successful: {len(successful)} | "
                        f"Failed: {len(failed)}"
                    )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
