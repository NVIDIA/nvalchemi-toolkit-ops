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
Batched Molecular Dynamics Benchmarks (nvalchemiops only)
=========================================================

Benchmark batched MD integrators using nvalchemiops GPU-accelerated implementations.

Usage
-----
    python benchmark_md_batch.py --config benchmark_config.yaml

Output
------
CSV file with batched schema (14 columns):
- dynamics_md_batch_nvalchemiops_<gpu_sku>.csv
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from benchmarks.dynamics.shared_utils import (
    NvalchemiOpsBenchmark,
    create_fcc_argon,
    get_gpu_sku,
    load_config,
    print_batch_benchmark_header,
    print_batch_benchmark_result,
)


def create_batched_system(
    num_atoms_per_system: int,
    batch_size: int,
    lattice_constant: float = 5.26,
    temperature: float = 300.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Create batched FCC argon systems.

    Parameters
    ----------
    num_atoms_per_system : int
        Number of atoms per system.
    batch_size : int
        Number of independent systems.
    lattice_constant : float
        FCC lattice constant.
    temperature : float
        Initial temperature.
    device : str
        Torch device.
    dtype : torch.dtype
        Data type.

    Returns
    -------
    positions : torch.Tensor
        Batched positions, shape (total_atoms, 3).
    velocities : torch.Tensor
        Batched velocities, shape (total_atoms, 3).
    cell : torch.Tensor
        Cell matrix, shape (batch_size, 3, 3).
    masses : torch.Tensor
        Batched masses, shape (total_atoms,).
    batch_idx : torch.Tensor
        Batch index for each atom, shape (total_atoms,).
    atom_ptr : torch.Tensor
        Pointer array, shape (batch_size + 1,).
    """
    num_cells = int((num_atoms_per_system // 4 + 1) ** (1.0 / 3.0))
    pos, cell = create_fcc_argon(
        num_unit_cells=num_cells,
        a=lattice_constant,
    )

    actual_num_atoms = pos.shape[0]

    # Replicate for batch
    positions_list = []
    cell_list = []
    positions = torch.as_tensor(pos, dtype=dtype, device=device)
    cell = torch.as_tensor(cell, dtype=dtype, device=device)
    for i in range(batch_size):
        # Slightly perturb each system to make them independent
        pos_perturbed = positions + torch.randn_like(positions) * 0.01

        positions_list.append(pos_perturbed)
        cell_list.append(cell)

    # Stack into batched tensors
    positions = torch.cat(positions_list, dim=0)
    cell = torch.stack(cell_list, dim=0).reshape(batch_size, 3, 3)

    # Create batch_idx: [0,0,0,...,1,1,1,...,2,2,2,...]
    batch_idx = torch.repeat_interleave(
        torch.arange(batch_size, device=device), actual_num_atoms
    ).to(torch.int32)

    # Create atom_ptr: [0, N, 2N, 3N, ..., batch_size*N]
    atom_ptr = torch.arange(
        0,
        (batch_size + 1) * actual_num_atoms,
        actual_num_atoms,
        device=device,
        dtype=torch.int32,
    )
    return positions, cell, batch_idx, atom_ptr


def run_benchmarks(config: dict, output_dir: Path) -> None:
    """Run batched MD benchmarks.

    Parameters
    ----------
    config : dict
        Benchmark configuration.
    output_dir : Path
        Output directory for CSV files.
    """
    batch_config = config.get("md_batch", {})
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = batch_config.get("dtype", torch.float32)
    if not batch_config.get("enabled", False):
        print("Batched MD benchmarks disabled in config")
        return

    system_sizes = batch_config.get("system_sizes", [256, 512, 1024])
    batch_sizes = batch_config.get("batch_sizes", [1, 2, 4, 8, 16, 32])
    integrators = batch_config.get("integrators", {})

    # Extract potential configuration parameters
    potential_config = config.get("potential", {})
    epsilon = potential_config.get("epsilon", 0.0104)
    sigma = potential_config.get("sigma", 3.40)
    cutoff = potential_config.get("cutoff", 8.5)
    skin = potential_config.get("skin", 1.0)
    neighbor_rebuild_interval = potential_config.get("neighbor_rebuild_interval", 10)

    gpu_sku = get_gpu_sku()
    results = []

    # Print header with title and GPU info
    print("\nRunning Batched MD Benchmarks (nvalchemiops)")
    print(f"GPU: {gpu_sku}")
    print_batch_benchmark_header()

    for num_atoms in system_sizes:
        for batch_size in batch_sizes:
            # Create batched system
            (
                batch_positions,
                batch_cells,
                batch_idx,
                atom_ptr,
            ) = create_batched_system(
                num_atoms_per_system=num_atoms,
                batch_size=batch_size,
                lattice_constant=5.26,
                temperature=300.0,
                device=device,
                dtype=dtype,
            )

            pbc = torch.tensor([True, True, True], device=batch_positions.device)

            # Run nvalchemiops benchmarks
            nv_bench = NvalchemiOpsBenchmark(
                positions=batch_positions.clone(),
                cell=batch_cells.clone(),
                pbc=pbc.clone(),
                skin=skin,
                epsilon=epsilon,
                sigma=sigma,
                cutoff=cutoff,
                neighbor_rebuild_interval=neighbor_rebuild_interval,
                batch_idx=batch_idx.clone(),
            )

            # Velocity Verlet
            if integrators.get("velocity_verlet", {}).get("enabled", False):
                vv_config = integrators["velocity_verlet"]
                result = nv_bench.run_velocity_verlet(
                    dt=vv_config.get("dt", 0.001),
                    num_steps=vv_config.get("steps", 10000),
                    warmup_steps=vv_config.get("warmup_steps", 100),
                )
                results.append(result)
                print_batch_benchmark_result(result, is_md=True)

            # Langevin
            if integrators.get("langevin", {}).get("enabled", False):
                lang_config = integrators["langevin"]
                result = nv_bench.run_langevin(
                    dt=lang_config.get("dt", 0.001),
                    num_steps=lang_config.get("steps", 10000),
                    temperature=lang_config.get("temperature", 94.4),
                    friction=lang_config.get("friction", 0.01),
                    warmup_steps=lang_config.get("warmup_steps", 100),
                )
                results.append(result)
                print_batch_benchmark_result(result, is_md=True)

            # NPT
            if integrators.get("npt", {}).get("enabled", False):
                npt_config = integrators["npt"]
                result = nv_bench.run_npt(
                    dt=npt_config.get("dt", 0.001),
                    num_steps=npt_config.get("steps", 10000),
                    temperature=npt_config.get("temperature", 94.4),
                    pressure=npt_config.get("pressure", 1.0),
                    tau_t=npt_config.get("tau_t", 500.0),
                    tau_p=npt_config.get("tau_p", 5000.0),
                    chain_length=npt_config.get("chain_length", 3),
                    warmup_steps=npt_config.get("warmup_steps", 100),
                )
                results.append(result)
                print_batch_benchmark_result(result, is_md=True)

            # NPH
            if integrators.get("nph", {}).get("enabled", False):
                nph_config = integrators["nph"]
                result = nv_bench.run_nph(
                    dt=nph_config.get("dt", 0.001),
                    num_steps=nph_config.get("steps", 10000),
                    temperature=nph_config.get("temperature", 94.4),
                    pressure=nph_config.get("pressure", 1.0),
                    tau_p=nph_config.get("tau_p", 5000.0),
                    warmup_steps=nph_config.get("warmup_steps", 100),
                )
                results.append(result)
                print_batch_benchmark_result(result, is_md=True)

        del nv_bench
        torch.cuda.empty_cache()
        gc.collect()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batched MD benchmarks for nvalchemiops"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for CSV files",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_benchmarks(config, output_dir)


if __name__ == "__main__":
    main()
