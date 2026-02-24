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
Single-System Molecular Dynamics Benchmarks
===========================================

Benchmark MD integrators using Lennard-Jones systems.

Usage
-----
    python benchmark_md_single.py --config benchmark_config.yaml

Backends
--------
- nvalchemiops: GPU-accelerated MD integrators (VelocityVerlet, Langevin, NoseHoover, NPT, NPH)

Output
------
CSV file with single-system schema (11 columns):
- dynamics_md_single_nvalchemiops_<gpu_sku>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmarks.dynamics.shared_utils import (
    NvalchemiOpsBenchmark,
    create_fcc_argon,
    get_gpu_sku,
    load_config,
    print_benchmark_footer,
    print_benchmark_header,
    print_benchmark_result,
    write_results_csv,
)


def run_benchmarks(config: dict, output_dir: Path) -> None:
    """Run single-system MD benchmarks.

    Parameters
    ----------
    config : dict
        Benchmark configuration.
    output_dir : Path
        Output directory for CSV files.
    """
    md_config = config.get("md_single", {})
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = md_config.get("dtype", torch.float32)
    if not md_config.get("enabled", True):
        print("MD single-system benchmarks disabled in config")
        return

    system_sizes = md_config.get("system_sizes", [256, 512, 1024, 2048, 4096])
    integrators = md_config.get("integrators", {})

    # Get potential parameters
    potential_config = config.get("potential", {})
    epsilon = potential_config.get("epsilon", 0.0104)
    sigma = potential_config.get("sigma", 3.40)
    cutoff = potential_config.get("cutoff", 8.5)
    skin = potential_config.get("skin", 1.0)
    neighbor_rebuild_interval = potential_config.get("neighbor_rebuild_interval", 10)

    gpu_sku = get_gpu_sku()
    results_nvalchemiops = []

    print("\nRunning Single-System MD Benchmarks (nvalchemiops)")
    print(f"GPU: {gpu_sku}")
    print_benchmark_header("MD")

    for num_atoms in system_sizes:
        # Create system
        num_cells = int((num_atoms // 4 + 1) ** (1.0 / 3.0))
        positions, cell = create_fcc_argon(
            num_unit_cells=num_cells,
            a=5.26,
        )

        positions = torch.as_tensor(positions, dtype=dtype, device=device)
        cell = torch.as_tensor(cell, dtype=dtype, device=device)
        pbc = torch.tensor([True, True, True], device=positions.device)

        # Run nvalchemiops benchmarks
        nv_bench = NvalchemiOpsBenchmark(
            positions=positions,
            cell=cell,
            pbc=pbc,
            skin=skin,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_rebuild_interval=neighbor_rebuild_interval,
        )

        # Velocity Verlet
        if integrators.get("velocity_verlet", {}).get("enabled", False):
            vv_config = integrators["velocity_verlet"]
            result = nv_bench.run_velocity_verlet(
                dt=vv_config.get("dt", 0.001),
                num_steps=vv_config.get("steps", 10000),
                warmup_steps=vv_config.get("warmup_steps", 100),
            )
            results_nvalchemiops.append(result)
            print_benchmark_result(result, is_md=True)

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
            results_nvalchemiops.append(result)
            print_benchmark_result(result, is_md=True)

        # NPT
        if integrators.get("npt", {}).get("enabled", False):
            npt_config = integrators["npt"]
            result = nv_bench.run_npt(
                dt=npt_config.get("dt", 0.001),
                num_steps=npt_config.get("steps", 10000),
                temperature=npt_config.get("temperature", 94.4),
                pressure=npt_config.get("pressure", 0.0),
                tau_t=npt_config.get("tau_t", 500.0),
                tau_p=npt_config.get("tau_p", 5000.0),
                chain_length=npt_config.get("chain_length", 3),
                warmup_steps=npt_config.get("warmup_steps", 100),
            )
            results_nvalchemiops.append(result)
            print_benchmark_result(result, is_md=True)

        # NPH
        if integrators.get("nph", {}).get("enabled", False):
            nph_config = integrators["nph"]
            result = nv_bench.run_nph(
                dt=nph_config.get("dt", 0.001),
                num_steps=nph_config.get("steps", 10000),
                temperature=nph_config.get("temperature", 94.4),
                pressure=nph_config.get("pressure", 1.0),
                tau_p=nph_config.get("tau_p", 1000.0),
                warmup_steps=nph_config.get("warmup_steps", 100),
            )
            results_nvalchemiops.append(result)
            print_benchmark_result(result, is_md=True)

    print_benchmark_footer()

    # Write CSV results
    if results_nvalchemiops:
        output_path = output_dir / f"dynamics_md_single_nvalchemiops_{gpu_sku}.csv"
        write_results_csv(results_nvalchemiops, output_path)
        print(f"\nWrote nvalchemiops results to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Single-system MD benchmarks for nvalchemiops"
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
