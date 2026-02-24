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
Single-System Geometry Optimization Benchmarks
==============================================

Benchmark optimization methods using Lennard-Jones systems.

Usage
-----
    python benchmark_opt_single.py --config benchmark_config.yaml

Backends
--------
- nvalchemiops: GPU-accelerated FIRE optimizer

Output
------
CSV file with single-system schema (11 columns):
- dynamics_opt_single_nvalchemiops_<gpu_sku>.csv
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
    """Run single-system optimization benchmarks.

    Parameters
    ----------
    config : dict
        Benchmark configuration.
    output_dir : Path
        Output directory for CSV files.
    """
    opt_config = config.get("opt_single", {})
    if not opt_config.get("enabled", True):
        print("Optimization single-system benchmarks disabled in config")
        return
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = opt_config.get("dtype", torch.float32)
    system_sizes = opt_config.get("system_sizes", [256, 512, 1024, 2048])
    optimizers = opt_config.get("optimizers", {})

    # Get potential parameters
    potential_config = config.get("potential", {})
    epsilon = potential_config.get("epsilon", 0.0104)
    sigma = potential_config.get("sigma", 3.40)
    cutoff = potential_config.get("cutoff", 8.5)
    skin = potential_config.get("skin", 1.0)
    neighbor_rebuild_interval = potential_config.get("neighbor_rebuild_interval", 10)

    gpu_sku = get_gpu_sku()
    results_nvalchemiops = []

    print("\nRunning Single-System Optimization Benchmarks (nvalchemiops)")
    print(f"GPU: {gpu_sku}")
    print_benchmark_header("Optimization")

    for num_atoms in system_sizes:
        # Create system (perturbed from equilibrium for optimization)
        num_cells = int((num_atoms // 4 + 1) ** (1 / 3))
        positions, cell = create_fcc_argon(
            num_unit_cells=num_cells,
            a=5.26,
        )

        positions = torch.as_tensor(positions, dtype=dtype, device=device)
        positions += torch.randn_like(positions) * 0.1
        cell = torch.as_tensor(cell, dtype=dtype, device=device)
        pbc = torch.tensor([True, True, True], device=positions.device)

        # Run nvalchemiops benchmarks
        nv_bench = NvalchemiOpsBenchmark(
            positions=positions.clone(),
            cell=cell.clone(),
            pbc=pbc,
            skin=skin,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_rebuild_interval=neighbor_rebuild_interval,
        )

        # FIRE
        if optimizers.get("fire", {}).get("enabled", False):
            fire_config = optimizers["fire"]
            result = nv_bench.run_fire(
                max_steps=fire_config.get("max_steps", 1000),
                force_tolerance=fire_config.get("force_tolerance", 0.01),
                dt_start=fire_config.get("dt_start", 1.0),
                dt_max=fire_config.get("dt_max", 10.0),
                dt_min=fire_config.get("dt_min", 0.001),
                alpha_start=fire_config.get("alpha_start", 0.1),
                n_min=fire_config.get("n_min", 5),
                f_inc=fire_config.get("f_inc", 1.1),
                f_dec=fire_config.get("f_dec", 0.5),
                f_alpha=fire_config.get("f_alpha", 0.99),
                maxstep=fire_config.get("maxstep", 0.2),
                warmup_steps=fire_config.get("warmup_steps", 0),
                log_interval=fire_config.get("log_interval", 100),
                check_interval=fire_config.get("check_interval", 20),
            )
            results_nvalchemiops.append(result)
            print_benchmark_result(result, is_md=False)

        nv_bench = NvalchemiOpsBenchmark(
            positions=positions.clone(),
            cell=cell.clone(),
            pbc=pbc,
            skin=skin,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_rebuild_interval=neighbor_rebuild_interval,
        )
        if optimizers.get("fire2", {}).get("enabled", False):
            fire2_config = optimizers["fire2"]
            result = nv_bench.run_fire2(
                max_steps=fire2_config.get("max_steps", 1000),
                force_tolerance=fire2_config.get("force_tolerance", 0.01),
                dt_start=fire2_config.get("dt_start", 0.045),
                tmax=fire2_config.get("tmax", 0.10),
                tmin=fire2_config.get("tmin", 0.005),
                delaystep=fire2_config.get("delaystep", 50),
                dtgrow=fire2_config.get("dtgrow", 1.09),
                dtshrink=fire2_config.get("dtshrink", 0.95),
                alpha0=fire2_config.get("alpha0", 0.20),
                alphashrink=fire2_config.get("alphashrink", 0.985),
                maxstep=fire2_config.get("maxstep", 0.25),
                warmup_steps=fire2_config.get("warmup_steps", 0),
                log_interval=fire2_config.get("log_interval", 100),
                check_interval=fire2_config.get("check_interval", 20),
            )
            results_nvalchemiops.append(result)
            print_benchmark_result(result, is_md=False)

    print_benchmark_footer()

    # Write CSV results
    if results_nvalchemiops:
        output_path = output_dir / f"dynamics_opt_single_nvalchemiops_{gpu_sku}.csv"
        write_results_csv(results_nvalchemiops, output_path)
        print(f"\nWrote nvalchemiops results to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Single-system optimization benchmarks for nvalchemiops"
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
