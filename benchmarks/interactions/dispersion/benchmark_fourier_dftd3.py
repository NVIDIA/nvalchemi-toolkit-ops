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
FourierD3 benchmarks
====================

Measures the particle-mesh dispersion correction against the real-space ``dftd3``.

The comparison that matters is **at matched accuracy**. A ``1/r^6`` interaction summed over
three dimensions leaves a truncation error decaying as ``1/r^3``, so a real-space cutoff of
6 Angstrom is not a converged calculation; converging it takes 15 Angstrom or more. Timing
FourierD3 against a 6 Angstrom ``dftd3`` compares two different calculations and flatters
the truncated one. Both are reported here so the distinction stays visible.

Timings include the neighbour-list build, because that is what a caller pays. FourierD3 needs
only the short coordination-number list, which an MLFF already builds, so a second set of
columns reports the evaluation alone -- the marginal cost when the list comes for free.

Usage
-----
::

    python benchmarks/interactions/dispersion/benchmark_fourier_dftd3.py \\
        --output-dir /tmp/fd3-bench
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.config import load_yaml_config  # noqa: E402
from benchmarks.suite_utils import (  # noqa: E402
    build_failure_result,
    build_result,
    create_run_directory,
    cuda_timed_runs,
    failure_error_type,
    make_row_meta,
    measure_memory_torch,
    save_results,
)

DEFAULT_CONFIG = Path(__file__).with_name("benchmark_fourier_config.yaml")

__all__ = [
    "benchmark_fourier_d3",
    "dry_run_from_config",
    "main",
    "merge_cli_overrides",
    "parse_args",
    "run_from_config",
]

# Mesh size against system size, from the FourierD3 paper's Table I.
MESH_SCHEDULE = ((200, 16), (2000, 32), (20000, 64), (None, 128))

# Coordination-number cutoff. Six Angstrom is the cutoff of foundation MLFFs such as
# MACE-MP, which is the list FourierD3 is designed to reuse.
CN_CUTOFF = 6.0

# PBE-D3(BJ).
DAMPING = dict(a1=0.4289, a2=4.4407, s8=0.7875, s6=1.0)


def mesh_for(num_atoms: int) -> int:
    """Mesh edge length for a system size, following the paper's schedule."""
    for threshold, mesh in MESH_SCHEDULE:
        if threshold is None or num_atoms <= threshold:
            return mesh
    return MESH_SCHEDULE[-1][1]


def _synthetic_tables(max_z: int = 10):
    """Reference tables shaped like the DFT-D3 parametrisation.

    The benchmark measures cost, not chemistry, and generating the tables keeps the run free
    of a network fetch. Rank and therefore mesh channel count follow the real tables closely
    enough for timing to be representative.
    """
    rng = np.random.default_rng(0)
    n_ref = 5
    used = {1: 2, 6: 5}
    c6ab = np.zeros((max_z, max_z, n_ref, n_ref))
    cn_ref = np.zeros_like(c6ab)
    factors = {z: rng.normal(size=(n, 3)) for z, n in used.items()}
    for z_i, n_i in used.items():
        for z_j, n_j in used.items():
            c6ab[z_i, z_j, :n_i, :n_j] = factors[z_i] @ factors[z_j].T + 20.0
            for p in range(n_i):
                cn_ref[z_i, z_j, p, :n_j] = np.linspace(0.0, 3.5, n_i)[p]
    rcov = np.zeros(max_z)
    rcov[[1, 6]] = [0.6, 1.2]
    r4r2 = np.zeros(max_z)
    r4r2[[1, 6]] = [1.0, 1.4]
    return rcov, r4r2, c6ab, cn_ref


def _make_system(num_atoms: int, density: float, device: str, dtype=None):
    """A random periodic cell at the requested number density.

    The precision is single by default, matching how MLFF inference runs and keeping every
    method in the sweep on the same footing. Double precision is a large penalty on hardware
    with a reduced FP64 rate, so mixing the two across methods would not be a comparison.
    """
    import torch

    dtype = dtype or torch.float32
    rng = np.random.default_rng(1)
    box = (num_atoms / density) ** (1.0 / 3.0)
    positions = torch.tensor(
        rng.uniform(0.0, box, (num_atoms, 3)), dtype=dtype, device=device
    )
    numbers = torch.tensor(
        rng.choice([1, 6], num_atoms), dtype=torch.int32, device=device
    )
    cell = torch.eye(3, dtype=dtype, device=device) * box
    return positions, numbers, cell, box


def benchmark_fourier_d3(
    num_atoms, density, device, num_runs, warmup_runs, reuse_setup=False, dtype=None
):
    """Time one FourierD3 evaluation, and the neighbour-list build separately.

    Parameters
    ----------
    reuse_setup : bool
        Build the cell- and mesh-derived quantities once and reuse them, which is what a
        constant-volume trajectory does. They cost a matrix inversion and a set of spline
        moduli per call otherwise, neither of which depends on the positions.

    Returns
    -------
    dict
        ``time_eval_seconds`` is the evaluation alone and is the reported metric, because
        the kernel style guide requires neighbour-list construction to be pre-computed
        separately rather than timed with the kernel. ``time_neighbor_seconds`` and
        ``time_total_seconds`` are carried alongside it: the list a method needs is part of
        what a converged dispersion correction costs, and for real-space D3 it grows with the
        cutoff, so the total is worth seeing even though it is not the headline.
    """
    import torch

    from nvalchemiops.torch.interactions.dispersion import (
        FourierD3Parameters,
        fourier_dftd3,
    )
    from nvalchemiops.torch.interactions.dispersion._fourier_dftd3 import FourierD3Setup
    from nvalchemiops.torch.neighbors import neighbor_list

    dtype = dtype or torch.float32
    positions, numbers, cell, _box = _make_system(num_atoms, density, device, dtype)
    rcov, r4r2, c6ab, cn_ref = _synthetic_tables()

    def tensor(array):
        return torch.tensor(array, dtype=dtype, device=device)

    parameters = FourierD3Parameters.from_tables(
        tensor(rcov),
        tensor(r4r2),
        tensor(c6ab),
        tensor(cn_ref),
        species=[1, 6],
        device=device,
    )
    pbc = torch.tensor([True, True, True], device=device)
    mesh = mesh_for(num_atoms)

    def build_list():
        return neighbor_list(
            positions,
            cutoff=CN_CUTOFF,
            cell=cell,
            pbc=pbc,
            return_neighbor_list=True,
            method="cell_list",
        )

    neighbors, pointer, shifts = build_list()
    setup = (
        FourierD3Setup.build(cell, parameters.n_species, (mesh, mesh, mesh))
        if reuse_setup
        else None
    )

    def evaluate():
        return fourier_dftd3(
            positions,
            numbers,
            fd3_params=parameters,
            cell=cell,
            r_cut=CN_CUTOFF,
            mesh_dimensions=(mesh, mesh, mesh),
            neighbor_list=neighbors,
            neighbor_ptr=pointer,
            unit_shifts=shifts,
            setup=setup,
            **DAMPING,
        )

    _, mem_info = measure_memory_torch(evaluate)
    time_list = cuda_timed_runs(build_list, num_runs, warmup_runs=warmup_runs)
    time_eval = cuda_timed_runs(evaluate, num_runs, warmup_runs=warmup_runs)
    return {
        "time_total_seconds": time_list + time_eval,
        "time_eval_seconds": time_eval,
        "time_neighbor_seconds": time_list,
        "mesh": mesh,
        "edges": int(neighbors.shape[1]),
        "mem_info": mem_info,
    }


def benchmark_real_space_d3(
    num_atoms, density, cutoff, device, num_runs, warmup_runs, dtype=None
):
    """Time one real-space ``dftd3`` evaluation at a given interaction cutoff."""
    import torch

    from benchmarks.constants import DEFAULT_NL_SAFETY_FACTOR
    from nvalchemiops.neighbors import estimate_max_neighbors
    from nvalchemiops.torch.interactions.dispersion import D3Parameters, dftd3
    from nvalchemiops.torch.neighbors import neighbor_list

    dtype = dtype or torch.float32
    positions, numbers, cell, _box = _make_system(num_atoms, density, device, dtype)
    rcov, r4r2, c6ab, cn_ref = _synthetic_tables()

    def tensor(array):
        return torch.tensor(array, dtype=dtype, device=device)

    parameters = D3Parameters(
        rcov=tensor(rcov), r4r2=tensor(r4r2), c6ab=tensor(c6ab), cn_ref=tensor(cn_ref)
    )
    pbc = torch.tensor([True, True, True], device=device)

    # Sized from the cutoff and the density rather than pinned. The dense kernel scans every
    # slot in a row, so a fixed width is paid for directly: at 6 A there are about ninety
    # neighbours, and a width of 8192 made the real-space method look an order of magnitude
    # slower than it is.
    max_neighbors = estimate_max_neighbors(
        cutoff, atomic_density=density * DEFAULT_NL_SAFETY_FACTOR
    )

    def build_list():
        return neighbor_list(
            positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list",
            max_neighbors=max_neighbors,
        )

    matrix, _counts, matrix_shifts = build_list()

    def evaluate():
        return dftd3(
            positions=positions.float(),
            numbers=numbers,
            cell=cell.float(),
            neighbor_matrix=matrix,
            neighbor_matrix_shifts=matrix_shifts,
            d3_params=parameters,
            a1=DAMPING["a1"],
            a2=DAMPING["a2"],
            s8=DAMPING["s8"],
        )

    _, mem_info = measure_memory_torch(evaluate)
    time_list = cuda_timed_runs(build_list, num_runs, warmup_runs=warmup_runs)
    time_eval = cuda_timed_runs(evaluate, num_runs, warmup_runs=warmup_runs)
    return {
        "time_total_seconds": time_list + time_eval,
        "time_eval_seconds": time_eval,
        "time_neighbor_seconds": time_list,
        "mem_info": mem_info,
    }


def run_from_config(config: dict, output_dir, backend: str = "torch") -> list[dict]:
    """Sweep system size for FourierD3 and for ``dftd3`` at several cutoffs."""
    parameters = config.get("parameters", {})
    atom_counts = parameters.get("atom_counts", [500, 2000, 8000, 20000])
    density = parameters.get("density", 0.1)
    cutoffs = parameters.get("real_space_cutoffs", [6.0, 15.0, 20.0])
    num_runs = parameters.get("timing_runs", 10)
    warmup_runs = parameters.get("warmup_runs", 3)
    device = parameters.get("device", "cuda")
    results: list[dict] = []

    for num_atoms in atom_counts:
        row_meta = make_row_meta(
            "random", "system_size", backend, num_atoms, 1, num_atoms
        )
        for method, reuse in (("fourier_dftd3", False), ("fourier_dftd3_setup", True)):
            try:
                measured = benchmark_fourier_d3(
                    num_atoms, density, device, num_runs, warmup_runs, reuse_setup=reuse
                )
                results.append(
                    build_result(
                        method=method,
                        time_seconds=measured["time_eval_seconds"],
                        mem_info=measured["mem_info"],
                        timing_runs=num_runs,
                        warmup_runs=warmup_runs,
                        cutoff=CN_CUTOFF,
                        mesh=measured["mesh"],
                        density=density,
                        time_total_seconds=measured["time_total_seconds"],
                        time_eval_seconds=measured["time_eval_seconds"],
                        time_neighbor_seconds=measured["time_neighbor_seconds"],
                        **row_meta,
                    )
                )
            except Exception as error:  # noqa: BLE001 - one failure must not stop the sweep
                results.append(
                    build_failure_result(
                        method=method,
                        error=str(error),
                        error_type=failure_error_type(error),
                        timing_runs=num_runs,
                        warmup_runs=warmup_runs,
                        **row_meta,
                    )
                )

        for cutoff in cutoffs:
            try:
                measured = benchmark_real_space_d3(
                    num_atoms, density, cutoff, device, num_runs, warmup_runs
                )
                results.append(
                    build_result(
                        method=f"dftd3_cutoff_{cutoff:g}",
                        time_seconds=measured["time_eval_seconds"],
                        mem_info=measured["mem_info"],
                        timing_runs=num_runs,
                        warmup_runs=warmup_runs,
                        cutoff=cutoff,
                        density=density,
                        time_total_seconds=measured["time_total_seconds"],
                        time_eval_seconds=measured["time_eval_seconds"],
                        time_neighbor_seconds=measured["time_neighbor_seconds"],
                        **row_meta,
                    )
                )
            except Exception as error:  # noqa: BLE001
                results.append(
                    build_failure_result(
                        method=f"dftd3_cutoff_{cutoff:g}",
                        error=str(error),
                        error_type=failure_error_type(error),
                        timing_runs=num_runs,
                        warmup_runs=warmup_runs,
                        **row_meta,
                    )
                )

    if output_dir is None:
        # No --output-dir, so use the directory the config names, as the other runners do.
        base_dir = config.get("output", {}).get("base_dir")
        if base_dir is not None:
            output_dir = create_run_directory(base_dir, prefix="fd3")
    if output_dir is not None:
        save_results(
            results,
            Path(output_dir) / "fd3-random-system-size-scaling.csv",
            replace_backend=backend,
        )
    return results


def dry_run_from_config(config: dict, backend: str | None = None) -> list[dict]:
    """Expand the case matrix without allocating or timing anything."""
    parameters = config.get("parameters", {})
    atom_counts = parameters.get("atom_counts", [500, 2000, 8000, 20000])
    cutoffs = parameters.get("real_space_cutoffs", [6.0, 15.0, 20.0])
    return [
        {"method": method, "atoms_per_system": num_atoms, "backend": backend or "torch"}
        for num_atoms in atom_counts
        for method in ["fourier_dftd3", "fourier_dftd3_setup"]
        + [f"dftd3_cutoff_{c:g}" for c in cutoffs]
    ]


def parse_args():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="FourierD3 benchmarks")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="YAML configuration. Defaults to the one beside this script.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the results CSV. Defaults to output.base_dir from the config.",
    )
    parser.add_argument("--atom-counts", type=int, nargs="+", default=None)
    parser.add_argument("--density", type=float, default=None)
    parser.add_argument("--timing-runs", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply the CLI flags that were actually given on top of the YAML config."""
    parameters = config.setdefault("parameters", {})
    for flag, key in (
        ("atom_counts", "atom_counts"),
        ("density", "density"),
        ("timing_runs", "timing_runs"),
        ("warmup_runs", "warmup_runs"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            parameters[key] = value
    return config


def main():
    """Run the sweep and print a summary table."""
    args = parse_args()
    config = merge_cli_overrides(load_yaml_config(args.config), args)
    density = config["parameters"].get("density", 0.1)
    if args.dry_run:
        for case in dry_run_from_config(config):
            print(case)
        return 0

    results = run_from_config(config, args.output_dir)
    print(
        f"\ndensity {density} atoms/A^3;  times in ms. 'eval' is the reported metric: "
        f"the kernel style\nguide keeps neighbour-list construction out of kernel timing. "
        f"'nlist' and 'total' are shown\nbecause the list a method needs is part of what a "
        f"converged correction costs.\n"
    )
    print(f"{'N':>7}  {'method':<22} {'eval':>8} {'nlist':>8} {'total':>8}")
    for row in results:
        if not row.get("success", True):
            print(f"{row['atoms_per_system']:>7}  {row['method']:<22}    failed")
            continue
        print(
            f"{row['atoms_per_system']:>7}  {row['method']:<22} "
            f"{row['time_eval_seconds'] * 1e3:8.2f} "
            f"{row['time_neighbor_seconds'] * 1e3:8.2f} "
            f"{row['time_total_seconds'] * 1e3:8.2f}"
        )
    return 0 if any(r.get("success", True) for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
