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

Fixed-configuration throughput for the particle-mesh dispersion correction, alongside the
real-space ``dftd3`` at several cutoffs.

.. warning::
    This is **not a matched-accuracy comparison**. The mesh schedule follows the paper's
    atom-count table and the ``dftd3`` rows use fixed cutoffs; no accuracy relationship
    between the two has been established for this system, so the rows are not directly
    comparable as "same answer, different cost". Read each as the cost of one stated
    configuration.

Systems are CsCl (B2) supercells from the shared benchmark builders, so the geometry and the
density are a real crystal's rather than a chosen number. Positions, cells and cutoffs are
converted to Bohr, matching the published D3 tables and the existing ``dftd3`` benchmark.

``time_eval_seconds`` is the canonical metric, consistent with the other kernel benchmarks.
``time_neighbor_seconds`` and ``time_total_seconds`` are reported alongside it.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from benchmarks.config import load_yaml_config  # noqa: E402
from benchmarks.constants import ANGSTROM_TO_BOHR  # noqa: E402
from benchmarks.suite_systems import (  # noqa: E402
    compute_atomic_density,
    configs_for_mode,
    create_system,
    filter_configs_by_total_atoms,
    planned_atom_counts,
    resolve_nh3_dir,
)
from benchmarks.suite_utils import (  # noqa: E402
    build_failure_result,
    build_result,
    build_skipped_result,
    clean_gpu,
    configure_input_provenance,
    create_run_directory,
    cuda_timed_runs,
    failure_error_type,
    make_csv_name,
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
# Mesh spacing in Bohr, applied uniformly at every system size, so mesh resolution stays
# constant as the cell grows. Replaces the paper's atom-count schedule, which tied the mesh
# to a number of atoms without establishing what accuracy it buys for this system.
MESH_SPACING_BOHR = 0.55

# Coordination-number cutoff. Six Angstrom is the cutoff of foundation MLFFs such as
# MACE-MP, which is the list FourierD3 is designed to reuse.
CN_CUTOFF_ANGSTROM = 6.0

# Positions, cells and cutoffs go to the kernels in Bohr, because the published D3 tables
# and the damping parameters are atomic units. The shared CsCl builder returns Angstrom.
CN_CUTOFF = CN_CUTOFF_ANGSTROM * ANGSTROM_TO_BOHR

# Same cache the other dispersion benchmarks read.
D3_PARAMS_PATH = "~/.cache/nvalchemiops/dftd3_parameters.pt"

# PBE-D3(BJ).
DAMPING = dict(a1=0.4289, a2=4.4407, s8=0.7875, s6=1.0)


def _published_tables(device: str, dtype):
    """Grimme's published D3 tables, from the cache the other dispersion benchmarks use.

    Atomic units, so positions, cells and cutoffs must be in Bohr to match.
    """
    import torch

    from benchmarks.interactions.dispersion.benchmark_dftd3 import (
        _ensure_d3_parameter_file,
        _resolve_d3_params_path,
    )

    path = _resolve_d3_params_path(D3_PARAMS_PATH)
    _ensure_d3_parameter_file(path)
    tables = torch.load(path, map_location="cpu", weights_only=True)
    return tuple(
        tables[name].to(device=device, dtype=dtype)
        for name in ("rcov", "r4r2", "c6ab", "cn_ref")
    )


def _to_bohr(data):
    """Positions, cell and density from a shared system dict, converted to atomic units.

    The builders work in Angstrom; the published D3 tables are atomic units.
    """
    return (
        data["positions"] * ANGSTROM_TO_BOHR,
        data["atomic_numbers"].to(torch.int32),
        data["cell"] * ANGSTROM_TO_BOHR,
        compute_atomic_density(data) / ANGSTROM_TO_BOHR**3,
    )


def _mesh_for_cell(cell_bohr):
    """Mesh dimensions from a fixed spacing, rounded up to an FFT-friendly size.

    A fixed spacing keeps the mesh resolution constant as the cell grows. It is a stated
    configuration, not a setting calibrated against any ``dftd3`` cutoff.
    """
    from nvalchemiops.interactions.dispersion._fourier_dftd3 import _resolve_mesh

    lengths = cell_bohr.reshape(-1, 3, 3).norm(dim=-1).max(dim=0).values.tolist()
    return _resolve_mesh(None, MESH_SPACING_BOHR, lengths, 4)


def benchmark_fourier_d3(data, num_runs, warmup_runs, reuse_setup=False):
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
    from nvalchemiops.torch.interactions.dispersion import (
        FourierD3Parameters,
        fourier_dftd3,
    )
    from nvalchemiops.torch.interactions.dispersion._fourier_dftd3 import FourierD3Setup
    from nvalchemiops.torch.neighbors import neighbor_list

    positions, numbers, cell, density = _to_bohr(data)
    device, dtype = str(positions.device), positions.dtype
    species = sorted({int(z) for z in numbers.unique().tolist()})
    rcov, r4r2, c6ab, cn_ref = _published_tables(device, dtype)

    # Matching dtype matters here: ``fourier_dftd3`` coerces the bundle to the positions'
    # dtype on every call, so float64 parameters would put five tensor conversions inside
    # the timed region that the real-space comparison does not pay.
    parameters = FourierD3Parameters.from_tables(
        rcov,
        r4r2,
        c6ab,
        cn_ref,
        species=species,
        device=device,
        dtype=dtype,
    )
    pbc = data["pbc"]
    mesh = _mesh_for_cell(cell)
    # A batched run concatenates several systems, so the list has to be built per system.
    # Leaving ``method`` unset lets the dispatcher pick the batch builder when batch_idx is
    # present; pinning "cell_list" would link replicas across systems.
    batch_idx = data.get("batch_idx") if int(data.get("batch_size", 1)) > 1 else None

    def build_list():
        return neighbor_list(
            positions,
            cutoff=CN_CUTOFF,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            return_neighbor_list=True,
        )

    neighbors, pointer, shifts = build_list()
    setup = (
        FourierD3Setup.build(cell.reshape(-1, 3, 3), parameters.n_species, mesh)
        if reuse_setup
        else None
    )

    def evaluate():
        return fourier_dftd3(
            positions,
            numbers,
            fourier_d3_params=parameters,
            cell=cell,
            cutoff=CN_CUTOFF,
            mesh_dimensions=mesh,
            neighbor_list=neighbors,
            neighbor_ptr=pointer,
            unit_shifts=shifts,
            batch_idx=batch_idx,
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
        "atoms": int(positions.shape[0]),
        "density": density,
        "mesh": mesh[0],
        "edges": int(neighbors.shape[1]),
        "mem_info": mem_info,
    }


def benchmark_real_space_d3(data, cutoff_angstrom, num_runs, warmup_runs):
    """Time one real-space ``dftd3`` evaluation at a given interaction cutoff.

    ``cutoff_angstrom`` is in Angstrom, as the label on the row says; it is converted to
    Bohr here because the tables and damping parameters are atomic units.
    """
    cutoff = cutoff_angstrom * ANGSTROM_TO_BOHR
    import torch

    from benchmarks.constants import DEFAULT_NL_SAFETY_FACTOR
    from nvalchemiops.neighbors import estimate_max_neighbors
    from nvalchemiops.torch.interactions.dispersion import D3Parameters, dftd3
    from nvalchemiops.torch.neighbors import neighbor_list

    positions, numbers, cell, density = _to_bohr(data)
    device, dtype = str(positions.device), positions.dtype
    rcov, r4r2, c6ab, cn_ref = _published_tables(device, dtype)

    parameters = D3Parameters(rcov=rcov, r4r2=r4r2, c6ab=c6ab, cn_ref=cn_ref)
    pbc = torch.tensor([True, True, True], device=device)

    # Sized from the cutoff and the density rather than pinned. The dense kernel scans every
    # slot in a row, so a fixed width is paid for directly: at 6 A there are about ninety
    # neighbours, and a width of 8192 made the real-space method look an order of magnitude
    # slower than it is.
    max_neighbors = estimate_max_neighbors(
        cutoff, atomic_density=density * DEFAULT_NL_SAFETY_FACTOR
    )

    # As above: the dispatcher picks the batch builder when batch_idx is present.
    batch_idx = data.get("batch_idx") if int(data.get("batch_size", 1)) > 1 else None

    def build_list():
        return neighbor_list(
            positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
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
            batch_idx=batch_idx,
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
        "atoms": int(positions.shape[0]),
        "density": density,
        "mem_info": mem_info,
    }


def _resolve_backend(config: dict, backend: str | None) -> str:
    """Resolve the backend and refuse any this benchmark cannot actually run.

    ``backend`` only ever reached the row labels: the measurement functions import torch
    unconditionally. Passing ``"jax"`` would therefore have run the torch kernels and written
    a CSV claiming they were JAX, which is worse than failing -- a backend comparison would
    show identical numbers and look like a finding.
    """
    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")
    if backend != "torch":
        raise ValueError(
            f"FourierD3 benchmark supports only the torch backend, got {backend!r}. "
            "There is no JAX measurement path here, so the rows would be mislabelled."
        )
    return backend


def _fourier_rows(data, config, num_runs, warmup_runs, row_meta, backend):
    """One row per FourierD3 variant and per real-space cutoff, for one configuration."""
    cutoffs = config.get("parameters", {}).get("real_space_cutoffs", [6.0, 15.0, 20.0])
    rows = []

    def record(method, measured, cutoff):
        rows.append(
            build_result(
                method=method,
                time_seconds=measured["time_eval_seconds"],
                mem_info=measured["mem_info"],
                timing_runs=num_runs,
                warmup_runs=warmup_runs,
                cutoff=cutoff,
                mesh=measured.get("mesh"),
                density=measured["density"],
                time_total_seconds=measured["time_total_seconds"],
                time_eval_seconds=measured["time_eval_seconds"],
                time_neighbor_seconds=measured["time_neighbor_seconds"],
                **row_meta,
            )
        )

    def fail(method, error, cutoff):
        rows.append(
            build_failure_result(
                method=method,
                error=str(error),
                error_type=failure_error_type(error),
                cutoff=cutoff,
                timing_runs=num_runs,
                warmup_runs=warmup_runs,
                **row_meta,
            )
        )

    for method, reuse in (("fourier_dftd3", False), ("fourier_dftd3_setup", True)):
        try:
            record(
                method,
                benchmark_fourier_d3(data, num_runs, warmup_runs, reuse_setup=reuse),
                CN_CUTOFF_ANGSTROM,
            )
        except Exception as error:  # noqa: BLE001 - one failure must not stop the sweep
            fail(method, error, CN_CUTOFF_ANGSTROM)

    for cutoff_angstrom in cutoffs:
        method = f"dftd3_cutoff_{cutoff_angstrom:g}"
        try:
            record(
                method,
                benchmark_real_space_d3(data, cutoff_angstrom, num_runs, warmup_runs),
                cutoff_angstrom,
            )
        except Exception as error:  # noqa: BLE001
            fail(method, error, cutoff_angstrom)
    return rows


def run_from_config(config: dict, output_dir, backend: str | None = None) -> list[dict]:
    """Run the FourierD3 sweep over the shared systems and scaling modes.

    Same matrix and result contract as the DFT-D3 runner: every enabled system crossed with
    every enabled scaling mode, one standardized CSV per pair.
    """
    backend = _resolve_backend(config, backend)
    if config.get("runtime", {}).get("dry_run", False):
        return dry_run_from_config(config, backend=backend)

    parameters = config.get("parameters", {})
    num_runs = parameters.get("timing_runs", 10)
    warmup_runs = parameters.get("warmup_runs", 3)
    max_total_atoms = parameters.get("max_total_atoms")

    if output_dir is None:
        output_dir = create_run_directory(config["output"]["base_dir"], prefix="fd3")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from benchmarks.interactions.dispersion.benchmark_dftd3 import (
        _resolve_d3_params_path,
    )

    params_path = _resolve_d3_params_path(D3_PARAMS_PATH)
    configure_input_provenance(
        {"d3_parameters": params_path}, metadata_values={"benchmark": "fd3"}
    )

    all_results: list[dict] = []
    for sys_name, sys_config in config["systems"].items():
        if not sys_config.get("enabled", True):
            continue
        nh3_dir = resolve_nh3_dir(sys_config)
        for mode_name, mode_config in config["scaling"].items():
            if not isinstance(mode_config, dict) or not mode_config.get(
                "enabled", True
            ):
                continue
            print(
                f"\n{'=' * 70}\nFourierD3: {sys_name.upper()} / {mode_name}\n{'=' * 70}"
            )
            configs = configs_for_mode(
                mode_name, mode_config, sys_name, sys_config, nh3_dir
            )
            configs, skipped = filter_configs_by_total_atoms(
                configs, sys_name, max_total_atoms
            )
            results: list[dict] = []
            for cfg, skipped_total in skipped:
                n, bs, total = planned_atom_counts(sys_name, cfg)
                results.append(
                    build_skipped_result(
                        method="fourier_dftd3",
                        reason=f">{max_total_atoms} max_total_atoms",
                        timing_runs=num_runs,
                        warmup_runs=warmup_runs,
                        **make_row_meta(sys_name, mode_name, backend, n, bs, total),
                    )
                )
            for cfg in configs:
                try:
                    data = create_system(
                        sys_name,
                        num_atoms=cfg["num_atoms"],
                        pdb_path=cfg.get("pdb_path"),
                        batch_size=cfg["batch_size"],
                        backend=backend,
                    )
                except Exception as error:  # noqa: BLE001
                    n, bs, total = planned_atom_counts(sys_name, cfg)
                    results.append(
                        build_failure_result(
                            method="fourier_dftd3",
                            error=str(error),
                            error_type=failure_error_type(error),
                            failure_stage="system_setup",
                            timing_runs=num_runs,
                            warmup_runs=warmup_runs,
                            **make_row_meta(sys_name, mode_name, backend, n, bs, total),
                        )
                    )
                    continue
                try:
                    actual_n = data["atoms_per_system"]
                    actual_total = data.get("total_atoms", actual_n)
                    print(f"\n  {actual_n} atoms x {cfg['batch_size']} batch")
                    row_meta = make_row_meta(
                        sys_name,
                        mode_name,
                        backend,
                        actual_n,
                        data.get("batch_size", 1),
                        actual_total,
                    )
                    results.extend(
                        _fourier_rows(
                            data, config, num_runs, warmup_runs, row_meta, backend
                        )
                    )
                finally:
                    del data
                    clean_gpu()
            if results:
                save_results(
                    results,
                    output_dir / make_csv_name("fd3", sys_name, mode_name),
                    replace_backend=backend,
                )
                all_results.extend(results)

    print(f"\nCOMPLETE: {len(all_results)} results in {output_dir}")
    return all_results


def dry_run_from_config(config: dict, backend: str | None = None) -> list[dict]:
    """Expand the case matrix without allocating or timing anything."""
    backend = _resolve_backend(config, backend)
    cutoffs = config.get("parameters", {}).get("real_space_cutoffs", [6.0, 15.0, 20.0])
    methods = ["fourier_dftd3", "fourier_dftd3_setup"] + [
        f"dftd3_cutoff_{c:g}" for c in cutoffs
    ]
    rows = []
    for sys_name, sys_config in config["systems"].items():
        if not sys_config.get("enabled", True):
            continue
        nh3_dir = resolve_nh3_dir(sys_config)
        for mode_name, mode_config in config["scaling"].items():
            if not isinstance(mode_config, dict) or not mode_config.get(
                "enabled", True
            ):
                continue
            for cfg in configs_for_mode(
                mode_name, mode_config, sys_name, sys_config, nh3_dir, plan_only=True
            ):
                n, bs, total = planned_atom_counts(sys_name, cfg)
                rows.extend(
                    {
                        "method": method,
                        "system": sys_name,
                        "mode": mode_name,
                        "atoms_per_system": n,
                        "batch_size": bs,
                        "total_atoms": total,
                        "backend": backend,
                    }
                    for method in methods
                )
    return rows


# Flags that steer the run itself rather than overriding a config value.
_STRUCTURAL_FLAGS = frozenset({"config", "output_dir", "dry_run", "help"})


def build_parser():
    """Command-line interface, separate from parsing so it can be inspected."""
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
    # Sweep sizes come from ``systems.<name>.atom_counts`` in the YAML, as in the DFT-D3
    # benchmark. A single flat CLI list cannot serve both systems: CsCl is only valid at
    # 2*k^3, so one list would be silently rounded for one system or the other.
    parser.add_argument("--timing-runs", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args():
    """Parse the command line."""
    return build_parser().parse_args()


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply the CLI flags that were actually given on top of the YAML config."""
    parameters = config.setdefault("parameters", {})
    for key in ("timing_runs", "warmup_runs"):
        value = getattr(args, key, None)
        if value is not None:
            parameters[key] = value
    return config


def main():
    """Run the sweep and print a summary table."""
    args = parse_args()
    config = merge_cli_overrides(load_yaml_config(args.config), args)
    if args.dry_run:
        for case in dry_run_from_config(config):
            print(case)
        return 0

    results = run_from_config(config, args.output_dir)
    density = next(
        (row["density"] for row in results if row.get("success", True)), float("nan")
    )
    print(
        f"\nCsCl supercells at {density:.4f} atoms/A^3;  times in ms. 'eval' is the "
        f"reported metric: "
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
