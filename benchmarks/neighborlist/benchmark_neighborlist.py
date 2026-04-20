#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neighbor List Benchmark.

Benchmarks naive O(N²) and cell-list O(N) neighbor list construction
across two chemical systems (CsCl, NH3) and three scaling modes.
Configuration is loaded from a per-module YAML file.

Usage (run from the repository root):
    python -m benchmarks.neighborlist.benchmark_neighborlist \
        --config benchmarks/neighborlist/benchmark_config.yaml
    python -m benchmarks.neighborlist.benchmark_neighborlist \
        --config benchmarks/neighborlist/benchmark_config.yaml \
        --system cscl --mode system_size
    python -m benchmarks.neighborlist.benchmark_neighborlist \
        --config benchmarks/neighborlist/benchmark_config.yaml \
        --output-dir docs/benchmarks/benchmark_results

    # JAX backend
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python -m benchmarks.neighborlist.benchmark_neighborlist \
        --config benchmarks/neighborlist/benchmark_config.yaml --backend jax

Backends
--------
``--backend torch`` (default) uses the warp-based torch kernels and CUDA
events for timing. ``--backend jax`` uses the JAX wrappers in
``nvalchemiops.jax.neighbors`` and wall-clock timing with
``jax.block_until_ready``.

Environment variables for ``--backend jax``:

- ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` — required, or XLA grabs all VRAM.
- ``JAX_ENABLE_X64=True`` — optional (electrostatics is the only benchmark
  that hard-requires this).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

__all__ = [
    "benchmark_nl",
    "main",
    "merge_cli_overrides",
    "parse_args",
    "run_from_config",
]

from benchmarks.config import (
    add_common_cli_args,
    load_yaml_config,
    merge_common_cli_overrides,
)
from benchmarks.constants import DEFAULT_ATOMIC_DENSITY, DEFAULT_NL_SAFETY_FACTOR
from benchmarks.systems import (
    configs_for_mode,
    create_system,
    resolve_nh3_dir,
)
from benchmarks.utils import (
    build_result,
    clean_gpu,
    create_run_directory,
    cuda_timed_runs,
    current_alloc_gb,
    ensure_jax_available,
    format_num,
    lazy_import_jax,
    make_csv_name,
    make_row_meta,
    measure_memory_jax,
    measure_memory_torch,
    save_results,
)

# Official nvalchemiops public APIs used by the neighbor-list runner.
from nvalchemiops.neighbors import estimate_max_neighbors
from nvalchemiops.torch.neighbors import (
    batch_cell_list,
    batch_naive_neighbor_list,
)


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides on top of YAML config.

    Common flags are merged by :func:`merge_common_cli_overrides`; this
    wrapper adds the NL-specific ``--cutoffs`` and ``--methods`` handling.
    """
    config = merge_common_cli_overrides(config, args)
    if args.cutoffs is not None:
        config["parameters"]["cutoffs"] = args.cutoffs
    if args.methods is not None:
        for method in config["methods"]:
            method["enabled"] = method["name"] in args.methods
    return config


# =============================================================================
# Core Benchmark Function
# =============================================================================


def benchmark_nl(
    data: dict,
    cutoff: float,
    method: str,
    num_runs: int,
    warmup_runs: int = 3,
    backend: str = "torch",
) -> dict:
    """Benchmark a single NL configuration.

    Optimized: no redundant clean_gpu or extra kernel calls.
    Memory is measured from a single warmup run. Neighbor count
    captured from warmup result. clean_gpu() is the caller's
    responsibility (once per atom-size group, not per config).

    Parameters
    ----------
    data : dict
        System data from create_system() (backend-specific arrays).
    cutoff : float
        Cutoff distance in Angstroms.
    method : str
        'naive' or 'cell'.
    num_runs : int
        Number of timing iterations.
    warmup_runs : int
        Number of warmup iterations.
    backend : str, default='torch'
        ``'torch'`` or ``'jax'``.

    Returns
    -------
    dict
        Timing and memory results with NL-specific extras.
    """
    if backend == "jax":
        return _benchmark_nl_jax(data, cutoff, method, num_runs, warmup_runs)

    positions = data["positions"]
    cell = data["cell"]
    pbc = data["pbc"]
    batch_idx = data["batch_idx"]
    total_atoms = data.get("total_atoms", data["atoms_per_system"])

    maxnb = estimate_max_neighbors(
        cutoff,
        atomic_density=DEFAULT_ATOMIC_DENSITY,
        safety_factor=DEFAULT_NL_SAFETY_FACTOR,
    )
    nl_func = batch_cell_list if method == "cell" else batch_naive_neighbor_list

    def run_nl():
        return nl_func(
            positions=positions,
            cell=cell,
            pbc=pbc,
            cutoff=cutoff,
            batch_idx=batch_idx,
            max_neighbors=maxnb,
        )

    # Single warmup run: captures neighbor count + peak memory
    result, mem_info = measure_memory_torch(run_nl)
    n_neighbors = int(result[0].shape[1]) if hasattr(result[0], "shape") else 0

    # Timing (warmup inside cuda_timed_runs handles GPU pipeline warmup)
    time_sec = cuda_timed_runs(run_nl, num_runs, warmup_runs=warmup_runs)

    return {
        "time_seconds": time_sec,
        "mem_info": mem_info,
        "max_neighbors": n_neighbors,
        "total_neighbor_pairs": total_atoms * n_neighbors,
    }


def _benchmark_nl_jax(data, cutoff, method, num_runs, warmup_runs):
    """JAX backend implementation of :func:`benchmark_nl`.

    Uses wall-clock timing (CUDA events cannot observe JAX work) and
    NVML-based memory deltas since ``torch.cuda.max_memory_allocated``
    cannot see XLA allocations. ``mem_peak_gb`` is set to 0 — the XLA pool
    makes peak memory meaningless; the plotter filters zero rows.
    """
    jax_api = lazy_import_jax()
    jax = jax_api["jax"]
    jnp = jax_api["jnp"]
    jax_nl = jax_api["neighbor_list"]
    estimate_bcl_sizes = jax_api["estimate_batch_cell_list_sizes"]
    compute_naive_shifts = jax_api["compute_naive_num_shifts"]

    positions = data["positions"]
    cell = data["cell"]
    pbc = data["pbc"]
    batch_idx = data["batch_idx"]
    atoms_per_system = int(data["atoms_per_system"])
    batch_size = int(data.get("batch_size", 1))
    total_atoms = data.get("total_atoms", atoms_per_system)

    # Under jax.jit the NL wrapper cannot infer num_systems from
    # batch_idx.max() or allocate shape-dependent buffers from traced
    # cell geometry. Compute batch_ptr and the cell-list / naive sizing
    # once outside the timed closure; the library's own helpers
    # (estimate_batch_cell_list_sizes, compute_naive_num_shifts) are
    # designed to be called here.
    batch_ptr = jnp.arange(batch_size + 1, dtype=jnp.int32) * atoms_per_system

    # We always pass a batch_idx, so always use the batched variants. The
    # wrapper's auto-prefix rule only fires when ``method=None``.
    jax_method = "batch_cell_list" if method == "cell" else "batch_naive"

    maxnb = estimate_max_neighbors(
        cutoff,
        atomic_density=DEFAULT_ATOMIC_DENSITY,
        safety_factor=DEFAULT_NL_SAFETY_FACTOR,
    )

    # Pre-compute sizing outside jit so the closure can be traced.
    if jax_method == "batch_cell_list":
        max_total_cells, _, _ = estimate_bcl_sizes(
            positions=positions,
            batch_ptr=batch_ptr,
            cell=cell,
            cutoff=float(cutoff),
            pbc=pbc,
        )
        nl_kwargs = dict(max_total_cells=int(max_total_cells))
    else:
        shift_range, num_shifts, max_shifts = compute_naive_shifts(
            cell=cell, pbc=pbc, cutoff=float(cutoff)
        )
        nl_kwargs = dict(
            shift_range_per_dimension=shift_range,
            num_shifts_per_system=num_shifts,
            max_shifts_per_system=int(max_shifts),
            max_atoms_per_system=atoms_per_system,
        )

    def run_nl():
        return jax_nl(
            positions=positions,
            cutoff=float(cutoff),
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method=jax_method,
            return_neighbor_list=False,
            max_neighbors=int(maxnb),
            **nl_kwargs,
        )

    # jit so the timed loop reflects steady-state per-call cost (the
    # number an MD/MC user would observe), not Python-side tracing.
    run_nl_jit = jax.jit(run_nl)

    # Memory via NVML (XLA pool makes torch-side measurement useless)
    result, mem_info = measure_memory_jax(run_nl_jit, jax)
    n_neighbors = int(result[0].shape[1]) if hasattr(result[0], "shape") else 0

    time_sec = cuda_timed_runs(
        run_nl_jit, num_runs, warmup_runs=warmup_runs, backend="jax"
    )

    return {
        "time_seconds": time_sec,
        "mem_info": mem_info,
        "max_neighbors": n_neighbors,
        "total_neighbor_pairs": total_atoms * n_neighbors,
    }


# =============================================================================
# Config-Driven Runner
# =============================================================================


def _nl_run_one_method(
    data, cutoff, method, num_runs, warmup_runs, backend, row_meta
):
    """Run :func:`benchmark_nl` for one ``(cutoff, method)`` and build a result row.

    Catches OOM (prints, clean_gpu, returns None) and other exceptions
    (prints, returns None). Keeps the inner loop in :func:`run_from_config`
    free of try/except nesting.
    """
    try:
        r = benchmark_nl(
            data, cutoff, method, num_runs, warmup_runs, backend=backend
        )
        result = build_result(
            method=method,
            time_seconds=r["time_seconds"],
            mem_info=r["mem_info"],
            cutoff=cutoff,
            **row_meta,
        )
        throughput_matoms = result["throughput_atoms_per_sec"] / 1e6
        print(
            f"    {cutoff}Å {method:5s}: "
            f"{result['time_us_per_atom']:.3f} μs/atom | "
            f"{throughput_matoms:.1f} Matom/s | "
            f"{result['mem_delta_mb']:.1f} MB"
        )
        return result
    except torch.cuda.OutOfMemoryError:
        print(f"    {cutoff}Å {method:5s}: OOM")
        clean_gpu()
        return None
    except Exception as e:
        print(f"    {cutoff}Å {method:5s}: FAILED - {e}")
        return None


def run_from_config(
    config: dict,
    output_dir: Path | str | None = None,
    backend: str | None = None,
) -> list[dict]:
    """Run NL benchmarks driven entirely by YAML config.

    This is the main entry point, used both standalone and from benchmark_suite.py.

    Parameters
    ----------
    config : dict
        Merged config (YAML + CLI overrides).
    output_dir : Path, optional
        Override output directory. If None, uses config['output']['base_dir'].
    backend : str, optional
        ``'torch'`` or ``'jax'``. If None, pulled from
        ``config['runtime']['backend']`` (merged in by ``merge_cli_overrides``),
        defaulting to ``'torch'``.

    Returns
    -------
    list[dict]
        All benchmark results.
    """
    params = config["parameters"]
    num_runs = params["timing_runs"]
    warmup_runs = params["warmup_runs"]
    cutoffs = params["cutoffs"]
    cutoff_limits = params.get("cutoff_limits", {})

    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")

    # Resolve output directory
    if output_dir is None:
        output_dir = create_run_directory(config["output"]["base_dir"], prefix="nl")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect enabled methods
    methods = [m["name"] for m in config.get("methods", []) if m.get("enabled", True)]

    # Eagerly validate JAX availability so the error surfaces now, not
    # partway through the benchmark loop below.
    if backend == "jax":
        lazy_import_jax()

    try:
        gpu_name = torch.cuda.get_device_name(0)
    except (AssertionError, RuntimeError):
        gpu_name = "N/A (no CUDA)"
    print("NL Benchmark Suite")
    print(f"GPU: {gpu_name}")
    print(f"Backend: {backend}")
    print(f"Cutoffs: {cutoffs} Å | Methods: {methods}")
    print(f"Timing: {num_runs} runs")
    print(f"Output: {output_dir}")

    all_results = []

    # Iterate: systems × scaling modes
    for sys_name, sys_config in config["systems"].items():
        if not sys_config.get("enabled", True):
            continue

        nh3_dir = resolve_nh3_dir(sys_config)

        for mode_name, mode_config in config["scaling"].items():
            if not isinstance(mode_config, dict) or not mode_config.get(
                "enabled", True
            ):
                continue

            print(f"\n{'=' * 70}")
            print(f"NL: {sys_name.upper()} / {mode_name}")
            print(f"{'=' * 70}")

            configs = configs_for_mode(
                mode_name, mode_config, sys_name, sys_config, nh3_dir
            )
            if not configs:
                continue

            results = []

            for cfg in configs:
                n, bs = cfg["num_atoms"], cfg["batch_size"]

                clean_gpu()

                try:
                    data = create_system(
                        sys_name,
                        num_atoms=n,
                        pdb_path=cfg.get("pdb_path"),
                        batch_size=bs,
                        backend=backend,
                    )
                except (FileNotFoundError, RuntimeError, ValueError) as e:
                    print(f"    SKIP: {e}")
                    continue

                actual_total = data.get("total_atoms", data["atoms_per_system"])
                actual_n = data["atoms_per_system"]
                print(
                    f"\n  {format_num(actual_n)} atoms × {bs} batch = {format_num(actual_total)} total"
                )
                print(f"  [GPU: {current_alloc_gb(backend):.1f} GB allocated]")
                row_meta = make_row_meta(
                    sys_name,
                    mode_name,
                    backend,
                    actual_n,
                    data.get("batch_size", 1),
                    actual_total,
                )

                for cutoff in cutoffs:
                    limit = cutoff_limits.get(cutoff) or cutoff_limits.get(str(cutoff))
                    if limit and actual_total > limit:
                        print(f"    {cutoff}Å: SKIP (>{format_num(limit)} limit)")
                        continue

                    # Note: cell_size < 2*cutoff violates minimum image convention
                    # but we still benchmark for completeness. Document in sphinx docs.
                    if data["cell_size"] < 2 * cutoff:
                        print(
                            f"    {cutoff}Å: WARNING cell {data['cell_size']:.1f}Å < 2×cutoff (benchmarking anyway)"
                        )

                    for method in methods:
                        result = _nl_run_one_method(
                            data,
                            cutoff,
                            method,
                            num_runs,
                            warmup_runs,
                            backend,
                            row_meta,
                        )
                        if result is not None:
                            results.append(result)

                del data

            # Save per-(system, mode) CSV with standardized name
            if results:
                csv_name = make_csv_name("nl", sys_name, mode_name)
                save_results(results, output_dir / csv_name)
                all_results.extend(results)

    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {len(all_results)} results saved to {output_dir}")
    print(f"{'=' * 70}")

    return all_results


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command-line arguments for neighbor list benchmarks."""
    parser = argparse.ArgumentParser(
        description="Neighbor List Benchmark (2 systems × 3 modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from the repository root):
    python -m benchmarks.neighborlist.benchmark_neighborlist \\
        --config benchmarks/neighborlist/benchmark_config.yaml
    python -m benchmarks.neighborlist.benchmark_neighborlist \\
        --config benchmarks/neighborlist/benchmark_config.yaml \\
        --system cscl --mode system_size
    python -m benchmarks.neighborlist.benchmark_neighborlist \\
        --config benchmarks/neighborlist/benchmark_config.yaml \\
        --cutoffs 6 15 --methods cell
    python -m benchmarks.neighborlist.benchmark_neighborlist \\
        --config benchmarks/neighborlist/benchmark_config.yaml \\
        --output-dir docs/benchmarks/benchmark_results
        """,
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to benchmark_config.yaml"
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--cutoffs",
        "-c",
        type=float,
        nargs="+",
        default=None,
        help="Override cutoff radii in Angstroms",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None, help="Override NL methods (naive, cell)"
    )
    return parser.parse_args()


def main():
    """Run neighbor list benchmarks."""
    args = parse_args()

    # Load YAML config, merge CLI overrides
    config = load_yaml_config(args.config)
    config = merge_cli_overrides(config, args)

    # Resolve backend: explicit CLI arg wins; otherwise honor config; else torch.
    backend = args.backend or config.get("runtime", {}).get("backend", "torch")

    if backend == "jax":
        ensure_jax_available()

    run_from_config(config, output_dir=args.output_dir, backend=backend)


if __name__ == "__main__":
    main()
