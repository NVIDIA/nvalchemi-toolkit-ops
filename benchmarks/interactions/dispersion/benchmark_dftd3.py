#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFT-D3 Dispersion Benchmark.

CRITICAL: D3 operates in atomic units (Bohr). All positions, cells, and
cutoffs are converted from Angstroms to Bohr before calling the D3 API.
Times NL and D3 separately.

Usage (run from the repository root):
    python -m benchmarks.interactions.dispersion.benchmark_dftd3 \
        --config benchmarks/interactions/dispersion/benchmark_config.yaml
    python -m benchmarks.interactions.dispersion.benchmark_dftd3 \
        --config benchmarks/interactions/dispersion/benchmark_config.yaml \
        --output-dir docs/benchmarks/benchmark_results

    # JAX backend
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python -m benchmarks.interactions.dispersion.benchmark_dftd3 \
        --config benchmarks/interactions/dispersion/benchmark_config.yaml --backend jax

Backends
--------
``--backend torch`` (default) uses the warp-based torch kernels and CUDA
events for timing. ``--backend jax`` uses the JAX wrappers in
``nvalchemiops.jax.interactions.dispersion`` and
``nvalchemiops.jax.neighbors``. D3 reference parameters are loaded via
``torch.load`` and converted to ``jax.numpy.asarray`` for the JAX backend.

Environment variables for ``--backend jax``:

- ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` — required, or XLA grabs all VRAM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

__all__ = [
    "benchmark_d3",
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
from benchmarks.constants import (
    ANGSTROM_TO_BOHR,
    DEFAULT_ATOMIC_DENSITY,
    DEFAULT_NL_SAFETY_FACTOR,
)
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
from nvalchemiops.neighbors import estimate_max_neighbors
from nvalchemiops.torch.interactions.dispersion import dftd3
from nvalchemiops.torch.neighbors import batch_cell_list


def _torch_d3_params_to_jax(torch_params, jnp):
    """Convert a dict of torch tensors to jax arrays (for D3Parameters)."""
    out = {}
    for k, v in torch_params.items():
        out[k] = jnp.asarray(v.detach().cpu().numpy())
    return out


# =============================================================================
# Config Loading
# =============================================================================


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides on top of YAML config. Adds D3-specific
    ``--cutoffs`` on top of the shared flags."""
    config = merge_common_cli_overrides(config, args)
    if args.cutoffs is not None:
        config["parameters"]["cutoffs"] = args.cutoffs
    return config


# =============================================================================
# Core Benchmark
# =============================================================================


def benchmark_d3(
    data: dict,
    cutoff: float,
    d3_params,
    d3_func_params: dict,
    num_runs: int,
    warmup_runs: int = 3,
    backend: str = "torch",
) -> dict:
    """Benchmark D3 for a single configuration. Times NL and D3 separately.

    Parameters
    ----------
    backend : str, default='torch'
        ``'torch'`` or ``'jax'``. For ``'jax'``, ``d3_params`` must be a dict
        of jax arrays with keys ``rcov``, ``r4r2``, ``c6ab``, ``cn_ref``.
    """
    if backend == "jax":
        return _benchmark_d3_jax(
            data, cutoff, d3_params, d3_func_params, num_runs, warmup_runs
        )

    positions = data["positions"]
    cell = data["cell"]
    pbc = data["pbc"]
    batch_idx = data["batch_idx"]
    numbers = data["atomic_numbers"]

    # Convert to Bohr
    pos_bohr = positions * ANGSTROM_TO_BOHR
    cell_bohr = cell * ANGSTROM_TO_BOHR
    cutoff_bohr = cutoff * ANGSTROM_TO_BOHR

    maxnb = estimate_max_neighbors(
        cutoff,
        atomic_density=DEFAULT_ATOMIC_DENSITY,
        safety_factor=DEFAULT_NL_SAFETY_FACTOR,
    )

    # YAML is authoritative for D3 damping parameters.
    a1 = d3_func_params["a1"]
    a2 = d3_func_params["a2"]
    s8 = d3_func_params["s8"]

    # Single warmup: build NL + run D3, capture memory
    def warmup_d3():
        nb, _, nb_shifts = batch_cell_list(
            positions=pos_bohr,
            cell=cell_bohr,
            pbc=pbc,
            cutoff=cutoff_bohr,
            batch_idx=batch_idx,
            max_neighbors=maxnb,
        )
        dftd3(
            positions=pos_bohr,
            cell=cell_bohr,
            numbers=numbers,
            batch_idx=batch_idx,
            neighbor_matrix=nb,
            neighbor_matrix_shifts=nb_shifts,
            d3_params=d3_params,
            a1=a1,
            a2=a2,
            s8=s8,
        )
        return nb, nb_shifts

    (nbmat, nbmat_shifts), mem_info = measure_memory_torch(warmup_d3)

    # NL timing
    def run_nl():
        nonlocal nbmat, nbmat_shifts
        nbmat, _, nbmat_shifts = batch_cell_list(
            positions=pos_bohr,
            cell=cell_bohr,
            pbc=pbc,
            cutoff=cutoff_bohr,
            batch_idx=batch_idx,
            max_neighbors=maxnb,
        )

    time_nl = cuda_timed_runs(run_nl, num_runs, warmup_runs=warmup_runs)

    # D3 timing (uses pre-computed NL from last run_nl call)
    def run_d3():
        dftd3(
            positions=pos_bohr,
            cell=cell_bohr,
            numbers=numbers,
            batch_idx=batch_idx,
            neighbor_matrix=nbmat,
            neighbor_matrix_shifts=nbmat_shifts,
            d3_params=d3_params,
            a1=a1,
            a2=a2,
            s8=s8,
        )

    time_d3 = cuda_timed_runs(run_d3, num_runs, warmup_runs=warmup_runs)

    return {
        "time_nl_seconds": time_nl,
        "time_d3_seconds": time_d3,
        "time_total_seconds": time_nl + time_d3,
        "mem_info": mem_info,
    }


def _benchmark_d3_jax(data, cutoff, d3_params, d3_func_params, num_runs, warmup_runs):
    """JAX backend implementation of :func:`benchmark_d3`.

    Uses wall-clock timing and NVML memory deltas. ``mem_peak_gb=0`` because
    the XLA allocator pool makes peak memory meaningless.
    """
    jax_api = lazy_import_jax(need_dispersion=True)
    jax = jax_api["jax"]
    jnp = jax_api["jnp"]
    jax_dftd3 = jax_api["dftd3"]
    jax_nl = jax_api["neighbor_list"]
    estimate_bcl_sizes = jax_api["estimate_batch_cell_list_sizes"]

    positions = data["positions"]
    cell = data["cell"]
    pbc = data["pbc"]
    batch_idx = data["batch_idx"]
    numbers = data["atomic_numbers"]
    atoms_per_system = int(data["atoms_per_system"])
    batch_size = int(data.get("batch_size", 1))

    # Under jax.jit, NL needs explicit max_total_cells (cell list sizing
    # reads traced cell geometry) and D3 needs num_systems / batch_ptr
    # (can't infer from ``batch_idx.max()`` inside a trace). All computed
    # once here so the timed closures can be traced cleanly.
    batch_ptr = jnp.arange(batch_size + 1, dtype=jnp.int32) * atoms_per_system

    # Convert to Bohr
    pos_bohr = positions * ANGSTROM_TO_BOHR
    cell_bohr = cell * ANGSTROM_TO_BOHR
    cutoff_bohr = cutoff * ANGSTROM_TO_BOHR

    maxnb = estimate_max_neighbors(
        cutoff,
        atomic_density=DEFAULT_ATOMIC_DENSITY,
        safety_factor=DEFAULT_NL_SAFETY_FACTOR,
    )

    max_total_cells, _, _ = estimate_bcl_sizes(
        positions=pos_bohr,
        batch_ptr=batch_ptr,
        cell=cell_bohr,
        cutoff=float(cutoff_bohr),
        pbc=pbc,
    )
    max_total_cells = int(max_total_cells)

    # YAML is authoritative for D3 damping parameters.
    a1 = d3_func_params["a1"]
    a2 = d3_func_params["a2"]
    s8 = d3_func_params["s8"]

    # Warmup: build NL + run D3, capture memory via NVML. Return the NL
    # buffers so the per-kernel timing runs below can reuse them.
    def warmup_d3_jax():
        nb, _, nb_shifts = jax_nl(
            positions=pos_bohr,
            cutoff=float(cutoff_bohr),
            cell=cell_bohr,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_cell_list",
            return_neighbor_list=False,
            max_neighbors=int(maxnb),
            max_total_cells=max_total_cells,
        )
        jax.block_until_ready(nb)
        out = jax_dftd3(
            positions=pos_bohr,
            numbers=numbers,
            a1=float(a1),
            a2=float(a2),
            s8=float(s8),
            d3_params=d3_params,
            batch_idx=batch_idx,
            num_systems=batch_size,
            cell=cell_bohr,
            neighbor_matrix=nb,
            neighbor_matrix_shifts=nb_shifts,
        )
        return nb, nb_shifts, out

    (nbmat, nbmat_shifts, _), mem_info = measure_memory_jax(warmup_d3_jax, jax)

    # NL timing — each call returns fresh buffers
    def run_nl():
        return jax_nl(
            positions=pos_bohr,
            cutoff=float(cutoff_bohr),
            cell=cell_bohr,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method="batch_cell_list",
            return_neighbor_list=False,
            max_neighbors=int(maxnb),
            max_total_cells=max_total_cells,
        )

    # jit so the timed loop reflects steady-state per-call cost, not
    # Python-side tracing on every iteration.
    run_nl_jit = jax.jit(run_nl)
    time_nl = cuda_timed_runs(
        run_nl_jit, num_runs, warmup_runs=warmup_runs, backend="jax"
    )

    # D3 timing — reuse the NL buffers we already have. Pass d3_params
    # and other large arrays as explicit jit arguments (not closure
    # captures), otherwise XLA const-folds them into the compiled
    # graph and peak memory grows to double-digit GB (the 3.89→13.74GB
    # "constants captured during lowering" warnings that crashed the
    # XLA compiler on prior runs).
    def _run_d3_kernel(pos, numbers_, cell_, bi, nbm, nbm_shifts, d3p):
        return jax_dftd3(
            positions=pos,
            numbers=numbers_,
            a1=float(a1),
            a2=float(a2),
            s8=float(s8),
            d3_params=d3p,
            batch_idx=bi,
            num_systems=batch_size,
            cell=cell_,
            neighbor_matrix=nbm,
            neighbor_matrix_shifts=nbm_shifts,
        )

    _run_d3_kernel_jit = jax.jit(_run_d3_kernel)

    def run_d3_jit():
        return _run_d3_kernel_jit(
            pos_bohr, numbers, cell_bohr, batch_idx, nbmat, nbmat_shifts, d3_params
        )
    time_d3 = cuda_timed_runs(
        run_d3_jit, num_runs, warmup_runs=warmup_runs, backend="jax"
    )

    return {
        "time_nl_seconds": time_nl,
        "time_d3_seconds": time_d3,
        "time_total_seconds": time_nl + time_d3,
        "mem_info": mem_info,
    }


# =============================================================================
# Config-Driven Runner
# =============================================================================


def _d3_run_one_cutoff(
    data,
    cutoff,
    d3_params,
    d3_func_params,
    num_runs,
    warmup_runs,
    backend,
    actual_total,
    row_meta,
):
    """Run :func:`benchmark_d3` for one ``cutoff`` and build a result row.

    Catches OOM and other exceptions; returns None on failure so the caller
    can continue. Keeps the inner loop in :func:`run_from_config` free of
    try/except nesting.
    """
    try:
        r = benchmark_d3(
            data,
            cutoff,
            d3_params,
            d3_func_params,
            num_runs,
            warmup_runs,
            backend=backend,
        )
        time_d3_us_per_atom = (
            (r["time_d3_seconds"] * 1e6) / actual_total if actual_total > 0 else 0.0
        )
        result = build_result(
            method="dftd3",
            time_seconds=r["time_total_seconds"],
            mem_info=r["mem_info"],
            cutoff=cutoff,
            time_d3_us_per_atom=time_d3_us_per_atom,
            **row_meta,
        )
        mem_suffix = (
            f" | {result['mem_delta_mb']:.1f} MB" if backend == "torch" else ""
        )
        print(
            f"    {cutoff}Å: D3={time_d3_us_per_atom:.3f} μs/atom{mem_suffix}"
        )
        return result
    except torch.cuda.OutOfMemoryError:
        print(f"    {cutoff}Å: OOM")
        clean_gpu()
        return None
    except Exception as e:
        print(f"    {cutoff}Å: FAILED - {e}")
        return None


def run_from_config(
    config: dict,
    output_dir: Path | str | None = None,
    backend: str | None = None,
) -> list[dict]:
    """Run D3 benchmarks driven by YAML config.

    Parameters
    ----------
    backend : str, optional
        ``'torch'`` or ``'jax'``. If None, pulled from
        ``config['runtime']['backend']``, defaulting to ``'torch'``.
    """
    params = config["parameters"]
    num_runs = params["timing_runs"]
    warmup_runs = params["warmup_runs"]
    cutoffs = params["cutoffs"]
    d3_func_params = config["dftd3_parameters"]

    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")

    # Load D3 reference parameters (YAML is authoritative for the path)
    d3_params_path = Path(config["params_path"]).expanduser()
    if not d3_params_path.exists():
        print(f"ERROR: D3 parameters not found at {d3_params_path}")
        print("Run: python examples/dispersion/01_dftd3_molecule.py (downloads params)")
        return []
    d3_params_torch = torch.load(d3_params_path, map_location="cuda", weights_only=True)
    print(f"Loaded D3 parameters from {d3_params_path}")

    if backend == "jax":
        jax_api = lazy_import_jax(need_dispersion=True)  # fail fast if jax missing
        d3_params = _torch_d3_params_to_jax(d3_params_torch, jax_api["jnp"])
    else:
        d3_params = d3_params_torch

    if output_dir is None:
        output_dir = create_run_directory(config["output"]["base_dir"], prefix="d3")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        gpu_name = torch.cuda.get_device_name(0)
    except (AssertionError, RuntimeError):
        gpu_name = "N/A (no CUDA)"
    print(f"D3 Benchmark Suite | GPU: {gpu_name}")
    print(f"Backend: {backend}")
    print(f"Cutoffs: {cutoffs} Å | Timing: {num_runs} runs")
    print(f"Output: {output_dir}")

    all_results = []

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
            print(f"D3: {sys_name.upper()} / {mode_name}")
            print(f"{'=' * 70}")

            configs = configs_for_mode(
                mode_name, mode_config, sys_name, sys_config, nh3_dir
            )
            if not configs:
                continue

            results = []

            for cfg in configs:
                n, bs = cfg["num_atoms"], cfg["batch_size"]

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

                try:
                    actual_total = data.get("total_atoms", data["atoms_per_system"])
                    actual_n = data["atoms_per_system"]
                    print(
                        f"\n  {format_num(actual_n)} atoms × {bs} batch = {format_num(actual_total)} total"
                    )
                    print(f"    [GPU: {current_alloc_gb(backend):.1f} GB allocated]")
                    row_meta = make_row_meta(
                        sys_name,
                        mode_name,
                        backend,
                        actual_n,
                        data.get("batch_size", 1),
                        actual_total,
                    )

                    for cutoff in cutoffs:
                        if data["cell_size"] < 2 * cutoff:
                            print(
                                f"    {cutoff}Å: WARNING cell {data['cell_size']:.1f}Å < 2×cutoff (benchmarking anyway)"
                            )

                        result = _d3_run_one_cutoff(
                            data,
                            cutoff,
                            d3_params,
                            d3_func_params,
                            num_runs,
                            warmup_runs,
                            backend,
                            actual_total,
                            row_meta,
                        )
                        if result is not None:
                            results.append(result)
                finally:
                    # Free GPU tensors so gc.collect() in clean_gpu() can reclaim memory
                    del data
                    clean_gpu()

            if results:
                csv_name = make_csv_name("d3", sys_name, mode_name)
                save_results(results, output_dir / csv_name)
                all_results.extend(results)

    print(f"\nCOMPLETE: {len(all_results)} results in {output_dir}")
    return all_results


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command-line arguments for DFT-D3 benchmarks."""
    parser = argparse.ArgumentParser(
        description="DFT-D3 Benchmark (2 systems × 3 modes)"
    )
    parser.add_argument("--config", type=Path, required=True)
    add_common_cli_args(parser)
    parser.add_argument(
        "--cutoffs",
        "-c",
        type=float,
        nargs="+",
        default=None,
        help="Override cutoff radii in Angstroms",
    )
    return parser.parse_args()


def main():
    """Run DFT-D3 dispersion benchmarks."""
    args = parse_args()
    config = load_yaml_config(args.config)
    config = merge_cli_overrides(config, args)

    backend = args.backend or config.get("runtime", {}).get("backend", "torch")
    if backend == "jax":
        ensure_jax_available(need_dispersion=True)

    run_from_config(config, output_dir=args.output_dir, backend=backend)


if __name__ == "__main__":
    main()
