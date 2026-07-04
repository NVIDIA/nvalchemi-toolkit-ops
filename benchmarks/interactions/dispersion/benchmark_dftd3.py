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

"""DFT-D3 Dispersion Benchmark.

CRITICAL: D3 operates in atomic units (Bohr). All positions, cells, and
cutoffs are converted from Angstroms to Bohr before calling the D3 API.
Neighbor-list setup is built outside the timed D3 region.

Usage (run from the repository root):
    python -m benchmarks.interactions.dispersion.benchmark_dftd3 \
        --config benchmarks/interactions/dispersion/benchmark_config.yaml
    python -m benchmarks.interactions.dispersion.benchmark_dftd3 \
        --config benchmarks/interactions/dispersion/benchmark_config.yaml \
        --output-dir docs/benchmarks/benchmark_results

    # JAX backend (the runner sets JAX env defaults before importing JAX)
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
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

- ``XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`` — set by the runner before
  importing JAX unless the user already configured JAX memory behavior. JAX's
  normal preallocator avoids the fragmentation seen with on-demand allocation
  in large benchmark sweeps.
- ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` — optional user override. If unset,
  the runner prints a one-line note before JAX import and keeps the default
  preallocator capped by ``XLA_PYTHON_CLIENT_MEM_FRACTION``.
"""

from __future__ import annotations

import argparse
import os
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
    enabled_method_names,
    load_yaml_config,
    merge_common_cli_overrides,
)
from benchmarks.constants import (
    ANGSTROM_TO_BOHR,
    DEFAULT_ATOMIC_DENSITY,
    DEFAULT_NL_SAFETY_FACTOR,
)
from benchmarks.suite_systems import (
    configs_for_mode,
    configured_nh3_artifacts,
    create_system,
    filter_configs_by_total_atoms,
    planned_atom_counts,
    resolve_nh3_dir,
)
from benchmarks.suite_utils import (
    build_failure_result,
    build_result,
    build_skipped_result,
    clean_gpu,
    clean_jax,
    configure_input_provenance,
    create_run_directory,
    cuda_timed_runs,
    current_alloc_gb,
    ensure_jax_available,
    failure_error_type,
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
from nvalchemiops.torch.neighbors import batch_cell_list, cell_list


def _torch_d3_params_to_jax(torch_params, jnp):
    """Convert a dict of torch tensors to jax arrays (for D3Parameters)."""
    out = {}
    for k, v in torch_params.items():
        out[k] = jnp.asarray(v.detach().cpu().numpy())
    return out


def _torch_d3_params_to_device(torch_params, device: str):
    """Move torch D3 parameter tensors to the selected backend device."""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in torch_params.items()
    }


def _ensure_d3_parameter_file(params_path: Path) -> None:
    """Create the configured D3 parameter cache file when it is absent."""
    if params_path.exists():
        return
    print(f"D3 parameters not found at {params_path}; generating from reference data")
    try:
        from examples.dispersion.utils import extract_dftd3_parameters

        params = extract_dftd3_parameters()
        params_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(params, params_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"D3 parameters not found at {params_path}, and automatic "
            f"generation failed: {exc}"
        ) from exc


def _resolve_d3_params_path(params_path: str | Path) -> Path:
    """Resolve the D3 parameter cache path with XDG cache override support."""
    raw_path = str(params_path)
    cache_prefix = "~/.cache/"
    if raw_path.startswith(cache_prefix) and (
        xdg_cache_home := os.environ.get("XDG_CACHE_HOME")
    ):
        return Path(xdg_cache_home).expanduser() / raw_path[len(cache_prefix) :]
    return Path(raw_path).expanduser()


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
    """Benchmark D3 for a single configuration.

    Neighbor-list setup is performed outside the timed and memory-measured
    D3 closure so the canonical CSV timing measures DFT-D3 only.

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
    batch_size = int(data.get("batch_size", 1))

    # Convert to Bohr
    pos_bohr = positions * ANGSTROM_TO_BOHR
    cell_bohr = cell * ANGSTROM_TO_BOHR
    cutoff_bohr = cutoff * ANGSTROM_TO_BOHR

    maxnb = estimate_max_neighbors(
        cutoff,
        atomic_density=DEFAULT_ATOMIC_DENSITY * DEFAULT_NL_SAFETY_FACTOR,
    )

    # YAML is authoritative for D3 damping parameters.
    a1 = d3_func_params["a1"]
    a2 = d3_func_params["a2"]
    s8 = d3_func_params["s8"]

    if batch_size == 1:
        nbmat, _, nbmat_shifts = cell_list(
            positions=pos_bohr,
            cell=cell_bohr,
            pbc=pbc,
            cutoff=cutoff_bohr,
            max_neighbors=maxnb,
        )
        d3_batch_idx = None
        neighbor_setup_method = "cell_list"
    else:
        nbmat, _, nbmat_shifts = batch_cell_list(
            positions=pos_bohr,
            cell=cell_bohr,
            pbc=pbc,
            cutoff=cutoff_bohr,
            batch_idx=batch_idx,
            max_neighbors=maxnb,
        )
        d3_batch_idx = batch_idx
        neighbor_setup_method = "batch_cell_list"

    def run_d3():
        dftd3(
            positions=pos_bohr,
            cell=cell_bohr,
            numbers=numbers,
            batch_idx=d3_batch_idx,
            neighbor_matrix=nbmat,
            neighbor_matrix_shifts=nbmat_shifts,
            d3_params=d3_params,
            a1=a1,
            a2=a2,
            s8=s8,
        )

    _, mem_info = measure_memory_torch(run_d3)
    time_d3 = cuda_timed_runs(run_d3, num_runs, warmup_runs=warmup_runs)

    return {
        "time_d3_seconds": time_d3,
        "mem_info": mem_info,
        "neighbor_setup_method": neighbor_setup_method,
    }


def _benchmark_d3_jax(data, cutoff, d3_params, d3_func_params, num_runs, warmup_runs):
    """JAX backend implementation of :func:`benchmark_d3`.

    Uses wall-clock timing. JAX memory fields are reported as NaN because
    the XLA allocator pool makes per-call memory attribution misleading.
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

    # Under jax.jit, NL needs explicit max_total_cells because cell-list
    # sizing reads traced cell geometry.
    batch_ptr = jnp.arange(batch_size + 1, dtype=jnp.int32) * atoms_per_system

    # Convert to Bohr
    pos_bohr = positions * ANGSTROM_TO_BOHR
    cell_bohr = cell * ANGSTROM_TO_BOHR
    cutoff_bohr = cutoff * ANGSTROM_TO_BOHR

    maxnb = estimate_max_neighbors(
        cutoff,
        atomic_density=DEFAULT_ATOMIC_DENSITY * DEFAULT_NL_SAFETY_FACTOR,
    )

    setup_batch_ptr = batch_ptr
    if batch_size == 1:
        setup_batch_ptr = jnp.asarray([0, atoms_per_system], dtype=jnp.int32)
    max_total_cells, _, neighbor_search_radius = estimate_bcl_sizes(
        positions=pos_bohr,
        batch_ptr=setup_batch_ptr,
        cell=cell_bohr,
        cutoff=float(cutoff_bohr),
        pbc=pbc,
    )
    max_total_cells = int(max_total_cells)

    # YAML is authoritative for D3 damping parameters.
    a1 = d3_func_params["a1"]
    a2 = d3_func_params["a2"]
    s8 = d3_func_params["s8"]

    if batch_size == 1:
        cell_neighbor_search_radius = (
            neighbor_search_radius[0]
            if getattr(neighbor_search_radius, "ndim", 1) == 2
            else neighbor_search_radius
        )

        def build_nl(pos, cell_, pbc_):
            return jax_nl(
                positions=pos,
                cutoff=float(cutoff_bohr),
                cell=cell_,
                pbc=pbc_,
                method="cell_list",
                return_neighbor_list=False,
                max_neighbors=int(maxnb),
                max_total_cells=max_total_cells,
                neighbor_search_radius=cell_neighbor_search_radius,
            )

        nbmat, _, nbmat_shifts = jax.jit(build_nl)(pos_bohr, cell_bohr, pbc)
        d3_batch_idx = None
        neighbor_setup_method = "cell_list"
    else:

        def build_nl(pos, cell_, pbc_, batch_idx_, batch_ptr_):
            return jax_nl(
                positions=pos,
                cutoff=float(cutoff_bohr),
                cell=cell_,
                pbc=pbc_,
                batch_idx=batch_idx_,
                batch_ptr=batch_ptr_,
                method="batch_cell_list",
                return_neighbor_list=False,
                max_neighbors=int(maxnb),
                max_total_cells=max_total_cells,
            )

        nbmat, _, nbmat_shifts = jax.jit(build_nl)(
            pos_bohr, cell_bohr, pbc, batch_idx, batch_ptr
        )
        d3_batch_idx = batch_idx
        neighbor_setup_method = "batch_cell_list"
    jax.block_until_ready(
        (pos_bohr, cell_bohr, numbers, nbmat, nbmat_shifts, d3_params)
    )

    # Reuse the NL buffers and time a compiled, fixed-shape D3 callable so
    # compilation remains outside the measured region.
    if d3_batch_idx is None:

        def _run_d3_kernel(pos, numbers_, cell_, nbm, nbm_shifts, d3p):
            return jax_dftd3(
                positions=pos,
                numbers=numbers_,
                a1=float(a1),
                a2=float(a2),
                s8=float(s8),
                d3_params=d3p,
                batch_idx=None,
                num_systems=batch_size,
                cell=cell_,
                neighbor_matrix=nbm,
                neighbor_matrix_shifts=nbm_shifts,
            )

        _run_d3_kernel_jit = jax.jit(_run_d3_kernel)

        def run_d3():
            return _run_d3_kernel_jit(
                pos_bohr, numbers, cell_bohr, nbmat, nbmat_shifts, d3_params
            )

    else:

        def _run_d3_kernel(pos, numbers_, cell_, batch_idx_, nbm, nbm_shifts, d3p):
            return jax_dftd3(
                positions=pos,
                numbers=numbers_,
                a1=float(a1),
                a2=float(a2),
                s8=float(s8),
                d3_params=d3p,
                batch_idx=batch_idx_,
                num_systems=batch_size,
                cell=cell_,
                neighbor_matrix=nbm,
                neighbor_matrix_shifts=nbm_shifts,
            )

        _run_d3_kernel_jit = jax.jit(_run_d3_kernel)

        def run_d3():
            return _run_d3_kernel_jit(
                pos_bohr,
                numbers,
                cell_bohr,
                d3_batch_idx,
                nbmat,
                nbmat_shifts,
                d3_params,
            )

    # JAX memory is unavailable in this suite; avoid an extra allocation-heavy
    # execution that would only produce NaN memory fields.
    _, mem_info = measure_memory_jax(run_d3, jax)
    time_d3 = cuda_timed_runs(run_d3, num_runs, warmup_runs=warmup_runs, backend="jax")

    return {
        "time_d3_seconds": time_d3,
        "mem_info": mem_info,
        "neighbor_setup_method": neighbor_setup_method,
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

    Catches OOM and other exceptions and emits ``success=False`` rows.
    Keeps the inner loop in :func:`run_from_config` free of try/except nesting.
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
            time_seconds=r["time_d3_seconds"],
            mem_info=r["mem_info"],
            cutoff=cutoff,
            timing_runs=num_runs,
            warmup_runs=warmup_runs,
            time_d3_us_per_atom=time_d3_us_per_atom,
            neighbor_setup_method=r["neighbor_setup_method"],
            **row_meta,
        )
        mem_suffix = f" | {result['mem_delta_mb']:.1f} MB" if backend == "torch" else ""
        print(f"    {cutoff}Å: D3={time_d3_us_per_atom:.3f} μs/atom{mem_suffix}")
        return result
    except torch.cuda.OutOfMemoryError as e:
        print(f"    {cutoff}Å: OOM - {e}")
        clean_gpu()
        return build_failure_result(
            method="dftd3",
            cutoff=cutoff,
            error=str(e),
            error_type=failure_error_type(e),
            timing_runs=num_runs,
            warmup_runs=warmup_runs,
            **row_meta,
        )
    except Exception as e:
        error_type = failure_error_type(e)
        if error_type == "OutOfMemoryError":
            clean_gpu()
        print(f"    {cutoff}Å: FAILED - {e}")
        return build_failure_result(
            method="dftd3",
            cutoff=cutoff,
            error=str(e),
            error_type=error_type,
            timing_runs=num_runs,
            warmup_runs=warmup_runs,
            **row_meta,
        )


def dry_run_from_config(config: dict, backend: str | None = None) -> list[dict]:
    """Print and return the expanded D3 benchmark plan without allocation."""
    params = config["parameters"]
    cutoffs = params["cutoffs"]
    max_total_atoms = params.get("max_total_atoms")
    methods = enabled_method_names(config) if "methods" in config else ["dftd3"]
    plan_output = config.get("runtime", {}).get("plan_output", "dry_run")
    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")
    if "dftd3" not in methods:
        if plan_output != "count":
            print("D3 dry-run plan")
            selected = ", ".join(methods) if methods else "(none)"
            print(f"D3 dry-run no enabled methods for selected methods: {selected}")
        print("D3 dry-run rows: 0")
        return []
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
            configs = configs_for_mode(
                mode_name,
                mode_config,
                sys_name,
                sys_config,
                nh3_dir,
                plan_only=True,
            )
            configs, skipped = filter_configs_by_total_atoms(
                configs, sys_name, max_total_atoms
            )
            for cfg, total_atoms in skipped:
                atoms_per_system, batch_size, _ = planned_atom_counts(sys_name, cfg)
                rows.extend(
                    {
                        "benchmark": "d3",
                        "backend": backend,
                        "system": sys_name,
                        "mode": mode_name,
                        "atoms_per_system": atoms_per_system,
                        "batch_size": batch_size,
                        "total_atoms": total_atoms,
                        "method": "dftd3",
                        "cutoff": cutoff,
                        "reason": f">{max_total_atoms} max_total_atoms",
                    }
                    for cutoff in cutoffs
                )
            for cfg in configs:
                atoms_per_system, batch_size, total_atoms = planned_atom_counts(
                    sys_name, cfg
                )
                rows.extend(
                    {
                        "benchmark": "d3",
                        "backend": backend,
                        "system": sys_name,
                        "mode": mode_name,
                        "atoms_per_system": atoms_per_system,
                        "batch_size": batch_size,
                        "total_atoms": total_atoms,
                        "method": "dftd3",
                        "cutoff": cutoff,
                        "reason": "",
                    }
                    for cutoff in cutoffs
                )
    if plan_output != "count":
        print("D3 dry-run plan")
        for row in rows:
            suffix = f" SKIP {row['reason']}" if row["reason"] else ""
            print(
                "  {system}/{mode} backend={backend} method={method} "
                "N={atoms_per_system} batch={batch_size} total={total_atoms} "
                "cutoff={cutoff}{suffix}".format(**row, suffix=suffix)
            )
    print(f"D3 dry-run rows: {len(rows)}")
    return rows


def _save_planned_failure_rows(
    config: dict,
    *,
    backend: str,
    cutoffs: list[float],
    max_total_atoms: int | None,
    num_runs: int,
    warmup_runs: int,
    output_dir: Path,
    error: str,
    error_type: str,
    failure_stage: str,
) -> list[dict]:
    """Write failed D3 rows for each planned case before backend setup."""
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

            configs = configs_for_mode(
                mode_name, mode_config, sys_name, sys_config, nh3_dir
            )
            configs, skipped = filter_configs_by_total_atoms(
                configs, sys_name, max_total_atoms
            )
            results = []

            for cfg, _skipped_total in skipped:
                atoms_per_system, batch_size, total_atoms = planned_atom_counts(
                    sys_name, cfg
                )
                row_meta = make_row_meta(
                    sys_name,
                    mode_name,
                    backend,
                    atoms_per_system,
                    batch_size,
                    total_atoms,
                )
                reason = f">{max_total_atoms} max_total_atoms"
                results.extend(
                    build_skipped_result(
                        method="dftd3",
                        cutoff=cutoff,
                        reason=reason,
                        timing_runs=num_runs,
                        warmup_runs=warmup_runs,
                        **row_meta,
                    )
                    for cutoff in cutoffs
                )

            for cfg in configs:
                atoms_per_system, batch_size, total_atoms = planned_atom_counts(
                    sys_name, cfg
                )
                row_meta = make_row_meta(
                    sys_name,
                    mode_name,
                    backend,
                    atoms_per_system,
                    batch_size,
                    total_atoms,
                )
                results.extend(
                    build_failure_result(
                        method="dftd3",
                        cutoff=cutoff,
                        error=error,
                        error_type=error_type,
                        failure_stage=failure_stage,
                        timing_runs=num_runs,
                        warmup_runs=warmup_runs,
                        **row_meta,
                    )
                    for cutoff in cutoffs
                )

            if results:
                csv_name = make_csv_name("d3", sys_name, mode_name)
                save_results(results, output_dir / csv_name, replace_backend=backend)
                all_results.extend(results)

    return all_results


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
    max_total_atoms = params.get("max_total_atoms")
    d3_func_params = config["dftd3_parameters"]
    methods = enabled_method_names(config) if "methods" in config else ["dftd3"]

    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")
    if backend == "warp":
        raise ValueError("D3 benchmark supports torch and jax backends, not warp.")
    if config.get("runtime", {}).get("dry_run", False):
        return dry_run_from_config(config, backend=backend)
    if "dftd3" not in methods:
        selected = ", ".join(methods) if methods else "(none)"
        print(f"D3 benchmark no enabled methods for selected methods: {selected}")
        return []

    if output_dir is None:
        output_dir = create_run_directory(config["output"]["base_dir"], prefix="d3")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d3_params_path = _resolve_d3_params_path(config["params_path"])
    input_artifacts = configured_nh3_artifacts(config)
    input_artifacts["d3_parameters"] = d3_params_path
    configure_input_provenance(
        input_artifacts,
        metadata_values={"benchmark": "d3"},
    )
    try:
        _ensure_d3_parameter_file(d3_params_path)
        configure_input_provenance(
            input_artifacts,
            metadata_values={"benchmark": "d3"},
        )
        d3_params_torch = torch.load(
            d3_params_path,
            map_location="cpu",
            weights_only=True,
        )
        if backend == "jax":
            jax_api = lazy_import_jax(need_dispersion=True)
            d3_params = _torch_d3_params_to_jax(d3_params_torch, jax_api["jnp"])
        else:
            d3_params = _torch_d3_params_to_device(d3_params_torch, "cuda")
    except Exception as exc:
        error_type = failure_error_type(exc)
        error = str(exc)
        if isinstance(exc, FileNotFoundError):
            error += (
                ". Run: python examples/dispersion/01_dftd3_molecule.py "
                "to populate the cache manually."
            )
        if error_type == "OutOfMemoryError":
            clean_gpu()
        print(f"ERROR (D3 parameter setup): {error}")
        return _save_planned_failure_rows(
            config,
            backend=backend,
            cutoffs=cutoffs,
            max_total_atoms=max_total_atoms,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            output_dir=output_dir,
            error=error,
            error_type=error_type,
            failure_stage="parameter_setup",
        )
    print(f"Loaded D3 parameters from {d3_params_path}")

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
            configs, skipped = filter_configs_by_total_atoms(
                configs, sys_name, max_total_atoms
            )
            results = []
            for cfg, skipped_total in skipped:
                print(
                    f"  SKIP total atoms {format_num(skipped_total)} "
                    f"(>{format_num(max_total_atoms)})"
                )
                atoms_per_system, batch_size, total_atoms = planned_atom_counts(
                    sys_name, cfg
                )
                row_meta = make_row_meta(
                    sys_name,
                    mode_name,
                    backend,
                    atoms_per_system,
                    batch_size,
                    total_atoms,
                )
                reason = f">{max_total_atoms} max_total_atoms"
                results.extend(
                    build_skipped_result(
                        method="dftd3",
                        cutoff=cutoff,
                        reason=reason,
                        timing_runs=num_runs,
                        warmup_runs=warmup_runs,
                        **row_meta,
                    )
                    for cutoff in cutoffs
                )
            if not configs:
                if results:
                    csv_name = make_csv_name("d3", sys_name, mode_name)
                    save_results(
                        results, output_dir / csv_name, replace_backend=backend
                    )
                    all_results.extend(results)
                continue

            for cfg in configs:
                n, bs = cfg["num_atoms"], cfg["batch_size"]
                planned_n, planned_bs, planned_total = planned_atom_counts(
                    sys_name, cfg
                )
                planned_row_meta = make_row_meta(
                    sys_name,
                    mode_name,
                    backend,
                    planned_n,
                    planned_bs,
                    planned_total,
                )
                try:
                    data = create_system(
                        sys_name,
                        num_atoms=n,
                        pdb_path=cfg.get("pdb_path"),
                        batch_size=bs,
                        backend=backend,
                    )
                except Exception as e:
                    error_type = failure_error_type(e)
                    if error_type == "OutOfMemoryError":
                        clean_gpu()
                    print(f"    FAILED (system setup): {e}")
                    results.extend(
                        build_failure_result(
                            method="dftd3",
                            cutoff=cutoff,
                            error=str(e),
                            error_type=error_type,
                            failure_stage="system_setup",
                            timing_runs=num_runs,
                            warmup_runs=warmup_runs,
                            **planned_row_meta,
                        )
                        for cutoff in cutoffs
                    )
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
                    if backend == "jax":
                        clean_jax()
                    clean_gpu()

            if results:
                csv_name = make_csv_name("d3", sys_name, mode_name)
                save_results(results, output_dir / csv_name, replace_backend=backend)
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
    add_common_cli_args(
        parser,
        backends=("torch", "jax"),
        include_d3_params_path=True,
    )
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
    plan_only = (
        getattr(args, "dry_run", False)
        or getattr(args, "list_plan", False)
        or getattr(args, "count_plan", False)
    )
    if backend == "jax" and not plan_only:
        ensure_jax_available(need_dispersion=True)

    results = run_from_config(config, output_dir=args.output_dir, backend=backend)
    if not results:
        return 1
    if not any(row.get("success", True) is not False for row in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
