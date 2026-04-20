#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Electrostatics Benchmark (Ewald + PME).

CRITICAL: Electrostatics requires float64 for positions and cells.
K-vectors are pre-computed ONCE outside the timing loop.

Usage (run from the repository root):
    python -m benchmarks.interactions.electrostatics.benchmark_electrostatics \
        --config benchmarks/interactions/electrostatics/benchmark_config.yaml
    python -m benchmarks.interactions.electrostatics.benchmark_electrostatics \
        --config benchmarks/interactions/electrostatics/benchmark_config.yaml \
        --output-dir docs/benchmarks/benchmark_results

    # JAX backend
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python -m benchmarks.interactions.electrostatics.benchmark_electrostatics \
        --config benchmarks/interactions/electrostatics/benchmark_config.yaml --backend jax

Backends
--------
``--backend torch`` (default) uses the warp-based torch kernels and CUDA
events for timing. ``--backend jax`` uses the JAX wrappers in
``nvalchemiops.jax.interactions.electrostatics`` and wall-clock timing.

JAX caveats:

- ``jax_enable_x64`` is enabled at module import; electrostatics requires
  float64 positions and cells.
- The JAX ``ewald_summation`` API does not currently support
  ``compute_charge_gradients`` for the combined (real+reciprocal) call.
  Rows with ``method='ewald_cg'`` and ``backend='jax'`` are written with
  ``success=False`` so they are filtered by the plotter.

Environment variables for ``--backend jax``:

- ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` — required, or XLA grabs all VRAM.
- ``JAX_ENABLE_X64=True`` — programmatically set by nvalchemiops on import;
  exporting it explicitly is a safe alternative.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch

__all__ = [
    "ElectrostaticsInputs",
    "benchmark_ewald",
    "benchmark_pme",
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
    cscl_actual_atoms,
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
from nvalchemiops.torch.interactions.electrostatics import (
    estimate_ewald_parameters,
    estimate_pme_parameters,
    ewald_summation,
    generate_k_vectors_ewald_summation,
    generate_k_vectors_pme,
    particle_mesh_ewald,
)
from nvalchemiops.torch.neighbors import batch_naive_neighbor_list

# =============================================================================
# Config Loading
# =============================================================================


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides on top of YAML config. Adds EL-specific
    ``--methods`` and ``--accuracies`` on top of the shared flags."""
    config = merge_common_cli_overrides(config, args)
    if args.accuracies is not None:
        config["accuracies"] = args.accuracies
    if args.methods is not None:
        for method in config.get("methods", []):
            method["enabled"] = method["name"] in args.methods
    return config


# =============================================================================
# Inputs Bundle
# =============================================================================


@dataclass(frozen=True)
class ElectrostaticsInputs:
    """Per-config tensor bundle passed to the benchmark kernels.

    Bundles the system state (positions/charges/cell/pbc/batch_idx) with the
    precomputed neighbor list (nl_data/nl_shifts/nl_ptr) so the kernel-level
    benchmark functions take one object instead of eight positional args.

    Frozen — the runner builds one per (system, accuracy) config and passes
    it through. Field types are ``Any`` because the same shape applies to
    both torch tensors and jax arrays; the ``backend`` field disambiguates.
    """

    positions: Any
    charges: Any
    cell: Any
    pbc: Any
    batch_idx: Any
    nl_data: Any
    nl_shifts: Any
    nl_ptr: Any
    backend: str
    # max_atoms_per_system is needed by the JAX ewald_summation API under
    # jax.jit (shape inference can't query batch_idx.max() inside a trace).
    # Our systems are uniform, so this equals the per-system atom count.
    max_atoms_per_system: int = 0


# =============================================================================
# Core Benchmarks
# =============================================================================


def benchmark_pme(
    inputs: ElectrostaticsInputs,
    alpha: Any,
    mesh_dims: tuple[int, int, int],
    spline_order: int,
    accuracy: float,
    compute_cg: bool,
    num_runs: int,
    warmup_runs: int,
    jax_api: dict | None = None,
) -> dict:
    """Benchmark Particle Mesh Ewald for one config.

    Dispatches on ``inputs.backend``. Accepts pre-converted f64 tensors/arrays
    on ``inputs`` to avoid redundant GPU copies.
    """
    if inputs.backend == "jax":
        return _benchmark_pme_jax(
            inputs,
            alpha,
            mesh_dims,
            spline_order,
            accuracy,
            compute_cg,
            num_runs,
            warmup_runs,
            jax_api,
        )

    k_vectors, k_squared = generate_k_vectors_pme(inputs.cell, mesh_dims)

    def run_pme():
        particle_mesh_ewald(
            positions=inputs.positions,
            charges=inputs.charges,
            cell=inputs.cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=spline_order,
            batch_idx=inputs.batch_idx,
            k_vectors=k_vectors,
            k_squared=k_squared,
            neighbor_list=inputs.nl_data,
            neighbor_ptr=inputs.nl_ptr,
            neighbor_shifts=inputs.nl_shifts,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
            accuracy=accuracy,
        )

    _, mem_info = measure_memory_torch(run_pme)
    time_sec = cuda_timed_runs(run_pme, num_runs, warmup_runs=warmup_runs)
    return {"time_seconds": time_sec, "mem_info": mem_info}


def benchmark_ewald(
    inputs: ElectrostaticsInputs,
    alpha: Any,
    k_cutoff: float,
    accuracy: float,
    compute_cg: bool,
    num_runs: int,
    warmup_runs: int,
    jax_api: dict | None = None,
) -> dict:
    """Benchmark Ewald summation for one config via the unified API.

    Raises
    ------
    NotImplementedError
        When ``inputs.backend='jax'`` and ``compute_cg=True``. The JAX
        ``ewald_summation`` API does not yet support charge gradients.
    """
    if inputs.backend == "jax":
        return _benchmark_ewald_jax(
            inputs,
            alpha,
            k_cutoff,
            accuracy,
            compute_cg,
            num_runs,
            warmup_runs,
            jax_api,
        )

    k_vectors = generate_k_vectors_ewald_summation(inputs.cell, k_cutoff)
    if k_vectors.ndim == 2:
        k_vectors = k_vectors.unsqueeze(0)

    def run_ewald():
        ewald_summation(
            positions=inputs.positions,
            charges=inputs.charges,
            cell=inputs.cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            batch_idx=inputs.batch_idx,
            neighbor_list=inputs.nl_data,
            neighbor_ptr=inputs.nl_ptr,
            neighbor_shifts=inputs.nl_shifts,
            k_vectors=k_vectors,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
            accuracy=accuracy,
        )

    _, mem_info = measure_memory_torch(run_ewald)
    time_sec = cuda_timed_runs(run_ewald, num_runs, warmup_runs=warmup_runs)
    return {"time_seconds": time_sec, "mem_info": mem_info}


def _benchmark_pme_jax(
    inputs,
    alpha,
    mesh_dims,
    spline_order,
    accuracy,
    compute_cg,
    num_runs,
    warmup_runs,
    jax_api,
):
    """JAX backend implementation of :func:`benchmark_pme`."""
    jax = jax_api["jax"]
    jax_pme = jax_api["particle_mesh_ewald"]
    k_pme = jax_api["generate_k_vectors_pme"]

    k_vectors, k_squared = k_pme(inputs.cell, mesh_dims)

    def run_pme():
        return jax_pme(
            positions=inputs.positions,
            charges=inputs.charges,
            cell=inputs.cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=spline_order,
            batch_idx=inputs.batch_idx,
            k_vectors=k_vectors,
            k_squared=k_squared,
            neighbor_list=inputs.nl_data,
            neighbor_ptr=inputs.nl_ptr,
            neighbor_shifts=inputs.nl_shifts,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
            accuracy=accuracy,
        )

    # jit so the timed loop reflects steady-state per-call cost, not
    # Python-side tracing on every iteration.
    run_pme_jit = jax.jit(run_pme)
    _, mem_info = measure_memory_jax(run_pme_jit, jax)
    time_sec = cuda_timed_runs(
        run_pme_jit, num_runs, warmup_runs=warmup_runs, backend="jax"
    )
    return {"time_seconds": time_sec, "mem_info": mem_info}


def _benchmark_ewald_jax(
    inputs,
    alpha,
    k_cutoff,
    accuracy,
    compute_cg,
    num_runs,
    warmup_runs,
    jax_api,
):
    """JAX backend implementation of :func:`benchmark_ewald`.

    Raises ``NotImplementedError`` when ``compute_cg=True`` because the JAX
    ``ewald_summation`` API hard-codes ``compute_charge_gradients=False`` for
    the combined real+reciprocal call. The caller logs a ``success=False`` row.
    """
    if compute_cg:
        raise NotImplementedError(
            "jax_cg_unsupported: ewald_summation does not accept "
            "compute_charge_gradients in the current JAX API"
        )

    jax = jax_api["jax"]
    jax_ewald = jax_api["ewald_summation"]
    k_ewald = jax_api["generate_k_vectors_ewald_summation"]

    k_vectors = k_ewald(inputs.cell, k_cutoff)
    if k_vectors.ndim == 2:
        k_vectors = jax_api["jnp"].expand_dims(k_vectors, 0)

    def run_ewald():
        return jax_ewald(
            positions=inputs.positions,
            charges=inputs.charges,
            cell=inputs.cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            batch_idx=inputs.batch_idx,
            max_atoms_per_system=inputs.max_atoms_per_system,
            neighbor_list=inputs.nl_data,
            neighbor_ptr=inputs.nl_ptr,
            neighbor_shifts=inputs.nl_shifts,
            k_vectors=k_vectors,
            compute_forces=True,
            accuracy=accuracy,
        )

    run_ewald_jit = jax.jit(run_ewald)
    _, mem_info = measure_memory_jax(run_ewald_jit, jax)
    time_sec = cuda_timed_runs(
        run_ewald_jit, num_runs, warmup_runs=warmup_runs, backend="jax"
    )
    return {"time_seconds": time_sec, "mem_info": mem_info}


# =============================================================================
# run_from_config helpers
# =============================================================================


def _el_tensors_from_data(data, backend):
    """Convert a ``create_system`` dict to f64 tensors/arrays for electrostatics.

    Returns ``(positions, charges, cell, pbc, batch_idx)`` as a tuple. The
    caller drops its reference to ``data`` afterward so the f32 originals can
    be released.
    """
    pbc = data["pbc"]
    batch_idx = data["batch_idx"]
    if backend == "torch":
        positions = data["positions"].to(torch.float64)
        charges = data["charges"].to(torch.float64)
        cell = data["cell"].to(torch.float64)
    else:
        import jax.numpy as jnp

        positions = data["positions"].astype(jnp.float64)
        charges = data["charges"].astype(jnp.float64)
        cell = data["cell"].astype(jnp.float64)
    return positions, charges, cell, pbc, batch_idx


def _el_estimate_params(positions, cell, batch_idx, backend, accuracy, jax_api):
    """Estimate PME + Ewald parameters for one system at the given accuracy.

    Dispatches on backend; returns ``(pme_params, ewald_params)`` parameter
    dataclasses from the underlying library.
    """
    if backend == "torch":
        pme_params = estimate_pme_parameters(
            positions, cell, batch_idx=batch_idx, accuracy=accuracy
        )
        ewald_params = estimate_ewald_parameters(
            positions, cell, batch_idx=batch_idx, accuracy=accuracy
        )
    else:
        pme_params = jax_api["estimate_pme_parameters"](
            positions, cell, batch_idx=batch_idx, accuracy=accuracy
        )
        ewald_params = jax_api["estimate_ewald_parameters"](
            positions, cell, batch_idx=batch_idx, accuracy=accuracy
        )
    return pme_params, ewald_params


def _el_unpack_params(pme_params, ewald_params, backend):
    """Extract alpha / cutoffs / mesh_dims from the parameter dataclasses.

    Returns ``(alpha, real_cutoff, mesh_dims, k_cutoff)``. ``alpha`` is the
    scalar shape the kernels consume (torch 0-dim tensor for torch, python
    float for jax). For diagnostic printing, ``float(alpha)`` works in both
    cases.
    """
    if backend == "torch":
        alpha = pme_params.alpha.clone()
        if alpha.dim() > 0:
            alpha = alpha.mean()
        real_cutoff = (
            pme_params.real_space_cutoff[0].item()
            if pme_params.real_space_cutoff.dim() > 0
            else pme_params.real_space_cutoff.item()
        )
        mesh_dims = tuple(pme_params.mesh_dimensions)
        k_cutoff = ewald_params.reciprocal_space_cutoff.max().item()
    else:
        alpha = float(pme_params.alpha.mean())
        real_cutoff = float(pme_params.real_space_cutoff[0])
        md = pme_params.mesh_dimensions
        mesh_dims = (int(md[0]), int(md[1]), int(md[2]))
        k_cutoff = float(ewald_params.reciprocal_space_cutoff.max())
    return alpha, real_cutoff, mesh_dims, k_cutoff


def _el_build_nl(positions, cell, pbc, batch_idx, real_cutoff, backend, jax_api):
    """Build the naive-style neighbor list in LIST (COO + ptr) format."""
    maxnb = estimate_max_neighbors(
        real_cutoff,
        atomic_density=DEFAULT_ATOMIC_DENSITY,
        safety_factor=DEFAULT_NL_SAFETY_FACTOR,
    )
    if backend == "torch":
        return batch_naive_neighbor_list(
            positions=positions,
            cutoff=real_cutoff,
            batch_idx=batch_idx,
            pbc=pbc,
            cell=cell,
            max_neighbors=maxnb,
            return_neighbor_list=True,
        )
    nl = jax_api["neighbor_list"](
        positions=positions,
        cutoff=real_cutoff,
        cell=cell,
        pbc=pbc,
        batch_idx=batch_idx,
        method="batch_naive",
        return_neighbor_list=True,
        max_neighbors=int(maxnb),
    )
    jax_api["jax"].block_until_ready(nl[0])
    return nl


# =============================================================================
# Config-Driven Runner
# =============================================================================


class ElConfigSetup(NamedTuple):
    """Return shape of :func:`_el_setup_config`.

    Named over positional unpacking — the eight fields are a mix of
    backend-polymorphic tensors (``inputs``, ``alpha``) and plain scalars
    that the method loop consumes.
    """

    inputs: ElectrostaticsInputs
    alpha: Any  # torch 0-dim tensor (torch) or python float (jax)
    real_cutoff: float
    mesh_dims: tuple[int, int, int]
    k_cutoff: float
    atoms_per_system: int
    batch_size: int
    actual_total: int


def _el_setup_config(
    cfg: dict, sys_name: str, accuracy: float, backend: str, jax_api: dict | None
) -> ElConfigSetup | None:
    """Build the per-config :class:`ElectrostaticsInputs` + derived params.

    Returns ``None`` on expected failures (create_system error, params
    estimation failure, NL build failure) after printing a diagnostic line.
    """
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
        return None

    atoms_per_system = data["atoms_per_system"]
    actual_total = data.get("total_atoms", atoms_per_system)
    batch_size = data.get("batch_size", 1)
    print(
        f"\n  {format_num(atoms_per_system)} atoms × {batch_size} batch = "
        f"{format_num(actual_total)} total  [GPU: {current_alloc_gb(backend):.1f} GB allocated]"
    )

    positions, charges, cell, pbc, batch_idx = _el_tensors_from_data(data, backend)
    del data

    try:
        pme_params, ewald_params = _el_estimate_params(
            positions, cell, batch_idx, backend, accuracy, jax_api
        )
    except (RuntimeError, ValueError) as e:
        print(f"    SKIP (params): {e}")
        del positions, charges, cell, batch_idx
        return None

    alpha, real_cutoff, mesh_dims, k_cutoff = _el_unpack_params(
        pme_params, ewald_params, backend
    )
    del pme_params, ewald_params

    try:
        nl_data, nl_ptr, nl_shifts = _el_build_nl(
            positions, cell, pbc, batch_idx, real_cutoff, backend, jax_api
        )
    except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
        print(f"    SKIP (NL): {e}")
        del positions, charges, cell, batch_idx
        return None

    inputs = ElectrostaticsInputs(
        positions=positions,
        charges=charges,
        cell=cell,
        pbc=pbc,
        batch_idx=batch_idx,
        nl_data=nl_data,
        nl_shifts=nl_shifts,
        nl_ptr=nl_ptr,
        backend=backend,
        max_atoms_per_system=int(atoms_per_system),
    )
    return ElConfigSetup(
        inputs=inputs,
        alpha=alpha,
        real_cutoff=real_cutoff,
        mesh_dims=mesh_dims,
        k_cutoff=k_cutoff,
        atoms_per_system=atoms_per_system,
        batch_size=batch_size,
        actual_total=actual_total,
    )


def _el_run_method(
    method,
    inputs,
    alpha,
    mesh_dims,
    k_cutoff,
    spline_order,
    accuracy,
    compute_cg,
    num_runs,
    warmup_runs,
    jax_api,
    row_meta,
):
    """Run one ``(method, compute_cg)`` combination and build a result row.

    Catches OOM (prints, returns None), ``NotImplementedError`` (emits a
    ``success=False`` row for plotter filtering), and other exceptions
    (prints, returns None). ``row_meta`` carries the identity fields for
    :func:`build_result`.
    """
    cg_label = "+cg" if compute_cg else ""
    label = f"{method.upper()}{cg_label}"
    method_col = f"{method}{'_cg' if compute_cg else ''}"
    try:
        if method == "pme":
            r = benchmark_pme(
                inputs,
                alpha,
                mesh_dims,
                spline_order,
                accuracy,
                compute_cg,
                num_runs,
                warmup_runs,
                jax_api=jax_api,
            )
        else:
            r = benchmark_ewald(
                inputs,
                alpha,
                k_cutoff,
                accuracy,
                compute_cg,
                num_runs,
                warmup_runs,
                jax_api=jax_api,
            )
        result = build_result(
            method=method_col,
            time_seconds=r["time_seconds"],
            mem_info=r["mem_info"],
            accuracy=accuracy,
            **row_meta,
        )
        print(
            f"    {label:10s}: {result['time_us_per_atom']:.3f} μs/atom | "
            f"{result['mem_peak_gb']:.1f} GB"
        )
        return result
    except NotImplementedError as e:
        # Expected for JAX ewald+cg. Emit success=False so the plotter can filter.
        print(f"    {label:10s}: UNSUPPORTED - {e}")
        return build_result(
            method=method_col,
            time_seconds=0.0,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
            success=False,
            accuracy=accuracy,
            **row_meta,
        )
    except torch.cuda.OutOfMemoryError:
        print(f"    {label:10s}: OOM")
        clean_gpu()
        return None
    except Exception as e:
        print(f"    {label:10s}: FAILED - {e}")
        return None


def run_from_config(
    config: dict,
    output_dir: Path | str | None = None,
    backend: str | None = None,
) -> list[dict]:
    """Run electrostatics benchmarks driven by YAML config.

    Parameters
    ----------
    backend : str, optional
        ``'torch'`` or ``'jax'``. Pulled from ``config['runtime']['backend']``
        when None. Default is ``'torch'``.
    """
    params = config["parameters"]
    num_runs = params["timing_runs"]
    warmup_runs = params["warmup_runs"]
    accuracies = config["accuracies"]
    max_atoms = config["max_atoms"]
    skip_accuracy_for_large = config.get("skip_accuracy_for_large", {})
    cg_options = config["compute_charge_gradients"]
    methods_config = config["methods"]
    method_names = [m["name"] for m in methods_config if m.get("enabled", True)]
    # YAML is authoritative for spline_order; None when PME isn't enabled.
    pme_spline_order = next(
        (m["spline_order"] for m in methods_config if m["name"] == "pme"),
        None,
    )

    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")
    jax_api = lazy_import_jax(need_electrostatics=True) if backend == "jax" else None

    if output_dir is None:
        output_dir = create_run_directory(config["output"]["base_dir"], prefix="el")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        gpu_name = torch.cuda.get_device_name(0)
    except (AssertionError, RuntimeError):
        gpu_name = "N/A (no CUDA)"
    print(f"Electrostatics Benchmark | GPU: {gpu_name}")
    print(f"Backend: {backend}")
    print(f"Methods: {method_names} | Accuracies: {accuracies}")
    print(f"Timing: {num_runs} runs")
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
            print(f"ELECTROSTATICS: {sys_name.upper()} / {mode_name}")
            print(f"{'=' * 70}")

            configs = configs_for_mode(
                mode_name, mode_config, sys_name, sys_config, nh3_dir
            )
            if not configs:
                continue

            results = []
            for accuracy in accuracies:
                print(f"\n  --- Accuracy: {accuracy:.0e} ---")

                for cfg in configs:
                    actual_n_est = (
                        cscl_actual_atoms(cfg["num_atoms"])
                        if sys_name == "cscl"
                        else cfg["num_atoms"]
                    )
                    skip_threshold = skip_accuracy_for_large.get(
                        accuracy
                    ) or skip_accuracy_for_large.get(str(accuracy))
                    if skip_threshold and actual_n_est >= skip_threshold:
                        print(
                            f"  {format_num(actual_n_est)}: SKIP (OOM risk at {accuracy:.0e})"
                        )
                        continue
                    if actual_n_est * cfg["batch_size"] > max_atoms:
                        print(
                            f"  {format_num(actual_n_est)}×{cfg['batch_size']}: "
                            f"SKIP (>{format_num(max_atoms)})"
                        )
                        continue

                    clean_gpu()
                    setup = _el_setup_config(cfg, sys_name, accuracy, backend, jax_api)
                    if setup is None:
                        continue

                    print(
                        f"    alpha={float(setup.alpha):.4f}, "
                        f"r_cut={setup.real_cutoff:.2f}Å, "
                        f"NL pairs={setup.inputs.nl_data.shape[1]:,}"
                    )

                    row_meta = make_row_meta(
                        sys_name,
                        mode_name,
                        backend,
                        setup.atoms_per_system,
                        setup.batch_size,
                        setup.actual_total,
                    )
                    for method in method_names:
                        for compute_cg in cg_options:
                            result = _el_run_method(
                                method,
                                setup.inputs,
                                setup.alpha,
                                setup.mesh_dims,
                                setup.k_cutoff,
                                pme_spline_order,
                                accuracy,
                                compute_cg,
                                num_runs,
                                warmup_runs,
                                jax_api,
                                row_meta,
                            )
                            if result is not None:
                                results.append(result)

                    del setup

            if results:
                csv_name = make_csv_name("el", sys_name, mode_name)
                save_results(results, output_dir / csv_name)
                all_results.extend(results)

    print(f"\nCOMPLETE: {len(all_results)} results in {output_dir}")
    return all_results


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command-line arguments for electrostatics benchmarks."""
    parser = argparse.ArgumentParser(
        description="Electrostatics Benchmark (2 systems × 3 modes)"
    )
    parser.add_argument("--config", type=Path, required=True)
    add_common_cli_args(parser)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=["pme", "ewald"],
        help="Override which electrostatics methods to benchmark",
    )
    parser.add_argument(
        "--accuracies",
        "-a",
        type=float,
        nargs="+",
        default=None,
        help="Override target accuracies (Hartree/atom)",
    )
    return parser.parse_args()


def main():
    """Run electrostatics benchmarks."""
    args = parse_args()
    config = load_yaml_config(args.config)
    config = merge_cli_overrides(config, args)

    backend = args.backend or config.get("runtime", {}).get("backend", "torch")
    if backend == "jax":
        ensure_jax_available(need_electrostatics=True)

    run_from_config(config, output_dir=args.output_dir, backend=backend)


if __name__ == "__main__":
    main()
