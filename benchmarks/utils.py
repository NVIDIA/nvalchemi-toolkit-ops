# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark timing, memory measurement, and utility functions.

Timing: batched mean across N back-to-back calls with synchronisation only
at the start and end of the batch.

    Torch (CUDA events):
        sync → start.record() → N × fn() → end.record() → sync
        return elapsed_seconds / N

    JAX (wall-clock):
        block_until_ready(warmup) → t0 = perf_counter()
        → N × fn() → block_until_ready(last) → elapsed / N
"""

from __future__ import annotations

import csv
import gc
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import torch
import warp as wp

__all__ = [
    "MemInfo",
    "build_result",
    "clean_gpu",
    "create_run_directory",
    "cuda_timed_batch",
    "cuda_timed_runs",
    "current_alloc_gb",
    "ensure_jax_available",
    "format_num",
    "get_gpu_memory_info",
    "get_timestamp",
    "jax_timed_batch",
    "lazy_import_jax",
    "make_csv_name",
    "make_row_meta",
    "measure_memory_jax",
    "measure_memory_torch",
    "save_results",
    "write_run_log",
]


class MemInfo(TypedDict):
    """Minimal memory info dict returned by :func:`measure_memory_torch`
    and :func:`measure_memory_jax` and consumed by :func:`build_result`.

    The two keys map directly to CSV columns. For torch, ``mem_delta_mb``
    is the delta from the pre-timing measurement call and
    ``mem_peak_gb`` is the allocator peak. For JAX, both are always
    0.0 — the XLA pool (BFC or platform) makes per-call memory
    attribution unreliable, so the suite does not track JAX memory.
    The plotter filters zero-valued JAX rows out of memory panels.
    """

    mem_delta_mb: float
    mem_peak_gb: float


# =============================================================================
# GPU Utilities
# =============================================================================

# GPU memory tracking -- initialize pynvml once at module level
try:
    import pynvml as _pynvml

    _pynvml.nvmlInit()
    _GPU_HANDLE = _pynvml.nvmlDeviceGetHandleByIndex(0)
    _PYNVML_AVAILABLE = True
except Exception:
    _PYNVML_AVAILABLE = False
    _GPU_HANDLE = None


def clean_gpu() -> None:
    """Clean GPU memory between benchmark atom-size groups.

    Call once per atom-size change, NOT per (method, cutoff) config.
    """
    torch.cuda.synchronize()
    wp.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def get_gpu_memory_info() -> dict:
    """Get GPU memory usage. Uses cached pynvml handle.

    Returns
    -------
    dict
        Keys: 'used' (bytes), 'total' (bytes), 'percent' (float).
    """
    if _PYNVML_AVAILABLE:
        info = _pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
        return {
            "used": info.used,
            "total": info.total,
            "percent": 100.0 * info.used / info.total,
        }
    else:
        used = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        return {"used": used, "total": total, "percent": 100.0 * used / total}


# =============================================================================
# Lazy JAX Imports
# =============================================================================


def lazy_import_jax(
    *,
    need_electrostatics: bool = False,
    need_dispersion: bool = False,
) -> dict:
    """Lazy-import JAX and the selected nvalchemiops.jax submodules.

    Each benchmark runner needs a different subset of the JAX API.
    Consolidating the try/except and install-hint here keeps the error
    message consistent and centralizes the fp64 ordering constraint.

    Parameters
    ----------
    need_electrostatics : bool
        If True, import the electrostatics symbols used by the EL runner
        and enable x64 beforehand. ``jax_enable_x64`` must be set before
        the first electrostatics import — module-level code there captures
        dtypes that cannot be changed post-import.
    need_dispersion : bool
        If True, import the ``dftd3`` symbol used by the D3 runner.

    Returns
    -------
    dict
        Always contains ``jax``, ``jnp``, and ``neighbor_list``. The
        electrostatics and dispersion keys are populated only when the
        corresponding ``need_*`` flag is True.

    Raises
    ------
    ImportError
        If JAX or any requested submodule is unavailable, with a single
        install hint: ``uv sync --extra torch --extra jax``.
    """
    try:
        # x64 MUST be set via env before JAX is imported. `jax.config.update`
        # after import is a no-op once any traced array has been created,
        # which is exactly what happens when the suite runs NL before EL
        # and both share the same Python process. Setting the env var
        # unconditionally here is safe — runners that don't need f64
        # still work correctly; only EL relies on it.
        if need_electrostatics:
            os.environ.setdefault("JAX_ENABLE_X64", "1")

        import jax
        import jax.numpy as jnp

        if need_electrostatics:
            jax.config.update("jax_enable_x64", True)

        from nvalchemiops.jax.neighbors import (
            compute_naive_num_shifts,
            estimate_batch_cell_list_sizes,
            neighbor_list as jax_neighbor_list,
        )

        api: dict = {
            "jax": jax,
            "jnp": jnp,
            "neighbor_list": jax_neighbor_list,
            # Sizing helpers: CPU-side, called outside jit to compute the
            # static kwargs the jitted `neighbor_list` needs.
            "estimate_batch_cell_list_sizes": estimate_batch_cell_list_sizes,
            "compute_naive_num_shifts": compute_naive_num_shifts,
        }

        if need_electrostatics:
            from nvalchemiops.jax.interactions.electrostatics import (
                estimate_ewald_parameters,
                estimate_pme_parameters,
                ewald_summation,
                generate_k_vectors_ewald_summation,
                generate_k_vectors_pme,
                particle_mesh_ewald,
            )

            api.update(
                {
                    "ewald_summation": ewald_summation,
                    "particle_mesh_ewald": particle_mesh_ewald,
                    "generate_k_vectors_pme": generate_k_vectors_pme,
                    "generate_k_vectors_ewald_summation": generate_k_vectors_ewald_summation,
                    "estimate_pme_parameters": estimate_pme_parameters,
                    "estimate_ewald_parameters": estimate_ewald_parameters,
                }
            )

        if need_dispersion:
            from nvalchemiops.jax.interactions.dispersion import dftd3 as jax_dftd3

            api["dftd3"] = jax_dftd3
    except ImportError as e:
        raise ImportError(
            "JAX backend requested but jax is not installed. "
            "Install with: uv sync --extra torch --extra jax"
        ) from e
    return api


def ensure_jax_available(
    *,
    need_electrostatics: bool = False,
    need_dispersion: bool = False,
) -> None:
    """Fail-fast wrapper around :func:`lazy_import_jax` for runners' ``main()``.

    Prints the error message and exits with code 1 when jax is unavailable.
    Each benchmark runner calls this at startup when the user selects the
    jax backend so the failure surfaces immediately instead of partway
    through the benchmark loop.
    """
    try:
        lazy_import_jax(
            need_electrostatics=need_electrostatics,
            need_dispersion=need_dispersion,
        )
    except ImportError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1) from e


def current_alloc_gb(backend: str) -> float:
    """Return the current GPU memory footprint (in GB) for the given backend.

    Uses ``torch.cuda.memory_allocated`` for torch (allocator-level) and the
    NVML-based :func:`get_gpu_memory_info` for jax (since XLA allocations
    are invisible to torch). Used for the pre-kernel ``[GPU: X.X GB]``
    diagnostic line printed by each runner.
    """
    if backend == "torch":
        return torch.cuda.memory_allocated() / 1024**3
    return get_gpu_memory_info()["used"] / 1024**3


def make_row_meta(
    sys_name: str,
    mode_name: str,
    backend: str,
    atoms_per_system: int,
    batch_size: int,
    total_atoms: int,
) -> dict:
    """Build the six identity columns shared by every CSV row.

    Pulled from the three runners where the same dict construction appeared
    verbatim. Callers pass this to :func:`build_result` as ``**row_meta``
    along with the method-specific fields. Accepts the three atom-count
    values directly so EL (which extracts them from ``_el_setup_config``
    after ``data`` is released) can use it without reconstructing a dict.
    """
    return {
        "system": sys_name,
        "scaling_mode": mode_name,
        "backend": backend,
        "atoms_per_system": atoms_per_system,
        "batch_size": batch_size,
        "total_atoms": total_atoms,
    }


# =============================================================================
# Timing Functions
# =============================================================================


def cuda_timed_batch(
    fn: Callable[[], Any], num_runs: int, warmup_runs: int = 3
) -> float:
    """Time a function using CUDA events — batch pattern (throughput).

    Pattern (verified by senior engineer):
        sync() → start.record() → N × fn() → end.record() → sync()
        No sync inside loop. Returns total_elapsed / N.

    This measures sustained GPU throughput without sync overhead pollution.

    Parameters
    ----------
    fn : callable
        Zero-argument function to time.
    num_runs : int
        Number of timing iterations.
    warmup_runs : int, default=3
        Number of warmup runs (not timed).

    Returns
    -------
    float
        Mean time per run in seconds.
    """
    # Warmup (separate from timing)
    for _ in range(warmup_runs):
        fn()
    torch.cuda.synchronize()
    wp.synchronize()

    # Batch timing: sync only at start and end
    torch.cuda.synchronize()
    wp.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_runs):
        fn()  # NO sync inside loop
    end.record()

    torch.cuda.synchronize()
    wp.synchronize()

    elapsed_ms = start.elapsed_time(end)
    return (elapsed_ms / 1000.0) / num_runs  # seconds per run


def cuda_timed_runs(
    fn: Callable[[], Any],
    num_runs: int,
    warmup_runs: int = 3,
    backend: str = "torch",
) -> float:
    """Time ``fn`` and return the mean seconds per call, dispatching on backend.

    Torch uses CUDA events (:func:`cuda_timed_batch`); JAX uses wall-clock
    + ``jax.block_until_ready`` (:func:`jax_timed_batch`). See the module
    docstring for the batch-timing rationale.

    Parameters
    ----------
    fn : callable
        Zero-argument function to time. For JAX, ``fn`` must return a
        jax.Array (or pytree) so ``jax.block_until_ready`` can flush
        async dispatch.
    num_runs : int
        Number of timing iterations.
    warmup_runs : int, default=3
        Number of warmup calls (not counted; the first JAX warmup
        triggers JIT compile).
    backend : str, default='torch'
        ``'torch'`` or ``'jax'``.

    Returns
    -------
    float
        Mean seconds per call across ``num_runs`` back-to-back executions.
    """
    if backend == "jax":
        return jax_timed_batch(fn, num_runs, warmup_runs)
    return cuda_timed_batch(fn, num_runs, warmup_runs)


# =============================================================================
# JAX Timing
# =============================================================================


def jax_timed_batch(
    fn: Callable[[], Any], num_runs: int, warmup_runs: int = 3
) -> float:
    """Time a JAX function using wall-clock timing — batch pattern.

    CUDA events do not see JAX work. Instead, warm up (to trigger JIT
    compile, which is NOT counted in timing), then measure N calls with
    ``time.perf_counter`` and flush async dispatch with
    :func:`jax.block_until_ready` once at the end.

    Parameters
    ----------
    fn : callable
        Zero-argument function. Must return a jax.Array (or pytree of arrays)
        so ``jax.block_until_ready`` can complete dispatch.
    num_runs : int
        Number of timing iterations.
    warmup_runs : int, default=3
        Number of warmup runs. First warmup triggers JIT compile.

    Returns
    -------
    float
        Mean time per run in seconds (excludes JIT compile).
    """
    import time

    import jax

    # Warmup — first call triggers JIT. Block so compile time doesn't bleed in.
    for _ in range(warmup_runs):
        out = fn()
        jax.block_until_ready(out)

    # Batch timing: dispatch N calls, block once at the end.
    t0 = time.perf_counter()
    last = None
    for _ in range(num_runs):
        last = fn()
    jax.block_until_ready(last)
    elapsed = time.perf_counter() - t0
    return elapsed / num_runs


# =============================================================================
# Memory Measurement
# =============================================================================


def measure_memory_torch(fn: Callable[[], Any]) -> tuple[Any, MemInfo]:
    """Run ``fn`` once and capture torch-side peak allocator memory.

    Does NOT call :func:`clean_gpu`. The runner clears caches per
    atom-size group, not per (method, cutoff) config, so allocator
    peaks accumulate inside a group intentionally.

    Parameters
    ----------
    fn : callable
        Zero-argument function that executes the kernel under
        measurement. Its return value is passed through unchanged.

    Returns
    -------
    (result, mem_info)
        ``result`` is whatever ``fn`` returned; ``mem_info`` has keys
        ``mem_delta_mb`` and ``mem_peak_gb`` — the two fields the lean
        CSV schema consumes.
    """
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    wp.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    mem_info: MemInfo = {
        "mem_delta_mb": (mem_peak - mem_before) / 1024**2,
        "mem_peak_gb": mem_peak / 1024**3,
    }
    return result, mem_info


def measure_memory_jax(fn: Callable[[], Any], jax_module: Any) -> tuple[Any, MemInfo]:
    """Run ``fn`` once and return zero memory info.

    JAX memory is not tracked by this suite. XLA's BFC pool preallocates
    a large fraction of VRAM at process start and reuses across calls,
    which makes NVML-based deltas and ``mem_peak_gb`` both uninformative.
    We run the kernel once (so the caller sees a consistent control-flow
    shape and gets a real result back for inspection) but do not sample
    memory. Kept as a thin wrapper for API symmetry with
    :func:`measure_memory_torch`.

    Parameters
    ----------
    fn : callable
        Zero-argument function returning a jax.Array (or pytree).
    jax_module : module
        The ``jax`` module (typically ``lazy_import_jax()["jax"]``),
        used to block on the result before returning.

    Returns
    -------
    (result, mem_info)
        ``mem_info`` is always ``{"mem_delta_mb": 0.0, "mem_peak_gb": 0.0}``.
    """
    result = fn()
    jax_module.block_until_ready(result)
    return result, {"mem_delta_mb": 0.0, "mem_peak_gb": 0.0}


# =============================================================================
# Result Building & CSV Output
# =============================================================================


def build_result(
    *,
    # Identity
    system: str,
    scaling_mode: str,
    method: str,
    backend: str = "torch",
    atoms_per_system: int,
    batch_size: int,
    total_atoms: int,
    # Timing
    time_seconds: float,
    # Memory
    mem_info: MemInfo,
    # Status
    success: bool = True,
    # Optional extras (method-specific: cutoff, accuracy, time_d3_us_per_atom)
    **extra: Any,
) -> dict:
    """Build a standardized benchmark result dictionary.

    Emits only the columns the plotter reads; unused metrics are not stored.
    All benchmark scripts should use this to ensure consistent CSV columns.

    Parameters
    ----------
    system : str
        Chemical system ('cscl' or 'nh3').
    scaling_mode : str
        'system_size', 'constant_workload', or 'batch_scaling'.
    method : str
        Method name (e.g. 'naive', 'cell', 'dftd3', 'pme_cg', 'ewald_cg').
    backend : str, default='torch'
        Framework backend ('torch' or 'jax').
    atoms_per_system : int
        Atoms in each individual system.
    batch_size : int
        Number of systems in the batch.
    total_atoms : int
        atoms_per_system * batch_size.
    time_seconds : float
        Mean time per call in seconds.
    mem_info : dict
        Output from :func:`measure_memory_torch` or :func:`measure_memory_jax`;
        must contain 'mem_delta_mb' and 'mem_peak_gb'.
    success : bool, default=True
        Whether the benchmark completed successfully. Failed rows are filtered
        out by the plotter on load.
    **extra
        Additional method-specific fields:
        - cutoff (NL, D3)
        - accuracy (EL)
        - time_d3_us_per_atom (D3-only time, excludes NL)

    Returns
    -------
    dict
        Standardized result with minimal columns required by the plotter.
    """
    time_us_per_atom = (time_seconds * 1e6) / total_atoms if total_atoms > 0 else 0.0
    throughput_atoms_per_sec = total_atoms / time_seconds if time_seconds > 0 else 0.0

    result = {
        # Identity
        "system": system,
        "scaling_mode": scaling_mode,
        "method": method,
        "backend": backend,
        "atoms_per_system": atoms_per_system,
        "batch_size": batch_size,
        "total_atoms": total_atoms,
        # Time
        "time_us_per_atom": time_us_per_atom,
        # Throughput
        "throughput_atoms_per_sec": throughput_atoms_per_sec,
        # Memory — mem_peak_gb for torch, mem_delta_mb for jax
        "mem_delta_mb": mem_info["mem_delta_mb"],
        "mem_peak_gb": mem_info["mem_peak_gb"],
        # Status
        "success": success,
    }

    # Add any extra method-specific fields (cutoff, accuracy, time_d3_us_per_atom)
    result.update(extra)
    return result


def save_results(results: list[dict], output_path: Path | str) -> None:
    """Save benchmark results to CSV.

    If ``output_path`` already exists and its header matches ``results`` field
    names, new rows are appended. This lets torch and jax runs share one
    output directory without clobbering each other. If the existing header
    differs, the file is overwritten fresh.

    Parameters
    ----------
    results : list[dict]
        List of result dicts from build_result().
    output_path : str or Path
        Path to output CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        print(f"No results to save to {output_path}")
        return

    fieldnames = list(results[0].keys())

    # Append if existing file has same schema, else overwrite fresh
    if output_path.exists():
        with open(output_path, newline="") as f:
            existing_fields = csv.DictReader(f).fieldnames or []
        if list(existing_fields) == fieldnames:
            with open(output_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(results)
            print(f"Appended {len(results)} results to {output_path}")
            return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} results to {output_path}")


# =============================================================================
# Formatting Utilities
# =============================================================================


def format_num(n: int) -> str:
    """Format atom count using binary prefix: 1024→'1k', 131072→'128k'."""
    if n >= 1048576:
        return f"{n // 1048576}M"
    elif n >= 1024:
        return f"{n // 1024}k"
    return str(n)


def get_timestamp() -> str:
    """Generate timestamp string: YYYY-MM-DD_HH-MM-SS."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_run_directory(base_dir: Path | str, prefix: str = "run") -> Path:
    """Create a timestamped run directory for benchmark results.

    Parameters
    ----------
    base_dir : str or Path
        Base directory (e.g., benchmark-results/).
    prefix : str
        Directory prefix (e.g., 'run').

    Returns
    -------
    Path
        Created directory path.
    """
    run_dir = Path(base_dir) / f"{prefix}_{get_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    return run_dir


def make_csv_name(module: str, system: str, mode: str) -> str:
    """Generate standardized CSV filename.

    Pattern: {module}-{system}-{mode_label}.csv
    Examples: nl-cscl-system-size-scaling.csv, d3-nh3-batch-scaling.csv

    Parameters
    ----------
    module : str
        'nl', 'd3', or 'el'.
    system : str
        'cscl' or 'nh3'.
    mode : str
        Scaling mode key from YAML.

    Returns
    -------
    str
        Filename string.
    """
    mode_labels = {
        "system_size": "system-size-scaling",
        "constant_workload": "constant-workload-scaling",
        "batch_scaling": "batch-scaling",
    }
    label = mode_labels.get(mode, mode.replace("_", "-"))
    return f"{module}-{system}-{label}.csv"


def write_run_log(
    run_dir: Path | str,
    start_time: datetime,
    end_time: datetime | None = None,
    extra_info: dict | None = None,
) -> None:
    """Write a RUN_LOG.md with environment and reproducibility info.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    start_time : datetime
        When the run started.
    end_time : datetime, optional
        When the run ended.
    extra_info : dict, optional
        Additional key-value pairs to include.
    """
    import platform

    # Normalise to Path up front: ``run_dir.name`` below would crash on str.
    run_dir = Path(run_dir)

    gpu_name = "N/A"
    gpu_mem = "N/A"
    cuda_version = "N/A"
    try:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_mem = f"{props.total_memory / 1024**3:.0f} GB"
    except (AssertionError, RuntimeError):
        pass
    try:
        cuda_version = torch.version.cuda or "N/A"
    except AttributeError:
        pass

    try:
        import nvalchemiops

        toolkit_version = nvalchemiops.__version__
    except (ImportError, AttributeError):
        toolkit_version = "N/A"

    try:
        import warp as wp_ver

        warp_version = wp_ver.__version__
    except (ImportError, AttributeError):
        warp_version = "N/A"

    elapsed = ""
    if end_time:
        delta = end_time - start_time
        minutes = delta.total_seconds() / 60
        elapsed = f"\n- **Total runtime**: {minutes:.1f} minutes"

    lines = [
        f"# Benchmark Run: {run_dir.name}",
        "",
        "## Environment",
        "",
        f"- **GPU**: {gpu_name} ({gpu_mem})",
        f"- **CUDA**: {cuda_version}",
        f"- **PyTorch**: {torch.__version__}",
        f"- **nvalchemi-toolkit-ops**: {toolkit_version}",
        f"- **Warp**: {warp_version}",
        f"- **Python**: {platform.python_version()}",
        f"- **OS**: {platform.system()} {platform.release()}",
        f"- **Date**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        elapsed,
        "",
        "## Timing Methodology",
        "",
        "Batch timing pattern (sync outside loop):",
        "```",
        "sync() -> start.record() -> N x fn() -> end.record() -> sync()",
        "```",
        "Returns mean time per call. Measures sustained GPU throughput.",
        "",
        "## Files",
        "",
        "CSV naming: `{module}-{system}-{scaling-mode}.csv`",
        "",
        "| Column | Description |",
        "|--------|-------------|",
        "| `time_us_per_atom` | Time per atom [microseconds] |",
        "| `throughput_atoms_per_sec` | Atoms processed per second |",
        "| `mem_delta_mb` | Memory delta from the pre-timing measurement call [MB] (torch only; 0 for jax) |",
        "| `mem_peak_gb` | Peak GPU VRAM usage [GB] (torch only; 0 for jax) |",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "uv sync --all-extras",
        "python -m benchmarks.benchmark_suite --benchmark all",
        "```",
        "",
    ]

    if extra_info:
        lines.append("## Additional Info")
        lines.append("")
        for k, v in extra_info.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    log_path = run_dir / "RUN_LOG.md"
    log_path.write_text("\n".join(lines))
