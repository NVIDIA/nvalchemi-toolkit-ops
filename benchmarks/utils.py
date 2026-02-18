# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark timing, memory measurement, and utility functions.

Provides two timing strategies:
- Batch timing (default): measures sustained GPU throughput, returns mean.
  Pattern: sync → start.record() → N × fn() → end.record() → sync()
- Per-run timing: measures per-call latency, returns median.
  Pattern: for each run: sync → start → fn() → end → sync → append()

Batch timing is preferred for these benchmarks because:
1. No sync overhead pollution (~20μs per sync, significant for fast kernels)
2. GPU pipeline stays full (realistic sustained throughput)
3. Mean of continuous execution = true throughput metric

References:
    benchmarks-temp/benchmark_suite.py — verified by senior engineer (DF)
"""

import csv
import gc
import signal
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import warp as wp

# =============================================================================
# Timeout Handling
# =============================================================================


class TimeoutError(Exception):
    """Exception raised when a benchmark times out."""

    pass


@contextmanager
def timeout(seconds):
    """Context manager for timing out a code block (Unix only).

    Parameters
    ----------
    seconds : int
        Number of seconds before timeout.
    """

    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        yield  # No timeout on Windows


# =============================================================================
# GPU Utilities
# =============================================================================

# GPU memory tracking -- initialize pynvml once at module level
try:
    import pynvml as _pynvml

    _pynvml.nvmlInit()
    _GPU_HANDLE = _pynvml.nvmlDeviceGetHandleByIndex(0)
    _PYNVML_AVAILABLE = True
except (ImportError, Exception):
    _PYNVML_AVAILABLE = False
    _GPU_HANDLE = None


def clean_gpu():
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


def get_gpu_sku() -> str:
    """Get GPU name for filenames (e.g. 'h100-80gb')."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        name = torch.cuda.get_device_name(0)
        return name.replace(" ", "-").replace("NVIDIA-", "").lower()
    except Exception:
        return "unknown-gpu"


# =============================================================================
# Timing Functions
# =============================================================================


def cuda_timed_batch(fn, num_runs, warmup_runs=3):
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


def cuda_timed_per_run(fn, num_runs, warmup_runs=3):
    """Time a function using CUDA events — per-run pattern (latency).

    Creates event pairs per iteration, syncs between runs.
    Returns median to reject outliers.

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
        Median time per run in seconds.
    """
    # Warmup
    for _ in range(warmup_runs):
        fn()
        torch.cuda.synchronize()
        wp.synchronize()

    # Per-run timing
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        wp.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        wp.synchronize()
        times.append(start.elapsed_time(end))

    return float(np.median(times)) / 1000.0  # seconds


def cuda_timed_runs(fn, num_runs, mode="batch", warmup_runs=3):
    """Unified timing interface with selectable mode.

    Parameters
    ----------
    fn : callable
        Zero-argument function to time.
    num_runs : int
        Number of timing iterations.
    mode : str, default='batch'
        'batch' (throughput, mean) or 'per_run' (latency, median).
    warmup_runs : int, default=3
        Number of warmup runs.

    Returns
    -------
    float
        Time per run in seconds.
    """
    if mode == "batch":
        return cuda_timed_batch(fn, num_runs, warmup_runs)
    elif mode == "per_run":
        return cuda_timed_per_run(fn, num_runs, warmup_runs)
    else:
        raise ValueError(f"Unknown timing mode: {mode}. Use 'batch' or 'per_run'.")


# =============================================================================
# Memory Measurement
# =============================================================================


def measure_memory(fn):
    """Measure GPU memory consumed by a single function call.

    Runs fn() ONCE in a clean GPU state and measures peak memory delta.
    This is separate from timing to avoid interference.

    Parameters
    ----------
    fn : callable
        Zero-argument function to measure.

    Returns
    -------
    dict
        Keys: 'mem_delta_bytes', 'mem_peak_bytes', 'mem_delta_mb',
              'mem_peak_gb', 'mem_gpu_percent'.
    """
    clean_gpu()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    fn()

    torch.cuda.synchronize()
    wp.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    mem_delta = mem_peak - mem_before
    gpu_info = get_gpu_memory_info()

    return {
        "mem_delta_bytes": mem_delta,
        "mem_peak_bytes": mem_peak,
        "mem_delta_mb": mem_delta / 1024**2,
        "mem_peak_gb": mem_peak / 1024**3,
        "mem_gpu_percent": 100.0 * mem_peak / gpu_info["total"],
    }


# =============================================================================
# Result Building & CSV Output
# =============================================================================


def build_result(
    *,
    # Identity
    system: str,
    scaling_mode: str,
    method: str,
    atoms_per_system: int,
    batch_size: int,
    total_atoms: int,
    # Timing
    time_seconds: float,
    # Memory
    mem_info: dict,
    # Optional extras
    **extra,
) -> dict:
    """Build a standardized benchmark result dictionary.

    All benchmark scripts should use this to ensure consistent CSV columns.

    Parameters
    ----------
    system : str
        Chemical system ('cscl' or 'nh3').
    scaling_mode : str
        'system_size', 'constant_total', or 'constant_atoms_per_system'.
    method : str
        Method name (e.g. 'naive', 'cell', 'pme', 'ewald').
    atoms_per_system : int
        Atoms in each individual system.
    batch_size : int
        Number of systems in the batch.
    total_atoms : int
        atoms_per_system * batch_size.
    time_seconds : float
        Mean time per call in seconds.
    mem_info : dict
        Output from measure_memory().
    **extra
        Additional method-specific fields (cutoff, accuracy, etc.).

    Returns
    -------
    dict
        Standardized result with all required columns.
    """
    # Time metrics
    time_us_per_atom = (time_seconds * 1e6) / total_atoms if total_atoms > 0 else 0.0

    # Throughput metrics
    throughput_atoms_per_sec = total_atoms / time_seconds if time_seconds > 0 else 0.0

    # Memory per-atom metric
    mem_per_atom_kb = (
        (mem_info["mem_delta_bytes"] / 1024) / total_atoms if total_atoms > 0 else 0.0
    )

    result = {
        # Identity
        "system": system,
        "scaling_mode": scaling_mode,
        "method": method,
        "atoms_per_system": atoms_per_system,
        "batch_size": batch_size,
        "total_atoms": total_atoms,
        # Time
        "time_seconds": time_seconds,
        "time_us_per_atom": time_us_per_atom,
        # Throughput
        "throughput_atoms_per_sec": throughput_atoms_per_sec,
        "throughput_matoms_per_sec": throughput_atoms_per_sec / 1e6,
        # Memory
        "mem_delta_mb": mem_info["mem_delta_mb"],
        "mem_peak_gb": mem_info["mem_peak_gb"],
        "mem_per_atom_kb": mem_per_atom_kb,
        "mem_gpu_percent": mem_info["mem_gpu_percent"],
    }

    # Add any extra method-specific fields
    result.update(extra)
    return result


def save_results(results, output_path):
    """Save benchmark results to CSV.

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
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} results to {output_path}")


# =============================================================================
# Formatting Utilities
# =============================================================================


def format_num(n):
    """Format atom count using binary prefix: 1024→'1k', 131072→'128k'."""
    if n >= 1048576:
        return f"{n // 1048576}M"
    elif n >= 1024:
        return f"{n // 1024}k"
    return str(n)


def get_timestamp():
    """Generate timestamp string: YYYY-MM-DD_HH-MM-SS."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_run_directory(base_dir, prefix="run"):
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


def make_csv_name(module, system, mode):
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


def make_plot_name(module, system, mode):
    """Generate standardized plot filename (same pattern as CSV but .png)."""
    return make_csv_name(module, system, mode).replace(".csv", ".png")


def write_run_readme(run_dir, start_time, end_time=None, extra_info=None):
    """Write a README.md with reproducibility info to the run directory.

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

    gpu_name = "N/A"
    gpu_mem = "N/A"
    cuda_version = "N/A"
    try:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_mem = f"{props.total_mem / 1024**3:.0f} GB"
    except Exception:
        pass
    try:
        cuda_version = torch.version.cuda or "N/A"
    except Exception:
        pass

    try:
        import nvalchemiops

        toolkit_version = nvalchemiops.__version__
    except Exception:
        toolkit_version = "N/A"

    try:
        import warp as wp_ver

        warp_version = wp_ver.__version__
    except Exception:
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
        "| `mem_peak_gb` | Peak GPU VRAM usage [GB] |",
        "| `mem_per_atom_kb` | Memory per atom [KB] |",
        "",
        "## Reproducibility",
        "",
        "To reproduce:",
        "```bash",
        "cd benchmarks/neighborlist",
        "python benchmark_neighborlist.py --config benchmark_config.yaml",
        "```",
        "",
    ]

    if extra_info:
        lines.append("## Additional Info")
        lines.append("")
        for k, v in extra_info.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    readme_path = run_dir / "README.md"
    readme_path.write_text("\n".join(lines))
    return readme_path
