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
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, TypedDict

import torch
import warp as wp

__all__ = [
    "MemInfo",
    "build_result",
    "build_failure_result",
    "build_skipped_result",
    "clean_gpu",
    "clean_jax",
    "configure_input_provenance",
    "configure_jax_environment",
    "create_run_directory",
    "cuda_timed_batch",
    "cuda_timed_runs",
    "current_alloc_gb",
    "ensure_jax_available",
    "failure_error_type",
    "format_num",
    "get_gpu_memory_info",
    "get_timestamp",
    "jax_timed_batch",
    "jax_timed_serial",
    "jax_timed_stateful",
    "lazy_import_jax",
    "make_csv_name",
    "make_row_meta",
    "measure_memory_jax",
    "measure_memory_torch",
    "save_results",
    "sync_gpu",
    "validate_result_files",
    "write_run_log",
]


class MemInfo(TypedDict):
    """Minimal memory info dict returned by :func:`measure_memory_torch`
    and :func:`measure_memory_jax` and consumed by :func:`build_result`.

    The two keys map directly to CSV columns. For torch, ``mem_delta_mb``
    is the delta from the pre-timing measurement call and
    ``mem_peak_gb`` is the allocator peak. For JAX, both are ``NaN`` —
    the XLA pool (BFC or platform) makes per-call memory attribution
    unreliable, so the suite does not track JAX memory.
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
    sync_gpu()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sync_gpu()


def clean_jax(*, clear_executables: bool = False) -> None:
    """Release Python references after a JAX benchmark row.

    JAX owns its CUDA allocator separately from Torch/Warp. Successful benchmark
    rows should keep compiled executables available for later compatible rows;
    clearing them after every cutoff/method adds substantial compile/lowering
    wall time without changing the measured kernel timing. Pass
    ``clear_executables=True`` only for a deliberate hard reset after a severe
    failure or at process shutdown.
    """
    gc.collect()
    if clear_executables:
        with suppress(Exception):
            import jax

            jax.clear_caches()
    gc.collect()


def sync_gpu() -> None:
    """Synchronize Torch CUDA and Warp work queues once."""
    torch.cuda.synchronize()
    wp.synchronize()


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

_JAX_ALLOCATOR_WARNING_EMITTED = False


def configure_jax_environment(
    *,
    need_x64: bool = False,
    context: str = "JAX benchmark",
) -> None:
    """Set JAX env defaults before import and warn about allocator policy.

    The benchmark suite keeps JAX's default preallocator for steady-state
    throughput and caps it with ``XLA_PYTHON_CLIENT_MEM_FRACTION``. When users
    have not explicitly chosen a preallocation policy, emit a one-line note so
    the large XLA allocation is visible before the first JAX import.
    """
    global _JAX_ALLOCATOR_WARNING_EMITTED

    if need_x64:
        os.environ.setdefault("JAX_ENABLE_X64", "1")
    mem_fraction = os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

    if (
        "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ
        and "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ
        and not _JAX_ALLOCATOR_WARNING_EMITTED
    ):
        print(
            f"{context}: XLA_PYTHON_CLIENT_PREALLOCATE is unset; keeping JAX's "
            f"default preallocator capped by XLA_PYTHON_CLIENT_MEM_FRACTION="
            f"{mem_fraction}. Set XLA_PYTHON_CLIENT_PREALLOCATE=false to use "
            "on-demand allocation."
        )
        _JAX_ALLOCATOR_WARNING_EMITTED = True


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
        # Standalone NL/D3 runners keep JAX's default precision. EL opts into
        # x64 before its modules are imported; the combined suite performs that
        # setup before an earlier NL/D3 runner imports JAX in the shared process.
        configure_jax_environment(
            need_x64=need_electrostatics,
            context="JAX benchmark runner",
        )

        import jax
        import jax.numpy as jnp

        if need_electrostatics:
            jax.config.update("jax_enable_x64", True)

        from nvalchemiops.jax.neighbors import (
            batch_build_cell_list,
            batch_query_cell_list,
            build_cell_list,
            cluster_tile_neighbor_list,
            compute_naive_num_shifts,
            estimate_batch_cell_list_sizes,
            estimate_cell_list_sizes,
            query_cell_list,
        )
        from nvalchemiops.jax.neighbors import (
            neighbor_list as jax_neighbor_list,
        )

        api: dict = {
            "jax": jax,
            "jnp": jnp,
            "neighbor_list": jax_neighbor_list,
            "build_cell_list": build_cell_list,
            "query_cell_list": query_cell_list,
            "batch_build_cell_list": batch_build_cell_list,
            "batch_query_cell_list": batch_query_cell_list,
            "cluster_tile_neighbor_list": cluster_tile_neighbor_list,
            # Sizing helpers: CPU-side, called outside jit to compute the
            # static kwargs the jitted `neighbor_list` needs.
            "estimate_batch_cell_list_sizes": estimate_batch_cell_list_sizes,
            "estimate_cell_list_sizes": estimate_cell_list_sizes,
            "compute_naive_num_shifts": compute_naive_num_shifts,
        }

        if need_electrostatics:
            from nvalchemiops.jax.interactions.electrostatics import (
                compute_bspline_moduli_1d,
                estimate_ewald_parameters,
                estimate_pme_parameters,
                ewald_real_space,
                ewald_reciprocal_space,
                ewald_summation,
                generate_k_vectors_ewald_summation,
                generate_k_vectors_pme,
                particle_mesh_ewald,
                pme_reciprocal_space,
            )

            api.update(
                {
                    "ewald_summation": ewald_summation,
                    "ewald_real_space": ewald_real_space,
                    "ewald_reciprocal_space": ewald_reciprocal_space,
                    "particle_mesh_ewald": particle_mesh_ewald,
                    "pme_reciprocal_space": pme_reciprocal_space,
                    "generate_k_vectors_pme": generate_k_vectors_pme,
                    "generate_k_vectors_ewald_summation": generate_k_vectors_ewald_summation,
                    "estimate_pme_parameters": estimate_pme_parameters,
                    "estimate_ewald_parameters": estimate_ewald_parameters,
                    "compute_bspline_moduli_1d": compute_bspline_moduli_1d,
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
    # Batch timing: sync only once before start and once after end.
    sync_gpu()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_runs):
        fn()  # NO sync inside loop
    end.record()

    sync_gpu()

    elapsed_ms = start.elapsed_time(end)
    return (elapsed_ms / 1000.0) / num_runs  # seconds per run


def cuda_timed_runs(
    fn: Callable[[], Any],
    num_runs: int,
    warmup_runs: int = 3,
    backend: str = "torch",
) -> float:
    """Time ``fn`` and return the mean seconds per call, dispatching on backend.

    Torch and Warp use CUDA events (:func:`cuda_timed_batch`); JAX uses
    wall-clock + ``jax.block_until_ready`` (:func:`jax_timed_batch`). See the
    module docstring for the batch-timing rationale.

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
        ``'torch'``, ``'warp'``, or ``'jax'``.

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


def _jax_compile_warmups(warmup_runs: int) -> int:
    """Return warmup calls needed to keep JIT compile outside timed regions."""
    return max(1, int(warmup_runs))


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
    for _ in range(_jax_compile_warmups(warmup_runs)):
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


def jax_timed_serial(
    fn: Callable[[], Any], num_runs: int, warmup_runs: int = 3
) -> float:
    """Time JAX calls while blocking after every timed invocation.

    This fallback is slower as a timing harness, but it avoids holding a queue
    of allocation-heavy outputs live when the normal batched dispatch pattern
    exceeds the device memory envelope.
    """
    import time

    import jax

    for _ in range(_jax_compile_warmups(warmup_runs)):
        out = fn()
        jax.block_until_ready(out)

    elapsed = 0.0
    for _ in range(num_runs):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        elapsed += time.perf_counter() - t0
    return elapsed / num_runs


def jax_timed_stateful(
    step: Callable[[Any], Any],
    state: Any,
    num_runs: int,
    warmup_runs: int = 3,
) -> tuple[float, Any]:
    """Time a JAX step that threads donated buffers through each call.

    This is for JAX wrappers whose recommended steady-state contract is
    ``state = step(state)`` with output buffers supplied as JIT arguments and
    donated. Warmup blocks after each call so compile time does not bleed into
    timing; the timed loop dispatches back-to-back dependent calls and blocks
    on the final state.

    Parameters
    ----------
    step : callable
        Function accepting the previous state and returning the next state.
    state : Any
        Initial state, usually a pytree of JAX arrays.
    num_runs : int
        Number of timing iterations.
    warmup_runs : int, default=3
        Number of warmup calls. First warmup triggers JIT compile.

    Returns
    -------
    tuple[float, Any]
        Mean time per run in seconds and the final state.
    """
    import time

    import jax

    for _ in range(_jax_compile_warmups(warmup_runs)):
        state = step(state)
        jax.block_until_ready(state)

    t0 = time.perf_counter()
    for _ in range(num_runs):
        state = step(state)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - t0
    return elapsed / num_runs, state


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
    sync_gpu()
    mem_peak = torch.cuda.max_memory_allocated()
    mem_info: MemInfo = {
        "mem_delta_mb": (mem_peak - mem_before) / 1024**2,
        "mem_peak_gb": mem_peak / 1024**3,
    }
    return result, mem_info


def measure_memory_jax(fn: Callable[[], Any], jax_module: Any) -> tuple[Any, MemInfo]:
    """Return unavailable JAX memory info without executing ``fn``.

    JAX memory is not tracked by this suite. XLA's BFC pool preallocates
    a large fraction of VRAM at process start and reuses across calls,
    which makes NVML-based deltas and ``mem_peak_gb`` both uninformative.
    Running a large JAX benchmark solely to return ``NaN`` memory can leave
    output buffers live before timing begins, so this helper intentionally
    does not call ``fn``. Kept as a thin wrapper for API symmetry with
    :func:`measure_memory_torch`.

    Parameters
    ----------
    fn : callable
        Zero-argument function returning a jax.Array (or pytree). Accepted
        for API symmetry; it is not executed.
    jax_module : module
        The ``jax`` module (typically ``lazy_import_jax()["jax"]``),
        accepted for API symmetry; it is not used.

    Returns
    -------
    (result, mem_info)
        ``result`` is always ``None``.
        ``mem_info`` is always ``{"mem_delta_mb": NaN, "mem_peak_gb": NaN}``.
    """
    return None, {"mem_delta_mb": math.nan, "mem_peak_gb": math.nan}


# =============================================================================
# Result Building & CSV Output
# =============================================================================

_PROVENANCE_FIELDS = (
    "provenance_version",
    "run_id",
    "gpu_context",
    "software_context",
)
_PRESERVED_PROVENANCE_FIELDS = (*_PROVENANCE_FIELDS, "input_context")
_ROW_PROVENANCE_FIELDS = (
    *_PRESERVED_PROVENANCE_FIELDS,
    "execution_context",
    "runtime_context",
)
_MEASUREMENT_PROTOCOL_FIELDS = (
    "timing_runs",
    "warmup_runs",
    "compile_policy",
    "derivative_contract",
    "workload",
    "compute_forces",
    "compute_charge_gradients",
    "component_profiled",
)
_PROVENANCE_VERSION = "2"
_RUN_ID_FILENAME = ".benchmark-run-id"
_RUN_IDS_BY_DIRECTORY: dict[Path, str] = {}
_EXECUTION_PROVENANCE: dict[str, str] | None = None
_INPUT_CONTEXT = json.dumps(
    {"artifacts": {}, "metadata": {"benchmark": "unconfigured"}},
    sort_keys=True,
    separators=(",", ":"),
)


def _distribution_version(name: str) -> str:
    """Return an installed distribution version without importing it."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _source_revision() -> dict[str, str]:
    """Return the Git revision and a hash of benchmark-relevant source files."""
    repo_root = Path(__file__).resolve().parents[1]
    source_suffixes = {
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".lock",
        ".py",
        ".sh",
        ".toml",
        ".yaml",
        ".yml",
    }
    head = "unavailable"
    relative_paths: list[bytes] = []
    git_executable = shutil.which("git")
    if git_executable is not None:
        with suppress(OSError, subprocess.CalledProcessError):
            head = subprocess.run(  # noqa: S603 -- fixed local Git query
                [git_executable, "-C", str(repo_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        with suppress(OSError, subprocess.CalledProcessError):
            listed = subprocess.run(  # noqa: S603 -- fixed local Git query
                [
                    git_executable,
                    "-C",
                    str(repo_root),
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                    "--",
                    "benchmarks",
                    "nvalchemiops",
                    "pyproject.toml",
                    "uv.lock",
                ],
                check=True,
                capture_output=True,
            ).stdout
            relative_paths = sorted(path for path in listed.split(b"\0") if path)

    if not relative_paths:
        candidates: list[Path] = []
        for directory in (repo_root / "benchmarks", repo_root / "nvalchemiops"):
            if directory.is_dir():
                candidates.extend(
                    path for path in directory.rglob("*") if path.is_file()
                )
        candidates.extend(
            path
            for path in (repo_root / "pyproject.toml", repo_root / "uv.lock")
            if path.is_file()
        )
        relative_paths = sorted(
            os.fsencode(path.relative_to(repo_root).as_posix())
            for path in candidates
            if path.suffix in source_suffixes
        )

    if not relative_paths:
        return {"git_head": head, "source_sha256": "unknown"}

    digest = hashlib.sha256()
    for raw_path in relative_paths:
        relative_path = os.fsdecode(raw_path)
        path = repo_root / relative_path
        if path.suffix not in source_suffixes or not path.is_file():
            continue
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"git_head": head, "source_sha256": digest.hexdigest()}


def _text(value: Any) -> str:
    """Normalize provenance values returned as either bytes or strings."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _gpu_context() -> str:
    """Return hardware properties that must match across reportable shards."""
    context: dict[str, Any] = {
        "name": "unavailable",
        "compute_capability": "unavailable",
        "memory_bytes": "unavailable",
        "driver": "unavailable",
    }
    try:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        context.update(
            {
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "memory_bytes": properties.total_memory,
            }
        )
    except (AssertionError, RuntimeError):
        pass

    if _PYNVML_AVAILABLE:
        with suppress(Exception):
            context["driver"] = _text(_pynvml.nvmlSystemGetDriverVersion())

    return json.dumps(context, sort_keys=True, separators=(",", ":"))


def _execution_context() -> str:
    """Return the node and physical device identity for one benchmark shard."""
    context = {
        "host": platform.node() or "unknown",
        "gpu_uuid": "unavailable",
    }
    try:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        context["gpu_uuid"] = _text(getattr(properties, "uuid", "unavailable"))
    except (AssertionError, RuntimeError):
        pass
    return json.dumps(context, sort_keys=True, separators=(",", ":"))


def _runtime_context() -> str:
    """Return backend runtime settings that may affect benchmark behavior."""
    names = (
        "JAX_DISABLE_JIT",
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
        "TF_GPU_ALLOCATOR",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
    context = {name: os.environ.get(name, "<unset>") for name in names}
    jax_module = sys.modules.get("jax")
    if jax_module is None:
        context["JAX_DISABLE_JIT_ACTUAL"] = "<not-imported>"
        context["JAX_ENABLE_X64_ACTUAL"] = "<not-imported>"
    else:
        context["JAX_DISABLE_JIT_ACTUAL"] = "unavailable"
        context["JAX_ENABLE_X64_ACTUAL"] = "unavailable"
        with suppress(AttributeError, TypeError, ValueError):
            context["JAX_DISABLE_JIT_ACTUAL"] = str(
                bool(jax_module.config.jax_disable_jit)
            )
        with suppress(AttributeError, TypeError, ValueError):
            context["JAX_ENABLE_X64_ACTUAL"] = str(
                bool(jax_module.config.jax_enable_x64)
            )
    return json.dumps(context, sort_keys=True, separators=(",", ":"))


def _artifact_fingerprint(path: Path) -> dict[str, Any]:
    """Return a content fingerprint without embedding machine-specific paths."""
    if not path.exists():
        return {"state": "missing"}

    digest = hashlib.sha256()
    if path.is_file():
        data = path.read_bytes()
        digest.update(data)
        return {
            "state": "file",
            "bytes": len(data),
            "sha256": digest.hexdigest(),
        }

    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    total_bytes = 0
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode()
        data = candidate.read_bytes()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
        total_bytes += len(data)
    return {
        "state": "directory",
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def configure_input_provenance(
    artifacts: Mapping[str, Path | str] | None = None,
    *,
    metadata_values: Mapping[str, object] | None = None,
) -> None:
    """Set content-addressed external-input provenance for subsequent rows.

    Artifact labels are stored, but absolute paths are intentionally omitted so
    compatible scheduler nodes can use different scratch mount points.

    Parameters
    ----------
    artifacts : mapping, optional
        Stable artifact label to file or directory path.
    metadata_values : mapping, optional
        Additional non-secret workload identifiers to record as strings.
    """
    global _INPUT_CONTEXT

    artifact_context = {
        str(label): _artifact_fingerprint(Path(path).expanduser())
        for label, path in sorted((artifacts or {}).items())
    }
    metadata_context = {
        str(key): str(value) for key, value in sorted((metadata_values or {}).items())
    }
    _INPUT_CONTEXT = json.dumps(
        {"artifacts": artifact_context, "metadata": metadata_context},
        sort_keys=True,
        separators=(",", ":"),
    )


def _software_context() -> str:
    """Return versions and source identity shared by all benchmark backends."""
    context = {
        "cuda": str(torch.version.cuda or "unavailable"),
        "jax": _distribution_version("jax"),
        "jaxlib": _distribution_version("jaxlib"),
        "nvalchemi_toolkit_ops": _distribution_version("nvalchemi-toolkit-ops"),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "warp": _distribution_version("warp-lang"),
        **_source_revision(),
    }
    return json.dumps(context, sort_keys=True, separators=(",", ":"))


_LOADED_SOFTWARE_CONTEXT = _software_context()


def _execution_provenance() -> dict[str, str]:
    """Collect stable provenance once and current runtime settings per write."""
    global _EXECUTION_PROVENANCE

    if _EXECUTION_PROVENANCE is None:
        _EXECUTION_PROVENANCE = {
            "provenance_version": _PROVENANCE_VERSION,
            "gpu_context": _gpu_context(),
            "software_context": _LOADED_SOFTWARE_CONTEXT,
            "execution_context": _execution_context(),
        }
    return {**_EXECUTION_PROVENANCE, "runtime_context": _runtime_context()}


def _run_id_for_directory(output_dir: Path) -> str:
    """Return the UUID shared by every backend writing to ``output_dir``."""
    output_dir = output_dir.resolve()
    requested_run_id = os.environ.get("BENCHMARK_RUN_ID", "").strip()
    if requested_run_id:
        try:
            uuid.UUID(requested_run_id)
        except ValueError as exc:
            raise RuntimeError("BENCHMARK_RUN_ID must be a valid UUID") from exc
    if output_dir in _RUN_IDS_BY_DIRECTORY:
        run_id = _RUN_IDS_BY_DIRECTORY[output_dir]
        if requested_run_id and requested_run_id != run_id:
            raise RuntimeError(
                f"BENCHMARK_RUN_ID does not match the active run in {output_dir}"
            )
        return run_id

    marker_path = output_dir / _RUN_ID_FILENAME
    marker_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import fcntl

        with marker_path.open("a+", encoding="ascii") as marker:
            fcntl.flock(marker.fileno(), fcntl.LOCK_EX)
            marker.seek(0)
            run_id = marker.read().strip()
            if run_id and requested_run_id and run_id != requested_run_id:
                raise RuntimeError(f"BENCHMARK_RUN_ID does not match {marker_path}")
            if not run_id:
                run_id = requested_run_id or uuid.uuid4().hex
                marker.seek(0)
                marker.write(f"{run_id}\n")
                marker.truncate()
                marker.flush()
                os.fsync(marker.fileno())
            fcntl.flock(marker.fileno(), fcntl.LOCK_UN)
    except ImportError:
        try:
            with marker_path.open("x", encoding="ascii") as marker:
                run_id = requested_run_id or uuid.uuid4().hex
                marker.write(f"{run_id}\n")
        except FileExistsError:
            run_id = marker_path.read_text(encoding="ascii").strip()
            if requested_run_id and run_id != requested_run_id:
                raise RuntimeError(f"BENCHMARK_RUN_ID does not match {marker_path}")

    try:
        uuid.UUID(run_id)
    except ValueError as exc:
        raise RuntimeError(f"Invalid benchmark run ID in {marker_path}") from exc

    _RUN_IDS_BY_DIRECTORY[output_dir] = run_id
    return run_id


def _run_provenance(output_path: Path | str) -> dict[str, str]:
    """Return the provenance stamped onto rows written to ``output_path``."""
    output_path = Path(output_path)
    return {
        **_execution_provenance(),
        "input_context": _INPUT_CONTEXT,
        "run_id": _run_id_for_directory(output_path.parent),
    }


def _stamp_provenance(results: list[dict], provenance: dict[str, str]) -> list[dict]:
    """Copy result rows and stamp authoritative run provenance."""
    return [{**result, **provenance} for result in results]


def _validate_preserved_provenance(
    rows: list[dict[str, str]],
    provenance: dict[str, str],
    output_path: Path,
) -> None:
    """Reject backend merges that would preserve incompatible rows."""
    if not rows:
        return

    missing = sorted(
        field
        for field in _PRESERVED_PROVENANCE_FIELDS
        if any(not row.get(field) for row in rows)
    )
    if missing:
        fields = ", ".join(missing)
        raise ValueError(
            f"Refusing to preserve rows with missing provenance in {output_path}: "
            f"{fields}. Use a fresh output directory or replace the complete CSV."
        )

    mismatched = sorted(
        field
        for field in _PRESERVED_PROVENANCE_FIELDS
        if any(row[field] != provenance[field] for row in rows)
    )
    if mismatched:
        fields = ", ".join(mismatched)
        raise ValueError(
            f"Refusing to combine benchmark rows from different contexts in "
            f"{output_path}; mismatched provenance: {fields}. Use a fresh output "
            "directory or replace the complete CSV."
        )


def _validate_backend_runtime_contexts(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    """Reject rows that mix runtime settings within one backend."""
    contexts: dict[str, str] = {}
    for row in rows:
        backend = str(row.get("backend", ""))
        runtime_context = str(row.get("runtime_context", ""))
        if not runtime_context:
            raise ValueError(
                f"Refusing to write rows with missing runtime_context in {output_path}"
            )
        previous = contexts.setdefault(backend, runtime_context)
        if runtime_context != previous:
            raise ValueError(
                f"Refusing to combine {backend} rows with different runtime settings "
                f"in {output_path}"
            )


def _measurement_protocol(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> dict[str, str]:
    """Return one consistent measurement protocol for a result set."""
    protocol: dict[str, str] = {}
    for field in _MEASUREMENT_PROTOCOL_FIELDS:
        if not any(field in row for row in rows):
            continue
        values = {str(row.get(field, "")).strip() for row in rows}
        if "" in values:
            raise ValueError(
                f"Refusing to combine rows with missing measurement protocol "
                f"field {field} in {output_path}"
            )
        if len(values) != 1:
            raise ValueError(
                f"Refusing to combine rows with different measurement protocol "
                f"values for {field} in {output_path}: {sorted(values)}"
            )
        protocol[field] = values.pop()
    return protocol


@contextmanager
def _output_file_lock(output_path: Path) -> Iterator[None]:
    """Serialize read-modify-write operations for one benchmark CSV."""
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    try:
        import fcntl
    except ImportError:
        yield
        return

    with lock_path.open("a+", encoding="ascii") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_csv_atomic(
    output_path: Path,
    fieldnames: list[str],
    rows: Sequence[dict],
) -> None:
    """Replace one CSV atomically after fully writing and syncing a temp file."""
    temp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temp_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temp_path, output_path)
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()


def validate_result_files(
    csv_paths: Sequence[Path | str],
    *,
    expected_run_id: str | None = None,
) -> dict[str, int]:
    """Validate provenance and failure metadata across reportable CSV files.

    ``execution_context`` is required on every row but may differ across cluster
    shards. The run ID, hardware class, software/source context, and provenance
    schema must be identical across all files. External input fingerprints must
    be uniform within each CSV, and runtime/allocator settings must be uniform
    across shards of the same backend.
    """
    paths = [Path(path) for path in csv_paths]
    expected_context: dict[str, str] = {}
    expected_protocol: dict[str, str] = {}
    backend_runtime_contexts: dict[str, str] = {}
    counts = {"rows": 0, "successes": 0, "failures": 0}

    for path in paths:
        with path.open(newline="") as input_file:
            rows = list(csv.DictReader(input_file))
        if not rows:
            raise ValueError(f"Empty benchmark CSV: {path}")

        for field, value in _measurement_protocol(rows, path).items():
            previous = expected_protocol.setdefault(field, value)
            if value != previous:
                raise ValueError(
                    f"Mixed benchmark measurement protocol for {field} in {path}: "
                    f"expected {previous!r}, found {value!r}"
                )

        file_input_context: str | None = None
        for row_number, row in enumerate(rows, start=2):
            location = f"{path}:{row_number}"
            missing = [field for field in _ROW_PROVENANCE_FIELDS if not row.get(field)]
            if missing:
                raise ValueError(
                    f"Missing benchmark provenance at {location}: " + ", ".join(missing)
                )

            if expected_run_id is not None and row["run_id"] != expected_run_id:
                raise ValueError(
                    f"Run ID mismatch at {location}: expected {expected_run_id}, "
                    f"found {row['run_id']}"
                )

            for field in _PROVENANCE_FIELDS:
                previous = expected_context.setdefault(field, row[field])
                if row[field] != previous:
                    raise ValueError(
                        f"Mixed benchmark {field} at {location}: expected "
                        f"{previous!r}, found {row[field]!r}"
                    )

            if file_input_context is None:
                file_input_context = row["input_context"]
            elif row["input_context"] != file_input_context:
                raise ValueError(
                    f"Mixed benchmark input_context at {location}; one CSV cannot "
                    "combine different external artifacts"
                )

            backend = row.get("backend", "")
            previous_runtime = backend_runtime_contexts.setdefault(
                backend, row["runtime_context"]
            )
            if row["runtime_context"] != previous_runtime:
                raise ValueError(
                    f"Mixed {backend} runtime_context at {location}: allocator or "
                    "backend environment settings differ across shards"
                )

            success_value = row.get("success", "").strip().lower()
            if success_value not in {"true", "false"}:
                raise ValueError(
                    f"Invalid benchmark success value at {location}: "
                    f"{row.get('success')!r}"
                )
            counts["rows"] += 1
            if success_value == "true":
                counts["successes"] += 1
                continue

            counts["failures"] += 1
            missing_failure = [
                field
                for field in ("error_type", "error")
                if not row.get(field, "").strip()
            ]
            if missing_failure:
                raise ValueError(
                    f"Missing failure metadata at {location}: "
                    + ", ".join(missing_failure)
                )

    return counts


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
    timing_runs: int,
    warmup_runs: int,
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
        Method name (e.g. 'naive_neighbor_list', 'cell_list', 'dftd3',
        'pme', 'ewald').
    backend : str, default='torch'
        Framework backend ('torch', 'warp', or 'jax').
    atoms_per_system : int
        Atoms in each individual system.
    batch_size : int
        Number of systems in the batch.
    total_atoms : int
        atoms_per_system * batch_size.
    time_seconds : float
        Mean time per call in seconds.
    timing_runs : int
        Number of measured calls used for the timing row.
    warmup_runs : int
        Number of untimed warmup calls used before measurement.
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

    timing_runs = _require_run_count("timing_runs", timing_runs)
    warmup_runs = _require_run_count("warmup_runs", warmup_runs)

    error = extra.pop("error", "")
    error_type = extra.pop("error_type", "")
    timing_method = extra.pop("timing_method", None) or _timing_method_for_backend(
        backend
    )
    compile_policy = extra.pop("compile_policy", None) or "warmup_excluded"

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
        # Timing metadata
        "timing_runs": timing_runs,
        "warmup_runs": warmup_runs,
        "timing_method": timing_method,
        "compile_policy": compile_policy,
        # Status
        "success": success,
        "error": error,
        "error_type": error_type,
    }

    # Add any extra method-specific fields (cutoff, accuracy, time_d3_us_per_atom)
    result.update(extra)
    return result


def _require_run_count(name: str, value: Any) -> int:
    """Validate run-count metadata before writing result rows."""
    if value is None:
        raise ValueError(f"{name} is required for benchmark result rows")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


def build_failure_result(
    *,
    error: str,
    error_type: str,
    mem_info: MemInfo | None = None,
    **kwargs: Any,
) -> dict:
    """Build a standardized failed benchmark row."""
    if mem_info is None:
        mem_info = {"mem_delta_mb": math.nan, "mem_peak_gb": math.nan}
    error_type = str(error_type).strip()
    if not error_type:
        raise ValueError("error_type is required for failed benchmark rows")
    error = str(error)
    if not error.strip():
        error = error_type
    error = _sanitize_error_message(error, error_type)
    result = build_result(
        time_seconds=math.nan,
        mem_info=mem_info,
        success=False,
        error=error,
        error_type=error_type,
        **kwargs,
    )
    result["time_us_per_atom"] = math.nan
    result["throughput_atoms_per_sec"] = math.nan
    return result


def _timing_method_for_backend(backend: str) -> str:
    """Return the timing method label stored in CSV rows."""
    if backend == "jax":
        return "jax_wall_block_until_ready"
    if backend == "torch":
        return "torch_cuda_events"
    if backend == "warp":
        return "warp_cuda_events"
    return "unknown"


def _sanitize_error_message(error: str, error_type: str) -> str:
    """Keep user-facing CSV errors concise and stable."""
    if error_type == "OutOfMemoryError":
        return "Out of memory during benchmark execution; see run logs for backend details."
    return error


def failure_error_type(exc: BaseException) -> str:
    """Return a stable CSV error type for framework-specific failures."""
    message = str(exc).lower()
    exc_type = type(exc).__name__
    oom_tokens = (
        "out of memory",
        "resource_exhausted",
        "cuda error: out of memory",
        "cuda_error_out_of_memory",
    )
    if isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)) or any(
        token in message for token in oom_tokens
    ):
        return "OutOfMemoryError"
    if isinstance(exc, ValueError) and "safe linear launch limit" in message:
        return "UnsupportedConfiguration"
    return exc_type


def build_skipped_result(
    *,
    reason: str,
    policy: str = "SkippedByPolicy",
    **kwargs: Any,
) -> dict:
    """Build a standardized row for a planned case skipped before allocation."""
    return build_failure_result(
        error=reason,
        error_type=policy,
        **kwargs,
    )


def save_results(
    results: list[dict],
    output_path: Path | str,
    *,
    append: bool = False,
    replace_backend: str | None = None,
) -> None:
    """Save benchmark results to CSV.

    By default, an existing file is replaced so a rerun cannot silently mix
    stale rows, metadata, or backend versions. Pass ``append=True`` only for
    intentional append workflows. Pass ``replace_backend`` when Torch and JAX
    rows intentionally share a docs CSV: existing rows for that backend are
    replaced, rows for other backends are preserved only when their run, GPU,
    and software provenance matches. A hidden run-ID marker in the output
    directory lets separate backend processes share the same run identity.

    Parameters
    ----------
    results : list[dict]
        List of result dicts from build_result().
    output_path : str or Path
        Path to output CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if append and replace_backend is not None:
        raise ValueError("append and replace_backend are mutually exclusive")

    if not results:
        print(f"No results to save to {output_path}")
        return

    provenance = _run_provenance(output_path)
    stamped_results = _stamp_provenance(results, provenance)

    fieldnames: list[str] = []
    for result in stamped_results:
        for key in result:
            if key not in fieldnames:
                fieldnames.append(key)

    with _output_file_lock(output_path):
        if output_path.exists() and replace_backend is not None:
            with output_path.open(newline="") as input_file:
                reader = csv.DictReader(input_file)
                existing_fields = reader.fieldnames or []
                existing_rows = [
                    row for row in reader if row.get("backend") != replace_backend
                ]

            _validate_preserved_provenance(existing_rows, provenance, output_path)
            combined_rows = [*existing_rows, *stamped_results]
            _validate_backend_runtime_contexts(combined_rows, output_path)
            _measurement_protocol(combined_rows, output_path)

            combined_fields = list(existing_fields)
            for field in fieldnames:
                if field not in combined_fields:
                    combined_fields.append(field)

            _write_csv_atomic(
                output_path,
                combined_fields,
                combined_rows,
            )
            print(
                f"Replaced {replace_backend} rows with {len(stamped_results)} "
                f"results in {output_path}"
            )
            return

        if output_path.exists() and append:
            with output_path.open(newline="") as input_file:
                reader = csv.DictReader(input_file)
                existing_fields = reader.fieldnames or []
                existing_rows = list(reader)

            _validate_preserved_provenance(existing_rows, provenance, output_path)
            combined_rows = [*existing_rows, *stamped_results]
            _validate_backend_runtime_contexts(combined_rows, output_path)
            _measurement_protocol(combined_rows, output_path)
            combined_fields = list(existing_fields)
            for field in fieldnames:
                if field not in combined_fields:
                    combined_fields.append(field)

            _write_csv_atomic(
                output_path,
                combined_fields,
                combined_rows,
            )
            print(f"Appended {len(stamped_results)} results to {output_path}")
            return

        _validate_backend_runtime_contexts(stamped_results, output_path)
        _measurement_protocol(stamped_results, output_path)
        _write_csv_atomic(output_path, fieldnames, stamped_results)

        print(f"Saved {len(stamped_results)} results to {output_path}")


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
    *,
    filename: str = "RUN_LOG.md",
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
    filename : str, default="RUN_LOG.md"
        Basename for the run log. Parallel shards should use distinct names.
    """
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
    jax_env_names = (
        "JAX_ENABLE_X64",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "TF_GPU_ALLOCATOR",
    )
    jax_env = {name: os.environ.get(name, "<unset>") for name in jax_env_names}
    cache_env_names = (
        "XDG_CACHE_HOME",
        "UV_CACHE_DIR",
        "PRE_COMMIT_HOME",
        "WARP_CACHE_PATH",
        "TORCH_EXTENSIONS_DIR",
        "PYTORCH_KERNEL_CACHE_PATH",
        "JAX_COMPILATION_CACHE_DIR",
        "MPLCONFIGDIR",
    )
    cache_env = {name: os.environ.get(name, "<unset>") for name in cache_env_names}

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
        "## JAX/XLA Environment",
        "",
        *[f"- **{name}**: {value}" for name, value in jax_env.items()],
        "",
        "## Runtime Cache Environment",
        "",
        *[f"- **{name}**: {value}" for name, value in cache_env.items()],
        "",
        "## Timing Methodology",
        "",
        "Reported timings exclude warm-up/compile/load iterations.",
        "",
        "Torch/Warp CUDA timing pattern:",
        "```",
        "sync() -> start.record() -> N x fn() -> end.record() -> sync()",
        "```",
        "",
        "JAX timing pattern:",
        "```",
        "block_until_ready(warmup) -> t0 = perf_counter() -> N x fn() -> block_until_ready(last)",
        "```",
        "",
        "Returns mean steady-state time per call. CSV rows store "
        "`timing_method` and `compile_policy`; failed/OOM rows are retained "
        "with `success=False` and omitted from plots.",
        "",
        "## Files",
        "",
        "CSV naming: `{module}-{system}-{scaling-mode}.csv`",
        "",
        "| Column | Description |",
        "|--------|-------------|",
        "| `time_us_per_atom` | Time per atom [microseconds] |",
        "| `throughput_atoms_per_sec` | Atoms processed per second |",
        "| `mem_delta_mb` | Memory delta from the pre-timing measurement call [MB] (torch only; NaN for jax) |",
        "| `mem_peak_gb` | Peak GPU VRAM usage [GB] (torch only; NaN for jax) |",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "uv sync --extra torch --extra jax --group docs",
        "uv run python -m benchmarks.benchmark_suite --benchmark all",
        "```",
        "",
    ]

    if extra_info:
        lines.append("## Additional Info")
        lines.append("")
        for k, v in extra_info.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    if not filename or Path(filename).name != filename:
        raise ValueError("run-log filename must be a non-empty basename")
    log_path = run_dir / filename
    log_path.write_text("\n".join(lines))
