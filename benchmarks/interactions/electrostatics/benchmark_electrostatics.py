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
Electrostatics Benchmark
========================

CLI tool to benchmark electrostatic interaction methods (Ewald summation, Ewald
with slab correction, PME, PME with slab correction, and DSF) and generate CSV
files for documentation.
Results are saved with GPU- and dtype-specific naming:
`electrostatics_benchmark_<method>_<backend>_<dtype>_<gpu_sku>.csv`

Supports multiple backends:
1. torch (Warp kernels): Custom implementation using PyTorch + Warp
2. jax: Custom implementation using JAX + Warp (via XLA FFI)
3. torchpme: Reference PyTorch implementation
4. torch_dsf: Pure PyTorch DSF reference (torch.compile)

Methods:
- Ewald summation
- Ewald summation with 2D slab correction
- PME (Particle Mesh Ewald)
- PME with 2D slab correction
- DSF (Damped Shifted Force)

Usage:
    python benchmark_electrostatics.py --config benchmark_config.yaml --output-dir ./results
    python benchmark_electrostatics.py --config benchmark_config.yaml --backend jax
    python benchmark_electrostatics.py --config benchmark_config.yaml --backend torchpme --method ewald
    python benchmark_electrostatics.py --config benchmark_config.yaml --backend torch --method ewald_slab
    python benchmark_electrostatics.py --config benchmark_config.yaml --backend torch --method pme_slab
    python benchmark_electrostatics.py --config benchmark_config.yaml --backend jax --method ewald_slab
    python benchmark_electrostatics.py --config benchmark_config.yaml --backend jax --method pme_slab
    python benchmark_electrostatics.py --config benchmark_config.yaml --method dsf --backend both
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import sys
import traceback
import warnings
from pathlib import Path
from typing import Literal

# Add repo root to path for imports (4 levels up from this script)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import yaml

from benchmarks.systems import create_crystal_system
from benchmarks.utils import BackendType, BenchmarkTimer

BENCHMARK_CSV_FIELDNAMES = [
    "total_atoms",
    "batch_size",
    "supercell_size",
    "mode",
    "method",
    "backend",
    "component",
    "derivative_contract",
    "workload",
    "compute_forces",
    "compute_virial",
    "neighbor_format",
    "dtype",
    "config_path",
    "config_sha256",
    "accuracy",
    "real_space_cutoff",
    "cache_mode",
    "compile_policy",
    "backend_framework",
    "median_time_ms",
    "peak_memory_mb",
    "compile_ms",
    "warp_compile_ms",
    "framework_compile_ms",
    "framework",
    "success",
    "error",
    "error_type",
]

DerivativeContract = Literal["energy_autograd", "legacy_direct"]
BenchmarkWorkload = Literal[
    "forward",
    "backward",
    "double_backward",
    "legacy_direct",
    "autograd_reference",
]


def benchmark_output_file(
    output_dir: Path,
    method: str,
    backend: str,
    dtype_str: str,
    gpu_sku: str,
) -> Path:
    """Return the CSV path for one benchmark result group."""
    return (
        output_dir
        / f"electrostatics_benchmark_{method}_{backend}_{dtype_str}_{gpu_sku}.csv"
    )


# -- Torch backend -----------------------------------------------------------
try:
    import torch
    import warp as wp

    _torch_electrostatics = importlib.import_module(
        "nvalchemiops.torch.interactions.electrostatics"
    )
    _torch_neighbors = importlib.import_module("nvalchemiops.torch.neighbors")
    _neighbor_utils = importlib.import_module("nvalchemiops.neighbors.neighbor_utils")
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    wp = None  # type: ignore
    _torch_electrostatics = None
    _torch_neighbors = None
    _neighbor_utils = None

# -- JAX backend --------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp

    _jax_electrostatics = importlib.import_module(
        "nvalchemiops.jax.interactions.electrostatics"
    )
    _jax_neighbors = importlib.import_module("nvalchemiops.jax.neighbors")
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    _jax_electrostatics = None
    _jax_neighbors = None


SLAB_AXIS = 2
SLAB_VACUUM_FACTOR = 3.0
SLAB_METHODS = ("ewald_slab", "pme_slab")
EWALD_METHODS = ("ewald", "ewald_slab")
DEFAULT_METHODS = ["ewald", "pme"]
ALL_METHODS = ["ewald", "ewald_slab", "pme", "pme_slab", "dsf"]
CUSTOM_EWALD_PME_BACKENDS = ("torch", "jax")


def _electrostatic_method_family(method: str) -> Literal["ewald", "pme"]:
    """Return the parameter family used by an electrostatics benchmark method."""
    if method in EWALD_METHODS:
        return "ewald"
    if method in ("pme", "pme_slab"):
        return "pme"
    raise ValueError(f"Method {method!r} does not use Ewald/PME parameters")


def _system_cache_key(method: str) -> str:
    """Return the prepared-system cache key for an electrostatics method."""
    family = _electrostatic_method_family(method)
    suffix = "_slab" if method in SLAB_METHODS else ""
    return f"{family}{suffix}"


def _neighbor_cutoff_from_params(
    params: dict, family: Literal["ewald", "pme"]
) -> float:
    """Return the real-space neighbor cutoff for a method parameter family."""
    return float(params[family]["cutoff"])


def resolve_methods(
    cli_method: str | None, config_methods: list[str] | None
) -> list[str]:
    """Return the ordered benchmark methods requested by CLI/config."""
    if cli_method is not None:
        if cli_method == "both":
            return list(DEFAULT_METHODS)
        if cli_method == "all":
            return list(ALL_METHODS)
        return [cli_method]

    if not config_methods:
        return list(DEFAULT_METHODS)
    return list(config_methods)


def get_backends_for_method(cli_backend: str, method: str) -> list[str]:
    """Return concrete benchmark backends for one method/backend request."""
    if cli_backend == "both":
        if method in ("ewald", "pme"):
            result = ["torch"]
            if JAX_AVAILABLE:
                result.append("jax")
            if TORCHPME_AVAILABLE:
                result.append("torchpme")
            return result
        if method in SLAB_METHODS:
            result = ["torch"]
            if JAX_AVAILABLE:
                result.append("jax")
            return result
        if method == "dsf":
            return ["torch", "torch_dsf"]
    elif cli_backend == "torch":
        return ["torch"]
    elif cli_backend == "jax":
        if method in ("ewald", "ewald_slab", "pme", "pme_slab"):
            return ["jax"]
        return []
    elif cli_backend == "torchpme":
        if method in SLAB_METHODS:
            raise ValueError(f"torchpme does not support slab method {method!r}")
        if method in ("ewald", "pme"):
            return ["torchpme"] if TORCHPME_AVAILABLE else []
        return []
    elif cli_backend == "torch_dsf":
        if method == "dsf":
            return ["torch_dsf"]
        return []
    return ["torch"]


def compile_policy_for_backend(backend: str, torch_compile: bool = False) -> str:
    """Return the framework compilation policy label for a backend."""
    if torch_compile and backend in ("torch", "torchpme", "torch_dsf"):
        return "torch.compile"
    if backend == "jax":
        return "jax.jit"
    return "eager"


def resolve_derivative_contract(
    cli_contract: str | None,
    config_contract: str | None,
) -> DerivativeContract:
    """Return the benchmark derivative contract requested by CLI/config."""
    contract = cli_contract or config_contract or "energy_autograd"
    if contract not in ("energy_autograd", "legacy_direct"):
        raise ValueError(
            "derivative_contract must be 'energy_autograd' or 'legacy_direct'"
        )
    return contract


def benchmark_workloads(
    *,
    method: str,
    backend: str,
    derivative_contract: DerivativeContract,
    compute_forces: bool,
    compute_virial: bool,
) -> list[BenchmarkWorkload]:
    """Return the workload rows to emit for one benchmark request."""
    if method == "dsf" or backend in ("torchpme", "torch_dsf"):
        return ["autograd_reference"]
    if derivative_contract == "legacy_direct":
        return ["legacy_direct"]
    if method not in ("ewald", "ewald_slab", "pme", "pme_slab"):
        return ["forward"]

    workloads: list[BenchmarkWorkload] = ["forward"]
    if compute_forces or compute_virial:
        workloads.append("backward")
        if backend == "torch":
            workloads.append("double_backward")
    return workloads


def benchmark_result_row(
    *,
    system_data: dict,
    method: str,
    backend: str,
    component: str,
    compute_forces: bool,
    compute_virial: bool,
    derivative_contract: DerivativeContract = "energy_autograd",
    workload: BenchmarkWorkload = "forward",
    neighbor_format: str,
    torch_compile: bool = False,
    success: bool,
    median_time_ms: float | None = None,
    peak_memory_mb: float | None = None,
    compile_ms: float | None = None,
    warp_compile_ms: float | None = None,
    framework_compile_ms: float | None = None,
    framework: str = "none",
    error: str = "",
    error_type: str = "",
    **extra,
) -> dict:
    """Build one stable benchmark CSV row for success or failure."""
    row = {
        "total_atoms": system_data["total_atoms"],
        "batch_size": system_data.get("batch_size", 1),
        "supercell_size": "",
        "mode": "",
        "method": method,
        "backend": backend,
        "component": component,
        "derivative_contract": derivative_contract,
        "workload": workload,
        "compute_forces": compute_forces,
        "compute_virial": compute_virial,
        "neighbor_format": neighbor_format,
        "dtype": "",
        "config_path": "",
        "config_sha256": "",
        "accuracy": "",
        "real_space_cutoff": "",
        "cache_mode": system_data.get("cache_mode", "precomputed"),
        "compile_policy": compile_policy_for_backend(backend, torch_compile),
        "backend_framework": backend,
        "median_time_ms": (
            float(median_time_ms) if median_time_ms is not None else float("inf")
        ),
        "peak_memory_mb": peak_memory_mb,
        "compile_ms": compile_ms,
        "warp_compile_ms": warp_compile_ms,
        "framework_compile_ms": framework_compile_ms,
        "framework": framework,
        "success": bool(success),
        "error": "" if success else str(error),
        "error_type": "" if success else str(error_type or "Unknown"),
    }
    row.update(extra)
    return row


def annotate_result_row(
    row: dict,
    *,
    supercell_size: int,
    mode: str,
    dtype: str,
    config_path: str,
    config_sha256: str,
    accuracy: float,
    real_space_cutoff: float | None,
) -> dict:
    """Add run-level metadata to an existing benchmark row."""
    row.update(
        {
            "supercell_size": supercell_size,
            "mode": mode,
            "dtype": dtype,
            "config_path": config_path,
            "config_sha256": config_sha256,
            "accuracy": accuracy,
            "real_space_cutoff": "" if real_space_cutoff is None else real_space_cutoff,
        }
    )
    return row


def _get_backend_modules(
    backend: str,
) -> tuple:
    """Return (electrostatics_module, neighbors_module) for *backend*.

    Parameters
    ----------
    backend : str
        ``"torch"`` or ``"jax"``.

    Returns
    -------
    tuple
        ``(electrostatics_module, neighbors_module)``

    Raises
    ------
    ValueError
        If the backend is unknown or unavailable.
    """
    match backend:
        case "torch":
            if _torch_electrostatics is None:
                raise ValueError("torch backend is not available")
            return _torch_electrostatics, _torch_neighbors
        case "jax":
            if _jax_electrostatics is None:
                raise ValueError("jax backend is not available")
            return _jax_electrostatics, _jax_neighbors
        case _:
            raise ValueError(f"Unknown backend: {backend}")


# Optional torchpme imports
try:
    from torchpme import EwaldCalculator, PMECalculator
    from torchpme.potentials import CoulombPotential

    TORCHPME_AVAILABLE = True
except ImportError:
    TORCHPME_AVAILABLE = False
    EwaldCalculator = None
    PMECalculator = None
    CoulombPotential = None


# ==============================================================================
# Utilities
# ==============================================================================


def get_gpu_sku(backend: BackendType) -> str:
    """Get GPU SKU name for filename generation.

    Uses NVML for reliable, backend-agnostic GPU name detection.
    Falls back to "cpu" if no GPU is available.
    """
    has_gpu = False
    match backend:
        case "torch":
            has_gpu = torch is not None and torch.cuda.is_available()
        case "jax":
            try:
                has_gpu = jax is not None and any(
                    d.platform == "gpu" for d in jax.local_devices()
                )
            except Exception:
                has_gpu = False
        case "warp":
            has_gpu = False

    if not has_gpu:
        return "cpu"

    from benchmarks.utils import _nvml_get_gpu_sku

    return _nvml_get_gpu_sku()


def _resolve_backend_type(cli_backend: str) -> BackendType:
    """Map CLI backend string to BackendType."""
    match cli_backend:
        case "torch" | "torchpme" | "torch_dsf" | "both":
            return "torch"
        case "jax":
            return "jax"
        case _:
            raise ValueError(f"Unknown backend: {cli_backend}")


def _check_backend_available(cli_backend: str) -> None:
    """Validate that the requested backend is installed."""
    match cli_backend:
        case "torch" | "torch_dsf" | "both":
            if not TORCH_AVAILABLE:
                print("ERROR: torch backend requested but torch is not installed.")
                sys.exit(1)
        case "jax":
            if not JAX_AVAILABLE:
                print("ERROR: jax backend requested but JAX is not installed.")
                sys.exit(1)
        case "torchpme":
            if not TORCH_AVAILABLE:
                print("ERROR: torchpme backend requires torch.")
                sys.exit(1)
            if not TORCHPME_AVAILABLE:
                print("ERROR: torchpme backend requested but not installed.")
                print("Install via: pip install torch-pme")
                sys.exit(1)


def load_config(config_path: Path) -> dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


# ==============================================================================
# System Generation
# ==============================================================================


def prepare_system_numpy(
    supercell_size: int,
    batch_size: int = 1,
) -> dict:
    """Create crystal system(s) and return as numpy arrays (no backend dependency for data).

    Uses ``create_crystal_system`` internally (which returns torch tensors on CPU),
    then converts to numpy arrays. This decouples geometry generation from the
    compute backend.

    Parameters
    ----------
    supercell_size : int
        Linear dimension of the supercell. For BCC lattice (2 atoms per unit cell),
        each system has 2 * supercell_size³ atoms.
    batch_size : int, default=1
        Number of systems to batch together.

    Returns
    -------
    dict
        Dictionary containing numpy arrays:
        - positions: (N_total, 3) float64
        - charges: (N_total,) float64
        - cell: (batch_size, 3, 3) float64
        - pbc: (batch_size, 3) bool
        - batch_idx: (N_total,) int32 or None (single system)
        - total_atoms: int
        - num_atoms_per_system: int (for BCC: 2 * supercell_size³)
    """
    target_atoms_per_system = 2 * supercell_size**3

    if batch_size == 1:
        system = create_crystal_system(
            target_atoms_per_system,
            lattice_type="bcc",
            lattice_constant=4.14,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        total_atoms = system["num_atoms"]

        return {
            "positions": system["positions"].numpy(),
            "charges": system["atomic_charges"].numpy(),
            "cell": system["cell"].numpy(),  # shape (1, 3, 3)
            "pbc": system["pbc"].numpy()[np.newaxis, :],  # shape (1, 3)
            "batch_idx": None,
            "total_atoms": total_atoms,
            "num_atoms_per_system": total_atoms,
        }
    else:
        all_positions = []
        all_charges = []
        all_cells = []
        all_pbc = []
        batch_idx_list = []

        for i in range(batch_size):
            system = create_crystal_system(
                target_atoms_per_system,
                lattice_type="bcc",
                lattice_constant=4.14,
                device=torch.device("cpu"),
                dtype=torch.float64,
            )
            n_atoms = system["num_atoms"]

            all_positions.append(system["positions"].numpy())
            all_charges.append(system["atomic_charges"].numpy())
            all_cells.append(system["cell"].numpy())  # shape (1, 3, 3)
            all_pbc.append(system["pbc"].numpy())  # shape (3,)
            batch_idx_list.extend([i] * n_atoms)

        positions = np.concatenate(all_positions, axis=0)
        charges = np.concatenate(all_charges, axis=0)
        cells = np.concatenate(all_cells, axis=0)  # shape (batch_size, 3, 3)
        pbc = np.stack(all_pbc, axis=0)  # shape (batch_size, 3)
        batch_idx = np.array(batch_idx_list, dtype=np.int32)
        total_atoms = positions.shape[0]

        return {
            "positions": positions,
            "charges": charges,
            "cell": cells,
            "pbc": pbc,
            "batch_idx": batch_idx,
            "total_atoms": total_atoms,
            "num_atoms_per_system": target_atoms_per_system,
        }


def prepare_slab_system_numpy(
    supercell_size: int,
    batch_size: int = 1,
    vacuum_factor: float = SLAB_VACUUM_FACTOR,
    slab_axis: int = SLAB_AXIS,
) -> dict:
    """Create a slab benchmark system with vacuum and mixed periodicity."""
    np_data = prepare_system_numpy(supercell_size, batch_size=batch_size)

    cell = np_data["cell"].copy()
    cell[:, slab_axis, :] *= vacuum_factor

    pbc = np_data["pbc"].copy()
    pbc[..., slab_axis] = False

    return {**np_data, "cell": cell, "pbc": pbc}


def convert_to_backend(
    np_data: dict,
    backend: str,
    device: str = "cuda",
    dtype_str: str = "float64",
) -> dict:
    """Convert numpy arrays to backend-specific arrays.

    Parameters
    ----------
    np_data : dict
        Output from prepare_system_numpy().
    backend : str
        "torch" or "jax".
    device : str
        Device string (used by torch).
    dtype_str : str
        Dtype string like "float64".

    Returns
    -------
    dict
        Dictionary with backend arrays: positions, charges, cell, pbc, batch_idx, total_atoms.
    """
    result = {
        "total_atoms": np_data["total_atoms"],
        "num_atoms_per_system": np_data["num_atoms_per_system"],
    }

    match backend:
        case "torch":
            dtype = getattr(torch, dtype_str)
            result["positions"] = torch.tensor(
                np_data["positions"], dtype=dtype, device=device
            )
            result["charges"] = torch.tensor(
                np_data["charges"], dtype=dtype, device=device
            )
            result["cell"] = torch.tensor(np_data["cell"], dtype=dtype, device=device)
            result["pbc"] = torch.tensor(
                np_data["pbc"], dtype=torch.bool, device=device
            )
            if np_data["batch_idx"] is not None:
                result["batch_idx"] = torch.tensor(
                    np_data["batch_idx"], dtype=torch.int32, device=device
                )
            else:
                result["batch_idx"] = None
        case "jax":
            dtype = getattr(jnp, dtype_str)
            result["positions"] = jnp.array(np_data["positions"], dtype=dtype)
            result["charges"] = jnp.array(np_data["charges"], dtype=dtype)
            result["cell"] = jnp.array(np_data["cell"], dtype=dtype)
            result["pbc"] = jnp.array(np_data["pbc"], dtype=jnp.bool_)
            if np_data["batch_idx"] is not None:
                result["batch_idx"] = jnp.array(np_data["batch_idx"], dtype=jnp.int32)
            else:
                result["batch_idx"] = None
        case _:
            raise ValueError(f"Unknown backend: {backend}")

    return result


def compute_electrostatics_params(
    backend_data: dict,
    backend: str,
    real_space_cutoff: float | None = None,
    accuracy: float = 1e-4,
) -> dict:
    """Compute Ewald/PME parameters using the appropriate backend.

    Parameters
    ----------
    backend_data : dict
        Output from convert_to_backend(). Must contain positions, cell, and
        optionally batch_idx.
    backend : str
        "torch" or "jax".
    real_space_cutoff : float, optional
        If provided, forwarded to ``estimate_pme_parameters``; ``alpha``
        and mesh dimensions are then derived from this cutoff instead
        of the cost-optimal one. The Ewald-parameter side uses the
        cost-optimal cutoff regardless (Ewald has no auto-cutoff
        equivalent to PME's).
    accuracy : float, default=1e-4
        Target relative force accuracy passed to ``estimate_ewald_parameters``
        and ``estimate_pme_parameters``. Drives the cost-model choice of
        alpha, real-space cutoff, and mesh dimensions.

    Returns
    -------
    dict
        Dictionary containing alpha, k_cutoff, cutoff, mesh_dimensions,
        mesh_spacing, k_vectors_pme, k_squared_pme.
    """
    electrostatics_mod, _ = _get_backend_modules(backend)

    positions = backend_data["positions"]
    cell = backend_data["cell"]
    batch_idx = backend_data["batch_idx"]

    if batch_idx is None:
        ewald_params = electrostatics_mod.estimate_ewald_parameters(
            positions, cell, accuracy=accuracy
        )
        ewald_k_cutoff = ewald_params.reciprocal_space_cutoff.item()
        ewald_cutoff = ewald_params.real_space_cutoff.item()

        pme_params = electrostatics_mod.estimate_pme_parameters(
            positions,
            cell,
            accuracy=accuracy,
            real_space_cutoff=real_space_cutoff,
        )
    else:
        ewald_params = electrostatics_mod.estimate_ewald_parameters(
            positions, cell, batch_idx, accuracy=accuracy
        )
        ewald_k_cutoff = ewald_params.reciprocal_space_cutoff[0].item()
        ewald_cutoff = ewald_params.real_space_cutoff[0].item()

        pme_params = electrostatics_mod.estimate_pme_parameters(
            positions,
            cell,
            batch_idx,
            accuracy=accuracy,
            real_space_cutoff=real_space_cutoff,
        )

    pme_alpha = pme_params.alpha
    pme_cutoff = (
        float(real_space_cutoff)
        if real_space_cutoff is not None
        else (
            pme_params.real_space_cutoff.item()
            if batch_idx is None
            else pme_params.real_space_cutoff[0].item()
        )
    )
    mesh_dimensions = pme_params.mesh_dimensions
    mesh_spacing = pme_params.mesh_spacing

    cell_static = (
        cell.detach() if hasattr(cell, "detach") else jax.lax.stop_gradient(cell)
    )
    k_vectors_pme, k_squared_pme = electrostatics_mod.generate_k_vectors_pme(
        cell_static, mesh_dimensions
    )

    ewald_block = {
        "alpha": ewald_params.alpha,
        "k_cutoff": ewald_k_cutoff,
        "cutoff": ewald_cutoff,
    }
    pme_block = {
        "alpha": pme_alpha,
        "cutoff": pme_cutoff,
        "mesh_dimensions": mesh_dimensions,
        "mesh_spacing": mesh_spacing,
        "k_vectors_pme": k_vectors_pme,
        "k_squared_pme": k_squared_pme,
    }

    return {
        "ewald": ewald_block,
        "pme": pme_block,
        # Backward-compatible aliases used by older helpers. Prefer the
        # explicit method blocks for new benchmark code.
        "alpha": pme_alpha,
        "alpha_ewald": ewald_block["alpha"],
        "alpha_pme": pme_block["alpha"],
        "k_cutoff": ewald_k_cutoff,
        "cutoff": pme_cutoff,
        "cutoff_ewald": ewald_cutoff,
        "cutoff_pme": pme_cutoff,
        "mesh_dimensions": mesh_dimensions,
        "mesh_spacing": mesh_spacing,
        "k_vectors_pme": k_vectors_pme,
        "k_squared_pme": k_squared_pme,
    }


def compute_neighbor_list(
    backend_data: dict,
    backend: str,
    cutoff: float,
) -> tuple:
    """Compute neighbor list using the appropriate backend.

    Parameters
    ----------
    backend_data : dict
        Output from convert_to_backend().
    backend : str
        "torch" or "jax".
    cutoff : float
        Cutoff distance for neighbor list.

    Returns
    -------
    tuple
        (neighbor_matrix, num_neighbors, neighbor_matrix_shifts)
    """
    _, neighbors_mod = _get_backend_modules(backend)

    positions = backend_data["positions"]
    cell = backend_data["cell"]
    pbc = backend_data["pbc"]
    batch_idx = backend_data["batch_idx"]

    if batch_idx is None:
        return neighbors_mod.neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            return_neighbor_list=False,
        )
    else:
        return neighbors_mod.neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            method="batch_naive",
            return_neighbor_list=False,
        )


def _torch_pme_static_metadata(
    cell: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    spline_order: int,
) -> dict:
    """Return fixed-cell PME metadata tensors for repeated Torch timings."""
    cell_static = cell.detach()
    cell_inv_t = torch.linalg.inv(cell_static).transpose(-1, -2).contiguous()
    volume = torch.abs(torch.linalg.det(cell_static)).reshape(-1).to(cell.dtype)

    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
    device = cell.device
    miller_x = torch.fft.fftfreq(
        mesh_nx, d=1.0 / mesh_nx, device=device, dtype=cell.dtype
    )
    miller_y = torch.fft.fftfreq(
        mesh_ny, d=1.0 / mesh_ny, device=device, dtype=cell.dtype
    )
    miller_z = torch.fft.rfftfreq(
        mesh_nz, d=1.0 / mesh_nz, device=device, dtype=cell.dtype
    )

    return {
        "cell_inv_t": cell_inv_t,
        "volume": volume,
        "moduli_x": _torch_electrostatics.compute_bspline_moduli_1d(
            miller_x, mesh_nx, spline_order
        ),
        "moduli_y": _torch_electrostatics.compute_bspline_moduli_1d(
            miller_y, mesh_ny, spline_order
        ),
        "moduli_z": _torch_electrostatics.compute_bspline_moduli_1d(
            miller_z, mesh_nz, spline_order
        ),
    }


def _jax_pme_static_metadata(
    cell: jax.Array,
    mesh_dimensions: tuple[int, int, int],
    spline_order: int,
) -> dict:
    """Return fixed-cell PME metadata arrays for repeated JAX timings."""
    cell_static = jax.lax.stop_gradient(cell)
    cell_inv = jnp.linalg.inv(cell_static)
    cell_inv_t = cell_inv.T if cell.ndim == 2 else jnp.transpose(cell_inv, (0, 2, 1))
    volume = jnp.abs(jnp.linalg.det(cell_static)).reshape(-1).astype(cell.dtype)

    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
    miller_x = jnp.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx).astype(cell.dtype)
    miller_y = jnp.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny).astype(cell.dtype)
    miller_z = jnp.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz).astype(cell.dtype)

    return {
        "cell_inv_t": cell_inv_t,
        "volume": volume,
        "moduli_x": _jax_electrostatics.compute_bspline_moduli_1d(
            miller_x, mesh_nx, spline_order
        ),
        "moduli_y": _jax_electrostatics.compute_bspline_moduli_1d(
            miller_y, mesh_ny, spline_order
        ),
        "moduli_z": _jax_electrostatics.compute_bspline_moduli_1d(
            miller_z, mesh_nz, spline_order
        ),
    }


def prepare_single_system(
    supercell_size: int,
    device: str,
    dtype: torch.dtype,
    np_data: dict | None = None,
    real_space_cutoff: float | None = None,
    accuracy: float = 1e-4,
    build_neighbors: bool = True,
    neighbor_family: Literal["ewald", "pme"] = "pme",
) -> dict:
    """Prepare a single system for benchmarking.

    Parameters
    ----------
    supercell_size : int
        Linear dimension of the supercell. For BCC lattice (2 atoms per unit cell),
        this creates 2 * supercell_size³ atoms total.
    device : str
        Device string for torch tensors.
    dtype : torch.dtype
        Data type for torch tensors.
    real_space_cutoff : float, optional
        If provided, forwarded to ``estimate_pme_parameters`` (pins PME's
        rc / alpha) and also used as the cutoff for the neighbor matrix.
    accuracy : float, default=1e-4
        Target relative force accuracy for the Ewald/PME parameter estimator.
    build_neighbors : bool, default=True
        Build the neighbor matrix. Set to ``False`` to skip when the
        benchmark only exercises the reciprocal half of PME.
    neighbor_family : {"ewald", "pme"}, default="pme"
        Which method family's real-space cutoff to use for the prebuilt
        neighbor matrix.

    Returns
    -------
    dict
        System data ready for electrostatics benchmarks, containing positions,
        charges, cell, pbc, neighbor list data (or ``None`` when
        ``build_neighbors=False``), and computed parameters.
    """
    dtype_str = str(dtype).split(".")[-1]

    if np_data is None:
        np_data = prepare_system_numpy(supercell_size, batch_size=1)

    backend_data = convert_to_backend(
        np_data, "torch", device=device, dtype_str=dtype_str
    )

    params = compute_electrostatics_params(
        backend_data,
        "torch",
        real_space_cutoff=real_space_cutoff,
        accuracy=accuracy,
    )

    if build_neighbors:
        neighbor_cutoff = _neighbor_cutoff_from_params(params, neighbor_family)
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = compute_neighbor_list(
            backend_data, "torch", neighbor_cutoff
        )
    else:
        neighbor_cutoff = _neighbor_cutoff_from_params(params, neighbor_family)
        neighbor_matrix = None
        num_neighbors = None
        neighbor_matrix_shifts = None

    pbc = backend_data["pbc"]
    if pbc.dim() == 2 and pbc.shape[0] == 1:
        pbc = pbc.squeeze(0)

    pbc_slab = backend_data["pbc"].clone()
    pbc_slab[..., SLAB_AXIS] = False

    mesh_spacing = params["mesh_spacing"]
    if hasattr(mesh_spacing, "tolist"):
        mesh_spacing = mesh_spacing.tolist()

    cell_t = backend_data["cell"]
    pme_static = _torch_pme_static_metadata(
        cell_t, params["mesh_dimensions"], spline_order=4
    )

    return {
        "positions": backend_data["positions"],
        "charges": backend_data["charges"],
        "cell": cell_t,
        **pme_static,
        "pbc": pbc,
        "pbc_slab": pbc_slab,
        "neighbor_matrix": neighbor_matrix,
        "num_neighbors": num_neighbors,
        "neighbor_matrix_shifts": neighbor_matrix_shifts,
        "total_atoms": backend_data["total_atoms"],
        "batch_idx": None,
        "alpha": params["alpha"],
        "alpha_ewald": params["ewald"]["alpha"],
        "alpha_pme": params["pme"]["alpha"],
        "k_cutoff": params["k_cutoff"],
        "cutoff_ewald": params["ewald"]["cutoff"],
        "cutoff_pme": params["pme"]["cutoff"],
        "cutoff": neighbor_cutoff,
        "neighbor_family": neighbor_family,
        "mesh_dimensions": params["mesh_dimensions"],
        "mesh_spacing": mesh_spacing,
        "spline_order": 4,
        "k_vectors_pme": params["k_vectors_pme"],
        "k_squared_pme": params["k_squared_pme"],
    }


def prepare_batch_system(
    supercell_size: int,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
    np_data: dict | None = None,
    real_space_cutoff: float | None = None,
    accuracy: float = 1e-4,
    build_neighbors: bool = True,
    neighbor_family: Literal["ewald", "pme"] = "pme",
) -> dict:
    """Prepare a batched system for benchmarking.

    This is the batched Torch companion to :func:`prepare_single_system`. It
    computes method-family-specific Ewald/PME parameter blocks, optional
    prebuilt neighbor data, and fixed-cell PME metadata used by repeated timing
    calls.

    Parameters
    ----------
    supercell_size : int
        Linear dimension of each supercell. For BCC lattice (2 atoms per unit cell),
        each system has 2 * supercell_size³ atoms.
    batch_size : int
        Number of systems to batch together.
    device : str
        Device string for torch tensors.
    dtype : torch.dtype
        Data type for torch tensors.
    real_space_cutoff : float, optional
        If provided, pins PME's real-space cutoff and derived alpha. Ewald keeps
        its own estimator-derived reciprocal cutoff.
    accuracy : float, default=1e-4
        Target relative force accuracy for Ewald/PME parameter estimation.
    build_neighbors : bool, default=True
        Build neighbor matrices. When false, neighbor-list fields in the return
        value are ``None`` for reciprocal-only benchmark paths.
    neighbor_family : {"ewald", "pme"}, default="pme"
        Which method family's real-space cutoff to use for prebuilt neighbors.

    Returns
    -------
    dict
        System data ready for electrostatics benchmarks. Contains backend
        tensors, batch metadata, optional neighbor data, separate Ewald/PME
        alpha/cutoff entries, PME mesh/k-vector caches, and fixed-cell metadata
        (`cell_inv_t`, `volume`, `moduli_x`, `moduli_y`, `moduli_z`).
    """
    dtype_str = str(dtype).split(".")[-1]

    if np_data is None:
        np_data = prepare_system_numpy(supercell_size, batch_size=batch_size)

    backend_data = convert_to_backend(
        np_data, "torch", device=device, dtype_str=dtype_str
    )

    params = compute_electrostatics_params(
        backend_data,
        "torch",
        real_space_cutoff=real_space_cutoff,
        accuracy=accuracy,
    )

    pbc_slab = backend_data["pbc"].clone()
    pbc_slab[..., SLAB_AXIS] = False

    if build_neighbors:
        neighbor_cutoff = _neighbor_cutoff_from_params(params, neighbor_family)
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = compute_neighbor_list(
            backend_data, "torch", neighbor_cutoff
        )
    else:
        neighbor_cutoff = _neighbor_cutoff_from_params(params, neighbor_family)
        neighbor_matrix = None
        num_neighbors = None
        neighbor_matrix_shifts = None

    cell_t_b = backend_data["cell"]
    pme_static = _torch_pme_static_metadata(
        cell_t_b, params["mesh_dimensions"], spline_order=4
    )

    return {
        "positions": backend_data["positions"],
        "charges": backend_data["charges"],
        "cell": cell_t_b,
        **pme_static,
        "pbc": backend_data["pbc"],
        "pbc_slab": pbc_slab,
        "neighbor_matrix": neighbor_matrix,
        "num_neighbors": num_neighbors,
        "neighbor_matrix_shifts": neighbor_matrix_shifts,
        "total_atoms": backend_data["total_atoms"],
        "batch_idx": backend_data["batch_idx"],
        "batch_size": batch_size,
        "alpha": params["alpha"],
        "alpha_ewald": params["ewald"]["alpha"],
        "alpha_pme": params["pme"]["alpha"],
        "k_cutoff": params["k_cutoff"],
        "cutoff_ewald": params["ewald"]["cutoff"],
        "cutoff_pme": params["pme"]["cutoff"],
        "cutoff": neighbor_cutoff,
        "neighbor_family": neighbor_family,
        "mesh_dimensions": params["mesh_dimensions"],
        "mesh_spacing": params["mesh_spacing"],
        "spline_order": 4,
        "k_vectors_pme": params["k_vectors_pme"],
        "k_squared_pme": params["k_squared_pme"],
    }


def prepare_jax_ewald_pme_system(
    np_data: dict,
    dtype_str: str,
    real_space_cutoff: float | None = None,
    accuracy: float = 1e-4,
    build_neighbors: bool = True,
    neighbor_family: Literal["ewald", "pme"] = "pme",
) -> dict:
    """Prepare a JAX system dictionary for Ewald and PME benchmarks."""
    backend_data = convert_to_backend(np_data, "jax", dtype_str=dtype_str)
    params_data = compute_electrostatics_params(
        backend_data,
        "jax",
        real_space_cutoff=real_space_cutoff,
        accuracy=accuracy,
    )
    if build_neighbors:
        neighbor_cutoff = _neighbor_cutoff_from_params(params_data, neighbor_family)
        nl_matrix, nl_num_neighbors, nl_matrix_shifts = compute_neighbor_list(
            backend_data, "jax", neighbor_cutoff
        )
    else:
        neighbor_cutoff = _neighbor_cutoff_from_params(params_data, neighbor_family)
        nl_matrix = None
        nl_num_neighbors = None
        nl_matrix_shifts = None

    pbc_slab = backend_data["pbc"].at[..., SLAB_AXIS].set(False)
    cell_j = backend_data["cell"]
    pme_static = _jax_pme_static_metadata(
        cell_j, params_data["mesh_dimensions"], spline_order=4
    )

    system_data = {
        "positions": backend_data["positions"],
        "charges": backend_data["charges"],
        "cell": cell_j,
        **pme_static,
        "pbc": backend_data["pbc"],
        "pbc_slab": pbc_slab,
        "neighbor_matrix": nl_matrix,
        "num_neighbors": nl_num_neighbors,
        "neighbor_matrix_shifts": nl_matrix_shifts,
        "total_atoms": backend_data["total_atoms"],
        "num_atoms_per_system": backend_data["num_atoms_per_system"],
        "batch_idx": backend_data["batch_idx"],
        "alpha": params_data["alpha"],
        "alpha_ewald": params_data["ewald"]["alpha"],
        "alpha_pme": params_data["pme"]["alpha"],
        "k_cutoff": params_data["k_cutoff"],
        "cutoff_ewald": params_data["ewald"]["cutoff"],
        "cutoff_pme": params_data["pme"]["cutoff"],
        "cutoff": neighbor_cutoff,
        "neighbor_family": neighbor_family,
        "mesh_dimensions": params_data["mesh_dimensions"],
        "mesh_spacing": params_data["mesh_spacing"],
        "spline_order": 4,
        "k_vectors_pme": params_data["k_vectors_pme"],
        "k_squared_pme": params_data["k_squared_pme"],
    }
    if backend_data["batch_idx"] is not None:
        system_data["batch_size"] = int(backend_data["cell"].shape[0])
    return system_data


# ==============================================================================
# DSF System Preparation
# ==============================================================================


def build_neighbors(
    system_data: dict,
    neighbor_format: str,
) -> None:
    """Build neighbor data in-place for the requested format.

    Modifies *system_data* to add the neighbor keys for exactly one format
    (CSR or matrix).  Any previously-stored neighbor data is removed first
    so that only one representation is in GPU memory at a time.

    Parameters
    ----------
    system_data : dict
        System dictionary produced by one of the ``prepare_*`` functions.
    neighbor_format : str
        ``"list"`` for CSR (sparse), ``"matrix"`` for dense neighbor matrix,
        or ``"n/a"`` which is treated as CSR (used by torchpme / torch_dsf).
    """
    for key in [
        "neighbor_list",
        "neighbor_ptr",
        "neighbor_shifts",
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "fill_value",
        "num_neighbors",
    ]:
        system_data.pop(key, None)

    positions = system_data["positions"]
    cutoff = system_data["cutoff"]
    cell = system_data.get("cell")
    pbc = system_data.get("pbc")
    batch_idx = system_data.get("batch_idx")
    total_atoms = system_data["total_atoms"]

    nl_kwargs: dict = dict(cell=cell, pbc=pbc)
    if batch_idx is not None:
        nl_kwargs["batch_idx"] = batch_idx
        nl_kwargs["method"] = "batch_naive"

    if cell is not None:
        batch_size = system_data.get("batch_size", 1)
        cell_2d = cell[0] if cell.dim() == 3 else cell
        volume = torch.abs(torch.det(cell_2d)).item()
        density = (total_atoms / batch_size) / volume
        max_nbrs = _neighbor_utils.estimate_max_neighbors(
            cutoff, atomic_density=density, safety_factor=1.2
        )
        nl_kwargs["max_neighbors"] = max_nbrs

    if neighbor_format == "matrix":
        nm, num_nbrs, nm_shifts = _torch_neighbors.neighbor_list(
            positions, cutoff, **nl_kwargs
        )
        system_data["neighbor_matrix"] = nm
        system_data["num_neighbors"] = num_nbrs
        system_data["neighbor_matrix_shifts"] = nm_shifts
        system_data["fill_value"] = total_atoms
    else:  # "list" or "n/a" (CSR)
        nl_data, nl_ptr, nl_shifts = _torch_neighbors.neighbor_list(
            positions, cutoff, return_neighbor_list=True, **nl_kwargs
        )
        system_data["neighbor_list"] = nl_data
        system_data["neighbor_ptr"] = nl_ptr
        system_data["neighbor_shifts"] = nl_shifts


def prepare_dsf_single_system(
    supercell_size: int,
    device: str,
    dtype: torch.dtype,
    cutoff: float = 12.0,
    alpha: float = 0.2,
) -> dict:
    """Prepare a single system for DSF benchmarking.

    DSF does not need k-vectors, PME mesh, or Ewald parameter estimation.
    Neighbor data is built by ``build_neighbors()`` before each run.

    Parameters
    ----------
    supercell_size : int
        Linear dimension of the supercell. For BCC lattice (2 atoms per unit cell),
        this creates 2 * supercell_size^3 atoms total.
    """
    target_atoms = 2 * supercell_size**3
    system = create_crystal_system(
        target_atoms,
        lattice_type="bcc",
        lattice_constant=4.14,
        device=device,
        dtype=dtype,
    )
    total_atoms = system["num_atoms"]
    positions = system["positions"]
    charges = system["atomic_charges"]
    cell = system["cell"]
    pbc = system["pbc"]

    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "pbc": pbc,
        "total_atoms": total_atoms,
        "batch_idx": None,
        "cutoff": cutoff,
        "alpha": alpha,
    }


def prepare_dsf_batch_system(
    supercell_size: int,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
    cutoff: float = 12.0,
    alpha: float = 0.2,
) -> dict:
    """Prepare a batched system for DSF benchmarking.

    Neighbor data is built by ``build_neighbors()`` before each run.

    Parameters
    ----------
    supercell_size : int
        Linear dimension of each supercell.
    batch_size : int
        Number of systems to batch together.
    """
    target_atoms_per_system = 2 * supercell_size**3

    all_positions = []
    all_charges = []
    all_cells = []
    all_pbc = []
    batch_idx_list = []

    for i in range(batch_size):
        system = create_crystal_system(
            target_atoms_per_system,
            lattice_type="bcc",
            lattice_constant=4.14,
            device=device,
            dtype=dtype,
        )
        n_atoms = system["num_atoms"]
        all_positions.append(system["positions"])
        all_charges.append(system["atomic_charges"])
        all_cells.append(system["cell"])
        all_pbc.append(system["pbc"])
        batch_idx_list.extend([i] * n_atoms)

    positions = torch.cat(all_positions, dim=0)
    charges = torch.cat(all_charges, dim=0)
    cells = torch.cat(all_cells, dim=0)
    pbc = torch.stack(all_pbc, dim=0)
    batch_idx = torch.tensor(batch_idx_list, dtype=torch.int32, device=device)
    total_atoms = positions.shape[0]

    return {
        "positions": positions,
        "charges": charges,
        "cell": cells,
        "pbc": pbc,
        "total_atoms": total_atoms,
        "batch_idx": batch_idx,
        "batch_size": batch_size,
        "cutoff": cutoff,
        "alpha": alpha,
    }


# ==============================================================================
# nvalchemiops Backend
# ==============================================================================


def run_nvalchemiops_dsf(
    system_data: dict,
    compute_forces: bool,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run DSF using nvalchemiops backend (neighbor matrix format)."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    cutoff = system_data["cutoff"]
    alpha = system_data["alpha"]
    neighbor_matrix = system_data["neighbor_matrix"]
    neighbor_matrix_shifts = system_data["neighbor_matrix_shifts"]
    fill_value = system_data["fill_value"]
    num_systems = system_data.get("batch_size", 1)

    return _torch_electrostatics.dsf_coulomb(
        positions=positions,
        charges=charges,
        cutoff=cutoff,
        alpha=alpha,
        cell=cell,
        batch_idx=batch_idx,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        fill_value=fill_value,
        compute_forces=compute_forces,
        compute_virial=compute_virial,
        num_systems=num_systems,
    )


def run_nvalchemiops_dsf_csr(
    system_data: dict,
    compute_forces: bool,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run DSF using nvalchemiops backend (CSR neighbor list format)."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    cutoff = system_data["cutoff"]
    alpha = system_data["alpha"]
    neighbor_list_data = system_data["neighbor_list"]
    neighbor_ptr = system_data["neighbor_ptr"]
    neighbor_shifts = system_data["neighbor_shifts"]
    num_systems = system_data.get("batch_size", 1)

    return _torch_electrostatics.dsf_coulomb(
        positions=positions,
        charges=charges,
        cutoff=cutoff,
        alpha=alpha,
        cell=cell,
        batch_idx=batch_idx,
        neighbor_list=neighbor_list_data,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=neighbor_shifts,
        compute_forces=compute_forces,
        compute_virial=compute_virial,
        num_systems=num_systems,
    )


def run_nvalchemiops_ewald(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
    slab_correction: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Ewald summation using nvalchemiops backend."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha_ewald", system_data.get("alpha"))
    k_cutoff = system_data.get("k_cutoff")
    k_vectors = _torch_electrostatics.generate_k_vectors_ewald_summation(
        cell.detach(), k_cutoff
    )
    pbc_slab = system_data.get("pbc_slab")

    neighbor_matrix_data = system_data.get("neighbor_matrix")
    neighbor_matrix_shifts = system_data.get("neighbor_matrix_shifts")

    if batch_idx is None:
        if component == "real":
            return _torch_electrostatics.ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                neighbor_matrix=neighbor_matrix_data,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        elif component == "reciprocal":
            return _torch_electrostatics.ewald_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                k_vectors=k_vectors,
                alpha=alpha,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        else:  # full
            return _torch_electrostatics.ewald_summation(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                k_cutoff=k_cutoff,
                k_vectors=k_vectors,
                neighbor_matrix=neighbor_matrix_data,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                pbc=pbc_slab,
                slab_correction=slab_correction,
            )
    else:
        if component == "real":
            return _torch_electrostatics.ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                batch_idx=batch_idx,
                neighbor_matrix=neighbor_matrix_data,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        elif component == "reciprocal":
            return _torch_electrostatics.ewald_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                k_vectors=k_vectors,
                alpha=alpha,
                batch_idx=batch_idx,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        else:  # full
            return _torch_electrostatics.ewald_summation(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                k_cutoff=k_cutoff,
                k_vectors=k_vectors,
                batch_idx=batch_idx,
                neighbor_matrix=neighbor_matrix_data,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                pbc=pbc_slab,
                slab_correction=slab_correction,
            )


def run_nvalchemiops_pme(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
    slab_correction: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run PME using nvalchemiops backend."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    cell_inv_t = system_data.get("cell_inv_t")
    volume = system_data.get("volume")
    moduli_x = system_data.get("moduli_x")
    moduli_y = system_data.get("moduli_y")
    moduli_z = system_data.get("moduli_z")
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha_pme", system_data.get("alpha"))
    mesh_dimensions = system_data.get("mesh_dimensions")
    spline_order = system_data.get("spline_order")
    k_vectors_pme = system_data.get("k_vectors_pme")
    k_squared_pme = system_data.get("k_squared_pme")
    pbc_slab = system_data.get("pbc_slab")

    neighbor_matrix_data = system_data.get("neighbor_matrix")
    neighbor_matrix_shifts = system_data.get("neighbor_matrix_shifts")

    if batch_idx is None:
        if component == "real":
            return _torch_electrostatics.ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                neighbor_matrix=neighbor_matrix_data,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        elif component == "reciprocal":
            return _torch_electrostatics.pme_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                k_vectors=k_vectors_pme,
                k_squared=k_squared_pme,
                cell_inv_t=cell_inv_t,
                volume=volume,
                moduli_x=moduli_x,
                moduli_y=moduli_y,
                moduli_z=moduli_z,
            )
        else:  # full
            return _torch_electrostatics.particle_mesh_ewald(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                neighbor_matrix=neighbor_matrix_data,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                k_vectors=k_vectors_pme,
                k_squared=k_squared_pme,
                cell_inv_t=cell_inv_t,
                volume=volume,
                moduli_x=moduli_x,
                moduli_y=moduli_y,
                moduli_z=moduli_z,
                pbc=pbc_slab,
                slab_correction=slab_correction,
            )
    else:
        if component == "real":
            return _torch_electrostatics.ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                batch_idx=batch_idx,
                neighbor_matrix=neighbor_matrix_data,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        elif component == "reciprocal":
            return _torch_electrostatics.pme_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                batch_idx=batch_idx,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                k_vectors=k_vectors_pme,
                k_squared=k_squared_pme,
                cell_inv_t=cell_inv_t,
                volume=volume,
                moduli_x=moduli_x,
                moduli_y=moduli_y,
                moduli_z=moduli_z,
            )
        else:  # full
            return _torch_electrostatics.particle_mesh_ewald(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                batch_idx=batch_idx,
                neighbor_matrix=neighbor_matrix_data,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                k_vectors=k_vectors_pme,
                k_squared=k_squared_pme,
                cell_inv_t=cell_inv_t,
                volume=volume,
                moduli_x=moduli_x,
                moduli_y=moduli_y,
                moduli_z=moduli_z,
                pbc=pbc_slab,
                slab_correction=slab_correction,
            )


def _first_tensor(result):
    """Return the first tensor from a backend result tuple or the tensor itself."""
    return result[0] if isinstance(result, tuple) else result


def _torch_deformed_energy_inputs(
    system_data: dict,
    *,
    use_strain: bool,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return differentiable Torch inputs for energy-autograd benchmark rows."""
    positions = system_data["positions"].detach().clone().requires_grad_(True)
    charges = system_data["charges"].detach().clone().requires_grad_(True)
    cell = system_data["cell"].detach().clone()
    data = dict(system_data)
    data["positions"] = positions
    data["charges"] = charges

    if not use_strain:
        data["cell"] = cell
        return data, positions, charges, None

    cell_3d = cell if cell.ndim == 3 else cell.unsqueeze(0)
    num_systems = cell_3d.shape[0]
    strain = torch.zeros(
        num_systems,
        3,
        3,
        dtype=positions.dtype,
        device=positions.device,
        requires_grad=True,
    )
    deform = torch.eye(3, dtype=positions.dtype, device=positions.device).unsqueeze(0)
    deform = deform + strain
    batch_idx = system_data.get("batch_idx")
    if batch_idx is None:
        atom_system = torch.zeros(
            positions.shape[0], dtype=torch.long, device=positions.device
        )
    else:
        atom_system = batch_idx.to(device=positions.device, dtype=torch.long)

    data["positions"] = torch.einsum("ni,nij->nj", positions, deform[atom_system])
    data["cell"] = torch.einsum("bij,bjk->bik", cell_3d, deform)
    for key in (
        "cell_inv_t",
        "volume",
        "moduli_x",
        "moduli_y",
        "moduli_z",
        "k_vectors_pme",
        "k_squared_pme",
    ):
        data.pop(key, None)
    return data, positions, charges, strain


def _run_torch_energy_autograd(
    runner,
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool,
    workload: BenchmarkWorkload,
    slab_correction: bool,
):
    """Run one Torch energy-autograd benchmark workload."""
    needs_strain = compute_virial and workload in ("backward", "double_backward")
    data, positions, charges, strain = _torch_deformed_energy_inputs(
        system_data, use_strain=needs_strain
    )
    energy = _first_tensor(
        runner(
            data,
            component,
            False,
            False,
            slab_correction=slab_correction,
        )
    )
    if workload == "forward" or not (compute_forces or compute_virial):
        return energy

    targets: list[torch.Tensor] = []
    if compute_forces:
        targets.append(positions)
    if compute_virial and strain is not None:
        targets.append(strain)
    grads = torch.autograd.grad(
        energy.sum(),
        tuple(targets),
        create_graph=workload == "double_backward",
        allow_unused=True,
    )

    outputs: list[torch.Tensor | None] = [energy]
    offset = 0
    force_grad = None
    strain_grad = None
    if compute_forces:
        force_grad = grads[offset]
        outputs.append(None if force_grad is None else -force_grad)
        offset += 1
    if compute_virial and strain is not None:
        strain_grad = grads[offset]
        outputs.append(None if strain_grad is None else -strain_grad)

    if workload != "double_backward":
        return tuple(outputs)

    loss = energy.new_zeros(())
    if force_grad is not None:
        loss = loss + force_grad.square().mean()
    if strain_grad is not None:
        loss = loss + strain_grad.square().mean()
    second_targets: list[torch.Tensor] = [positions, charges]
    if strain is not None:
        second_targets.append(strain)
    second_grads = torch.autograd.grad(
        loss,
        tuple(second_targets),
        allow_unused=True,
    )
    return (*outputs, loss, *second_grads)


def run_nvalchemiops_ewald_energy_autograd(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
    workload: BenchmarkWorkload = "backward",
    slab_correction: bool = False,
):
    """Run Torch Ewald via public energy API plus autograd-derived derivatives."""
    return _run_torch_energy_autograd(
        run_nvalchemiops_ewald,
        system_data,
        component,
        compute_forces,
        compute_virial,
        workload,
        slab_correction,
    )


def run_nvalchemiops_pme_energy_autograd(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
    workload: BenchmarkWorkload = "backward",
    slab_correction: bool = False,
):
    """Run Torch PME via public energy API plus autograd-derived derivatives."""
    return _run_torch_energy_autograd(
        run_nvalchemiops_pme,
        system_data,
        component,
        compute_forces,
        compute_virial,
        workload,
        slab_correction,
    )


# ==============================================================================
# nvalchemiops JAX Backend
# ==============================================================================


def prepare_jax_ewald(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
    slab_correction: bool = False,
):
    """Prepare a JIT-compiled Ewald callable for benchmarking.

    Creates the ``@jax.jit`` function **once** and returns a zero-argument
    callable that executes it.  This avoids re-tracing and recompilation on
    every timing iteration.

    Parameters
    ----------
    system_data : dict
        Dictionary containing system data with JAX arrays.
    component : {"real", "reciprocal", "full"}
        Which component of Ewald summation to compute.
    compute_forces : bool
        Whether to compute forces.
    compute_virial : bool, optional
        Whether to compute virial tensor, by default False.
    slab_correction : bool, optional
        Whether to run the full Ewald wrapper with slab correction enabled.

    Returns
    -------
    callable
        A zero-argument function that runs the JIT-compiled Ewald computation.
    """
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha_ewald", system_data.get("alpha"))
    k_cutoff = system_data.get("k_cutoff")
    num_atoms_per_system = system_data.get("num_atoms_per_system")
    pbc_slab = system_data.get("pbc_slab")

    neighbor_matrix_data = system_data.get("neighbor_matrix")
    neighbor_matrix_shifts = system_data.get("neighbor_matrix_shifts")

    cell_for_miller = cell if cell.ndim == 3 else cell[None, ...]
    _bounds = _jax_electrostatics.generate_miller_indices(cell_for_miller, k_cutoff)
    _miller_bounds = (int(_bounds[0]), int(_bounds[1]), int(_bounds[2]))

    _compute_forces = compute_forces
    _compute_virial = compute_virial
    _k_cutoff = k_cutoff
    _slab_correction = slab_correction

    if component == "real":

        @jax.jit
        def _jit_fn(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix,
            neighbor_matrix_shifts,
            batch_idx,
        ):
            return _jax_electrostatics.ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                batch_idx=batch_idx,
                compute_forces=_compute_forces,
                compute_virial=_compute_virial,
            )

        def call():
            return _jit_fn(
                positions,
                charges,
                cell,
                alpha,
                neighbor_matrix_data,
                neighbor_matrix_shifts,
                batch_idx,
            )

    elif component == "reciprocal":

        @jax.jit
        def _jit_fn(positions, charges, cell, alpha, batch_idx):
            k_vectors = _jax_electrostatics.generate_k_vectors_ewald_summation(
                jax.lax.stop_gradient(cell),
                _k_cutoff,
                miller_bounds=_miller_bounds,
            )
            return _jax_electrostatics.ewald_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                k_vectors=k_vectors,
                alpha=alpha,
                batch_idx=batch_idx,
                max_atoms_per_system=num_atoms_per_system,
                compute_forces=_compute_forces,
                compute_virial=_compute_virial,
            )

        def call():
            return _jit_fn(positions, charges, cell, alpha, batch_idx)

    else:  # full

        @jax.jit
        def _jit_fn(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix,
            neighbor_matrix_shifts,
            batch_idx,
            pbc_slab,
        ):
            return _jax_electrostatics.ewald_summation(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                k_cutoff=_k_cutoff,
                k_vectors=None,
                miller_bounds=_miller_bounds,
                batch_idx=batch_idx,
                max_atoms_per_system=num_atoms_per_system,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                compute_forces=_compute_forces,
                compute_virial=_compute_virial,
                pbc=pbc_slab,
                slab_correction=_slab_correction,
            )

        def call():
            return _jit_fn(
                positions,
                charges,
                cell,
                alpha,
                neighbor_matrix_data,
                neighbor_matrix_shifts,
                batch_idx,
                pbc_slab,
            )

    return call


def prepare_jax_pme(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
    slab_correction: bool = False,
):
    """Prepare a JIT-compiled PME callable for benchmarking.

    Creates the ``@jax.jit`` function **once** and returns a zero-argument
    callable that executes it.

    Parameters
    ----------
    system_data : dict
        Dictionary containing system data with JAX arrays.
    component : {"real", "reciprocal", "full"}
        Which component of PME to compute.
    compute_forces : bool
        Whether to compute forces.
    compute_virial : bool, optional
        Whether to compute virial tensor, by default False.
    slab_correction : bool, optional
        Whether to run the full PME wrapper with slab correction enabled.

    Returns
    -------
    callable
        A zero-argument function that runs the JIT-compiled PME computation.
    """
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha_pme", system_data.get("alpha"))
    mesh_dimensions = system_data.get("mesh_dimensions")
    spline_order = system_data.get("spline_order")
    pbc_slab = system_data.get("pbc_slab")
    cell_inv_t = system_data.get("cell_inv_t")
    volume = system_data.get("volume")
    moduli_x = system_data.get("moduli_x")
    moduli_y = system_data.get("moduli_y")
    moduli_z = system_data.get("moduli_z")
    k_vectors_pme = system_data.get("k_vectors_pme")
    k_squared_pme = system_data.get("k_squared_pme")

    neighbor_matrix_data = system_data.get("neighbor_matrix")
    neighbor_matrix_shifts = system_data.get("neighbor_matrix_shifts")

    _compute_forces = compute_forces
    _compute_virial = compute_virial
    _spline_order = spline_order
    _mesh_dimensions = mesh_dimensions
    _slab_correction = slab_correction

    if component == "real":

        @jax.jit
        def _jit_fn(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix,
            neighbor_matrix_shifts,
            batch_idx,
        ):
            return _jax_electrostatics.ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                batch_idx=batch_idx,
                compute_forces=_compute_forces,
                compute_virial=_compute_virial,
            )

        def call():
            return _jit_fn(
                positions,
                charges,
                cell,
                alpha,
                neighbor_matrix_data,
                neighbor_matrix_shifts,
                batch_idx,
            )

    elif component == "reciprocal":

        @jax.jit
        def _jit_fn(
            positions,
            charges,
            cell,
            alpha,
            batch_idx,
            cell_inv_t,
            volume,
            moduli_x,
            moduli_y,
            moduli_z,
            k_vectors,
            k_squared,
        ):
            return _jax_electrostatics.pme_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=_mesh_dimensions,
                spline_order=_spline_order,
                batch_idx=batch_idx,
                k_vectors=k_vectors,
                k_squared=k_squared,
                volume=volume,
                cell_inv_t=cell_inv_t,
                moduli_x=moduli_x,
                moduli_y=moduli_y,
                moduli_z=moduli_z,
                compute_forces=_compute_forces,
                compute_virial=_compute_virial,
            )

        def call():
            return _jit_fn(
                positions,
                charges,
                cell,
                alpha,
                batch_idx,
                cell_inv_t,
                volume,
                moduli_x,
                moduli_y,
                moduli_z,
                k_vectors_pme,
                k_squared_pme,
            )

    else:  # full

        @jax.jit
        def _jit_fn(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix,
            neighbor_matrix_shifts,
            batch_idx,
            cell_inv_t,
            volume,
            moduli_x,
            moduli_y,
            moduli_z,
            k_vectors,
            k_squared,
            pbc_slab,
        ):
            return _jax_electrostatics.particle_mesh_ewald(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=_mesh_dimensions,
                spline_order=_spline_order,
                batch_idx=batch_idx,
                k_vectors=k_vectors,
                k_squared=k_squared,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                volume=volume,
                cell_inv_t=cell_inv_t,
                moduli_x=moduli_x,
                moduli_y=moduli_y,
                moduli_z=moduli_z,
                compute_forces=_compute_forces,
                compute_virial=_compute_virial,
                pbc=pbc_slab,
                slab_correction=_slab_correction,
            )

        def call():
            return _jit_fn(
                positions,
                charges,
                cell,
                alpha,
                neighbor_matrix_data,
                neighbor_matrix_shifts,
                batch_idx,
                cell_inv_t,
                volume,
                moduli_x,
                moduli_y,
                moduli_z,
                k_vectors_pme,
                k_squared_pme,
                pbc_slab,
            )

    return call


def _jax_deformed_inputs(
    positions,
    cell,
    batch_idx,
    strain,
):
    """Apply row-vector strain to JAX positions and cells."""
    cell_3d = cell if cell.ndim == 3 else cell[jnp.newaxis, ...]
    deform = jnp.eye(3, dtype=positions.dtype)[jnp.newaxis, ...] + strain
    if batch_idx is None:
        atom_system = jnp.zeros((positions.shape[0],), dtype=jnp.int32)
    else:
        atom_system = batch_idx.astype(jnp.int32)
    positions_def = jnp.einsum("ni,nij->nj", positions, deform[atom_system])
    cell_def = jnp.einsum("bij,bjk->bik", cell_3d, deform)
    return positions_def, cell_def


def prepare_jax_ewald_energy_autograd(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
    workload: BenchmarkWorkload = "backward",
    slab_correction: bool = False,
):
    """Prepare a JIT-compiled JAX Ewald energy-autograd benchmark callable."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha_ewald", system_data.get("alpha"))
    k_cutoff = system_data.get("k_cutoff")
    num_atoms_per_system = system_data.get("num_atoms_per_system")
    pbc_slab = system_data.get("pbc_slab")
    neighbor_matrix_data = system_data.get("neighbor_matrix")
    neighbor_matrix_shifts = system_data.get("neighbor_matrix_shifts")
    cell_for_miller = cell if cell.ndim == 3 else cell[jnp.newaxis, ...]
    _bounds = _jax_electrostatics.generate_miller_indices(cell_for_miller, k_cutoff)
    _miller_bounds = (int(_bounds[0]), int(_bounds[1]), int(_bounds[2]))
    _k_cutoff = k_cutoff
    _slab_correction = slab_correction

    def _energy(pos, cell_arg):
        if component == "real":
            return _jax_electrostatics.ewald_real_space(
                positions=pos,
                charges=charges,
                cell=cell_arg,
                alpha=alpha,
                neighbor_matrix=neighbor_matrix_data,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                batch_idx=batch_idx,
            )
        if component == "reciprocal":
            k_vectors = _jax_electrostatics.generate_k_vectors_ewald_summation(
                jax.lax.stop_gradient(cell_arg),
                _k_cutoff,
                miller_bounds=_miller_bounds,
            )
            return _jax_electrostatics.ewald_reciprocal_space(
                positions=pos,
                charges=charges,
                cell=cell_arg,
                k_vectors=k_vectors,
                alpha=alpha,
                batch_idx=batch_idx,
                max_atoms_per_system=num_atoms_per_system,
            )
        return _jax_electrostatics.ewald_summation(
            positions=pos,
            charges=charges,
            cell=cell_arg,
            alpha=alpha,
            k_cutoff=_k_cutoff,
            k_vectors=None,
            miller_bounds=_miller_bounds,
            batch_idx=batch_idx,
            max_atoms_per_system=num_atoms_per_system,
            neighbor_matrix=neighbor_matrix_data,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            pbc=pbc_slab,
            slab_correction=_slab_correction,
        )

    if workload == "forward" or not (compute_forces or compute_virial):

        @jax.jit
        def _jit_forward(pos, cell_arg):
            return _energy(pos, cell_arg)

        def call():
            return _jit_forward(positions, cell)

        return call

    if compute_virial:
        cell_3d = cell if cell.ndim == 3 else cell[jnp.newaxis, ...]
        strain0 = jnp.zeros(cell_3d.shape, dtype=positions.dtype)

        def _strained_total(pos, strain):
            pos_def, cell_def = _jax_deformed_inputs(pos, cell, batch_idx, strain)
            return _energy(pos_def, cell_def).sum()

        @jax.jit
        def _jit_backward(pos, strain):
            value, grads = jax.value_and_grad(
                _strained_total,
                argnums=(0, 1),
            )(pos, strain)
            return value, grads

        def call():
            return _jit_backward(positions, strain0)

        return call

    @jax.jit
    def _jit_backward(pos):
        return jax.value_and_grad(lambda p: _energy(p, cell).sum())(pos)

    def call():
        return _jit_backward(positions)

    return call


def prepare_jax_pme_energy_autograd(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
    workload: BenchmarkWorkload = "backward",
    slab_correction: bool = False,
):
    """Prepare a JIT-compiled JAX PME energy-autograd benchmark callable."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha_pme", system_data.get("alpha"))
    mesh_dimensions = system_data.get("mesh_dimensions")
    spline_order = system_data.get("spline_order")
    pbc_slab = system_data.get("pbc_slab")
    neighbor_matrix_data = system_data.get("neighbor_matrix")
    neighbor_matrix_shifts = system_data.get("neighbor_matrix_shifts")
    _mesh_dimensions = mesh_dimensions
    _spline_order = spline_order
    _slab_correction = slab_correction

    def _energy(pos, cell_arg):
        if component == "real":
            return _jax_electrostatics.ewald_real_space(
                positions=pos,
                charges=charges,
                cell=cell_arg,
                alpha=alpha,
                neighbor_matrix=neighbor_matrix_data,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                batch_idx=batch_idx,
            )
        if component == "reciprocal":
            return _jax_electrostatics.pme_reciprocal_space(
                positions=pos,
                charges=charges,
                cell=cell_arg,
                alpha=alpha,
                mesh_dimensions=_mesh_dimensions,
                spline_order=_spline_order,
                batch_idx=batch_idx,
            )
        return _jax_electrostatics.particle_mesh_ewald(
            positions=pos,
            charges=charges,
            cell=cell_arg,
            alpha=alpha,
            mesh_dimensions=_mesh_dimensions,
            spline_order=_spline_order,
            batch_idx=batch_idx,
            neighbor_matrix=neighbor_matrix_data,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            pbc=pbc_slab,
            slab_correction=_slab_correction,
        )

    if workload == "forward" or not (compute_forces or compute_virial):

        @jax.jit
        def _jit_forward(pos, cell_arg):
            return _energy(pos, cell_arg)

        def call():
            return _jit_forward(positions, cell)

        return call

    if compute_virial:
        cell_3d = cell if cell.ndim == 3 else cell[jnp.newaxis, ...]
        strain0 = jnp.zeros(cell_3d.shape, dtype=positions.dtype)

        def _strained_total(pos, strain):
            pos_def, cell_def = _jax_deformed_inputs(pos, cell, batch_idx, strain)
            return _energy(pos_def, cell_def).sum()

        @jax.jit
        def _jit_backward(pos, strain):
            value, grads = jax.value_and_grad(
                _strained_total,
                argnums=(0, 1),
            )(pos, strain)
            return value, grads

        def call():
            return _jit_backward(positions, strain0)

        return call

    @jax.jit
    def _jit_backward(pos):
        return jax.value_and_grad(lambda p: _energy(p, cell).sum())(pos)

    def call():
        return _jit_backward(positions)

    return call


# ==============================================================================
# torchpme Backend
# ==============================================================================


def prepare_torchpme_neighbors(
    system_data: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare neighbor data in torchpme format.

    Converts dense padded neighbor_matrix format to COO format required by torchpme.
    """
    positions = system_data["positions"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")

    if batch_idx is None:
        neighbor_matrix_data = system_data.get("neighbor_matrix")
        neighbor_matrix_shifts_data = system_data.get("neighbor_matrix_shifts")

        if neighbor_matrix_data is not None:
            total_atoms_val = positions.shape[0]
            row_idx = torch.arange(total_atoms_val, device=positions.device)
            row_idx = row_idx.unsqueeze(1).expand_as(neighbor_matrix_data)
            valid = neighbor_matrix_data < total_atoms_val
            src = row_idx[valid]
            dst = neighbor_matrix_data[valid]
            neighbor_indices = torch.stack([src, dst], dim=0).T  # (num_pairs, 2)
            if neighbor_matrix_shifts_data is not None:
                shifts = neighbor_matrix_shifts_data[valid]  # (num_pairs, 3)
            else:
                shifts = torch.zeros(
                    src.shape[0], 3, dtype=torch.int32, device=positions.device
                )
            cell_2d = cell.squeeze(0)
            neighbor_distances = torch.norm(
                positions[dst]
                - positions[src]
                + shifts.to(dtype=positions.dtype) @ cell_2d,
                dim=1,
            )
        else:
            neighbor_indices = torch.zeros(
                (0, 2), dtype=torch.int32, device=positions.device
            )
            neighbor_distances = torch.zeros(
                0, dtype=positions.dtype, device=positions.device
            )

        return neighbor_indices, neighbor_distances
    else:
        raise NotImplementedError("torchpme batch mode requires per-system handling")


def run_torchpme_ewald(
    system_data: dict,
    compute_forces: bool,
    compute_virial: bool = False,
    calculator: EwaldCalculator | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run Ewald summation using torchpme backend."""
    if not TORCHPME_AVAILABLE:
        raise ImportError("torchpme not available")

    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    alpha = system_data.get("alpha_ewald", system_data.get("alpha")).item()
    k_cutoff = system_data.get("k_cutoff")
    dtype = positions.dtype
    device = positions.device
    neighbor_indices, neighbor_distances = prepare_torchpme_neighbors(
        system_data,
    )

    if calculator is None:
        lr_wavelength = 2 * torch.pi / k_cutoff
        smearing = 1.0 / alpha
        calculator = EwaldCalculator(
            potential=CoulombPotential(smearing=smearing).to(
                device=device, dtype=dtype
            ),
            lr_wavelength=lr_wavelength,
        ).to(device=device, dtype=dtype)

    charges_expanded = charges.unsqueeze(1)
    cell_2d = cell.squeeze(0)

    if not compute_forces and not compute_virial:
        energy = calculator.forward(
            charges_expanded,
            cell_2d,
            positions,
            neighbor_indices,
            neighbor_distances,
        )
        return energy, None

    positions_grad = positions.clone().detach().requires_grad_(True)
    cell_grad = (
        cell_2d.clone().detach().requires_grad_(True) if compute_virial else cell_2d
    )
    potentials_grad = calculator.forward(
        charges_expanded,
        cell_grad,
        positions_grad,
        neighbor_indices,
        neighbor_distances,
    )
    energy_grad = (potentials_grad * charges_expanded).sum()
    energy_grad.backward()
    forces = -positions_grad.grad if compute_forces else None
    virial = cell_grad.grad if compute_virial else None

    return energy_grad, forces, virial


def run_torchpme_pme(
    system_data: dict,
    compute_forces: bool,
    compute_virial: bool = False,
    calculator: PMECalculator | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run PME using torchpme backend."""
    if not TORCHPME_AVAILABLE:
        raise ImportError("torchpme not available")

    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    alpha = system_data.get("alpha_pme", system_data.get("alpha")).item()
    mesh_spacing = system_data.get("mesh_spacing")[0][0]
    spline_order = system_data.get("spline_order")
    dtype = positions.dtype
    device = positions.device

    neighbor_indices, neighbor_distances = prepare_torchpme_neighbors(
        system_data,
    )
    if calculator is None:
        smearing = 1.0 / alpha
        calculator = PMECalculator(
            potential=CoulombPotential(smearing=smearing).to(
                device=device, dtype=dtype
            ),
            mesh_spacing=mesh_spacing,
            interpolation_nodes=spline_order,
            full_neighbor_list=True,
            prefactor=1.0,
        ).to(device=device, dtype=dtype)

    charges_expanded = charges.unsqueeze(1)
    cell_2d = cell.squeeze(0)

    if not compute_forces and not compute_virial:
        energy = calculator.forward(
            charges_expanded,
            cell_2d,
            positions,
            neighbor_indices,
            neighbor_distances,
        )
        return energy, None

    positions_grad = positions.clone().detach().requires_grad_(True)
    cell_grad = (
        cell_2d.clone().detach().requires_grad_(True) if compute_virial else cell_2d
    )
    potentials_grad = calculator.forward(
        charges_expanded,
        cell_grad,
        positions_grad,
        neighbor_indices,
        neighbor_distances,
    )
    energy_grad = (potentials_grad * charges_expanded).sum()
    energy_grad.backward()
    forces = -positions_grad.grad if compute_forces else None
    virial = cell_grad.grad if compute_virial else None

    return energy_grad, forces, virial


# ==============================================================================
# torch_dsf Backend -- Pure PyTorch DSF reference
# ==============================================================================


def dsf_reference(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cutoff: float,
    alpha: float,
    neighbor_list: torch.Tensor,
    cell: torch.Tensor | None = None,
    unit_shifts: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    num_systems: int = 1,
    compute_forces: bool = True,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Pure PyTorch DSF reference implementation (benchmark-oriented).

    Runs in input precision.  Uses autograd for force and virial computation.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates (float32 or float64).
    charges : torch.Tensor, shape (N,)
        Atomic charges. Must match positions dtype.
    cutoff : float
        Cutoff radius.
    alpha : float
        Damping parameter. 0.0 for shifted-force bare Coulomb.
    neighbor_list : torch.Tensor, shape (2, E)
        Full neighbor list in COO format [idx_i, idx_j].
    cell : torch.Tensor, shape (B, 3, 3), optional
        Unit cell matrices for PBC.
    unit_shifts : torch.Tensor, shape (E, 3), optional
        Integer unit cell shifts for PBC.
    batch_idx : torch.Tensor, shape (N,), optional
        System index per atom.
    num_systems : int
        Number of systems.
    compute_forces : bool
        Whether to compute forces.
    compute_virial : bool
        Whether to compute virial (requires cell).

    Returns
    -------
    energy : torch.Tensor, shape (num_systems,)
        Per-system electrostatic energy.
    forces : torch.Tensor or None, shape (N, 3)
        Per-atom forces if compute_forces=True, else None.
    virial : torch.Tensor or None, shape (B, 3, 3)
        Cell virial if compute_virial=True, else None.
    """
    if charges.dtype != positions.dtype:
        msg = f"charges dtype ({charges.dtype}) must match positions dtype ({positions.dtype})"
        raise TypeError(msg)
    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]

    if batch_idx is None:
        batch_idx = torch.zeros(N, dtype=torch.long, device=device)
    else:
        batch_idx = batch_idx.long()

    need_grad = compute_forces or compute_virial

    if need_grad:
        pos = positions.detach().clone().requires_grad_(True)
    else:
        pos = positions

    if compute_virial and cell is not None:
        cell_grad = cell.detach().clone().to(dtype=dtype).requires_grad_(True)
    else:
        cell_grad = cell.to(dtype=dtype) if cell is not None else None

    idx_i = neighbor_list[0].long()
    idx_j = neighbor_list[1].long()

    pos_i = torch.index_select(pos, 0, idx_i)
    pos_j = torch.index_select(pos, 0, idx_j)
    r_ij = pos_j - pos_i

    if cell_grad is not None and unit_shifts is not None:
        batch_i = torch.index_select(batch_idx, 0, idx_i)
        cell_per_pair = torch.index_select(cell_grad, 0, batch_i)
        shift_cart = torch.bmm(
            unit_shifts.to(dtype=dtype).unsqueeze(1), cell_per_pair
        ).squeeze(1)
        r_ij = r_ij + shift_cart

    dist = torch.norm(r_ij, dim=1)

    mask = dist < cutoff
    dist = dist[mask]
    idx_i_f = idx_i[mask]
    idx_j_f = idx_j[mask]

    q_i = torch.index_select(charges, 0, idx_i_f)
    q_j = torch.index_select(charges, 0, idx_j_f)

    alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
    cutoff_t = torch.tensor(cutoff, dtype=dtype, device=device)
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=dtype, device=device))

    if alpha > 0.0:
        erfc_Rc = torch.erfc(alpha_t * cutoff_t)
        exp_Rc = torch.exp(-(alpha_t**2) * cutoff_t**2)
    else:
        erfc_Rc = torch.ones(1, dtype=dtype, device=device)
        exp_Rc = torch.ones(1, dtype=dtype, device=device)

    V_shift = erfc_Rc / cutoff_t
    B = erfc_Rc / cutoff_t**2 + 2.0 * alpha_t / sqrt_pi * exp_Rc / cutoff_t
    self_coeff = -(erfc_Rc / (2.0 * cutoff_t) + alpha_t / sqrt_pi)

    if alpha > 0.0:
        erfc_r = torch.erfc(alpha_t * dist)
    else:
        erfc_r = torch.ones_like(dist)

    V_pair = erfc_r / dist - V_shift + B * (dist - cutoff_t)

    pair_energy_contrib = 0.5 * q_i * q_j * V_pair
    batch_i_f = torch.index_select(batch_idx, 0, idx_i_f)

    energy = torch.zeros(num_systems, dtype=dtype, device=device)
    if pair_energy_contrib.numel() > 0:
        energy = energy.index_add(0, batch_i_f, pair_energy_contrib)

    self_energy_per_atom = self_coeff * charges**2
    energy = energy.index_add(0, batch_idx, self_energy_per_atom)

    forces = None
    virial = None
    if need_grad:
        e_total = energy.sum()
        grad_targets = [pos] if compute_forces else []
        if compute_virial and cell_grad is not None:
            grad_targets.append(cell_grad)

        grads = torch.autograd.grad(e_total, grad_targets)

        idx = 0
        if compute_forces:
            forces = -grads[idx].detach()
            idx += 1
        if compute_virial and cell_grad is not None:
            virial = grads[idx].detach()

        energy = energy.detach()

    return energy, forces, virial


dsf_torch_compiled = (
    torch.compile(dsf_reference, mode="default") if TORCH_AVAILABLE else None
)


def run_torch_dsf(
    system_data: dict,
    compute_forces: bool,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run DSF using pure PyTorch reference (torch.compile).

    Supports both CSR (neighbor_list) and matrix (neighbor_matrix) formats.
    When matrix format is provided, it is converted to COO on-the-fly since
    the reference implementation only accepts COO neighbor lists.
    """
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    cutoff = system_data["cutoff"]
    alpha = system_data["alpha"]
    num_systems = system_data.get("batch_size", 1)

    if "neighbor_list" in system_data:
        neighbor_list_data = system_data["neighbor_list"]
        neighbor_shifts = system_data["neighbor_shifts"]
    elif "neighbor_matrix" in system_data:
        neighbor_matrix = system_data["neighbor_matrix"]
        fill_value = system_data["fill_value"]
        N, M = neighbor_matrix.shape
        atom_idx = torch.arange(N, device=positions.device).unsqueeze(1).expand(-1, M)
        mask = neighbor_matrix != fill_value
        idx_i = atom_idx[mask]
        idx_j = neighbor_matrix[mask]
        neighbor_list_data = torch.stack([idx_i, idx_j], dim=0).to(torch.int32)
        nm_shifts = system_data.get("neighbor_matrix_shifts")
        if nm_shifts is not None:
            neighbor_shifts = nm_shifts[mask]
        else:
            neighbor_shifts = None
    else:
        raise KeyError("system_data must contain 'neighbor_list' or 'neighbor_matrix'")

    return dsf_torch_compiled(
        positions=positions,
        charges=charges,
        cutoff=cutoff,
        alpha=alpha,
        neighbor_list=neighbor_list_data,
        cell=cell,
        unit_shifts=neighbor_shifts,
        batch_idx=batch_idx,
        num_systems=num_systems,
        compute_forces=compute_forces,
        compute_virial=compute_virial,
    )


# ==============================================================================
# Benchmark Runner
# ==============================================================================


def run_benchmark(
    method: Literal["ewald", "ewald_slab", "pme", "pme_slab", "dsf"],
    backend: Literal["torch", "jax", "torchpme", "torch_dsf"],
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool,
    timer: BenchmarkTimer,
    neighbor_format: str = "list",
    derivative_contract: DerivativeContract = "energy_autograd",
    workload: BenchmarkWorkload = "forward",
    torch_compile: bool = False,
) -> dict:
    """Run a single benchmark configuration.

    When ``torch_compile`` is True and the backend is ``torch`` /
    ``torchpme`` / ``torch_dsf``, the bench callable is wrapped in
    ``torch.compile(fullgraph=True)``. Framework-compile cost
    (``torch.compile`` Dynamo + Inductor trace; or for the jax backend
    the first XLA trace) is measured separately as
    ``framework_compile_ms`` in the returned dict, isolated from the
    warp NVRTC cost which is paid by a pre-warm raw call beforehand.
    """
    effective_virial = compute_virial

    try:
        if method == "dsf":
            if backend == "torch":
                if neighbor_format == "matrix":

                    def bench_fn():
                        return run_nvalchemiops_dsf(
                            system_data, compute_forces, effective_virial
                        )
                else:  # "list" (CSR)

                    def bench_fn():
                        return run_nvalchemiops_dsf_csr(
                            system_data, compute_forces, effective_virial
                        )
            elif backend == "torch_dsf":

                def bench_fn():
                    return run_torch_dsf(system_data, compute_forces, effective_virial)
            else:
                return benchmark_result_row(
                    system_data=system_data,
                    method=method,
                    backend=backend,
                    component=component,
                    compute_forces=compute_forces,
                    compute_virial=effective_virial,
                    derivative_contract=derivative_contract,
                    workload=workload,
                    neighbor_format=neighbor_format,
                    torch_compile=torch_compile,
                    success=False,
                    error=f"Backend '{backend}' not applicable for DSF",
                    error_type="NotApplicable",
                )
        elif backend == "torch":
            if method in ("ewald", "ewald_slab"):

                def bench_fn():
                    if derivative_contract == "legacy_direct":
                        return run_nvalchemiops_ewald(
                            system_data,
                            component,
                            compute_forces,
                            effective_virial,
                            slab_correction=method == "ewald_slab",
                        )
                    return run_nvalchemiops_ewald_energy_autograd(
                        system_data,
                        component,
                        compute_forces,
                        effective_virial,
                        workload=workload,
                        slab_correction=method == "ewald_slab",
                    )
            else:  # pme

                def bench_fn():
                    if derivative_contract == "legacy_direct":
                        return run_nvalchemiops_pme(
                            system_data,
                            component,
                            compute_forces,
                            effective_virial,
                            slab_correction=method == "pme_slab",
                        )
                    return run_nvalchemiops_pme_energy_autograd(
                        system_data,
                        component,
                        compute_forces,
                        effective_virial,
                        workload=workload,
                        slab_correction=method == "pme_slab",
                    )
        elif backend == "jax":
            if method in ("ewald", "ewald_slab"):
                if derivative_contract == "legacy_direct":
                    bench_fn = prepare_jax_ewald(
                        system_data,
                        component,
                        compute_forces,
                        effective_virial,
                        slab_correction=method == "ewald_slab",
                    )
                else:
                    bench_fn = prepare_jax_ewald_energy_autograd(
                        system_data,
                        component,
                        compute_forces,
                        effective_virial,
                        workload=workload,
                        slab_correction=method == "ewald_slab",
                    )
            else:  # pme
                if derivative_contract == "legacy_direct":
                    bench_fn = prepare_jax_pme(
                        system_data,
                        component,
                        compute_forces,
                        effective_virial,
                        slab_correction=method == "pme_slab",
                    )
                else:
                    bench_fn = prepare_jax_pme_energy_autograd(
                        system_data,
                        component,
                        compute_forces,
                        effective_virial,
                        workload=workload,
                        slab_correction=method == "pme_slab",
                    )
        elif backend == "torchpme":
            if method in SLAB_METHODS:
                return benchmark_result_row(
                    system_data=system_data,
                    method=method,
                    backend=backend,
                    component=component,
                    compute_forces=compute_forces,
                    compute_virial=effective_virial,
                    derivative_contract=derivative_contract,
                    workload=workload,
                    neighbor_format=neighbor_format,
                    torch_compile=torch_compile,
                    success=False,
                    error=f"torchpme does not support slab method {method!r}",
                    error_type="NotApplicable",
                )
            if system_data.get("batch_idx") is not None:
                return benchmark_result_row(
                    system_data=system_data,
                    method=method,
                    backend=backend,
                    component=component,
                    compute_forces=compute_forces,
                    compute_virial=effective_virial,
                    derivative_contract=derivative_contract,
                    workload=workload,
                    neighbor_format=neighbor_format,
                    torch_compile=torch_compile,
                    success=False,
                    error="torchpme does not support native batched evaluation",
                    error_type="NotImplemented",
                )

            if method == "ewald":

                def bench_fn():
                    return run_torchpme_ewald(
                        system_data, compute_forces, effective_virial
                    )
            else:  # pme

                def bench_fn():
                    return run_torchpme_pme(
                        system_data,
                        compute_forces,
                        effective_virial,
                    )
        else:
            return benchmark_result_row(
                system_data=system_data,
                method=method,
                backend=backend,
                component=component,
                compute_forces=compute_forces,
                compute_virial=effective_virial,
                derivative_contract=derivative_contract,
                workload=workload,
                neighbor_format=neighbor_format,
                torch_compile=torch_compile,
                success=False,
                error=f"Backend '{backend}' not applicable for {method}",
                error_type="NotApplicable",
            )

        if derivative_contract == "legacy_direct":
            raw_bench_fn = bench_fn

            def bench_fn():
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=DeprecationWarning,
                    )
                    return raw_bench_fn()

        # Framework-compile cost (torch.compile or JAX jit) measured by a
        # one-shot wrap + first-call timing AFTER the raw bench_fn has been
        # called once to warm warp / cuFFT. This isolates Dynamo+Inductor
        # (or XLA) trace cost from NVRTC kernel compile cost.
        framework_compile_ms: float | None = None
        framework: str = "none"
        warp_compile_ms: float | None = None
        if backend in ("torch", "torchpme", "torch_dsf") and torch_compile:
            try:
                # 1) Raw pre-warm — pays warp NVRTC + any cuFFT plan creation.
                import time as _time

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                _t0 = _time.perf_counter()
                bench_fn()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                warp_compile_ms = (_time.perf_counter() - _t0) * 1000.0
                # Use the default torch.compile mode. CUDA-graph capture modes
                # clash with warp's stream binding in this benchmark.
                compiled_fn = torch.compile(bench_fn, fullgraph=True)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                _t0 = _time.perf_counter()
                compiled_fn()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                framework_compile_ms = (_time.perf_counter() - _t0) * 1000.0
                framework = "torch.compile"
                bench_fn = compiled_fn
            except Exception as _e:
                framework = f"torch.compile-FAILED:{type(_e).__name__}"
                framework_compile_ms = None
                return benchmark_result_row(
                    system_data=system_data,
                    method=method,
                    backend=backend,
                    component=component,
                    compute_forces=compute_forces,
                    compute_virial=effective_virial,
                    derivative_contract=derivative_contract,
                    workload=workload,
                    neighbor_format=neighbor_format,
                    torch_compile=torch_compile,
                    success=False,
                    warp_compile_ms=warp_compile_ms,
                    framework_compile_ms=framework_compile_ms,
                    framework=framework,
                    error=str(_e),
                    error_type=type(_e).__name__,
                )
        elif backend == "jax":
            # JAX path is already JIT-ed inside prepare_jax_pme; time the
            # first call separately as a proxy for XLA trace cost.
            try:
                import time as _time

                # First call: XLA trace + warp NVRTC + GPU kernel
                _t0 = _time.perf_counter()
                _res = bench_fn()
                if _res is not None:
                    jax.block_until_ready(_res)
                framework_compile_ms = (_time.perf_counter() - _t0) * 1000.0
                framework = "jax.jit"
            except Exception as _e:
                framework = f"jax.jit-FAILED:{type(_e).__name__}"
                framework_compile_ms = None
                return benchmark_result_row(
                    system_data=system_data,
                    method=method,
                    backend=backend,
                    component=component,
                    compute_forces=compute_forces,
                    compute_virial=effective_virial,
                    derivative_contract=derivative_contract,
                    workload=workload,
                    neighbor_format=neighbor_format,
                    torch_compile=torch_compile,
                    success=False,
                    warp_compile_ms=warp_compile_ms,
                    framework_compile_ms=framework_compile_ms,
                    framework=framework,
                    error=str(_e),
                    error_type=type(_e).__name__,
                )

        # Run steady-state benchmark
        timing_results = timer.time_function(bench_fn)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not timing_results["success"]:
            print(f"Benchmark failed: {timing_results.get('error', 'Unknown error')}")
            return benchmark_result_row(
                system_data=system_data,
                method=method,
                backend=backend,
                component=component,
                compute_forces=compute_forces,
                compute_virial=effective_virial,
                derivative_contract=derivative_contract,
                workload=workload,
                neighbor_format=neighbor_format,
                torch_compile=torch_compile,
                success=False,
                peak_memory_mb=timing_results.get("peak_memory_mb"),
                compile_ms=timing_results.get("compile_ms"),
                warp_compile_ms=warp_compile_ms,
                framework_compile_ms=framework_compile_ms,
                framework=framework,
                error=timing_results.get("error", "Unknown error"),
                error_type=timing_results.get("error_type", "Unknown"),
            )

        return benchmark_result_row(
            system_data=system_data,
            method=method,
            backend=backend,
            component=component,
            compute_forces=compute_forces,
            compute_virial=effective_virial,
            derivative_contract=derivative_contract,
            workload=workload,
            neighbor_format=neighbor_format,
            torch_compile=torch_compile,
            success=True,
            median_time_ms=timing_results["median"],
            peak_memory_mb=timing_results.get("peak_memory_mb"),
            compile_ms=timing_results.get("compile_ms"),
            warp_compile_ms=warp_compile_ms,
            framework_compile_ms=framework_compile_ms,
            framework=framework,
        )

    except Exception as e:
        print(f"Benchmark failed: {e}")
        return benchmark_result_row(
            system_data=system_data,
            method=method,
            backend=backend,
            component=component,
            compute_forces=compute_forces,
            compute_virial=effective_virial,
            derivative_contract=derivative_contract,
            workload=workload,
            neighbor_format=neighbor_format,
            torch_compile=torch_compile,
            success=False,
            error=str(e),
            error_type=type(e).__name__,
        )


# ==============================================================================
# Main
# ==============================================================================


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark electrostatic interaction methods and generate CSV files"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./benchmark_results"),
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "jax", "torchpme", "torch_dsf", "both"],
        default="torch",
        help=(
            "Backend to use for benchmarking (default: torch). "
            "'both' dispatches per-method: torch + torchpme for ewald/pme, "
            "torch + torch_dsf for dsf."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ewald", "ewald_slab", "pme", "pme_slab", "dsf", "both", "all"],
        default=None,
        help=(
            "Method to benchmark. When omitted, the YAML ``methods:`` list "
            "controls; if that is also empty, defaults to ewald + pme. "
            "'both' = ewald + pme. "
            "'all' = ewald + ewald_slab + pme + pme_slab + dsf."
        ),
    )
    parser.add_argument(
        "--gpu-sku",
        type=str,
        help="Override GPU SKU name for output files (default: auto-detect)",
    )
    parser.add_argument(
        "--neighbor-format",
        type=str,
        choices=["list", "matrix", "both"],
        default="list",
        help=(
            "Neighbor format for DSF torch benchmarks (default: list). "
            "'list' = CSR sparse format. 'matrix' = dense neighbor matrix. "
            "'both' = benchmark both formats."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default=None,
        help="Override dtype from config (default: use config value)",
    )
    parser.add_argument(
        "--real-space-cutoff",
        type=float,
        default=None,
        help=(
            "Override the PME real-space cutoff (Angstrom). When set, alpha "
            "and the mesh dimensions are derived from this rc instead of the "
            "cost-optimized value. Overrides the same field in config "
            "parameters."
        ),
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=None,
        help=(
            "Target relative force accuracy passed to the Ewald/PME parameter "
            "estimator. Drives alpha, real-space cutoff (when not pinned), and "
            "mesh dimensions. Overrides the ``accuracy`` field in config "
            "parameters (default 1e-4)."
        ),
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help=(
            "Wrap each torch-backend bench callable in "
            "``torch.compile(fullgraph=True)`` and "
            "record the framework-compile cost in a separate CSV column. "
            "The pre-warm pass calls the raw bench_fn once first so warp "
            "NVRTC compile and torch.compile (Dynamo + Inductor) costs "
            "appear in distinct columns. JAX backend always JITs; the "
            "first-call XLA trace cost is recorded under the same column."
        ),
    )
    parser.add_argument(
        "--derivative-contract",
        type=str,
        choices=["energy_autograd", "legacy_direct"],
        default=None,
        help=(
            "Derivative contract for nvalchemiops Ewald/PME rows. "
            "Default is energy_autograd, which benchmarks energy-only calls plus "
            "framework autograd for force/stress workloads. Use legacy_direct to "
            "benchmark deprecated compute_forces/compute_virial direct outputs."
        ),
    )

    args = parser.parse_args()

    # Validate backend availability
    _check_backend_available(args.backend)

    # Load config
    config = load_config(args.config)
    config_path = str(args.config)
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()

    # Resolve framework-level backend type
    backend_type = _resolve_backend_type(args.backend)

    # Get parameters
    params = config["parameters"]
    warmup = int(params["warmup_iterations"])
    timing = int(params["timing_iterations"])
    if args.dtype is not None:
        dtype_str = args.dtype
    else:
        dtype_str = params["dtype"]

    # Backend-specific setup
    device = "cpu"  # Default
    dtype = None
    match backend_type:
        case "torch":
            dtype = getattr(torch, dtype_str)
            device = "cuda" if torch.cuda.is_available() else "cpu"
        case "jax":
            dtype = None  # JAX uses dtype_str directly
            try:
                if any(d.platform == "gpu" for d in jax.local_devices()):
                    device = "gpu"
            except Exception:  # noqa: S110
                pass

    # Get GPU SKU
    gpu_sku = args.gpu_sku if args.gpu_sku else get_gpu_sku(backend_type)

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize timers. ``--backend both`` may dispatch Torch-family and JAX
    # rows in one process, and each timer owns framework-specific sync logic.
    timers = {}
    if args.backend != "jax":
        timers["torch"] = BenchmarkTimer(
            backend="torch", warmup_runs=warmup, timing_runs=timing
        )
    if args.backend == "jax" or (args.backend == "both" and JAX_AVAILABLE):
        timers["jax"] = BenchmarkTimer(
            backend="jax", warmup_runs=warmup, timing_runs=timing
        )

    # Initialize Warp (only needed for torch backend)
    if backend_type == "torch" and wp is not None:
        wp.init()

    # Determine what to benchmark. The CLI ``--method`` wins when explicitly
    # passed; otherwise the YAML's ``methods:`` list controls. If neither is
    # set, default to the historical ewald + pme pair.
    methods = resolve_methods(args.method, config.get("methods"))
    for method in methods:
        get_backends_for_method(args.backend, method)

    components = config.get("components", ["full"])
    compute_forces = config.get("compute_forces", True)
    compute_virial = config.get("compute_virial", False)
    derivative_contract = resolve_derivative_contract(
        args.derivative_contract,
        config.get("derivative_contract"),
    )

    # Optional PME real-space cutoff (CLI overrides config)
    if args.real_space_cutoff is not None:
        real_space_cutoff = float(args.real_space_cutoff)
    else:
        rc_cfg = params.get("real_space_cutoff", None)
        real_space_cutoff = float(rc_cfg) if rc_cfg is not None else None

    # Target relative force accuracy passed to the Ewald/PME parameter
    # estimator (CLI overrides config; default 1e-4 if neither is set).
    if args.accuracy is not None:
        accuracy = float(args.accuracy)
    else:
        accuracy = float(params.get("accuracy", 1e-4))

    # Skip neighbor-list construction only for pure reciprocal non-slab runs.
    should_build_neighbors = set(components) != {"reciprocal"} or any(
        method in SLAB_METHODS for method in methods
    )

    # DSF-specific parameters (hardcoded defaults)
    dsf_cutoff = 12.0
    dsf_alpha = 0.2

    # Print configuration
    print("=" * 70)
    print("ELECTROSTATICS BENCHMARK")
    print("=" * 70)
    print(f"Backend: {args.backend}")
    print(f"Device: {device}")
    print(f"GPU SKU: {gpu_sku}")
    print(f"Dtype: {dtype_str}")
    print(f"Methods: {methods}")
    print(f"Components: {components}")
    print(f"Derivative contract: {derivative_contract}")
    print(f"Compute forces: {compute_forces}")
    print(f"Compute virial: {compute_virial}")
    print(f"Accuracy: {accuracy:g}")
    if real_space_cutoff is not None:
        print(f"Real-space cutoff (pinned): {real_space_cutoff} A")
    print(f"Warmup iterations: {warmup}")
    print(f"Timing iterations: {timing}")
    print(f"Output directory: {output_dir}")
    if "dsf" in methods:
        print(f"DSF cutoff: {dsf_cutoff}, alpha: {dsf_alpha}")
        print(f"DSF neighbor format: {args.neighbor_format}")

    # Run benchmarks for each system configuration
    all_results = []

    def _print_result(result, method, backend, component):
        """Print benchmark result."""
        workload = result.get("workload", "forward")
        if result["success"]:
            throughput = result["total_atoms"] / result["median_time_ms"] * 1000
            mem_str = ""
            if result.get("peak_memory_mb"):
                mem_str = f" | {result['peak_memory_mb']:.1f} MB"
            compile_str = ""
            if result.get("compile_ms") is not None:
                compile_str = f" | warmup {result['compile_ms']:.0f} ms"
            print(
                f"    {method:5s} {backend:16s} {component:10s}: "
                f"{workload:15s} {result['median_time_ms']:.3f} ms "
                f"({throughput:.1f} atoms/s){mem_str}{compile_str}"
            )
        else:
            print(
                f"    {method:5s} {backend:16s} {component:10s}: "
                f"{workload:15s} FAILED ({result.get('error_type', 'Unknown')})"
            )

    for system_config in config["systems"]:
        system_name = system_config["name"]
        mode = system_config["mode"]

        print(f"\n{'=' * 70}")
        print(f"System: {system_name} ({mode})")
        print(f"{'=' * 70}")

        if mode == "single":
            supercell_sizes = system_config["supercell_sizes"]

            for size in supercell_sizes:
                expected_atoms = 2 * size**3  # BCC: 2 atoms per unit cell
                print(f"\n  ~{expected_atoms:,d} atoms (supercell {size}³)...")

                # Reset memory
                for timer in timers.values():
                    timer.clear_memory()

                # Prepare systems (method/backend-specific)
                system_data_cache = {}
                for method in methods:
                    backends = get_backends_for_method(args.backend, method)
                    for backend in backends:
                        if method == "dsf":
                            cache_key = ("dsf", "torch")
                            if cache_key in system_data_cache:
                                continue
                            try:
                                system_data_cache[cache_key] = (
                                    prepare_dsf_single_system(
                                        size, device, dtype, dsf_cutoff, dsf_alpha
                                    )
                                )
                            except Exception as e:
                                print(f"    Failed to prepare DSF system: {e}")
                                traceback.print_exc()
                                system_data_cache[cache_key] = None
                            continue

                        prep_backend = "jax" if backend == "jax" else "torch"
                        cache_key = (_system_cache_key(method), prep_backend)
                        if cache_key in system_data_cache:
                            continue
                        try:
                            np_data = (
                                prepare_slab_system_numpy(size, batch_size=1)
                                if method in SLAB_METHODS
                                else prepare_system_numpy(size, batch_size=1)
                            )
                            if prep_backend == "jax":
                                system_data_cache[cache_key] = (
                                    prepare_jax_ewald_pme_system(
                                        np_data,
                                        dtype_str,
                                        real_space_cutoff=real_space_cutoff,
                                        accuracy=accuracy,
                                        build_neighbors=should_build_neighbors,
                                        neighbor_family=_electrostatic_method_family(
                                            method
                                        ),
                                    )
                                )
                            else:
                                system_data_cache[cache_key] = prepare_single_system(
                                    size,
                                    device,
                                    dtype,
                                    np_data=np_data,
                                    real_space_cutoff=real_space_cutoff,
                                    accuracy=accuracy,
                                    build_neighbors=should_build_neighbors,
                                    neighbor_family=_electrostatic_method_family(
                                        method
                                    ),
                                )
                        except Exception as e:
                            print(f"    Failed to prepare system: {e}")
                            traceback.print_exc()
                            system_data_cache[cache_key] = None

                for method in methods:
                    backends = get_backends_for_method(args.backend, method)

                    method_components = (
                        ["full"]
                        if method == "dsf" or method in SLAB_METHODS
                        else components
                    )
                    for backend in backends:
                        prep_backend = "jax" if backend == "jax" else "torch"
                        cache_key = (
                            ("dsf", "torch")
                            if method == "dsf"
                            else (_system_cache_key(method), prep_backend)
                        )
                        system_data = system_data_cache.get(cache_key)
                        if system_data is None:
                            continue
                        timer = timers[prep_backend]
                        for component in method_components:
                            if method == "dsf" and backend in ("torch", "torch_dsf"):
                                nf_arg = args.neighbor_format
                                nf_list = (
                                    ["list", "matrix"] if nf_arg == "both" else [nf_arg]
                                )
                            else:
                                nf_list = ["n/a"]

                            for nf in nf_list:
                                workloads = benchmark_workloads(
                                    method=method,
                                    backend=backend,
                                    derivative_contract=derivative_contract,
                                    compute_forces=compute_forces,
                                    compute_virial=compute_virial,
                                )
                                for workload in workloads:
                                    try:
                                        if method == "dsf":
                                            build_neighbors(system_data, nf)
                                        result = run_benchmark(
                                            method,
                                            backend,
                                            system_data,
                                            component,
                                            compute_forces,
                                            compute_virial,
                                            timer,
                                            neighbor_format=nf,
                                            derivative_contract=derivative_contract,
                                            workload=workload,
                                            torch_compile=args.torch_compile,
                                        )
                                        annotate_result_row(
                                            result,
                                            supercell_size=size,
                                            mode=mode,
                                            dtype=dtype_str,
                                            config_path=config_path,
                                            config_sha256=config_sha256,
                                            accuracy=accuracy,
                                            real_space_cutoff=real_space_cutoff,
                                        )
                                        all_results.append(result)
                                        nf_tag = f" [{nf}]" if nf != "n/a" else ""
                                        _print_result(
                                            result, method, backend + nf_tag, component
                                        )
                                    except RuntimeError as oom:
                                        if "out of memory" not in str(oom).lower():
                                            raise
                                        if (
                                            TORCH_AVAILABLE
                                            and torch.cuda.is_available()
                                        ):
                                            torch.cuda.empty_cache()
                                        nf_tag = f" [{nf}]" if nf != "n/a" else ""
                                        result = benchmark_result_row(
                                            system_data=system_data,
                                            method=method,
                                            backend=backend,
                                            component=component,
                                            compute_forces=compute_forces,
                                            compute_virial=compute_virial,
                                            derivative_contract=derivative_contract,
                                            workload=workload,
                                            neighbor_format=nf,
                                            torch_compile=args.torch_compile,
                                            success=False,
                                            error=str(oom).split(".")[0],
                                            error_type=type(oom).__name__,
                                        )
                                        annotate_result_row(
                                            result,
                                            supercell_size=size,
                                            mode=mode,
                                            dtype=dtype_str,
                                            config_path=config_path,
                                            config_sha256=config_sha256,
                                            accuracy=accuracy,
                                            real_space_cutoff=real_space_cutoff,
                                        )
                                        all_results.append(result)
                                        print(
                                            f"    {method:5s} {backend + nf_tag:16s} "
                                            f"{component:10s}: {workload:15s} "
                                            "SKIPPED (OOM)"
                                        )

        else:  # batched
            base_size = system_config["base_supercell_size"]
            batch_sizes = system_config["batch_sizes"]
            atoms_per_system = 2 * base_size**3

            for batch_size in batch_sizes:
                total_atoms = atoms_per_system * batch_size
                print(
                    f"\n  {total_atoms:,d} atoms "
                    f"({atoms_per_system:,d} x {batch_size})..."
                )

                # Reset memory
                for timer in timers.values():
                    timer.clear_memory()

                # Prepare systems (method/backend-specific)
                system_data_cache = {}
                for method in methods:
                    backends = get_backends_for_method(args.backend, method)
                    for backend in backends:
                        if method == "dsf":
                            cache_key = ("dsf", "torch")
                            if cache_key in system_data_cache:
                                continue
                            try:
                                system_data_cache[cache_key] = prepare_dsf_batch_system(
                                    base_size,
                                    batch_size,
                                    device,
                                    dtype,
                                    dsf_cutoff,
                                    dsf_alpha,
                                )
                            except Exception as e:
                                print(f"    Failed to prepare DSF batch: {e}")
                                traceback.print_exc()
                                system_data_cache[cache_key] = None
                            continue

                        prep_backend = "jax" if backend == "jax" else "torch"
                        cache_key = (_system_cache_key(method), prep_backend)
                        if cache_key in system_data_cache:
                            continue
                        try:
                            np_data = (
                                prepare_slab_system_numpy(
                                    base_size, batch_size=batch_size
                                )
                                if method in SLAB_METHODS
                                else prepare_system_numpy(
                                    base_size, batch_size=batch_size
                                )
                            )
                            if prep_backend == "jax":
                                system_data_cache[cache_key] = (
                                    prepare_jax_ewald_pme_system(
                                        np_data,
                                        dtype_str,
                                        real_space_cutoff=real_space_cutoff,
                                        accuracy=accuracy,
                                        build_neighbors=should_build_neighbors,
                                        neighbor_family=_electrostatic_method_family(
                                            method
                                        ),
                                    )
                                )
                            else:
                                system_data_cache[cache_key] = prepare_batch_system(
                                    base_size,
                                    batch_size,
                                    device,
                                    dtype,
                                    np_data=np_data,
                                    real_space_cutoff=real_space_cutoff,
                                    accuracy=accuracy,
                                    build_neighbors=should_build_neighbors,
                                    neighbor_family=_electrostatic_method_family(
                                        method
                                    ),
                                )
                        except Exception as e:
                            print(f"    Failed to prepare system: {e}")
                            traceback.print_exc()
                            system_data_cache[cache_key] = None

                for method in methods:
                    backends = get_backends_for_method(args.backend, method)

                    method_components = (
                        ["full"]
                        if method == "dsf" or method in SLAB_METHODS
                        else components
                    )
                    for backend in backends:
                        prep_backend = "jax" if backend == "jax" else "torch"
                        cache_key = (
                            ("dsf", "torch")
                            if method == "dsf"
                            else (_system_cache_key(method), prep_backend)
                        )
                        system_data = system_data_cache.get(cache_key)
                        if system_data is None:
                            continue
                        timer = timers[prep_backend]
                        for component in method_components:
                            if method == "dsf" and backend in ("torch", "torch_dsf"):
                                nf_arg = args.neighbor_format
                                nf_list = (
                                    ["list", "matrix"] if nf_arg == "both" else [nf_arg]
                                )
                            else:
                                nf_list = ["n/a"]

                            for nf in nf_list:
                                workloads = benchmark_workloads(
                                    method=method,
                                    backend=backend,
                                    derivative_contract=derivative_contract,
                                    compute_forces=compute_forces,
                                    compute_virial=compute_virial,
                                )
                                for workload in workloads:
                                    try:
                                        if method == "dsf":
                                            build_neighbors(system_data, nf)
                                        result = run_benchmark(
                                            method,
                                            backend,
                                            system_data,
                                            component,
                                            compute_forces,
                                            compute_virial,
                                            timer,
                                            neighbor_format=nf,
                                            derivative_contract=derivative_contract,
                                            workload=workload,
                                            torch_compile=args.torch_compile,
                                        )
                                        annotate_result_row(
                                            result,
                                            supercell_size=base_size,
                                            mode=mode,
                                            dtype=dtype_str,
                                            config_path=config_path,
                                            config_sha256=config_sha256,
                                            accuracy=accuracy,
                                            real_space_cutoff=real_space_cutoff,
                                        )
                                        all_results.append(result)
                                        nf_tag = f" [{nf}]" if nf != "n/a" else ""
                                        _print_result(
                                            result, method, backend + nf_tag, component
                                        )
                                    except RuntimeError as oom:
                                        if "out of memory" not in str(oom).lower():
                                            raise
                                        if (
                                            TORCH_AVAILABLE
                                            and torch.cuda.is_available()
                                        ):
                                            torch.cuda.empty_cache()
                                        nf_tag = f" [{nf}]" if nf != "n/a" else ""
                                        result = benchmark_result_row(
                                            system_data=system_data,
                                            method=method,
                                            backend=backend,
                                            component=component,
                                            compute_forces=compute_forces,
                                            compute_virial=compute_virial,
                                            derivative_contract=derivative_contract,
                                            workload=workload,
                                            neighbor_format=nf,
                                            torch_compile=args.torch_compile,
                                            success=False,
                                            error=str(oom).split(".")[0],
                                            error_type=type(oom).__name__,
                                        )
                                        annotate_result_row(
                                            result,
                                            supercell_size=base_size,
                                            mode=mode,
                                            dtype=dtype_str,
                                            config_path=config_path,
                                            config_sha256=config_sha256,
                                            accuracy=accuracy,
                                            real_space_cutoff=real_space_cutoff,
                                        )
                                        all_results.append(result)
                                        print(
                                            f"    {method:5s} {backend + nf_tag:16s} "
                                            f"{component:10s}: {workload:15s} "
                                            "SKIPPED (OOM)"
                                        )

    # Save results
    if all_results:
        all_backends = sorted({r["backend"] for r in all_results})
        for method in methods:
            for backend in all_backends:
                method_results = [
                    r
                    for r in all_results
                    if r["method"] == method and r["backend"] == backend
                ]
                if method_results:
                    output_file = benchmark_output_file(
                        output_dir,
                        method,
                        backend,
                        dtype_str,
                        gpu_sku,
                    )
                    all_fieldnames = list(BENCHMARK_CSV_FIELDNAMES)
                    seen = set()
                    for r in method_results:
                        for k in r.keys():
                            if k not in all_fieldnames and k not in seen:
                                all_fieldnames.append(k)
                                seen.add(k)
                    with open(output_file, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=all_fieldnames, extrasaction="ignore"
                        )
                        writer.writeheader()
                        writer.writerows(method_results)
                    print(f"\n✓ Results saved to: {output_file}")

                    successful = [r for r in method_results if r.get("success", True)]
                    failed = [r for r in method_results if not r.get("success", True)]
                    print(
                        f"  Total: {len(method_results)} | "
                        f"Successful: {len(successful)} | "
                        f"Failed: {len(failed)}"
                    )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
