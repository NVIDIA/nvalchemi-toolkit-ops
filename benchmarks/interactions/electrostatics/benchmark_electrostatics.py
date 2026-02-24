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

CLI tool to benchmark electrostatic interaction methods (Ewald summation and PME)
and generate CSV files for documentation. Results are saved with GPU-specific naming:
`electrostatics_benchmark_<method>_<backend>_<gpu_sku>.csv`

Supports three backends:
1. torch (Warp kernels): Custom implementation using PyTorch + Warp
2. jax: Custom implementation using JAX + Warp (via XLA FFI)
3. torchpme: Reference PyTorch implementation

Usage:
    python benchmark_electrostatics.py --config benchmark_config.yaml --output-dir ./results
    python benchmark_electrostatics.py --config benchmark_config.yaml --backend jax
    python benchmark_electrostatics.py --config benchmark_config.yaml --backend torchpme --method ewald
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import traceback
from pathlib import Path
from typing import Literal

# Add repo root to path for imports (4 levels up from this script)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import yaml

from benchmarks.systems import create_crystal_system
from benchmarks.utils import BackendType, BenchmarkTimer

# -- Torch backend -----------------------------------------------------------
try:
    import torch
    import warp as wp

    _torch_electrostatics = importlib.import_module(
        "nvalchemiops.torch.interactions.electrostatics"
    )
    _torch_neighbors = importlib.import_module("nvalchemiops.torch.neighbors")
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    wp = None  # type: ignore
    _torch_electrostatics = None
    _torch_neighbors = None

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
        case "torch" | "torchpme":
            return "torch"
        case "jax":
            return "jax"
        case _:
            raise ValueError(f"Unknown backend: {cli_backend}")


def _check_backend_available(cli_backend: str) -> None:
    """Validate that the requested backend is installed."""
    match cli_backend:
        case "torch":
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
    # BCC lattice has 2 atoms per unit cell
    target_atoms_per_system = 2 * supercell_size**3

    if batch_size == 1:
        # Single system case
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
        # Batched system case
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
) -> dict:
    """Compute Ewald/PME parameters using the appropriate backend.

    Calls estimate_ewald_parameters, estimate_pme_parameters, and
    generate_k_vectors_pme on the backend module.

    Parameters
    ----------
    backend_data : dict
        Output from convert_to_backend(). Must contain positions, cell, and
        optionally batch_idx.
    backend : str
        "torch" or "jax".

    Returns
    -------
    dict
        Dictionary containing:
        - alpha: Ewald splitting parameter
        - k_cutoff: Reciprocal space cutoff for Ewald
        - cutoff: Real space cutoff
        - mesh_dimensions: PME mesh dimensions
        - mesh_spacing: PME mesh spacing
        - k_vectors_pme: Precomputed k-vectors for PME
        - k_squared_pme: Precomputed k² values for PME
    """
    electrostatics_mod, _ = _get_backend_modules(backend)

    positions = backend_data["positions"]
    cell = backend_data["cell"]
    batch_idx = backend_data["batch_idx"]

    # Compute Ewald parameters
    if batch_idx is None:
        # Single system
        ewald_params = electrostatics_mod.estimate_ewald_parameters(
            positions, cell, accuracy=1e-6
        )
        k_cutoff = ewald_params.reciprocal_space_cutoff.item()
        cutoff = ewald_params.real_space_cutoff.item()

        pme_params = electrostatics_mod.estimate_pme_parameters(
            positions, cell, accuracy=1e-6
        )
    else:
        # Batched system
        ewald_params = electrostatics_mod.estimate_ewald_parameters(
            positions, cell, batch_idx, accuracy=1e-6
        )
        # For batch, parameters are arrays - take the first element
        k_cutoff = ewald_params.reciprocal_space_cutoff[0].item()
        cutoff = ewald_params.real_space_cutoff[0].item()

        pme_params = electrostatics_mod.estimate_pme_parameters(
            positions, cell, batch_idx, accuracy=1e-6
        )

    alpha = pme_params.alpha
    mesh_dimensions = pme_params.mesh_dimensions
    mesh_spacing = pme_params.mesh_spacing

    # Precompute k-vectors for PME
    k_vectors_pme, k_squared_pme = electrostatics_mod.generate_k_vectors_pme(
        cell, mesh_dimensions
    )

    return {
        "alpha": alpha,
        "k_cutoff": k_cutoff,
        "cutoff": cutoff,
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
        # Single system
        return neighbors_mod.neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            return_neighbor_list=False,
        )
    else:
        # Batched system
        return neighbors_mod.neighbor_list(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            method="batch_naive",
            return_neighbor_list=False,
        )


def prepare_single_system(
    supercell_size: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """Prepare a single system for benchmarking.

    Backward-compatible wrapper that uses the new decoupled helpers internally.
    The return value structure is identical to the original implementation.

    Parameters
    ----------
    supercell_size : int
        Linear dimension of the supercell. For BCC lattice (2 atoms per unit cell),
        this creates 2 * supercell_size³ atoms total.
    device : str
        Device string for torch tensors.
    dtype : torch.dtype
        Data type for torch tensors.

    Returns
    -------
    dict
        System data ready for electrostatics benchmarks, containing positions,
        charges, cell, pbc, neighbor list data, and computed parameters.
    """
    # Extract dtype string from torch.dtype for convert_to_backend
    dtype_str = str(dtype).split(".")[-1]

    # Step 1: Generate system as numpy arrays
    np_data = prepare_system_numpy(supercell_size, batch_size=1)

    # Step 2: Convert to torch backend
    backend_data = convert_to_backend(
        np_data, "torch", device=device, dtype_str=dtype_str
    )

    # Step 3: Compute electrostatics parameters
    params = compute_electrostatics_params(backend_data, "torch")

    # Step 4: Compute neighbor list
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = compute_neighbor_list(
        backend_data, "torch", params["cutoff"]
    )

    # For single system, pbc should be shape (3,) not (1, 3) for backward compat
    # The convert_to_backend returns (1, 3), so squeeze it
    pbc = backend_data["pbc"]
    if pbc.dim() == 2 and pbc.shape[0] == 1:
        pbc = pbc.squeeze(0)

    # mesh_spacing needs .tolist() for single system (backward compat)
    mesh_spacing = params["mesh_spacing"]
    if hasattr(mesh_spacing, "tolist"):
        mesh_spacing = mesh_spacing.tolist()

    return {
        "positions": backend_data["positions"],
        "charges": backend_data["charges"],
        "cell": backend_data["cell"],
        "pbc": pbc,
        "neighbor_matrix": neighbor_matrix,
        "num_neighbors": num_neighbors,
        "neighbor_matrix_shifts": neighbor_matrix_shifts,
        "total_atoms": backend_data["total_atoms"],
        "batch_idx": None,
        "alpha": params["alpha"],
        "k_cutoff": params["k_cutoff"],
        "cutoff": params["cutoff"],
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
) -> dict:
    """Prepare a batched system for benchmarking.

    Backward-compatible wrapper that uses the new decoupled helpers internally.
    The return value structure is identical to the original implementation.

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

    Returns
    -------
    dict
        System data ready for electrostatics benchmarks, containing positions,
        charges, cell, pbc, neighbor list data, batch information, and computed parameters.
    """
    # Extract dtype string from torch.dtype for convert_to_backend
    dtype_str = str(dtype).split(".")[-1]  # e.g., "torch.float64" -> "float64"

    # Step 1: Generate system as numpy arrays
    np_data = prepare_system_numpy(supercell_size, batch_size=batch_size)

    # Step 2: Convert to torch backend
    backend_data = convert_to_backend(
        np_data, "torch", device=device, dtype_str=dtype_str
    )

    # Step 3: Compute electrostatics parameters
    params = compute_electrostatics_params(backend_data, "torch")

    # Step 4: Compute neighbor list
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = compute_neighbor_list(
        backend_data, "torch", params["cutoff"]
    )

    return {
        "positions": backend_data["positions"],
        "charges": backend_data["charges"],
        "cell": backend_data["cell"],
        "pbc": backend_data["pbc"],
        "neighbor_matrix": neighbor_matrix,
        "num_neighbors": num_neighbors,
        "neighbor_matrix_shifts": neighbor_matrix_shifts,
        "total_atoms": backend_data["total_atoms"],
        "batch_idx": backend_data["batch_idx"],
        "batch_size": batch_size,
        "alpha": params["alpha"],
        "k_cutoff": params["k_cutoff"],
        "cutoff": params["cutoff"],
        "mesh_dimensions": params["mesh_dimensions"],
        "mesh_spacing": params["mesh_spacing"],
        "spline_order": 4,
        "k_vectors_pme": params["k_vectors_pme"],
        "k_squared_pme": params["k_squared_pme"],
    }


# ==============================================================================
# nvalchemiops Backend
# ==============================================================================


def run_nvalchemiops_ewald(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Ewald summation using nvalchemiops backend."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha")
    k_cutoff = system_data.get("k_cutoff")
    k_vectors = _torch_electrostatics.generate_k_vectors_ewald_summation(cell, k_cutoff)

    neighbor_matrix_data = system_data.get("neighbor_matrix")
    neighbor_matrix_shifts = system_data.get("neighbor_matrix_shifts")

    if batch_idx is None:
        # Single system

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
            )
    else:
        # Batch system
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
            )


def run_nvalchemiops_pme(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run PME using nvalchemiops backend."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha")
    mesh_dimensions = system_data.get("mesh_dimensions")
    spline_order = system_data.get("spline_order")
    k_vectors_pme = system_data.get("k_vectors_pme")
    k_squared_pme = system_data.get("k_squared_pme")

    neighbor_matrix_data = system_data.get("neighbor_matrix")
    neighbor_matrix_shifts = system_data.get("neighbor_matrix_shifts")

    if batch_idx is None:
        # Single system

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
            )
    else:
        # Batch system

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
            )


# ==============================================================================
# nvalchemiops JAX Backend
# ==============================================================================


def prepare_jax_ewald(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
):
    """Prepare a JIT-compiled Ewald callable for benchmarking.

    Creates the ``@jax.jit`` function **once** and returns a zero-argument
    callable that executes it.  This avoids re-tracing and recompilation on
    every timing iteration (JAX's JIT cache is keyed on function-object
    identity, so recreating the decorator inside a loop defeats the cache).

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

    Returns
    -------
    callable
        A zero-argument function that runs the JIT-compiled Ewald computation
        and returns ``(energy, forces)`` (forces is ``None`` when disabled).
    """
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha")
    k_cutoff = system_data.get("k_cutoff")
    num_atoms_per_system = system_data.get("num_atoms_per_system")

    neighbor_matrix_data = system_data.get("neighbor_matrix")
    neighbor_matrix_shifts = system_data.get("neighbor_matrix_shifts")

    # Precompute Miller index bounds eagerly (3 integers that determine k-vector
    # grid shape). These must be concrete Python ints for jnp.arange inside JIT.
    # The k-vector grid construction and reciprocal-cell matmul run inside JIT.
    cell_for_miller = cell if cell.ndim == 3 else cell[None, ...]
    _bounds = _jax_electrostatics.generate_miller_indices(cell_for_miller, k_cutoff)
    _miller_bounds = (int(_bounds[0]), int(_bounds[1]), int(_bounds[2]))

    # Capture booleans and scalars in closure (concrete under JIT)
    _compute_forces = compute_forces
    _compute_virial = compute_virial
    _k_cutoff = k_cutoff

    # --- Define JIT functions ONCE (cached across all timing iterations) ------

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
            # Generate k-vectors inside JIT using precomputed Miller bounds
            k_vectors = _jax_electrostatics.generate_k_vectors_ewald_summation(
                cell, _k_cutoff, miller_bounds=_miller_bounds
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
        ):
            # Pass miller_bounds so ewald_summation generates k-vectors inside JIT
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

    return call


def prepare_jax_pme(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool = False,
):
    """Prepare a JIT-compiled PME callable for benchmarking.

    Creates the ``@jax.jit`` function **once** and returns a zero-argument
    callable that executes it.  This avoids re-tracing and recompilation on
    every timing iteration (JAX's JIT cache is keyed on function-object
    identity, so recreating the decorator inside a loop defeats the cache).

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

    Returns
    -------
    callable
        A zero-argument function that runs the JIT-compiled PME computation
        and returns ``(energy, forces)`` (forces is ``None`` when disabled).
    """
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha")
    mesh_dimensions = system_data.get("mesh_dimensions")
    spline_order = system_data.get("spline_order")

    neighbor_matrix_data = system_data.get("neighbor_matrix")
    neighbor_matrix_shifts = system_data.get("neighbor_matrix_shifts")

    # Capture booleans and scalars in closure (concrete under JIT)
    _compute_forces = compute_forces
    _compute_virial = compute_virial
    _spline_order = spline_order
    _mesh_dimensions = mesh_dimensions

    # --- Define JIT functions ONCE (cached across all timing iterations) ------

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
        ):
            # Pass k_vectors=None, k_squared=None so pme_reciprocal_space generates
            # them inside JIT boundary for fair benchmark comparison
            return _jax_electrostatics.pme_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=_mesh_dimensions,
                spline_order=_spline_order,
                batch_idx=batch_idx,
                k_vectors=None,
                k_squared=None,
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
        ):
            # Pass k_vectors=None, k_squared=None so particle_mesh_ewald generates
            # them inside JIT boundary for fair benchmark comparison
            return _jax_electrostatics.particle_mesh_ewald(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=_mesh_dimensions,
                spline_order=_spline_order,
                batch_idx=batch_idx,
                k_vectors=None,
                k_squared=None,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
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
        # Single system
        neighbor_matrix_data = system_data.get("neighbor_matrix")
        neighbor_matrix_shifts_data = system_data.get("neighbor_matrix_shifts")

        if neighbor_matrix_data is not None:
            total_atoms_val = positions.shape[0]
            # Build COO pairs from dense matrix
            # neighbor_matrix is (N, max_neighbors), fill_value=total_atoms
            # Create row indices
            row_idx = torch.arange(total_atoms_val, device=positions.device)
            row_idx = row_idx.unsqueeze(1).expand_as(neighbor_matrix_data)
            # Mask valid neighbors
            valid = neighbor_matrix_data < total_atoms_val
            src = row_idx[valid]
            dst = neighbor_matrix_data[valid]
            neighbor_indices = torch.stack([src, dst], dim=0).T  # (num_pairs, 2)
            # Compute shifts for valid pairs
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
        # For batch, we need to handle each system separately for torchpme
        # This is a limitation - torchpme doesn't natively support batched neighbors
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
    alpha = system_data.get("alpha").item()
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

    # Compute forces and/or virial via autograd
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
    alpha = system_data.get("alpha").item()
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

    # Compute forces and/or virial via autograd
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
# Benchmark Runner
# ==============================================================================


def run_benchmark(
    method: Literal["ewald", "pme"],
    backend: Literal["torch", "jax", "torchpme"],
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    compute_virial: bool,
    timer: BenchmarkTimer,
) -> dict:
    """Run a single benchmark configuration."""
    total_atoms = system_data["total_atoms"]
    batch_size = system_data.get("batch_size", 1)

    effective_virial = compute_virial

    try:
        # Define benchmark function based on method and backend
        if backend == "torch":
            if method == "ewald":

                def bench_fn():
                    return run_nvalchemiops_ewald(
                        system_data,
                        component,
                        compute_forces,
                        effective_virial,
                    )
            else:  # pme

                def bench_fn():
                    return run_nvalchemiops_pme(
                        system_data,
                        component,
                        compute_forces,
                        effective_virial,
                    )
        elif backend == "jax":
            # Create JIT function ONCE; bench_fn just executes the
            # pre-compiled function on every iteration (no retrace).
            if method == "ewald":
                bench_fn = prepare_jax_ewald(
                    system_data,
                    component,
                    compute_forces,
                    effective_virial,
                )
            else:  # pme
                bench_fn = prepare_jax_pme(
                    system_data,
                    component,
                    compute_forces,
                    effective_virial,
                )
        else:  # torchpme
            if system_data.get("batch_idx") is not None:
                return {
                    "total_atoms": total_atoms,
                    "batch_size": batch_size,
                    "method": method,
                    "backend": backend,
                    "component": component,
                    "compute_forces": compute_forces,
                    "compute_virial": effective_virial,
                    "median_time_ms": float("inf"),
                    "peak_memory_mb": None,
                    "success": False,
                    "error": "torchpme does not support native batched evaluation",
                    "error_type": "NotImplemented",
                }

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

        # Run benchmark
        timing_results = timer.time_function(bench_fn)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not timing_results["success"]:
            print(f"Benchmark failed: {timing_results.get('error', 'Unknown error')}")
            return {
                "total_atoms": total_atoms,
                "batch_size": batch_size,
                "method": method,
                "backend": backend,
                "component": component,
                "compute_forces": compute_forces,
                "compute_virial": effective_virial,
                "median_time_ms": float("inf"),
                "peak_memory_mb": timing_results.get("peak_memory_mb"),
                "success": False,
                "error": timing_results.get("error", "Unknown error"),
                "error_type": timing_results.get("error_type", "Unknown"),
            }

        return {
            "total_atoms": total_atoms,
            "batch_size": batch_size,
            "method": method,
            "backend": backend,
            "component": component,
            "compute_forces": compute_forces,
            "compute_virial": effective_virial,
            "median_time_ms": float(timing_results["median"]),
            "peak_memory_mb": timing_results.get("peak_memory_mb"),
            "compile_ms": timing_results.get("compile_ms"),
            "success": True,
        }

    except Exception as e:
        print(f"Benchmark failed: {e}")
        return {
            "total_atoms": total_atoms,
            "batch_size": batch_size,
            "method": method,
            "backend": backend,
            "component": component,
            "compute_forces": compute_forces,
            "compute_virial": effective_virial,
            "median_time_ms": float("inf"),
            "peak_memory_mb": None,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


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
        choices=["torch", "jax", "torchpme"],
        default="torch",
        help="Backend to use for benchmarking (default: torch)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ewald", "pme", "both"],
        default="both",
        help="Method to benchmark (default: both)",
    )
    parser.add_argument(
        "--gpu-sku",
        type=str,
        help="Override GPU SKU name for output files (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate backend availability
    _check_backend_available(args.backend)

    # Load config
    config = load_config(args.config)

    # Resolve framework-level backend type
    backend_type = _resolve_backend_type(args.backend)

    # Get parameters
    params = config["parameters"]
    warmup = int(params["warmup_iterations"])
    timing = int(params["timing_iterations"])
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

    # Initialize timer
    timer = BenchmarkTimer(backend=backend_type, warmup_runs=warmup, timing_runs=timing)

    # Initialize Warp (only needed for torch backend)
    if backend_type == "torch" and wp is not None:
        wp.init()

    # The CLI now only allows a single backend, so just use it directly
    methods = ["ewald", "pme"] if args.method == "both" else [args.method]
    backends = [args.backend]

    components = config.get("components", ["full"])
    compute_forces = config.get("compute_forces", True)
    compute_virial = config.get("compute_virial", False)

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
    print(f"Compute forces: {compute_forces}")
    print(f"Compute virial: {compute_virial}")
    print(f"Warmup iterations: {warmup}")
    print(f"Timing iterations: {timing}")
    print(f"Output directory: {output_dir}")

    # Run benchmarks for each system configuration
    all_results = []

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
                timer.clear_memory()

                # Prepare system
                try:
                    if args.backend == "jax":
                        np_data = prepare_system_numpy(size, batch_size=1)
                        backend_data = convert_to_backend(
                            np_data, "jax", dtype_str=dtype_str
                        )
                        params_data = compute_electrostatics_params(backend_data, "jax")
                        nl_matrix, nl_num_neighbors, nl_matrix_shifts = (
                            compute_neighbor_list(
                                backend_data, "jax", params_data["cutoff"]
                            )
                        )
                        # Assemble system_data dict matching the expected format
                        system_data = {
                            "positions": backend_data["positions"],
                            "charges": backend_data["charges"],
                            "cell": backend_data["cell"],
                            "pbc": backend_data["pbc"],
                            "neighbor_matrix": nl_matrix,
                            "num_neighbors": nl_num_neighbors,
                            "neighbor_matrix_shifts": nl_matrix_shifts,
                            "total_atoms": backend_data["total_atoms"],
                            "num_atoms_per_system": backend_data[
                                "num_atoms_per_system"
                            ],
                            "batch_idx": None,
                            "alpha": params_data["alpha"],
                            "k_cutoff": params_data["k_cutoff"],
                            "cutoff": params_data["cutoff"],
                            "mesh_dimensions": params_data["mesh_dimensions"],
                            "mesh_spacing": params_data["mesh_spacing"],
                            "spline_order": 4,
                            "k_vectors_pme": params_data["k_vectors_pme"],
                            "k_squared_pme": params_data["k_squared_pme"],
                        }
                    else:
                        system_data = prepare_single_system(size, device, dtype)
                except Exception as e:
                    print(f"    Failed to prepare system: {e}")
                    traceback.print_exc()
                    continue

                for method in methods:
                    for backend in backends:
                        for component in components:
                            result = run_benchmark(
                                method,
                                backend,
                                system_data,
                                component,
                                compute_forces,
                                compute_virial,
                                timer,
                            )
                            result["supercell_size"] = size
                            result["mode"] = mode
                            all_results.append(result)

                            if result["success"]:
                                throughput = (
                                    result["total_atoms"]
                                    / result["median_time_ms"]
                                    * 1000
                                )
                                mem_str = ""
                                if result.get("peak_memory_mb"):
                                    mem_str = f" | {result['peak_memory_mb']:.1f} MB"
                                compile_str = ""
                                if result.get("compile_ms") is not None:
                                    compile_str = (
                                        f" | warmup {result['compile_ms']:.0f} ms"
                                    )
                                print(
                                    f"    {method:5s} {backend:16s} {component:10s}: "
                                    f"{result['median_time_ms']:.3f} ms "
                                    f"({throughput:.1f} atoms/s){mem_str}{compile_str}"
                                )
                            else:
                                print(
                                    f"    {method:5s} {backend:16s} {component:10s}: "
                                    f"FAILED ({result.get('error_type', 'Unknown')})"
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
                timer.clear_memory()

                # Prepare system
                try:
                    if args.backend == "jax":
                        np_data = prepare_system_numpy(base_size, batch_size=batch_size)
                        backend_data = convert_to_backend(
                            np_data, "jax", dtype_str=dtype_str
                        )
                        params_data = compute_electrostatics_params(backend_data, "jax")
                        nl_matrix, nl_num_neighbors, nl_matrix_shifts = (
                            compute_neighbor_list(
                                backend_data, "jax", params_data["cutoff"]
                            )
                        )
                        system_data = {
                            "positions": backend_data["positions"],
                            "charges": backend_data["charges"],
                            "cell": backend_data["cell"],
                            "pbc": backend_data["pbc"],
                            "neighbor_matrix": nl_matrix,
                            "num_neighbors": nl_num_neighbors,
                            "neighbor_matrix_shifts": nl_matrix_shifts,
                            "total_atoms": backend_data["total_atoms"],
                            "num_atoms_per_system": backend_data[
                                "num_atoms_per_system"
                            ],
                            "batch_idx": backend_data["batch_idx"],
                            "batch_size": batch_size,
                            "alpha": params_data["alpha"],
                            "k_cutoff": params_data["k_cutoff"],
                            "cutoff": params_data["cutoff"],
                            "mesh_dimensions": params_data["mesh_dimensions"],
                            "mesh_spacing": params_data["mesh_spacing"],
                            "spline_order": 4,
                            "k_vectors_pme": params_data["k_vectors_pme"],
                            "k_squared_pme": params_data["k_squared_pme"],
                        }
                    else:
                        system_data = prepare_batch_system(
                            base_size, batch_size, device, dtype
                        )
                except Exception as e:
                    print(f"    Failed to prepare system: {e}")
                    traceback.print_exc()
                    continue

                for method in methods:
                    for backend in backends:
                        for component in components:
                            result = run_benchmark(
                                method,
                                backend,
                                system_data,
                                component,
                                compute_forces,
                                compute_virial,
                                timer,
                            )
                            result["supercell_size"] = base_size
                            result["mode"] = mode
                            all_results.append(result)

                            if result["success"]:
                                throughput = (
                                    result["total_atoms"]
                                    / result["median_time_ms"]
                                    * 1000
                                )
                                mem_str = ""
                                if result.get("peak_memory_mb"):
                                    mem_str = f" | {result['peak_memory_mb']:.1f} MB"
                                compile_str = ""
                                if result.get("compile_ms") is not None:
                                    compile_str = (
                                        f" | warmup {result['compile_ms']:.0f} ms"
                                    )
                                print(
                                    f"    {method:5s} {backend:16s} {component:10s}: "
                                    f"{result['median_time_ms']:.3f} ms "
                                    f"({throughput:.1f} atoms/s){mem_str}{compile_str}"
                                )
                            else:
                                print(
                                    f"    {method:5s} {backend:16s} {component:10s}: "
                                    f"FAILED ({result.get('error_type', 'Unknown')})"
                                )

    # Save results
    if all_results:
        # Group by method and backend
        for method in methods:
            for backend in backends:
                method_results = [
                    r
                    for r in all_results
                    if r["method"] == method and r["backend"] == backend
                ]
                if method_results:
                    output_file = (
                        output_dir
                        / f"electrostatics_benchmark_{method}_{backend}_{gpu_sku}.csv"
                    )
                    # Collect all fieldnames across all results
                    all_fieldnames = []
                    seen = set()
                    for r in method_results:
                        for k in r.keys():
                            if k not in seen:
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
