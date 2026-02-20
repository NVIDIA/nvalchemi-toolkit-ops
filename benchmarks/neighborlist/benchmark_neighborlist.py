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
Neighbor List Scaling Benchmarks
=================================

CLI tool to benchmark neighbor list algorithms and generate CSV files
for documentation. Supports multiple backends (torch, jax) with results
saved using GPU-specific naming:
`neighbor_list_benchmark_<method>_<backend>_<gpu_sku>.csv`

Usage:
    python benchmark_neighborlist.py --config benchmark_config.yaml --backend torch
    python benchmark_neighborlist.py --config benchmark_config.yaml --backend jax

The config file specifies which methods to benchmark and their parameters.
Results are saved per-method to allow selective benchmarking.
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.utils import BackendType, BenchmarkTimer

# Guarded torch imports
try:
    import torch

    from nvalchemiops.torch.neighbors import neighbor_list as torch_neighbor_list
    from nvalchemiops.torch.neighbors.batch_cell_list import (
        estimate_batch_cell_list_sizes as torch_estimate_batch_cell_list_sizes,
    )
    from nvalchemiops.torch.neighbors.cell_list import (
        estimate_cell_list_sizes as torch_estimate_cell_list_sizes,
    )
    from nvalchemiops.torch.neighbors.neighbor_utils import (
        allocate_cell_list as torch_allocate_cell_list,
    )
    from nvalchemiops.torch.neighbors.neighbor_utils import (
        compute_naive_num_shifts as torch_compute_naive_num_shifts,
    )
    from nvalchemiops.torch.neighbors.neighbor_utils import (
        estimate_max_neighbors as torch_estimate_max_neighbors,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    torch_neighbor_list = None  # type: ignore
    torch_estimate_batch_cell_list_sizes = None  # type: ignore
    torch_estimate_cell_list_sizes = None  # type: ignore
    torch_allocate_cell_list = None  # type: ignore
    torch_compute_naive_num_shifts = None  # type: ignore
    torch_estimate_max_neighbors = None  # type: ignore

# Guarded JAX imports
try:
    import jax
    import jax.numpy as jnp

    from nvalchemiops.jax.neighbors import neighbor_list as jax_neighbor_list
    from nvalchemiops.jax.neighbors.batch_cell_list import (
        batch_cell_list as jax_batch_cell_list,
    )
    from nvalchemiops.jax.neighbors.batch_cell_list import (
        estimate_batch_cell_list_sizes as jax_estimate_batch_cell_list_sizes,
    )
    from nvalchemiops.jax.neighbors.cell_list import (
        estimate_cell_list_sizes as jax_estimate_cell_list_sizes,
    )
    from nvalchemiops.jax.neighbors.neighbor_utils import (
        allocate_cell_list as jax_allocate_cell_list,
    )
    from nvalchemiops.jax.neighbors.neighbor_utils import (
        compute_naive_num_shifts as jax_compute_naive_num_shifts,
    )
    from nvalchemiops.jax.neighbors.neighbor_utils import (
        estimate_max_neighbors as jax_estimate_max_neighbors,
    )

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    jax_neighbor_list = None  # type: ignore
    jax_batch_cell_list = None  # type: ignore
    jax_estimate_batch_cell_list_sizes = None  # type: ignore
    jax_estimate_cell_list_sizes = None  # type: ignore
    jax_allocate_cell_list = None  # type: ignore
    jax_compute_naive_num_shifts = None  # type: ignore
    jax_estimate_max_neighbors = None  # type: ignore


def get_gpu_sku(backend: BackendType) -> str:
    """Get GPU SKU name for filename generation.

    Uses NVML for reliable, backend-agnostic GPU name detection.
    Falls back to "cpu" if no GPU is available.

    Parameters
    ----------
    backend : BackendType
        Backend in use (used to check GPU availability).

    Returns
    -------
    str
        Cleaned GPU SKU string suitable for filenames (e.g., "h100-80gb-hbm3").
    """
    # First check if we even have a GPU based on the backend
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
            has_gpu = False  # Will be implemented later

    if not has_gpu:
        return "cpu"

    # Use NVML for reliable GPU name detection
    from benchmarks.utils import _nvml_get_gpu_sku

    return _nvml_get_gpu_sku()


def _check_backend_available(backend: str) -> None:
    """Validate that the requested backend is installed.

    Parameters
    ----------
    backend : str
        CLI backend choice.

    Raises
    ------
    SystemExit
        If the required backend is not available.
    """
    match backend:
        case "torch":
            if not TORCH_AVAILABLE:
                print("ERROR: torch backend requested but torch is not installed.")
                sys.exit(1)
        case "jax":
            if not JAX_AVAILABLE:
                print("ERROR: jax backend requested but JAX is not installed.")
                sys.exit(1)
        case _:
            print("ERROR: selected backend is not supported.")
            sys.exit(1)


def load_config(config_path: Path) -> dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict) -> None:
    """Validate benchmark configuration structure."""
    required_keys = ["methods", "parameters"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config missing required key: {key}")

    param_keys = ["cutoff", "warmup_iterations", "timing_iterations", "dtype"]
    for key in param_keys:
        if key not in config["parameters"]:
            raise ValueError(f"Config parameters missing required key: {key}")

    for method_config in config["methods"]:
        if (
            "name" not in method_config
            or "atom_counts" not in method_config
            or "batch_sizes" not in method_config
        ):
            raise ValueError(f"Method config missing required keys: {method_config}")


def create_crystal_system_numpy(
    num_atoms: int,
    lattice_type: str = "fcc",
    lattice_constant: float = 4.0,
    dtype_str: str = "float64",
) -> dict[str, np.ndarray | int]:
    """Create a crystalline system as numpy arrays (no framework dependency).

    This is a numpy-only port of ``benchmarks.systems.create_crystal_system``,
    used to build backend-agnostic system data before converting to a specific
    framework.

    Parameters
    ----------
    num_atoms : int
        Target number of atoms in the system.
    lattice_type : str, default="fcc"
        Type of crystal lattice ("fcc", "bcc", "simple_cubic").
    lattice_constant : float, default=4.0
        Lattice constant in Angstroms.
    dtype_str : str, default="float64"
        Dtype string (e.g., "float32", "float64").

    Returns
    -------
    dict
        Dictionary containing numpy arrays:
        - positions: (num_atoms, 3) float
        - atomic_charges: (num_atoms,) float
        - atomic_numbers: (num_atoms,) int32
        - cell: (1, 3, 3) float
        - pbc: (3,) bool
        - num_atoms: int
    """
    dtype = getattr(np, dtype_str)

    if lattice_type == "fcc":
        atoms_per_cell = 4
        basis = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
            dtype=dtype,
        )
    elif lattice_type == "bcc":
        atoms_per_cell = 2
        basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=dtype)
    elif lattice_type == "simple_cubic":
        atoms_per_cell = 1
        basis = np.array([[0.0, 0.0, 0.0]], dtype=dtype)
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")

    n_cells = int(np.ceil((num_atoms / atoms_per_cell) ** (1 / 3)))

    positions = []
    charges = []
    atomic_numbers = []

    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                if len(positions) >= num_atoms:
                    break
                cell_origin = np.array([i, j, k], dtype=dtype) * lattice_constant
                for atom_idx, pos in enumerate(basis):
                    if len(positions) >= num_atoms:
                        break
                    position = cell_origin + pos * lattice_constant
                    positions.append(position)
                    charge = 1.0 if (i + j + k + atom_idx) % 2 == 0 else -1.0
                    charges.append(charge)
                    atomic_num = 6 if (i + j + k + atom_idx) % 2 == 0 else 8
                    atomic_numbers.append(atomic_num)

    positions = np.array(positions[:num_atoms], dtype=dtype)
    charges = np.array(charges[:num_atoms], dtype=dtype)
    atomic_numbers_arr = np.array(atomic_numbers[:num_atoms], dtype=np.int32)

    # Ensure charge neutrality
    if abs(charges.sum()) > 1e-10:
        charges[-1] -= charges.sum()

    cell_size = n_cells * lattice_constant
    cell = np.zeros((1, 3, 3), dtype=dtype)
    cell[0, 0, 0] = cell_size
    cell[0, 1, 1] = cell_size
    cell[0, 2, 2] = cell_size

    pbc = np.array([True, True, True])

    return {
        "positions": positions,
        "atomic_charges": charges,
        "atomic_numbers": atomic_numbers_arr,
        "cell": cell,
        "pbc": pbc,
        "num_atoms": num_atoms,
    }


def convert_system_to_backend(
    np_data: dict[str, np.ndarray | int],
    backend: BackendType,
    device: str = "cuda",
    dtype_str: str = "float32",
) -> dict:
    """Convert numpy system arrays to backend-specific tensors.

    Parameters
    ----------
    np_data : dict
        Dictionary from ``create_crystal_system_numpy()``.
    backend : BackendType
        Target backend ("torch", "jax").
    device : str, default="cuda"
        Device string (used by torch backend).
    dtype_str : str, default="float32"
        Dtype string like "float32".

    Returns
    -------
    dict
        Dictionary with backend-specific arrays.
    """
    float_keys = ["positions", "atomic_charges"]
    int_keys = ["atomic_numbers"]
    bool_keys = ["pbc"]
    # cell needs special handling for shape

    result = {"num_atoms": np_data["num_atoms"]}

    match backend:
        case "torch":
            dtype = getattr(torch, dtype_str)
            for key in float_keys:
                result[key] = torch.tensor(np_data[key], dtype=dtype, device=device)
            for key in int_keys:
                result[key] = torch.tensor(
                    np_data[key], dtype=torch.int32, device=device
                )
            for key in bool_keys:
                result[key] = torch.tensor(
                    np_data[key], dtype=torch.bool, device=device
                )
            # Cell: keep (1, 3, 3) shape
            result["cell"] = torch.tensor(np_data["cell"], dtype=dtype, device=device)
        case "jax":
            dtype = getattr(jnp, dtype_str)
            for key in float_keys:
                result[key] = jnp.array(np_data[key], dtype=dtype)
            for key in int_keys:
                result[key] = jnp.array(np_data[key], dtype=jnp.int32)
            for key in bool_keys:
                result[key] = jnp.array(np_data[key], dtype=jnp.bool_)
            # Cell: keep (1, 3, 3) shape
            result["cell"] = jnp.array(np_data["cell"], dtype=dtype)
        case _:
            raise NotImplementedError(f"{backend} backend not yet supported")

    return result


# %%
# Utility Functions
# -----------------
# Helper functions for preparing inputs and running benchmarks.


def prepare_inputs(
    method,
    atoms_per_system,
    batch_size,
    cutoff,
    device,
    dtype_str,
    backend: BackendType = "torch",
):
    """Prepare inputs for a specific neighbor list method.

    Builds system data in numpy, converts to the target backend, and
    pre-allocates method-specific buffers.

    Parameters
    ----------
    method : str
        Neighbor list method name.
    atoms_per_system : int
        Number of atoms per system.
    batch_size : int
        Number of systems in batch.
    cutoff : float
        Cutoff distance.
    device : str
        Device string (used by torch backend).
    dtype_str : str
        Dtype string like "float32".
    backend : BackendType, default="torch"
        Backend to use.

    Returns
    -------
    dict
        Input dictionary ready for the backend's neighbor_list function.
    """
    is_batch = "batch" in method

    if is_batch:
        # --- Build numpy data for each system, then concatenate ---
        all_np_systems = []
        try:
            for i in range(batch_size):
                np_system = create_crystal_system_numpy(
                    atoms_per_system,
                    lattice_type="fcc",
                    dtype_str=dtype_str,
                )
                if np_system["positions"].shape[0] != atoms_per_system:
                    raise ValueError(
                        f"System {i}: requested {atoms_per_system} atoms, got {np_system['positions'].shape[0]}. "
                        f"FCC lattice may not support exact atom count."
                    )
                all_np_systems.append(np_system)
        except Exception as e:
            raise ValueError(
                f"Error creating batch systems (atoms_per_system={atoms_per_system}, batch_size={batch_size}): {e}"
            ) from e

        # Concatenate numpy arrays
        positions_np = np.concatenate([s["positions"] for s in all_np_systems], axis=0)
        cells_np = np.concatenate(
            [s["cell"] for s in all_np_systems], axis=0
        )  # Each is (1,3,3) -> stack to (batch_size,3,3)
        pbc_np = np.stack([s["pbc"] for s in all_np_systems], axis=0)  # (batch_size, 3)
        total_atoms_actual = positions_np.shape[0]

        batch_idx_np = np.zeros(total_atoms_actual, dtype=np.int32)
        for i in range(batch_size):
            start = i * atoms_per_system
            end = (i + 1) * atoms_per_system
            batch_idx_np[start:end] = i

        batch_ptr_np = np.arange(
            0, (batch_size + 1) * atoms_per_system, atoms_per_system, dtype=np.int32
        )

        # --- Convert to backend ---
        match backend:
            case "torch":
                dtype = getattr(torch, dtype_str)
                positions = torch.tensor(positions_np, dtype=dtype, device=device)
                cells = torch.tensor(cells_np, dtype=dtype, device=device)
                pbc = torch.tensor(pbc_np, dtype=torch.bool, device=device)
                batch_idx = torch.tensor(batch_idx_np, dtype=torch.int32, device=device)
                batch_ptr = torch.tensor(batch_ptr_np, dtype=torch.int32, device=device)
            case "jax":
                dtype = getattr(jnp, dtype_str)
                positions = jnp.array(positions_np, dtype=dtype)
                cells = jnp.array(cells_np, dtype=dtype)
                pbc = jnp.array(pbc_np, dtype=jnp.bool_)
                batch_idx = jnp.array(batch_idx_np, dtype=jnp.int32)
                batch_ptr = jnp.array(batch_ptr_np, dtype=jnp.int32)
            case _:
                raise NotImplementedError(f"{backend} backend not yet supported")

        # --- Pre-allocate buffers ---
        match backend:
            case "torch":
                max_neighbors = torch_estimate_max_neighbors(
                    cutoff, atomic_density=0.35, safety_factor=1.0
                )
                neighbor_matrix = torch.full(
                    (total_atoms_actual, max_neighbors),
                    total_atoms_actual,
                    dtype=torch.int32,
                    device=device,
                )
                neighbor_matrix_shifts = torch.zeros(
                    (total_atoms_actual, max_neighbors, 3),
                    dtype=torch.int32,
                    device=device,
                )
                num_neighbors = torch.zeros(
                    total_atoms_actual, dtype=torch.int32, device=device
                )
            case "jax":
                max_neighbors = jax_estimate_max_neighbors(
                    cutoff, atomic_density=0.35, safety_factor=1.0
                )
                neighbor_matrix = jnp.full(
                    (total_atoms_actual, max_neighbors),
                    total_atoms_actual,
                    dtype=jnp.int32,
                )
                neighbor_matrix_shifts = jnp.zeros(
                    (total_atoms_actual, max_neighbors, 3),
                    dtype=jnp.int32,
                )
                num_neighbors = jnp.zeros(total_atoms_actual, dtype=jnp.int32)

        inputs = {
            "positions": positions,
            "cutoff": cutoff,
            "cell": cells,
            "pbc": pbc,
            "method": method,
            "batch_idx": batch_idx,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "num_neighbors": num_neighbors,
        }

        # Method-specific allocations
        if "naive" in method:
            match backend:
                case "torch":
                    shift_range_per_dimension, shift_offset, total_shifts = (
                        torch_compute_naive_num_shifts(cells, cutoff, pbc)
                    )
                case "jax":
                    shift_range_per_dimension, shift_offset, total_shifts = (
                        jax_compute_naive_num_shifts(cells, cutoff, pbc)
                    )
            inputs["shift_range_per_dimension"] = shift_range_per_dimension
            inputs["shift_offset"] = shift_offset
            inputs["total_shifts"] = total_shifts
            inputs["batch_ptr"] = batch_ptr
        elif "cell_list" in method:
            match backend:
                case "torch":
                    max_total_cells, neighbor_search_radius = (
                        torch_estimate_batch_cell_list_sizes(cells, pbc, cutoff)
                    )
                    cell_list_cache = torch_allocate_cell_list(
                        total_atoms_actual,
                        max_total_cells,
                        neighbor_search_radius,
                        device,
                    )
                case "jax":
                    # JAX version has different signature and return values
                    max_total_cells, _cells_per_dim, neighbor_search_radius = (
                        jax_estimate_batch_cell_list_sizes(
                            positions,
                            batch_ptr=batch_ptr,
                            batch_idx=batch_idx,
                            cell=cells,
                            cutoff=cutoff,
                            pbc=pbc,
                        )
                    )
                    cell_list_cache = jax_allocate_cell_list(
                        total_atoms_actual, max_total_cells, neighbor_search_radius
                    )
            inputs["cells_per_dimension"] = cell_list_cache[0]
            inputs["neighbor_search_radius"] = cell_list_cache[1]
            inputs["atom_periodic_shifts"] = cell_list_cache[2]
            inputs["atom_to_cell_mapping"] = cell_list_cache[3]
            inputs["atoms_per_cell_count"] = cell_list_cache[4]
            inputs["cell_atom_start_indices"] = cell_list_cache[5]
            inputs["cell_atom_list"] = cell_list_cache[6]

        return inputs

    else:
        # --- Single system ---
        np_system = create_crystal_system_numpy(
            atoms_per_system,
            lattice_type="fcc",
            dtype_str=dtype_str,
        )

        positions_np = np_system["positions"]
        cell_np = np_system["cell"]  # Already (1, 3, 3)
        pbc_np = np_system["pbc"].reshape(1, 3)
        total_atoms_actual = positions_np.shape[0]

        # --- Convert to backend ---
        match backend:
            case "torch":
                dtype = getattr(torch, dtype_str)
                positions = torch.tensor(positions_np, dtype=dtype, device=device)
                cell = torch.tensor(cell_np, dtype=dtype, device=device)
                pbc = torch.tensor(pbc_np, dtype=torch.bool, device=device)
            case "jax":
                dtype = getattr(jnp, dtype_str)
                positions = jnp.array(positions_np, dtype=dtype)
                cell = jnp.array(cell_np, dtype=dtype)
                pbc = jnp.array(pbc_np, dtype=jnp.bool_)
            case _:
                raise NotImplementedError(f"{backend} backend not yet supported")

        # --- Pre-allocate buffers ---
        match backend:
            case "torch":
                max_neighbors = torch_estimate_max_neighbors(
                    cutoff, atomic_density=0.35, safety_factor=1.0
                )
                neighbor_matrix = torch.full(
                    (total_atoms_actual, max_neighbors),
                    total_atoms_actual,
                    dtype=torch.int32,
                    device=device,
                )
                neighbor_matrix_shifts = torch.zeros(
                    (total_atoms_actual, max_neighbors, 3),
                    dtype=torch.int32,
                    device=device,
                )
                num_neighbors = torch.zeros(
                    total_atoms_actual, dtype=torch.int32, device=device
                )
            case "jax":
                max_neighbors = jax_estimate_max_neighbors(
                    cutoff, atomic_density=0.35, safety_factor=1.0
                )
                neighbor_matrix = jnp.full(
                    (total_atoms_actual, max_neighbors),
                    total_atoms_actual,
                    dtype=jnp.int32,
                )
                neighbor_matrix_shifts = jnp.zeros(
                    (total_atoms_actual, max_neighbors, 3),
                    dtype=jnp.int32,
                )
                num_neighbors = jnp.zeros(total_atoms_actual, dtype=jnp.int32)

        inputs = {
            "positions": positions,
            "cutoff": cutoff,
            "cell": cell,
            "pbc": pbc,
            "method": method,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "num_neighbors": num_neighbors,
        }

        # Method-specific allocations
        if "naive" in method:
            match backend:
                case "torch":
                    shift_range_per_dimension, shift_offset, total_shifts = (
                        torch_compute_naive_num_shifts(cell, cutoff, pbc)
                    )
                case "jax":
                    shift_range_per_dimension, shift_offset, total_shifts = (
                        jax_compute_naive_num_shifts(cell, cutoff, pbc)
                    )
            inputs["shift_range_per_dimension"] = shift_range_per_dimension
            inputs["shift_offset"] = shift_offset
            inputs["total_shifts"] = total_shifts
        elif "cell_list" in method:
            match backend:
                case "torch":
                    max_total_cells, neighbor_search_radius = (
                        torch_estimate_cell_list_sizes(cell, pbc, cutoff)
                    )
                    cell_list_cache = torch_allocate_cell_list(
                        total_atoms_actual,
                        max_total_cells,
                        neighbor_search_radius,
                        device,
                    )
                case "jax":
                    # JAX version has different signature and return values
                    max_total_cells, _cells_per_dim, neighbor_search_radius = (
                        jax_estimate_cell_list_sizes(positions, cell, cutoff, pbc)
                    )
                    cell_list_cache = jax_allocate_cell_list(
                        total_atoms_actual, max_total_cells, neighbor_search_radius
                    )
            inputs["cells_per_dimension"] = cell_list_cache[0]
            inputs["neighbor_search_radius"] = cell_list_cache[1]
            inputs["atom_periodic_shifts"] = cell_list_cache[2]
            inputs["atom_to_cell_mapping"] = cell_list_cache[3]
            inputs["atoms_per_cell_count"] = cell_list_cache[4]
            inputs["cell_atom_start_indices"] = cell_list_cache[5]
            inputs["cell_atom_list"] = cell_list_cache[6]

        return inputs


def run_single_benchmark(
    method,
    num_atoms_per_system,
    batch_size,
    timer,
    cutoff,
    device,
    dtype_str,
    backend: BackendType = "torch",
):
    """Run a single benchmark configuration."""
    # Prepare inputs (includes pre-allocated tensors)
    inputs = prepare_inputs(
        method, num_atoms_per_system, batch_size, cutoff, device, dtype_str, backend
    )

    # Dispatch to correct backend and time the neighbor list construction
    match backend:
        case "torch":
            timing_results = timer.time_function(torch_neighbor_list, **inputs)
        case "jax":
            # The JAX neighbor_list API differs from Torch in several ways:
            # 1. Static values (cutoff, method, total_shifts) must be compile-time
            #    constants, so we capture them in the jit closure.
            # 2. For batch_naive method, max_atoms_per_system must be explicit inside jit.
            # 3. For cell_list methods, max_total_cells must be computed outside jit
            #    (estimate_cell_list_sizes is not JIT-compatible).
            # 4. The cell_list functions don't accept pre-allocated buffers
            #    (neighbor_matrix, cells_per_dimension, etc.) unlike Torch.
            #
            # We separate kwargs into static (closure) vs traced (jit args).

            # Static keys: non-array values that must be compile-time constants
            static_keys = {"cutoff", "method", "total_shifts"}

            method = inputs.get("method", "")

            # For batch_naive method, add max_atoms_per_system (required inside jit)
            if method == "batch_naive":
                # max_atoms_per_system = atoms per system (all systems same size)
                inputs["max_atoms_per_system"] = num_atoms_per_system
                static_keys.add("max_atoms_per_system")

            # For cell_list methods, compute max_total_cells outside jit
            # (estimate functions are not JIT-compatible as they compute concrete ints)
            if "cell_list" in method:
                positions = inputs["positions"]
                cell = inputs["cell"]
                pbc = inputs["pbc"]
                if method == "cell_list":
                    max_total_cells, _, _ = jax_estimate_cell_list_sizes(
                        positions, cell, cutoff, pbc
                    )
                else:  # batch_cell_list
                    # batch_cell_list needs batch_ptr explicitly (prepare_inputs
                    # doesn't add it for cell_list methods). Create it from known
                    # batch_size and num_atoms_per_system.
                    batch_ptr = jnp.arange(
                        0,
                        (batch_size + 1) * num_atoms_per_system,
                        num_atoms_per_system,
                        dtype=jnp.int32,
                    )
                    inputs["batch_ptr"] = batch_ptr
                    batch_idx = inputs.get("batch_idx")
                    max_total_cells, _, _ = jax_estimate_batch_cell_list_sizes(
                        positions,
                        batch_ptr=batch_ptr,
                        batch_idx=batch_idx,
                        cell=cell,
                        cutoff=cutoff,
                        pbc=pbc,
                    )
                inputs["max_total_cells"] = max_total_cells
                static_keys.add("max_total_cells")

            # Keys for buffers that JAX functions don't accept as kwargs.
            # The JAX cell_list/batch_cell_list have simplified signatures compared
            # to Torch — they don't accept pre-allocated output buffers or cell list
            # cache arrays. These are Torch-specific pre-allocation optimizations.
            jax_unsupported_keys = {
                # Output buffers (JAX allocates internally)
                "neighbor_matrix",
                "neighbor_matrix_shifts",
                "num_neighbors",
                # Cell list cache buffers (JAX allocates internally)
                "cells_per_dimension",
                "neighbor_search_radius",
                "atom_periodic_shifts",
                "atom_to_cell_mapping",
                "atoms_per_cell_count",
                "cell_atom_start_indices",
                "cell_atom_list",
            }

            static_kwargs = {k: inputs[k] for k in static_keys if k in inputs}
            array_kwargs = {
                k: v
                for k, v in inputs.items()
                if k not in static_keys and k not in jax_unsupported_keys
            }

            # Remove None values — JAX jit doesn't trace None well
            # and these correspond to optional args not used in this config
            array_kwargs = {k: v for k, v in array_kwargs.items() if v is not None}

            # For batch_cell_list, we need to call jax_batch_cell_list directly
            # because the neighbor_list() dispatcher doesn't properly forward
            # batch_ptr to the underlying function.
            if method == "batch_cell_list":
                # Remove 'method' from static_kwargs (batch_cell_list doesn't take it)
                static_kwargs.pop("method", None)

                @jax.jit
                def jit_neighbor_list(**kw):
                    return jax_batch_cell_list(**static_kwargs, **kw)
            else:

                @jax.jit
                def jit_neighbor_list(**kw):
                    return jax_neighbor_list(**static_kwargs, **kw)

            timing_results = timer.time_function(jit_neighbor_list, **array_kwargs)
        case _:
            raise NotImplementedError(f"{backend} backend not yet supported")

    # Check if benchmark was successful
    if not timing_results.get("success", False):
        # Return error result with inf for median_time_us
        return {
            "method": method,
            "total_atoms": num_atoms_per_system * batch_size
            if "batch" in method
            else num_atoms_per_system,
            "atoms_per_system": num_atoms_per_system,
            "total_neighbors": 0,
            "batch_size": batch_size,
            "backend": backend,
            "median_time_ms": float("inf"),
            "success": False,
            "error_type": timing_results.get("error_type", "Unknown"),
            "peak_memory_mb": timing_results.get("peak_memory_mb"),
        }

    # Extract number of neighbors from the pre-allocated num_neighbors tensor
    # (neighbor_list was already called during timing, results are in the tensors)
    match backend:
        case "torch":
            num_neighbors_total = inputs["num_neighbors"].sum().item()
        case "jax":
            last_result = timing_results.get("last_result")
            if last_result is not None:
                # neighbor_list returns (neighbor_matrix, num_neighbors, shifts)
                _, returned_num_neighbors, _ = last_result
                num_neighbors_total = int(jnp.sum(returned_num_neighbors))
            else:
                num_neighbors_total = 0

    # Convert from ms to us
    median_time_ms = timing_results.get("median")

    return {
        "method": method,
        "total_atoms": num_atoms_per_system * batch_size
        if "batch" in method
        else num_atoms_per_system,
        "atoms_per_system": num_atoms_per_system,
        "total_neighbors": num_neighbors_total,
        "batch_size": batch_size,
        "backend": backend,
        "median_time_ms": float(median_time_ms),
        "peak_memory_mb": timing_results.get("peak_memory_mb"),
        "success": True,
    }


def run_benchmarks_for_method(
    method_config: dict,
    gpu_sku: str,
    cutoff: float,
    device: str,
    dtype_str: str,
    timer: BenchmarkTimer,
    output_dir: Path,
    backend: BackendType = "torch",
) -> None:
    """Run benchmarks for a single method and save results."""
    method = method_config["name"]
    atom_counts = method_config["atom_counts"]
    batch_sizes = method_config["batch_sizes"]
    is_batch_method = "batch" in method

    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {method}")
    print(f"{'=' * 70}")

    all_results = []

    for atoms in atom_counts:
        for batch_size in batch_sizes:
            # Pre-validate configuration for batch methods
            if is_batch_method:
                atoms_per_system = atoms
                total_atoms = atoms_per_system * batch_size

                if atoms_per_system < 1:
                    # Skip invalid configuration silently
                    continue
            else:
                atoms_per_system = atoms
                total_atoms = atoms_per_system

            try:
                result = run_single_benchmark(
                    method,
                    atoms_per_system,
                    batch_size,
                    timer,
                    cutoff,
                    device,
                    dtype_str,
                    backend,
                )
                result["method"] = method.replace("_", "-")
                error_type = (
                    None if "error_type" not in result else result.pop("error_type")
                )
                all_results.append(result)

                if result.get("success", True):
                    # Successful benchmark
                    atoms_str = f"{result['total_atoms']:,}"
                    time_str = f"{result['median_time_ms']:.1f}"
                    neighbors_str = f"{result['total_neighbors']:,}"

                    print(
                        f"  {atoms_str:>8} atoms, batch={batch_size:2d}: "
                        f"{time_str:>8} ms, {neighbors_str:>10} neighbors"
                    )
                else:
                    # Failed benchmark
                    atoms_str = f"{result['total_atoms']:,}"

                    print(
                        f"  {atoms_str:>8} atoms, batch={batch_size:2d}: FAILED ({error_type})"
                    )
                    if error_type in ["OOM", "Timeout"]:
                        print("    └─ Skipping larger systems for this method")
                        break

            except ValueError as e:
                # Handle configuration errors
                print(
                    f"  {total_atoms:>8} atoms, batch={batch_size:2d}: SKIPPED - {str(e)}"
                )
                continue
            except Exception as e:
                # Print full traceback for debugging
                print(
                    f"  {total_atoms:>8} atoms, batch={batch_size:2d}: EXCEPTION - {type(e).__name__}: {e}"
                )
                print("\nFULL TRACEBACK:")
                traceback.print_exc()

                # Add failed result
                all_results.append(
                    {
                        "method": method.replace("_", "-"),
                        "total_atoms": total_atoms,
                        "atoms_per_system": atoms_per_system,
                        "total_neighbors": 0,
                        "batch_size": batch_size,
                        "median_time_ms": float("inf"),
                    }
                )

                # Break on critical errors
                if isinstance(e, (IndexError, KeyError, RuntimeError)):
                    print("  Critical error - skipping remaining configurations")
                    break

    # Save results to CSV with GPU-specific name
    if all_results:
        output_file = (
            output_dir
            / f"neighbor_list_benchmark_{method.replace('_', '-')}_{backend}_{gpu_sku}.csv"
        )
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✓ Results saved to: {output_file}")

        # Print summary
        successful = [r for r in all_results if r.get("success", True)]
        failed = [r for r in all_results if not r.get("success", True)]
        print(
            f"  Total: {len(all_results)} | Successful: {len(successful)} | Failed: {len(failed)}"
        )


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark neighbor list algorithms and generate CSV files for documentation"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./benchmark_results"),
        help="Output directory for CSV files (default: ./benchmark_results)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Specific methods to benchmark (default: all methods in config)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "jax"],
        default="torch",
        help="Backend to use for benchmarking (default: torch)",
    )
    parser.add_argument(
        "--gpu-sku",
        type=str,
        help="Override GPU SKU name for output files (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate backend availability
    _check_backend_available(args.backend)

    # Load and validate config
    config = load_config(args.config)
    validate_config(config)

    # Get parameters
    params = config["parameters"]
    cutoff = float(params["cutoff"])
    warmup = int(params["warmup_iterations"])
    timing = int(params["timing_iterations"])
    dtype_str = params["dtype"]

    # Resolve backend type
    backend_type: BackendType = args.backend

    # Backend-specific setup
    device = "cpu"  # Default
    match backend_type:
        case "torch":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        case "jax":
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
    timer = BenchmarkTimer(warmup_runs=warmup, timing_runs=timing, backend=backend_type)

    # Print configuration
    print("=" * 70)
    print("NEIGHBOR LIST BENCHMARK")
    print("=" * 70)
    print(f"Backend: {args.backend}")
    print(f"Device: {device}")
    print(f"GPU SKU: {gpu_sku}")
    print(f"Cutoff: {cutoff} Å")
    print(f"Dtype: {dtype_str}")
    print(f"Warmup iterations: {warmup}")
    print(f"Timing iterations: {timing}")
    print(f"Output directory: {output_dir}")

    # Filter methods if specified
    methods_to_run = config["methods"]
    if args.methods:
        methods_to_run = [m for m in methods_to_run if m["name"] in args.methods]
        print(f"Running methods: {[m['name'] for m in methods_to_run]}")
    else:
        print(f"Running all {len(methods_to_run)} methods from config")

    # Run benchmarks for each method
    for method_config in methods_to_run:
        run_benchmarks_for_method(
            method_config,
            gpu_sku,
            cutoff,
            device,
            dtype_str,
            timer,
            output_dir,
            backend_type,
        )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
