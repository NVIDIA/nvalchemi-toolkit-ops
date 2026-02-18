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
DFT-D3 Dispersion Scaling Benchmark
====================================

CLI tool to benchmark DFT-D3 dispersion corrections and generate CSV files
for documentation. Results are saved with GPU-specific naming:
`dftd3_benchmark_<system_type>_<gpu_sku>.csv`

Usage:
    python benchmark_dftd3.py --config benchmark_config.yaml --output-dir ../../docs/benchmark_results

The config file specifies system configurations and DFT-D3 parameters.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from pymatgen.core import Lattice, Structure

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.utils import BackendType, BenchmarkTimer

# Guarded torch imports
try:
    import torch

    from nvalchemiops.torch.interactions.dispersion import (
        D3Parameters as TorchD3Parameters,
    )
    from nvalchemiops.torch.interactions.dispersion import (
        dftd3 as torch_dftd3,
    )
    from nvalchemiops.torch.neighbors import neighbor_list as torch_neighbor_list

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    TorchD3Parameters = None  # type: ignore
    torch_dftd3 = None  # type: ignore
    torch_neighbor_list = None  # type: ignore

# JAX globals — populated lazily by _import_jax() so that env vars
# (XLA allocator mode) can be configured before the first import.
JAX_AVAILABLE = False
jax = None  # type: ignore
jnp = None  # type: ignore
JaxD3Parameters = None  # type: ignore
jax_dftd3 = None  # type: ignore
jax_neighbor_list = None  # type: ignore


def _setup_jax_allocator(mode: str) -> None:
    """Configure XLA memory allocator before JAX is imported.

    Parameters
    ----------
    mode : str
        ``"throughput"`` uses XLA's default preallocator (fast).
        ``"memory"`` uses the platform allocator (accurate memory accounting).
    """
    if mode == "memory":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _import_jax() -> None:
    """Import JAX and nvalchemiops JAX bindings, setting module globals."""
    global JAX_AVAILABLE, jax, jnp, JaxD3Parameters, jax_dftd3, jax_neighbor_list
    try:
        import jax as _jax
        import jax.numpy as _jnp

        from nvalchemiops.jax.interactions.dispersion import (
            D3Parameters as _JaxD3Parameters,
        )
        from nvalchemiops.jax.interactions.dispersion import (
            dftd3 as _jax_dftd3,
        )
        from nvalchemiops.jax.neighbors import neighbor_list as _jax_neighbor_list

        jax = _jax
        jnp = _jnp
        JaxD3Parameters = _JaxD3Parameters
        jax_dftd3 = _jax_dftd3
        jax_neighbor_list = _jax_neighbor_list
        JAX_AVAILABLE = True
    except ImportError:
        pass


# Optional torch-dftd imports (only needed for torch_dftd backend)
try:
    import torch_dftd
    from torch_dftd.functions.dftd3 import edisp
    from torch_dftd.functions.distance import calc_distances

    TORCH_DFTD_AVAILABLE = True
except ImportError:
    TORCH_DFTD_AVAILABLE = False
    torch_dftd = None  # type: ignore
    edisp = None  # type: ignore
    calc_distances = None  # type: ignore

# Constants
ANGSTROM_TO_BOHR = 1.88973


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


def load_config(config_path: Path) -> dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def create_cscl_supercell(size: int) -> Structure:
    """Create CsCl supercell of given linear size (2*size³ atoms)."""
    # Create cubic lattice
    lattice = Lattice.cubic(4.14)  # ~4.14 Å cubic cell

    # Create base unit cell with Cs and Cl atoms
    species = ["Cs", "Cl"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    base_unitcell = Structure(lattice, species, coords, coords_are_cartesian=False)

    # Create supercell
    supercell_matrix = [[size, 0, 0], [0, size, 0], [0, 0, size]]
    base_unitcell.make_supercell(supercell_matrix)

    return base_unitcell


def _create_d3_parameters_numpy(dtype_str: str = "float32") -> dict:
    """Create simplified D3 parameters for Cs and Cl as numpy arrays.

    Parameters
    ----------
    dtype_str : str, default="float32"
        Dtype string (e.g., "float32", "float64").

    Returns
    -------
    dict
        Dictionary with keys "rcov", "r4r2", "c6ab", "cn_ref" as numpy arrays.
    """
    dtype = getattr(np, dtype_str)

    rcov = np.zeros(56, dtype=dtype)
    rcov[17] = 1.88  # Cl
    rcov[55] = 4.91  # Cs

    r4r2 = np.zeros(56, dtype=dtype)
    r4r2[17] = 8.0  # Cl
    r4r2[55] = 18.0  # Cs

    c6ab = np.zeros((56, 56, 5, 5), dtype=dtype)
    c6ab[17, 17, :, :] = 50.0  # Cl-Cl
    c6ab[17, 55, :, :] = 200.0  # Cl-Cs
    c6ab[55, 17, :, :] = 200.0  # Cs-Cl
    c6ab[55, 55, :, :] = 800.0  # Cs-Cs

    cn_ref = np.zeros((56, 56, 5, 5), dtype=dtype)
    for i in range(5):
        for j in range(5):
            cn_ref[:, :, i, j] = i * 0.5

    return {"rcov": rcov, "r4r2": r4r2, "c6ab": c6ab, "cn_ref": cn_ref}


def create_d3_parameters(
    backend: BackendType,
    device: str = "cuda",
    dtype_str: str = "float32",
):
    """Create simplified D3 parameters for Cs and Cl using the specified backend.

    Core data is built once as numpy arrays, then converted to the target backend.
    """
    np_params = _create_d3_parameters_numpy(dtype_str)

    match backend:
        case "torch":
            dtype = getattr(torch, dtype_str)
            device_obj = torch.device(device)
            return TorchD3Parameters(
                rcov=torch.tensor(np_params["rcov"], dtype=dtype, device=device_obj),
                r4r2=torch.tensor(np_params["r4r2"], dtype=dtype, device=device_obj),
                c6ab=torch.tensor(np_params["c6ab"], dtype=dtype, device=device_obj),
                cn_ref=torch.tensor(
                    np_params["cn_ref"], dtype=dtype, device=device_obj
                ),
            )
        case "jax":
            dtype = getattr(jnp, dtype_str)
            return JaxD3Parameters(
                rcov=jnp.array(np_params["rcov"], dtype=dtype),
                r4r2=jnp.array(np_params["r4r2"], dtype=dtype),
                c6ab=jnp.array(np_params["c6ab"], dtype=dtype),
                cn_ref=jnp.array(np_params["cn_ref"], dtype=dtype),
            )
        case "warp":
            raise NotImplementedError("warp backend D3 parameters not yet supported")


def prepare_system_numpy(
    supercell_size: int,
    batch_size: int = 1,
) -> dict:
    """
    Create supercell(s) and prepare numpy arrays (no framework dependency).

    Parameters
    ----------
    supercell_size : int
        Linear size of the supercell (creates 2*size³ atoms per system).
    batch_size : int, default=1
        Number of systems to batch together.

    Returns
    -------
    dict
        Dictionary containing numpy arrays:
        - positions_bohr: Positions in Bohr (N_total, 3) float64
        - numbers: Atomic numbers (N_total,) int32
        - coords_angstrom: Positions in Angstroms (N_total, 3) float64
        - cell: Cell vectors. Shape (batch_size, 3, 3) if batched, else (3, 3) float64
        - pbc: PBC flags. Shape (batch_size, 3) if batched, else (3,) bool
        - batch_idx: Batch indices (N_total,) int32 or None if single system
        - batch_ptr: Batch pointer (batch_size+1,) int32 or None if single system
        - total_atoms: Total number of atoms
    """
    is_batched = batch_size > 1

    if is_batched:
        # Create multiple systems
        all_structures = [
            create_cscl_supercell(supercell_size) for _ in range(batch_size)
        ]

        # Concatenate all systems
        all_positions = []
        all_numbers = []
        all_coords = []
        all_cells = []
        all_pbc = []
        ptr = [0]

        for structure in all_structures:
            all_positions.append(structure.cart_coords * ANGSTROM_TO_BOHR)
            all_numbers.append(
                np.array([site.specie.Z for site in structure], dtype=np.int32)
            )
            all_coords.append(structure.cart_coords)
            all_cells.append(structure.lattice.matrix)
            all_pbc.append(np.array([True, True, True]))
            ptr.append(ptr[-1] + len(structure))

        positions_bohr = np.concatenate(all_positions, axis=0)
        numbers = np.concatenate(all_numbers, axis=0)
        coords_angstrom = np.concatenate(all_coords, axis=0)
        cell = np.stack(all_cells, axis=0)
        pbc = np.stack(all_pbc, axis=0)
        batch_ptr = np.array(ptr, dtype=np.int32)

        # Create batch_idx
        total_atoms = coords_angstrom.shape[0]
        batch_idx = np.zeros(total_atoms, dtype=np.int32)
        for i in range(batch_size):
            batch_idx[batch_ptr[i] : batch_ptr[i + 1]] = i
    else:
        # Single system
        structure = create_cscl_supercell(supercell_size)
        total_atoms = len(structure)

        positions_bohr = structure.cart_coords * ANGSTROM_TO_BOHR
        numbers = np.array([site.specie.Z for site in structure], dtype=np.int32)
        coords_angstrom = structure.cart_coords.copy()
        cell = structure.lattice.matrix.copy()
        pbc = np.array([True, True, True])
        batch_idx = None
        batch_ptr = None

    return {
        "positions_bohr": positions_bohr,
        "numbers": numbers,
        "coords_angstrom": coords_angstrom,
        "cell": cell,
        "pbc": pbc,
        "batch_idx": batch_idx,
        "batch_ptr": batch_ptr,
        "total_atoms": total_atoms,
    }


def convert_to_backend(
    np_data: dict,
    backend: BackendType,
    device: str = "cuda",
    dtype_str: str = "float32",
) -> dict:
    """
    Convert numpy arrays to backend-specific arrays.

    Core data structure is defined once as numpy arrays in ``np_data``
    (from :func:`prepare_system_numpy`), then converted to the target
    backend's array type.

    Parameters
    ----------
    np_data : dict
        Dictionary from prepare_system_numpy().
    backend : BackendType
        Target backend ("torch", "jax", "warp").
    device : str
        Device string (used by torch backend).
    dtype_str : str
        Dtype string like "float32".

    Returns
    -------
    dict
        Dictionary with backend-specific arrays, same keys as input
        plus the converted arrays.
    """
    # Define conversion specs once: (source_key, target_key)
    float_keys = [
        ("positions_bohr", "positions"),
        ("coords_angstrom", "coord"),
        ("cell", "cell"),
    ]
    int_keys = [
        ("numbers", "numbers"),
    ]
    bool_keys = [
        ("pbc", "pbc"),
    ]
    optional_int_keys = [
        ("batch_idx", "batch_idx"),
        ("batch_ptr", "batch_ptr"),
    ]

    result = {"total_atoms": np_data["total_atoms"]}

    match backend:
        case "torch":
            dtype = getattr(torch, dtype_str)
            for src, dst in float_keys:
                result[dst] = torch.tensor(np_data[src], dtype=dtype, device=device)
            for src, dst in int_keys:
                result[dst] = torch.tensor(
                    np_data[src], dtype=torch.int32, device=device
                )
            for src, dst in bool_keys:
                result[dst] = torch.tensor(
                    np_data[src], dtype=torch.bool, device=device
                )
            for src, dst in optional_int_keys:
                result[dst] = (
                    torch.tensor(np_data[src], dtype=torch.int32, device=device)
                    if np_data[src] is not None
                    else None
                )
        case "jax":
            dtype = getattr(jnp, dtype_str)
            for src, dst in float_keys:
                arr = jnp.array(np_data[src], dtype=dtype)
                # JAX dftd3() derives num_systems from cell.shape[0],
                # so cell must always be (batch, 3, 3) even for single systems.
                if dst == "cell" and arr.ndim == 2:
                    arr = arr[jnp.newaxis, :, :]
                result[dst] = arr
            for src, dst in int_keys:
                result[dst] = jnp.array(np_data[src], dtype=jnp.int32)
            for src, dst in bool_keys:
                arr = jnp.array(np_data[src], dtype=jnp.bool_)
                # Keep pbc shape consistent with cell: (batch, 3).
                if dst == "pbc" and arr.ndim == 1:
                    arr = arr[jnp.newaxis, :]
                result[dst] = arr
            for src, dst in optional_int_keys:
                result[dst] = (
                    jnp.array(np_data[src], dtype=jnp.int32)
                    if np_data[src] is not None
                    else None
                )
        case "warp":
            raise NotImplementedError("warp backend array conversion not yet supported")

    return result


def compute_neighbor_list(
    backend_data: dict,
    backend: BackendType,
    cutoff: float,
    max_neighbors: int,
    return_neighbor_list: bool = False,
) -> tuple:
    """
    Compute neighbor list using the appropriate backend.

    Parameters
    ----------
    backend_data : dict
        Dictionary from convert_to_backend().
    backend : BackendType
        Target backend ("torch", "jax", "warp").
    cutoff : float
        Cutoff distance in Angstroms for neighbor list.
    max_neighbors : int
        Maximum number of neighbors per atom.
    return_neighbor_list : bool, default=False
        If True, return neighbor list in COO format (2, num_pairs) instead of
        neighbor matrix (N_total, max_neighbors).

    Returns
    -------
    tuple
        (neighbor_data, num_neighbor_data, shifts_or_none)
        Where neighbor_data format depends on return_neighbor_list flag.
    """
    is_batched = backend_data["batch_idx"] is not None
    if is_batched:
        method = "batch_cell_list"
    else:
        method = "cell_list"

    dispatch_func = None
    match backend:
        case "torch":
            dispatch_func = torch_neighbor_list
        case "jax":
            dispatch_func = jax_neighbor_list
        case "warp":
            raise NotImplementedError("warp backend neighbor list not yet supported")
    return dispatch_func(
        backend_data["coord"],
        cutoff,
        cell=backend_data["cell"],
        pbc=backend_data["pbc"],
        batch_idx=backend_data["batch_idx"],
        batch_ptr=backend_data["batch_ptr"],
        method=method,
        max_neighbors=max_neighbors,
        return_neighbor_list=return_neighbor_list,
    )


def prepare_system_and_neighborlist(
    supercell_size: int,
    cutoff: float,
    max_neighbors: int,
    device: str,
    dtype: "torch.dtype",
    batch_size: int = 1,
    return_neighbor_list: bool = False,
) -> dict:
    """
    Create supercell(s), prepare tensors, and build neighbor list.

    This function preserves backward compatibility by wrapping the new
    numpy-based data preparation and backend dispatch functions.

    Parameters
    ----------
    supercell_size : int
        Linear size of the supercell (creates 2*size³ atoms per system).
    cutoff : float
        Cutoff distance in Angstroms for neighbor list.
    max_neighbors : int
        Maximum number of neighbors per atom.
    device : str
        Device string ('cuda' or 'cpu').
    dtype : torch.dtype
        Data type for floating point tensors.
    batch_size : int, default=1
        Number of systems to batch together.
    return_neighbor_list : bool, default=False
        If True, return neighbor list in COO format (2, num_pairs) instead of
        neighbor matrix (N_total, max_neighbors).

    Returns
    -------
    dict
        Dictionary containing:
        - positions: Tensor of positions in Bohr (N_total, 3)
        - numbers: Tensor of atomic numbers (N_total,)
        - coord: Tensor of positions in Angstroms (N_total, 3)
        - cell: Tensor of cell vectors (batch_size, 3, 3) or (3, 3)
        - pbc: Tensor of periodic boundary conditions (batch_size, 3) or (3,)
        - neighbor_data: Neighbor list matrix (N_total, max_neighbors) if return_neighbor_list=False,
                        or COO format (2, num_pairs) if return_neighbor_list=True
        - num_neighbor_data: Number of neighbors per atom (N_total,) if return_neighbor_list=False,
                            or neighbor pointer (N_total+1,) if return_neighbor_list=True
        - batch_idx: Batch indices (N_total,) or None for single system
        - batch_ptr: Batch pointer (batch_size+1,) or None for single system
        - total_atoms: Total number of atoms across all systems
        - total_neighbors: Total number of neighbor pairs
    """
    # Step 1: Prepare numpy data
    np_data = prepare_system_numpy(supercell_size, batch_size)

    # Step 2: Convert to torch backend (this function is only used by torch runners)
    dtype_str = str(dtype).split(".")[-1]  # "torch.float32" -> "float32"
    backend_data = convert_to_backend(np_data, "torch", device, dtype_str)

    # Step 3: Compute neighbor list
    neighbor_data, num_neighbor_data, _ = compute_neighbor_list(
        backend_data, "torch", cutoff, max_neighbors, return_neighbor_list
    )

    # Calculate total neighbors
    if return_neighbor_list:
        # neighbor_data is (2, num_pairs), so num_pairs is the total neighbors
        total_neighbors = neighbor_data.shape[1]
    else:
        # num_neighbor_data is (N_total,) with counts per atom
        total_neighbors = num_neighbor_data.sum().item()

    return {
        "positions": backend_data["positions"],
        "numbers": backend_data["numbers"],
        "coord": backend_data["coord"],
        "cell": backend_data["cell"],
        "pbc": backend_data["pbc"],
        "neighbor_data": neighbor_data,
        "num_neighbor_data": num_neighbor_data,
        "batch_idx": backend_data["batch_idx"],
        "batch_ptr": backend_data["batch_ptr"],
        "total_atoms": backend_data["total_atoms"],
        "total_neighbors": total_neighbors,
    }


def load_torch_dftd_parameters(
    device: "torch.device", dtype: "torch.dtype" = None
) -> dict:
    """Load DFT-D3 parameters from torch-dftd package.

    Note: dtype defaults to torch.float32 if not provided.
    """
    if not TORCH_DFTD_AVAILABLE:
        raise ImportError(
            "torch-dftd not installed. Install via: pip install torch-dftd"
        )

    if dtype is None:
        dtype = torch.float32

    # Load parameters from torch_dftd
    d3_filepath = str(
        Path(os.path.abspath(torch_dftd.__file__)).parent
        / "nn"
        / "params"
        / "dftd3_params.npz"
    )
    d3_params_np = np.load(d3_filepath)

    c6ab = torch.tensor(d3_params_np["c6ab"], dtype=dtype, device=device)
    r0ab = torch.tensor(d3_params_np["r0ab"], dtype=dtype, device=device)
    rcov = torch.tensor(d3_params_np["rcov"], dtype=dtype, device=device)
    r2r4 = torch.tensor(d3_params_np["r2r4"], dtype=dtype, device=device)

    # Convert rcov to Bohr
    rcov = rcov * ANGSTROM_TO_BOHR

    return {
        "c6ab": c6ab,
        "r0ab": r0ab,
        "rcov": rcov,
        "r2r4": r2r4,
    }


def run_dftd3_nvalchemiops_benchmark(
    supercell_size: int,
    cutoff: float,
    d3_params: "TorchD3Parameters",
    dftd3_config: dict,
    max_neighbors: int,
    timer: BenchmarkTimer,
    device: str,
    dtype: "torch.dtype",
    batch_size: int = 1,
) -> dict:
    """Run DFT-D3 benchmark using nvalchemiops backend (single or batched)."""
    try:
        # Prepare system and neighbor list (matrix format for nvalchemiops)
        system_data = prepare_system_and_neighborlist(
            supercell_size,
            cutoff,
            max_neighbors,
            device,
            dtype,
            batch_size,
            return_neighbor_list=False,
        )

        positions = system_data["positions"]
        numbers = system_data["numbers"]
        neighbor_matrix = system_data["neighbor_data"]
        total_atoms = system_data["total_atoms"]
        total_neighbors = system_data["total_neighbors"]
        batch_idx = system_data["batch_idx"]

        # Define the function to benchmark
        def dftd3_call():
            return torch_dftd3(
                positions=positions,
                numbers=numbers,
                d3_params=d3_params,
                neighbor_matrix=neighbor_matrix,
                fill_value=total_atoms,
                a1=dftd3_config["a1"],
                a2=dftd3_config["a2"],
                s6=dftd3_config["s6"],
                s8=dftd3_config["s8"],
                k1=dftd3_config["k1"],
                k3=dftd3_config["k3"],
                batch_idx=batch_idx,
                s5_smoothing_on=dftd3_config["s5_smoothing_on"],
                s5_smoothing_off=dftd3_config["s5_smoothing_off"],
                device=device,
            )

        # Time the function
        timing_results = timer.time_function(dftd3_call)

        if not timing_results["success"]:
            return {
                "total_atoms": total_atoms,
                "batch_size": batch_size,
                "supercell_size": supercell_size,
                "total_neighbors": 0,
                "median_time_ms": float("inf"),
                "peak_memory_mb": timing_results.get("peak_memory_mb"),
                "memory_note": "torch:cuda.max_memory_allocated",
                "success": False,
                "error": timing_results.get("error", "Unknown error"),
                "error_type": timing_results.get("error_type", "Unknown"),
            }

        median_time_ms = timing_results["median"]
        peak_memory_mb = timing_results.get("peak_memory_mb")

        return {
            "total_atoms": total_atoms,
            "batch_size": batch_size,
            "supercell_size": supercell_size,
            "total_neighbors": total_neighbors,
            "median_time_ms": float(median_time_ms),
            "peak_memory_mb": peak_memory_mb,
            "memory_note": "torch:cuda.max_memory_allocated",
            "success": True,
        }

    except Exception as e:
        total_atoms = 2 * supercell_size**3 * batch_size
        return {
            "total_atoms": total_atoms,
            "batch_size": batch_size,
            "supercell_size": supercell_size,
            "total_neighbors": 0,
            "median_time_ms": float("inf"),
            "peak_memory_mb": None,
            "memory_note": "torch:cuda.max_memory_allocated",
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def run_dftd3_nvalchemiops_jax_benchmark(
    supercell_size: int,
    cutoff: float,
    d3_params: "JaxD3Parameters",
    dftd3_config: dict,
    max_neighbors: int,
    timer: BenchmarkTimer,
    dtype_str: str = "float32",
    batch_size: int = 1,
) -> dict:
    """Run DFT-D3 benchmark using nvalchemiops JAX backend (single or batched)."""
    try:
        # Prepare system data as numpy, then convert to JAX
        np_data = prepare_system_numpy(supercell_size, batch_size)
        backend_data = convert_to_backend(np_data, "jax", dtype_str=dtype_str)

        # Compute neighbor list with JAX backend
        neighbor_data, num_neighbor_data, _ = compute_neighbor_list(
            backend_data, "jax", cutoff, max_neighbors, return_neighbor_list=False
        )

        positions = backend_data["positions"]
        numbers = backend_data["numbers"]
        neighbor_matrix = neighbor_data
        total_atoms = backend_data["total_atoms"]
        total_neighbors = int(jnp.sum(num_neighbor_data))
        batch_idx = backend_data["batch_idx"]
        cell = backend_data["cell"]

        # Closure-captured scalars stay concrete under JIT (not traced).
        # This is required for Warp FFI compatibility.
        a1_val = dftd3_config["a1"]
        a2_val = dftd3_config["a2"]
        s6_val = dftd3_config["s6"]
        s8_val = dftd3_config["s8"]
        k1_val = dftd3_config["k1"]
        k3_val = dftd3_config["k3"]
        s5_on_val = dftd3_config["s5_smoothing_on"]
        s5_off_val = dftd3_config["s5_smoothing_off"]
        fill_val = total_atoms

        @jax.jit
        def jitted_dftd3(
            positions,
            numbers,
            neighbor_matrix,
            batch_idx,
            cell,
            rcov,
            r4r2,
            c6ab,
            cn_ref,
        ):
            params = JaxD3Parameters(
                rcov=rcov,
                r4r2=r4r2,
                c6ab=c6ab,
                cn_ref=cn_ref,
            )
            return jax_dftd3(
                positions=positions,
                numbers=numbers,
                d3_params=params,
                neighbor_matrix=neighbor_matrix,
                fill_value=fill_val,
                a1=a1_val,
                a2=a2_val,
                s6=s6_val,
                s8=s8_val,
                k1=k1_val,
                k3=k3_val,
                batch_idx=batch_idx,
                cell=cell,
                s5_smoothing_on=s5_on_val,
                s5_smoothing_off=s5_off_val,
            )

        def dftd3_call():
            return jitted_dftd3(
                positions,
                numbers,
                neighbor_matrix,
                batch_idx,
                cell,
                d3_params.rcov,
                d3_params.r4r2,
                d3_params.c6ab,
                d3_params.cn_ref,
            )

        # Time the function
        timing_results = timer.time_function(dftd3_call)
        compile_ms = timing_results.get("compile_ms")

        if not timing_results["success"]:
            return {
                "total_atoms": total_atoms,
                "batch_size": batch_size,
                "supercell_size": supercell_size,
                "total_neighbors": 0,
                "compile_ms": compile_ms,
                "median_time_ms": float("inf"),
                "peak_memory_mb": timing_results.get("peak_memory_mb"),
                "memory_note": "jax:nvml_process_used",
                "success": False,
                "error": timing_results.get("error", "Unknown error"),
                "error_type": timing_results.get("error_type", "Unknown"),
            }

        median_time_ms = timing_results["median"]
        peak_memory_mb = timing_results.get("peak_memory_mb")

        return {
            "total_atoms": total_atoms,
            "batch_size": batch_size,
            "supercell_size": supercell_size,
            "total_neighbors": total_neighbors,
            "compile_ms": compile_ms,
            "median_time_ms": float(median_time_ms),
            "peak_memory_mb": peak_memory_mb,
            "memory_note": "jax:nvml_process_used",
            "success": True,
        }

    except Exception as e:
        total_atoms = 2 * supercell_size**3 * batch_size
        return {
            "total_atoms": total_atoms,
            "batch_size": batch_size,
            "supercell_size": supercell_size,
            "total_neighbors": 0,
            "compile_ms": None,
            "median_time_ms": float("inf"),
            "peak_memory_mb": None,
            "memory_note": "jax:nvml_process_used",
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def run_dftd3_torch_dftd_benchmark(
    supercell_size: int,
    cutoff: float,
    torch_dftd_params: dict,
    dftd3_config: dict,
    max_neighbors: int,
    timer: BenchmarkTimer,
    device: str,
    dtype: "torch.dtype",
    batch_size: int = 1,
) -> dict:
    """Run DFT-D3 benchmark using torch-dftd backend (single or batched)."""
    if not TORCH_DFTD_AVAILABLE:
        return {
            "total_atoms": 2 * supercell_size**3 * batch_size,
            "batch_size": batch_size,
            "supercell_size": supercell_size,
            "total_neighbors": 0,
            "median_time_ms": float("inf"),
            "peak_memory_mb": None,
            "success": False,
            "error": "torch-dftd not installed",
            "error_type": "ImportError",
        }

    try:
        # Prepare system and neighbor list in COO format (torch-dftd uses edge_index)
        system_data = prepare_system_and_neighborlist(
            supercell_size,
            cutoff,
            max_neighbors,
            device,
            dtype,
            batch_size,
            return_neighbor_list=True,
        )

        edge_index = system_data[
            "neighbor_data"
        ]  # Already in COO format (2, num_pairs)
        total_atoms = system_data["total_atoms"]
        total_neighbors = system_data["total_neighbors"]

        # Prepare positions with gradients for torch-dftd
        positions = system_data["positions"].clone().requires_grad_(True)
        Z = system_data["numbers"].to(torch.int64)

        # Note: torch-dftd does not require explicit batch tensors
        # It handles batched systems through concatenated positions/numbers
        # and the edge_index structure
        batch = None
        batch_edge = None

        # Prepare torch-dftd params dict
        params = {
            "s6": dftd3_config["s6"],
            "s18": dftd3_config["s8"],
            "rs6": dftd3_config["a1"],
            "rs18": dftd3_config["a2"],
            "alp": 14.0,  # Default for BJ damping
        }

        # Define the function to benchmark
        def torch_dftd_call():
            # Reset grad
            if positions.grad is not None:
                positions.grad.zero_()

            # Calculate distances
            r = calc_distances(positions, edge_index, cell=None, shift_pos=None)

            # Compute energy
            energy = edisp(
                Z=Z,
                r=r,
                edge_index=edge_index,
                c6ab=torch_dftd_params["c6ab"],
                r0ab=torch_dftd_params["r0ab"],
                rcov=torch_dftd_params["rcov"],
                r2r4=torch_dftd_params["r2r4"],
                params=params,
                cutoff=None,
                cnthr=None,
                batch=batch,
                batch_edge=batch_edge,
                shift_pos=None,
                pos=positions,
                cell=None,
                damping="bj",
                bidirectional=True,
                abc=False,
                k1=dftd3_config["k1"],
                k3=dftd3_config["k3"],
            )

            # Compute forces
            forces = -torch.autograd.grad(
                outputs=energy,
                inputs=positions,
                create_graph=False,
                retain_graph=False,
            )[0]

            return energy, forces

        # Time the function
        timing_results = timer.time_function(torch_dftd_call)

        if not timing_results["success"]:
            return {
                "total_atoms": total_atoms,
                "batch_size": batch_size,
                "supercell_size": supercell_size,
                "total_neighbors": 0,
                "median_time_ms": float("inf"),
                "peak_memory_mb": timing_results.get("peak_memory_mb"),
                "success": False,
                "error": timing_results.get("error", "Unknown error"),
                "error_type": timing_results.get("error_type", "Unknown"),
            }

        median_time_ms = timing_results["median"]
        peak_memory_mb = timing_results.get("peak_memory_mb")

        return {
            "total_atoms": total_atoms,
            "batch_size": batch_size,
            "supercell_size": supercell_size,
            "total_neighbors": total_neighbors,
            "median_time_ms": float(median_time_ms),
            "peak_memory_mb": peak_memory_mb,
            "success": True,
        }

    except Exception as e:
        return {
            "total_atoms": 2 * supercell_size**3 * batch_size,
            "batch_size": batch_size,
            "supercell_size": supercell_size,
            "total_neighbors": 0,
            "median_time_ms": float("inf"),
            "peak_memory_mb": None,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def _resolve_backend_type(cli_backend: str) -> BackendType:
    """Map CLI backend string to BackendType.

    Parameters
    ----------
    cli_backend : str
        CLI backend choice ("nvalchemiops", "nvalchemiops_jax", "torch_dftd").

    Returns
    -------
    BackendType
        Framework-level backend type ("torch" or "jax").
    """
    match cli_backend:
        case "torch" | "torch_dftd":
            return "torch"
        case "jax":
            return "jax"
        case _:
            raise ValueError(f"Unknown backend: {cli_backend}")


def _check_backend_available(cli_backend: str) -> None:
    """Validate that the requested backend is installed.

    Parameters
    ----------
    cli_backend : str
        CLI backend choice.

    Raises
    ------
    SystemExit
        If the required backend is not available.
    """
    match cli_backend:
        case "torch":
            if not TORCH_AVAILABLE:
                print(
                    "ERROR: nvalchemiops (torch) backend requested but torch is not installed."
                )
                sys.exit(1)
        case "jax":
            if not JAX_AVAILABLE:
                print(
                    "ERROR: nvalchemiops_jax backend requested but JAX is not installed."
                )
                sys.exit(1)
        case "torch_dftd":
            if not TORCH_DFTD_AVAILABLE:
                print("ERROR: torch-dftd backend requested but not installed.")
                print("Install via: pip install torch-dftd")
                sys.exit(1)


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark DFT-D3 dispersion corrections and generate CSV files"
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
        choices=["torch", "jax", "torch_dftd"],
        default="torch",
        help="Backend to use for benchmarking (default: torch)",
    )
    parser.add_argument(
        "--gpu-sku",
        type=str,
        help="Override GPU SKU name for output files (default: auto-detect)",
    )
    parser.add_argument(
        "--jax-allocator",
        type=str,
        choices=["throughput", "memory"],
        default="throughput",
        help=(
            "JAX XLA memory allocator mode (default: throughput). "
            "'throughput' uses XLA's preallocator for fast steady-state timing. "
            "'memory' uses the platform allocator for accurate memory accounting."
        ),
    )

    args = parser.parse_args()

    # Configure JAX allocator and import JAX (env vars must precede import)
    if args.backend == "jax":
        _setup_jax_allocator(args.jax_allocator)
    _import_jax()

    # Validate backend availability
    _check_backend_available(args.backend)

    # Load config
    config = load_config(args.config)

    # Resolve backend type
    backend_type = _resolve_backend_type(args.backend)

    # Get parameters
    params = config["parameters"]
    cutoff = float(params["cutoff"])
    warmup = int(params["warmup_iterations"])
    timing = int(params["timing_iterations"])
    dtype_str = params["dtype"]

    dftd3_config = config["dftd3_parameters"]

    # Backend-specific setup
    device = "cpu"  # Default
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
    timer = BenchmarkTimer(warmup_runs=warmup, timing_runs=timing, backend=backend_type)

    # Backend-specific parameter setup
    d3_params = None
    torch_dftd_params = None
    match args.backend:
        case "torch":
            d3_params = create_d3_parameters("torch", device, dtype_str)
        case "jax":
            d3_params = create_d3_parameters("jax", dtype_str=dtype_str)
        case "torch_dftd":
            torch_dftd_params = load_torch_dftd_parameters(torch.device(device), dtype)

    # Print configuration
    print("=" * 70)
    print("DFT-D3 DISPERSION BENCHMARK")
    print("=" * 70)
    print(f"Backend: {args.backend}")
    print(f"Device: {device}")
    print(f"GPU SKU: {gpu_sku}")
    print(f"Cutoff: {cutoff:.2f} Å ({cutoff * ANGSTROM_TO_BOHR:.1f} Bohr)")
    print(f"Dtype: {dtype}")
    print(f"Warmup iterations: {warmup}")
    print(f"Timing iterations: {timing}")
    if args.backend == "jax":
        print(f"JAX allocator mode: {args.jax_allocator}")
    print(f"Output directory: {output_dir}")
    print(
        f"Memory metric: "
        f"{'torch.cuda.max_memory_allocated (allocator peak)' if args.backend in ('torch', 'torch_dftd') else 'NVML used_memory (process-wide, not directly comparable to Torch)'}"
    )

    # Run benchmarks for each system configuration
    for system_config in config["systems"]:
        system_name = system_config["name"]
        system_type = system_config["system_type"]
        supercell_sizes = system_config["supercell_sizes"]
        batch_sizes = system_config.get("batch_sizes", [1])
        max_neighbors = system_config["max_neighbors"]

        # Group results by batch size
        results_by_batch = {}

        for batch_size in batch_sizes:
            is_batched = batch_size > 1

            print(f"\n{'=' * 70}")
            print(f"Benchmarking: {system_name} ({system_type})")
            if is_batched:
                print(f"Batch size: {batch_size}")
            print(f"{'=' * 70}")
            print(f"Supercell sizes: {supercell_sizes}")

            all_results = []

            for size in supercell_sizes:
                # Reset peak memory stats before each configuration
                timer.clear_memory()

                atoms_per_system = 2 * size**3
                total_atoms = atoms_per_system * batch_size

                if is_batched:
                    print(
                        f"\n  {total_atoms:6,d} atoms ({atoms_per_system:,d} x {batch_size})...",
                        end=" ",
                        flush=True,
                    )
                else:
                    print(
                        f"\n  {total_atoms:6,d} atoms (supercell {size}³)...",
                        end=" ",
                        flush=True,
                    )

                # Choose benchmark function based on backend
                match args.backend:
                    case "torch":
                        result = run_dftd3_nvalchemiops_benchmark(
                            size,
                            cutoff,
                            d3_params,
                            dftd3_config,
                            max_neighbors,
                            timer,
                            device,
                            dtype,
                            batch_size,
                        )
                    case "jax":
                        result = run_dftd3_nvalchemiops_jax_benchmark(
                            size,
                            cutoff,
                            d3_params,
                            dftd3_config,
                            max_neighbors,
                            timer,
                            dtype_str,
                            batch_size,
                        )
                    case "torch_dftd":
                        result = run_dftd3_torch_dftd_benchmark(
                            size,
                            cutoff,
                            torch_dftd_params,
                            dftd3_config,
                            max_neighbors,
                            timer,
                            device,
                            dtype,
                            batch_size,
                        )

                # Add backend to result for CSV
                result["backend"] = args.backend
                error_type = (
                    None if "error_type" not in result else result.pop("error_type")
                )
                error = None if "error" not in result else result.pop("error")
                all_results.append(result)

                if result["success"]:
                    throughput = result["total_atoms"] / result["median_time_ms"] * 1000
                    mem_str = ""
                    if result.get("peak_memory_mb") is not None:
                        mem_str = f" | {result['peak_memory_mb']:.1f} MB"
                    compile_str = ""
                    if result.get("compile_ms") is not None:
                        compile_str = f" | warmup {result['compile_ms']:.0f} ms"
                    neighbor_str = ""
                    if result.get("total_neighbors"):
                        neighbor_str = f" | {result['total_neighbors']:,d} neighbors"
                    print(
                        f"{result['median_time_ms']:.3f} ms "
                        f"({throughput:.1f} atoms/s){mem_str}{compile_str}{neighbor_str}"
                    )
                else:
                    print(f"FAILED ({error_type}): {error}")
                    if error_type in ["OOM", "Timeout"]:
                        print("  Skipping larger systems")
                        break

            # Store results for this batch size
            if all_results:
                results_by_batch[batch_size] = all_results

        # Separate batched and non-batched results
        non_batched_results = []
        batched_results = []

        for batch_size, results in results_by_batch.items():
            if batch_size == 1:
                non_batched_results.extend(results)
            else:
                batched_results.extend(results)

        # Save non-batched results
        if non_batched_results:
            output_file = output_dir / f"dftd3_benchmark_{args.backend}_{gpu_sku}.csv"
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=non_batched_results[0].keys())
                writer.writeheader()
                writer.writerows(non_batched_results)
            print(f"\n✓ Non-batched results saved to: {output_file}")

            successful = [r for r in non_batched_results if r.get("success", True)]
            failed = [r for r in non_batched_results if not r.get("success", True)]
            print(
                f"  Total: {len(non_batched_results)} | Successful: {len(successful)} | Failed: {len(failed)}"
            )

        # Save batched results (all batch sizes in one file)
        if batched_results:
            output_file = (
                output_dir / f"dftd3_benchmark_batch_{args.backend}_{gpu_sku}.csv"
            )
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=batched_results[0].keys())
                writer.writeheader()
                writer.writerows(batched_results)
            print(f"\n✓ Batched results saved to: {output_file}")

            successful = [r for r in batched_results if r.get("success", True)]
            failed = [r for r in batched_results if not r.get("success", True)]
            print(
                f"  Total: {len(batched_results)} | Successful: {len(successful)} | Failed: {len(failed)}"
            )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
