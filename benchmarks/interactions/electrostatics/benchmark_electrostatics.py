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

CLI tool to benchmark electrostatic interaction methods (Ewald summation, PME, and DSF)
and generate CSV files for documentation. Results are saved with GPU-specific naming:
`electrostatics_benchmark_<method>_<backend>_<gpu_sku>.csv`

Supports two backends:
1. nvalchemiops (Warp kernels): Custom implementation using PyTorch + Warp
2. torchpme: Reference PyTorch implementation

Methods:
- Ewald summation
- PME (Particle Mesh Ewald)
- DSF (Damped Shifted Force)

Usage:
    python benchmark_electrostatics.py --config benchmark_config.yaml --output-dir ./results
    python benchmark_electrostatics.py --config benchmark_config.yaml --backend both --method both
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Literal

import torch
import warp as wp

# Add repo root to path for imports (4 levels up from this script)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import yaml

from benchmarks.systems import create_crystal_system
from benchmarks.utils import BenchmarkTimer
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.interactions.electrostatics import (
    dsf_coulomb,
    estimate_ewald_parameters,
    estimate_pme_parameters,
    ewald_real_space,
    ewald_reciprocal_space,
    ewald_summation,
    particle_mesh_ewald,
    pme_reciprocal_space,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
    generate_k_vectors_pme,
)
from nvalchemiops.torch.neighbors import neighbor_list

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


def get_gpu_sku() -> str:
    """Get GPU SKU name for filename generation."""
    if not torch.cuda.is_available():
        return "cpu"

    try:
        gpu_name = torch.cuda.get_device_name(0)
        # Clean up GPU name for filename
        sku = gpu_name.replace(" ", "-").replace("_", "-")
        sku = sku.replace("NVIDIA-", "").replace("GeForce-", "")
        return sku.lower()
    except Exception:
        return "unknown_gpu"


def load_config(config_path: Path) -> dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


# ==============================================================================
# Neighbor Construction
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
    # Clear old neighbor data to free memory
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
        density = (total_atoms / batch_size) / volume  # atoms per cubic Angstrom
        max_nbrs = estimate_max_neighbors(
            cutoff, atomic_density=density, safety_factor=1.2
        )
        nl_kwargs["max_neighbors"] = max_nbrs

    if neighbor_format == "matrix":
        nm, num_nbrs, nm_shifts = neighbor_list(positions, cutoff, **nl_kwargs)
        system_data["neighbor_matrix"] = nm
        system_data["num_neighbors"] = num_nbrs
        system_data["neighbor_matrix_shifts"] = nm_shifts
        system_data["fill_value"] = total_atoms
    else:  # "list" or "n/a" (CSR)
        nl_data, nl_ptr, nl_shifts = neighbor_list(
            positions, cutoff, return_neighbor_list=True, **nl_kwargs
        )
        system_data["neighbor_list"] = nl_data
        system_data["neighbor_ptr"] = nl_ptr
        system_data["neighbor_shifts"] = nl_shifts


# ==============================================================================
# System Generation
# ==============================================================================


def prepare_single_system(
    supercell_size: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """Prepare a single system for benchmarking.

    Neighbor data is built by ``build_neighbors()`` before each run.

    Parameters
    ----------
    supercell_size : int
        Linear dimension of the supercell. For BCC lattice (2 atoms per unit cell),
        this creates 2 * supercell_size³ atoms total.
    """
    # BCC lattice has 2 atoms per unit cell, so total atoms = 2 * size³
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

    ewald_params = estimate_ewald_parameters(positions, cell, accuracy=1e-6)
    alpha = ewald_params.alpha

    k_cutoff = ewald_params.reciprocal_space_cutoff.item()
    cutoff = ewald_params.real_space_cutoff.item()

    pme_params = estimate_pme_parameters(positions, cell, accuracy=1e-6)
    alpha = pme_params.alpha

    mesh_dimensions = pme_params.mesh_dimensions
    mesh_spacing = pme_params.mesh_spacing.tolist()

    # Precompute k-vectors for PME (avoids regenerating them every iteration)
    k_vectors_pme, k_squared_pme = generate_k_vectors_pme(cell, mesh_dimensions)

    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "pbc": pbc,
        "total_atoms": total_atoms,
        "batch_idx": None,
        "alpha": alpha,
        "k_cutoff": k_cutoff,
        "cutoff": cutoff,
        "mesh_dimensions": mesh_dimensions,
        "mesh_spacing": mesh_spacing,
        "spline_order": 4,
        "k_vectors_pme": k_vectors_pme,
        "k_squared_pme": k_squared_pme,
    }


def prepare_batch_system(
    supercell_size: int,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """Prepare a batched system for benchmarking.

    Neighbor data is built by ``build_neighbors()`` before each run.

    Parameters
    ----------
    supercell_size : int
        Linear dimension of each supercell. For BCC lattice (2 atoms per unit cell),
        each system has 2 * supercell_size³ atoms.
    batch_size : int
        Number of systems to batch together.
    """
    # BCC lattice has 2 atoms per unit cell, so atoms per system = 2 * size³
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

        positions = system["positions"]
        charges = system["atomic_charges"]
        cell = system["cell"]
        pbc = system["pbc"]

        all_positions.append(positions)
        all_charges.append(charges)
        all_cells.append(cell)
        all_pbc.append(pbc)
        batch_idx_list.extend([i] * n_atoms)

    positions = torch.cat(all_positions, dim=0)
    charges = torch.cat(all_charges, dim=0)
    cells = torch.cat(all_cells, dim=0)
    pbc = torch.stack(all_pbc, dim=0)

    batch_idx = torch.tensor(batch_idx_list, dtype=torch.int32, device=device)
    total_atoms = positions.shape[0]
    ewald_params = estimate_ewald_parameters(positions, cells, batch_idx, accuracy=1e-6)
    alpha = ewald_params.alpha
    k_cutoff = ewald_params.reciprocal_space_cutoff[0].item()
    cutoff = ewald_params.real_space_cutoff[0].item()
    pme_params = estimate_pme_parameters(positions, cells, batch_idx, accuracy=1e-6)
    alpha = pme_params.alpha
    mesh_dimensions = pme_params.mesh_dimensions
    mesh_spacing = pme_params.mesh_spacing

    # Precompute k-vectors for PME (avoids regenerating them every iteration)
    k_vectors_pme, k_squared_pme = generate_k_vectors_pme(cells, mesh_dimensions)

    return {
        "positions": positions,
        "charges": charges,
        "cell": cells,
        "pbc": pbc,
        "total_atoms": total_atoms,
        "batch_idx": batch_idx,
        "batch_size": batch_size,
        "alpha": alpha,
        "k_cutoff": k_cutoff,
        "cutoff": cutoff,
        "mesh_dimensions": mesh_dimensions,
        "mesh_spacing": mesh_spacing,
        "spline_order": 4,
        "k_vectors_pme": k_vectors_pme,
        "k_squared_pme": k_squared_pme,
    }


# ==============================================================================
# DSF System Preparation
# ==============================================================================


def prepare_dsf_single_system(
    supercell_size: int,
    device: str,
    dtype: torch.dtype,
    cutoff: float = 12.0,
    alpha: float = 0.2,
) -> dict:
    """Prepare a single system for DSF benchmarking.

    DSF does not need k-vectors, PME mesh, or Ewald parameter estimation.
    Only positions, charges, cell, cutoff, and alpha.
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

    return dsf_coulomb(
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

    return dsf_coulomb(
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Ewald summation using nvalchemiops backend."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha")
    k_cutoff = system_data.get("k_cutoff")
    k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff)

    neighbor_list_data = system_data.get("neighbor_list")
    neighbor_ptr = system_data.get("neighbor_ptr")
    neighbor_shifts = system_data.get("neighbor_shifts")

    if batch_idx is None:
        # Single system

        if component == "real":
            return ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        elif component == "reciprocal":
            return ewald_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                k_vectors=k_vectors,
                alpha=alpha,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        else:  # full
            return ewald_summation(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                k_cutoff=k_cutoff,
                k_vectors=k_vectors,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
    else:
        # Batch system
        if component == "real":
            return ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                batch_idx=batch_idx,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        elif component == "reciprocal":
            return ewald_reciprocal_space(
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
            return ewald_summation(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                k_cutoff=k_cutoff,
                k_vectors=k_vectors,
                batch_idx=batch_idx,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
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

    neighbor_list_data = system_data.get("neighbor_list")
    neighbor_ptr = system_data.get("neighbor_ptr")
    neighbor_shifts = system_data.get("neighbor_shifts")

    if batch_idx is None:
        # Single system

        if component == "real":
            return ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        elif component == "reciprocal":
            return pme_reciprocal_space(
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
            return particle_mesh_ewald(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                k_vectors=k_vectors_pme,
                k_squared=k_squared_pme,
            )
    else:
        # Batch system

        if component == "real":
            return ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                batch_idx=batch_idx,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
            )
        elif component == "reciprocal":
            return pme_reciprocal_space(
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
            return particle_mesh_ewald(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                batch_idx=batch_idx,
                neighbor_list=neighbor_list_data,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
                compute_virial=compute_virial,
                k_vectors=k_vectors_pme,
                k_squared=k_squared_pme,
            )


def run_nvalchemiops_ewald_matrix(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Ewald summation using nvalchemiops backend (neighbor matrix format)."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha")
    k_cutoff = system_data.get("k_cutoff")
    k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff)

    neighbor_matrix = system_data.get("neighbor_matrix")
    neighbor_matrix_shifts = system_data.get("neighbor_matrix_shifts")
    mask_value = system_data.get("fill_value")

    if component == "reciprocal":
        # Reciprocal space does not use neighbors
        return ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
            batch_idx=batch_idx,
            compute_forces=compute_forces,
        )
    elif component == "real":
        return ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            batch_idx=batch_idx,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=mask_value,
            compute_forces=compute_forces,
        )
    else:  # full
        return ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            k_vectors=k_vectors,
            batch_idx=batch_idx,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=mask_value,
            compute_forces=compute_forces,
        )


def run_nvalchemiops_pme_matrix(
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run PME using nvalchemiops backend (neighbor matrix format)."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    alpha = system_data.get("alpha")
    mesh_dimensions = system_data.get("mesh_dimensions")
    spline_order = system_data.get("spline_order")
    k_vectors_pme = system_data.get("k_vectors_pme")
    k_squared_pme = system_data.get("k_squared_pme")

    neighbor_matrix = system_data.get("neighbor_matrix")
    neighbor_matrix_shifts = system_data.get("neighbor_matrix_shifts")
    mask_value = system_data.get("fill_value")

    if component == "reciprocal":
        # Reciprocal space does not use neighbors
        return pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            spline_order=spline_order,
            batch_idx=batch_idx,
            compute_forces=compute_forces,
            k_vectors=k_vectors_pme,
            k_squared=k_squared_pme,
        )
    elif component == "real":
        return ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            batch_idx=batch_idx,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=mask_value,
            compute_forces=compute_forces,
        )
    else:  # full
        return particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            spline_order=spline_order,
            batch_idx=batch_idx,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=mask_value,
            compute_forces=compute_forces,
            k_vectors=k_vectors_pme,
            k_squared=k_squared_pme,
        )


# ==============================================================================
# torchpme Backend
# ==============================================================================


def prepare_torchpme_neighbors(
    system_data: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare neighbor data in torchpme format."""
    positions = system_data["positions"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")

    if batch_idx is None:
        # Single system
        neighbor_list_data = system_data.get("neighbor_list")
        neighbor_shifts = system_data.get("neighbor_shifts")

        if neighbor_list_data is not None:
            neighbor_indices = neighbor_list_data.T
            cell_2d = cell.squeeze(0)
            neighbor_distances = torch.norm(
                positions[neighbor_list_data[1]]
                - positions[neighbor_list_data[0]]
                + neighbor_shifts.to(dtype=positions.dtype) @ cell_2d,
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
# torch_dsf Backend – Pure PyTorch DSF reference
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

    # Clone positions for autograd if needed
    if need_grad:
        pos = positions.detach().clone().requires_grad_(True)
    else:
        pos = positions

    # Clone cell for virial autograd if needed
    if compute_virial and cell is not None:
        cell_grad = cell.detach().clone().to(dtype=dtype).requires_grad_(True)
    else:
        cell_grad = cell.to(dtype=dtype) if cell is not None else None

    # Extract pair indices
    idx_i = neighbor_list[0].long()
    idx_j = neighbor_list[1].long()

    # Gather positions and compute displacement vectors
    pos_i = torch.index_select(pos, 0, idx_i)
    pos_j = torch.index_select(pos, 0, idx_j)
    r_ij = pos_j - pos_i

    # Apply PBC shifts
    if cell_grad is not None and unit_shifts is not None:
        batch_i = torch.index_select(batch_idx, 0, idx_i)
        cell_per_pair = torch.index_select(cell_grad, 0, batch_i)
        shift_cart = torch.bmm(
            unit_shifts.to(dtype=dtype).unsqueeze(1), cell_per_pair
        ).squeeze(1)
        r_ij = r_ij + shift_cart

    dist = torch.norm(r_ij, dim=1)

    # Filter to within-cutoff pairs ONCE
    mask = dist < cutoff
    dist = dist[mask]
    idx_i_f = idx_i[mask]
    idx_j_f = idx_j[mask]

    q_i = torch.index_select(charges, 0, idx_i_f)
    q_j = torch.index_select(charges, 0, idx_j_f)

    # Precompute cutoff constants
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

    # DSF pair potential
    if alpha > 0.0:
        erfc_r = torch.erfc(alpha_t * dist)
    else:
        erfc_r = torch.ones_like(dist)

    V_pair = erfc_r / dist - V_shift + B * (dist - cutoff_t)

    # Energy: 0.5 * sum qi*qj*V_pair + self_coeff * qi^2
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


dsf_torch_compiled = torch.compile(dsf_reference, mode="default")


def run_torch_dsf(
    system_data: dict,
    compute_forces: bool,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run DSF using pure PyTorch reference (torch.compile)."""
    positions = system_data["positions"]
    charges = system_data["charges"]
    cell = system_data["cell"]
    batch_idx = system_data.get("batch_idx")
    cutoff = system_data["cutoff"]
    alpha = system_data["alpha"]
    neighbor_list_data = system_data["neighbor_list"]
    neighbor_shifts = system_data["neighbor_shifts"]
    num_systems = system_data.get("batch_size", 1)

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
    method: Literal["ewald", "pme", "dsf"],
    backend: Literal["nvalchemiops", "torchpme", "torch_dsf"],
    system_data: dict,
    component: Literal["real", "reciprocal", "full"],
    compute_forces: bool,
    timer: BenchmarkTimer,
    compute_virial: bool = False,
    neighbor_format: str = "list",
) -> dict:
    """Run a single benchmark configuration."""
    total_atoms = system_data["total_atoms"]
    batch_size = system_data.get("batch_size", 1)

    try:
        # Define benchmark function based on method and backend
        if method == "dsf":
            if backend == "nvalchemiops":
                if neighbor_format == "matrix":

                    def bench_fn():
                        return run_nvalchemiops_dsf(
                            system_data, compute_forces, compute_virial
                        )
                else:  # "list" (CSR)

                    def bench_fn():
                        return run_nvalchemiops_dsf_csr(
                            system_data, compute_forces, compute_virial
                        )
            elif backend == "torch_dsf":

                def bench_fn():
                    return run_torch_dsf(system_data, compute_forces, compute_virial)
            else:
                return {
                    "total_atoms": total_atoms,
                    "batch_size": batch_size,
                    "method": method,
                    "backend": backend,
                    "component": component,
                    "compute_forces": compute_forces,
                    "neighbor_format": neighbor_format,
                    "median_time_ms": float("inf"),
                    "peak_memory_mb": None,
                    "success": False,
                    "error": f"Backend '{backend}' not applicable for DSF",
                    "error_type": "NotApplicable",
                }
        elif backend == "nvalchemiops":
            if method == "ewald":
                if neighbor_format == "matrix":

                    def bench_fn():
                        return run_nvalchemiops_ewald_matrix(
                            system_data, component, compute_forces
                        )
                else:

                    def bench_fn():
                        return run_nvalchemiops_ewald(
                            system_data, component, compute_forces, compute_virial
                        )
            else:  # pme
                if neighbor_format == "matrix":

                    def bench_fn():
                        return run_nvalchemiops_pme_matrix(
                            system_data,
                            component,
                            compute_forces,
                        )
                else:

                    def bench_fn():
                        return run_nvalchemiops_pme(
                            system_data,
                            component,
                            compute_forces,
                            compute_virial,
                        )
        elif backend == "torchpme":
            if system_data.get("batch_idx") is not None:
                return {
                    "total_atoms": total_atoms,
                    "batch_size": batch_size,
                    "method": method,
                    "backend": backend,
                    "component": component,
                    "compute_forces": compute_forces,
                    "compute_virial": compute_virial,
                    "neighbor_format": neighbor_format,
                    "median_time_ms": float("inf"),
                    "peak_memory_mb": None,
                    "success": False,
                    "error": "torchpme does not support native batched evaluation",
                    "error_type": "NotImplemented",
                }

            if method == "ewald":

                def bench_fn():
                    return run_torchpme_ewald(
                        system_data, compute_forces, compute_virial
                    )
            else:  # pme

                def bench_fn():
                    return run_torchpme_pme(
                        system_data,
                        compute_forces,
                        compute_virial,
                    )
        else:
            return {
                "total_atoms": total_atoms,
                "batch_size": batch_size,
                "method": method,
                "backend": backend,
                "component": component,
                "compute_forces": compute_forces,
                "neighbor_format": neighbor_format,
                "median_time_ms": float("inf"),
                "peak_memory_mb": None,
                "success": False,
                "error": f"Backend '{backend}' not applicable for {method}",
                "error_type": "NotApplicable",
            }

        # Run benchmark
        timing_results = timer.time_function(bench_fn)
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
                "compute_virial": compute_virial,
                "neighbor_format": neighbor_format,
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
            "compute_virial": compute_virial,
            "neighbor_format": neighbor_format,
            "median_time_ms": float(timing_results["median"]),
            "peak_memory_mb": timing_results.get("peak_memory_mb"),
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
            "compute_virial": compute_virial,
            "neighbor_format": neighbor_format,
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
        choices=["nvalchemiops", "torchpme", "torch_dsf", "both"],
        default="nvalchemiops",
        help=(
            "Backend to use for benchmarking (default: nvalchemiops). "
            "'both' dispatches per-method: torchpme for ewald/pme, "
            "torch_dsf for dsf."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ewald", "pme", "dsf", "both", "all"],
        default="both",
        help=(
            "Method to benchmark (default: both). "
            "'both' = ewald + pme (backward compat). "  # TODO: remove "both", use "all" instead
            "'all' = ewald + pme + dsf."
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
            "Neighbor format for DSF nvalchemiops benchmarks (default: list). "
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

    args = parser.parse_args()

    # Check if torchpme is available when requested
    if args.backend in ["torchpme", "both"] and not TORCHPME_AVAILABLE:
        if args.backend == "torchpme":
            print("ERROR: torchpme backend requested but not installed.")
            print("Install via: pip install torch-pme")
            sys.exit(1)
        else:
            print("WARNING: torchpme not installed, skipping torchpme benchmarks")

    # Load config
    config = load_config(args.config)

    # Get parameters
    params = config["parameters"]
    warmup = int(params["warmup_iterations"])
    timing = int(params["timing_iterations"])
    if args.dtype is not None:
        dtype_str = args.dtype
    else:
        dtype_str = params["dtype"]
    dtype = getattr(torch, dtype_str)
    device_str = params.get("device", "cuda")

    # Setup device
    device = device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu"
    device_obj = torch.device(device)

    # Get GPU SKU
    gpu_sku = args.gpu_sku if args.gpu_sku else get_gpu_sku()

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize timer
    timer = BenchmarkTimer(device_obj, warmup_runs=warmup, timing_runs=timing)

    # Initialize Warp
    wp.init()

    # Determine what to benchmark
    if args.method == "both":  # TODO: remove "both", use "all" instead
        methods = ["ewald", "pme"]
    elif args.method == "all":
        methods = ["ewald", "pme", "dsf"]
    else:
        methods = [args.method]

    # Build per-method backend list
    # "both" dispatches per-method: torchpme for ewald/pme, torch_dsf for dsf
    def get_backends_for_method(method: str) -> list[str]:
        if args.backend == "both":
            if method in ("ewald", "pme"):
                result = ["nvalchemiops"]
                if TORCHPME_AVAILABLE:
                    result.append("torchpme")
                return result
            elif method == "dsf":
                return ["nvalchemiops", "torch_dsf"]
        elif args.backend == "nvalchemiops":
            return ["nvalchemiops"]
        elif args.backend == "torchpme":
            # Only applicable for ewald/pme
            if method in ("ewald", "pme"):
                return ["torchpme"] if TORCHPME_AVAILABLE else []
            return []
        elif args.backend == "torch_dsf":
            # Only applicable for dsf
            if method == "dsf":
                return ["torch_dsf"]
            return []
        return ["nvalchemiops"]

    components = config.get("components", ["full"])
    compute_forces = config.get("compute_forces", True)
    compute_virial = config.get("compute_virial", False)

    # DSF-specific parameters (hardcoded defaults)
    dsf_cutoff = 12.0
    dsf_alpha = 0.2

    # Print configuration
    print("=" * 70)
    print("ELECTROSTATICS BENCHMARK")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"GPU SKU: {gpu_sku}")
    print(f"Dtype: {dtype}")
    print(f"Methods: {methods}")
    print(f"Backend flag: {args.backend}")
    print(f"Components: {components}")
    print(f"Compute forces: {compute_forces}")
    print(f"Compute virial: {compute_virial}")
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
        if result["success"]:
            throughput = result["total_atoms"] / result["median_time_ms"] * 1000
            mem_str = ""
            if result.get("peak_memory_mb"):
                mem_str = f" | {result['peak_memory_mb']:.1f} MB"
            print(
                f"    {method:5s} {backend:12s} {component:10s}: "
                f"{result['median_time_ms']:.3f} ms "
                f"({throughput:.1f} atoms/s){mem_str}"
            )
        else:
            print(
                f"    {method:5s} {backend:12s} {component:10s}: "
                f"FAILED ({result.get('error_type', 'Unknown')})"
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
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()

                # Prepare systems (method-specific)
                system_data_cache = {}
                for method in methods:
                    if method == "dsf":
                        if "dsf" not in system_data_cache:
                            try:
                                system_data_cache["dsf"] = prepare_dsf_single_system(
                                    size, device, dtype, dsf_cutoff, dsf_alpha
                                )
                            except Exception as e:
                                print(f"    Failed to prepare DSF system: {e}")
                                traceback.print_exc()
                                system_data_cache["dsf"] = None
                    else:
                        if "ewald_pme" not in system_data_cache:
                            try:
                                system_data_cache["ewald_pme"] = prepare_single_system(
                                    size, device, dtype
                                )
                            except Exception as e:
                                print(f"    Failed to prepare system: {e}")
                                traceback.print_exc()
                                system_data_cache["ewald_pme"] = None

                for method in methods:
                    backends = get_backends_for_method(method)
                    system_data = system_data_cache.get(
                        "dsf" if method == "dsf" else "ewald_pme"
                    )
                    if system_data is None:
                        continue

                    method_components = ["full"] if method == "dsf" else components
                    for backend in backends:
                        for component in method_components:
                            # Determine neighbor format(s) to benchmark
                            if backend == "nvalchemiops":
                                nf_arg = args.neighbor_format
                                nf_list = (
                                    ["list", "matrix"] if nf_arg == "both" else [nf_arg]
                                )
                            else:
                                nf_list = ["n/a"]

                            for nf in nf_list:
                                try:
                                    build_neighbors(system_data, nf)
                                    result = run_benchmark(
                                        method,
                                        backend,
                                        system_data,
                                        component,
                                        compute_forces,
                                        timer,
                                        compute_virial=compute_virial,
                                        neighbor_format=nf,
                                    )
                                    result["supercell_size"] = size
                                    result["mode"] = mode
                                    all_results.append(result)
                                    nf_tag = f" [{nf}]" if nf != "n/a" else ""
                                    _print_result(
                                        result, method, backend + nf_tag, component
                                    )
                                except (torch.OutOfMemoryError, RuntimeError) as oom:
                                    if (
                                        isinstance(oom, RuntimeError)
                                        and "out of memory" not in str(oom).lower()
                                    ):
                                        raise
                                    torch.cuda.empty_cache()
                                    nf_tag = f" [{nf}]" if nf != "n/a" else ""
                                    result = {
                                        "total_atoms": system_data["total_atoms"],
                                        "batch_size": system_data.get("batch_size", 1),
                                        "method": method,
                                        "backend": backend,
                                        "component": component,
                                        "compute_forces": compute_forces,
                                        "neighbor_format": nf,
                                        "median_time_ms": float("inf"),
                                        "peak_memory_mb": None,
                                        "success": False,
                                        "error": str(oom).split(".")[0],
                                        "error_type": type(oom).__name__,
                                        "supercell_size": size,
                                        "mode": mode,
                                    }
                                    all_results.append(result)
                                    print(
                                        f"    {method:5s} {backend + nf_tag:12s} "
                                        f"{component:10s}: SKIPPED (OOM)"
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
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()

                # Prepare systems (method-specific)
                system_data_cache = {}
                for method in methods:
                    if method == "dsf":
                        if "dsf" not in system_data_cache:
                            try:
                                system_data_cache["dsf"] = prepare_dsf_batch_system(
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
                                system_data_cache["dsf"] = None
                    else:
                        if "ewald_pme" not in system_data_cache:
                            try:
                                system_data_cache["ewald_pme"] = prepare_batch_system(
                                    base_size, batch_size, device, dtype
                                )
                            except Exception as e:
                                print(f"    Failed to prepare system: {e}")
                                traceback.print_exc()
                                system_data_cache["ewald_pme"] = None

                for method in methods:
                    backends = get_backends_for_method(method)
                    system_data = system_data_cache.get(
                        "dsf" if method == "dsf" else "ewald_pme"
                    )
                    if system_data is None:
                        continue

                    method_components = ["full"] if method == "dsf" else components
                    for backend in backends:
                        for component in method_components:
                            # Determine neighbor format(s) to benchmark
                            if backend == "nvalchemiops":
                                nf_arg = args.neighbor_format
                                nf_list = (
                                    ["list", "matrix"] if nf_arg == "both" else [nf_arg]
                                )
                            else:
                                nf_list = ["n/a"]

                            for nf in nf_list:
                                try:
                                    build_neighbors(system_data, nf)
                                    result = run_benchmark(
                                        method,
                                        backend,
                                        system_data,
                                        component,
                                        compute_forces,
                                        timer,
                                        compute_virial=compute_virial,
                                        neighbor_format=nf,
                                    )
                                    result["supercell_size"] = base_size
                                    result["mode"] = mode
                                    all_results.append(result)
                                    nf_tag = f" [{nf}]" if nf != "n/a" else ""
                                    _print_result(
                                        result, method, backend + nf_tag, component
                                    )
                                except (torch.OutOfMemoryError, RuntimeError) as oom:
                                    if (
                                        isinstance(oom, RuntimeError)
                                        and "out of memory" not in str(oom).lower()
                                    ):
                                        raise
                                    torch.cuda.empty_cache()
                                    nf_tag = f" [{nf}]" if nf != "n/a" else ""
                                    result = {
                                        "total_atoms": system_data["total_atoms"],
                                        "batch_size": system_data.get("batch_size", 1),
                                        "method": method,
                                        "backend": backend,
                                        "component": component,
                                        "compute_forces": compute_forces,
                                        "neighbor_format": nf,
                                        "median_time_ms": float("inf"),
                                        "peak_memory_mb": None,
                                        "success": False,
                                        "error": str(oom).split(".")[0],
                                        "error_type": type(oom).__name__,
                                        "supercell_size": base_size,
                                        "mode": mode,
                                    }
                                    all_results.append(result)
                                    print(
                                        f"    {method:5s} {backend + nf_tag:12s} "
                                        f"{component:10s}: SKIPPED (OOM)"
                                    )

    # Save results
    if all_results:
        # Collect all unique backends from results
        all_backends = sorted({r["backend"] for r in all_results})
        # Group by method and backend
        for method in methods:
            for backend in all_backends:
                method_results = [
                    r
                    for r in all_results
                    if r["method"] == method and r["backend"] == backend
                ]
                if method_results:
                    output_file = (
                        output_dir
                        / f"electrostatics_benchmark_{method}_{backend}_{dtype_str}_{gpu_sku}.csv"
                    )
                    # Collect all fieldnames across all results (some may have error fields)
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
