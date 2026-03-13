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
Shared utilities for dynamics benchmarks.

This module provides common functionality used across all dynamics benchmark scripts:
- Unit conversion constants
- System creation utilities
- Result data structures with CSV export
- Configuration loading
- GPU detection
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp
import yaml

from nvalchemiops.dynamics.integrators import (
    # NPT/NPH (MTK) high-level steps + utilities
    compute_barostat_mass,
    compute_pressure_tensor,
    compute_scalar_pressure,
    langevin_baoab_finalize,
    langevin_baoab_half_step,
    nhc_compute_masses,
    run_nph_step,
    run_npt_step,
    vec9d,
    vec9f,
    velocity_verlet_position_update,
    velocity_verlet_velocity_finalize,
)
from nvalchemiops.dynamics.utils import (
    compute_cell_inverse,
    compute_cell_volume,
    compute_kinetic_energy,
    compute_temperature,
    initialize_velocities,
    wrap_positions_to_cell,
)
from nvalchemiops.interactions import lj_energy_forces, lj_energy_forces_virial
from nvalchemiops.neighbors.batch_cell_list import (
    batch_build_cell_list,
    batch_query_cell_list,
)
from nvalchemiops.neighbors.cell_list import build_cell_list, query_cell_list
from nvalchemiops.neighbors.neighbor_utils import (
    selective_zero_num_neighbors,
    selective_zero_num_neighbors_single,
    zero_array,
)
from nvalchemiops.neighbors.rebuild_detection import (
    check_batch_neighbor_list_rebuild,
    check_neighbor_list_rebuild,
)
from nvalchemiops.torch.neighbors.batch_cell_list import estimate_batch_cell_list_sizes
from nvalchemiops.torch.neighbors.cell_list import estimate_cell_list_sizes

wp.init()


@wp.kernel
def _copy_virial_flat_to_vec9d(
    flat: wp.array2d(dtype=wp.float64),
    out: wp.array(dtype=vec9d),
):
    """Copy (B, 9) flat virial into (B,) vec9d for NPT/NPH."""
    i = wp.tid()
    zero_ = wp.float64(1.0)
    out[i] = zero_ * vec9d(
        flat[i, 0],
        flat[i, 1],
        flat[i, 2],
        flat[i, 3],
        flat[i, 4],
        flat[i, 5],
        flat[i, 6],
        flat[i, 7],
        flat[i, 8],
    )


@wp.kernel
def _copy_virial_flat_to_vec9f(
    flat: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=vec9f),
):
    """Copy (B, 9) flat virial into (B,) vec9f for NPT/NPH."""
    i = wp.tid()
    zero_ = wp.float32(1.0)
    out[i] = zero_ * vec9f(
        flat[i, 0],
        flat[i, 1],
        flat[i, 2],
        flat[i, 3],
        flat[i, 4],
        flat[i, 5],
        flat[i, 6],
        flat[i, 7],
        flat[i, 8],
    )


# ==============================================================================
# Physical Constants
# ==============================================================================

# Boltzmann constant in eV/K
KB_EV = 8.617333262e-5

# ------------------------------------------------------------------------------
# Unit conventions used by this example
# ------------------------------------------------------------------------------
#
# The nvalchemiops dynamics kernels are *unitless*: they assume the caller uses a
# self-consistent unit system for (x, v, t, m, E, F).
#
# In this example we choose:
# - length:        Angstrom (Å)
# - time:          femtosecond (fs)
# - energy:        electron-volt (eV)
# - mass (internal): eV * fs^2 / Å^2   (so that KE = 0.5 * m * v^2 is in eV when
#                                     v is in Å/fs)
#
# We accept user-facing masses in amu and convert them to the internal mass unit.
#
# Derivation:
#   1 [eV * fs^2 / Å^2] in SI is:
#     (eV→J) * (fs→s)^2 / (Å→m)^2 = kg
#   so:
#     1 amu = amu_kg / (eV*fs^2/Å^2)_kg
#
EV_TO_J = 1.602176634e-19
AMU_TO_KG = 1.66053906660e-27
FS_TO_S = 1.0e-15
ANGSTROM_TO_M = 1.0e-10

_MASS_UNIT_KG = EV_TO_J * (FS_TO_S**2) / (ANGSTROM_TO_M**2)  # kg per (eV*fs^2/Å^2)
AMU_TO_EV_FS2_PER_A2 = AMU_TO_KG / _MASS_UNIT_KG  # ~103.642691


def mass_amu_to_internal(mass_amu: np.ndarray) -> np.ndarray:
    """Convert masses from amu to internal units (eV*fs^2/Å^2)."""

    return mass_amu * AMU_TO_EV_FS2_PER_A2


# Pressure unit conversion
# 1 eV/Å^3 = 1.602176634e11 Pa
_EV_PER_A3_TO_PA = EV_TO_J / (ANGSTROM_TO_M**3)


def pressure_atm_to_ev_per_a3(pressure_atm: float) -> float:
    """Convert pressure from atm to eV/Å^3."""

    return float(pressure_atm) * 101325.0 / _EV_PER_A3_TO_PA


def pressure_ev_per_a3_to_atm(pressure_ev_per_a3: float) -> float:
    """Convert pressure from eV/Å^3 to atm."""

    return float(pressure_ev_per_a3) * _EV_PER_A3_TO_PA / 101325.0


def pressure_gpa_to_ev_per_a3(p_gpa: float) -> float:
    """Convert pressure from GPa to eV/Å³."""
    return p_gpa * 1e9 / _EV_PER_A3_TO_PA


def pressure_ev_per_a3_to_gpa(p_ev: float) -> float:
    """Convert pressure from eV/Å³ to GPa."""
    return p_ev * _EV_PER_A3_TO_PA / 1e9


# Argon LJ parameters (commonly used for liquid argon simulations)
EPSILON_AR = 0.0104  # eV
SIGMA_AR = 3.40  # Angstrom
MASS_AR = 39.948  # amu

# Default cutoff (2.5*sigma is typical for LJ)
DEFAULT_CUTOFF = 2.5 * SIGMA_AR  # ~8.5 Angstrom
DEFAULT_SKIN = 0.5  # Angstrom


@dataclass
class BenchmarkResult:
    """Container for benchmark results with CSV export support.

    Parameters
    ----------
    name : str
        Benchmark name (e.g., 'velocity_verlet', 'fire').
    backend : str
        Backend used ('nvalchemiops').
    model_type : str
        Model type used ('native_lj', 'nvalchemiops_lj', 'mace', or None for default).
    ensemble : str
        Ensemble or method type (e.g., 'NVE', 'NVT', 'optimization').
    num_atoms : int
        Number of atoms per system.
    num_steps : int
        Number of simulation/optimization steps.
    dt : float
        Timestep in fs (for MD) or max_step (for optimization).
    warmup_steps : int
        Number of warmup steps excluded from timing.
    total_time : float
        Total time in seconds (excluding warmup).
    step_times : list[float]
        List of individual step times in seconds.
    batch_size : Optional[int]
        Number of systems in batch (None for single-system).
    energies : list[float]
        Energy trajectory (optional).
    temperatures : list[float]
        Temperature trajectory (optional).
    final_ke : Optional[float]
        Final kinetic energy in eV (optional).
    final_pe : Optional[float]
        Final potential energy in eV (optional).
    final_temp : Optional[float]
        Final temperature in K (optional).
    """

    name: str
    backend: str
    ensemble: str
    num_atoms: int
    num_steps: int
    dt: float
    warmup_steps: int
    total_time: float
    step_times: list[float] = field(default_factory=list)
    batch_size: int | None = None
    model_type: str | None = None
    energies: list[float] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    final_ke: float | None = None
    final_pe: float | None = None
    final_temp: float | None = None
    neighbor_rebuilds: int | None = None
    cells_per_dimension: str | None = None
    total_cells: int | None = None

    @property
    def avg_step_time_ms(self) -> float:
        """Average time per step in milliseconds."""
        return np.mean(self.step_times) * 1000 if self.step_times else 0.0

    @property
    def throughput_steps_per_s(self) -> float:
        """Calculate steps per second."""
        return self.num_steps / self.total_time if self.total_time > 0 else 0.0

    @property
    def throughput_atom_steps_per_s(self) -> float:
        """Calculate atom-steps per second (throughput metric).

        For batched benchmarks, this uses total_atoms (num_atoms * batch_size)
        to give the total atom-steps across all systems in the batch.
        """
        return (
            self.total_atoms * self.num_steps / self.total_time
            if self.total_time > 0
            else 0.0
        )

    @property
    def total_atoms(self) -> int:
        """Total number of atoms (num_atoms * batch_size for batched)."""
        if self.batch_size is not None:
            return self.num_atoms * self.batch_size
        return self.num_atoms

    @property
    def batch_throughput_system_steps_per_s(self) -> float | None:
        """Complete system steps per second for batched benchmarks."""
        if self.batch_size is not None and self.total_time > 0:
            return self.batch_size * self.num_steps / self.total_time
        return None

    @property
    def is_batched(self) -> bool:
        """Check if this is a batched benchmark."""
        return self.batch_size is not None

    @property
    def ns_per_day(self) -> float:
        """Calculate nanoseconds of simulation time per day of wall-clock time.

        This metric is only meaningful for MD simulations (not optimization).
        For optimization benchmarks, returns NaN.

        Formula: ns/day = (timestep_fs * steps_per_second * 86400 * num_systems) / 1e6
        where timestep_fs is the MD timestep in femtoseconds.
        """
        # Check if this is an optimization benchmark
        if self.ensemble.lower() == "optimization" or self.name.lower() == "fire":
            return float("nan")

        # For MD: calculate ns/day
        num_systems = self.batch_size if self.is_batched else 1
        if self.total_time > 0:
            steps_per_second = self.throughput_steps_per_s
            # timestep (fs) * steps/s * seconds/day / 1e6 (fs to ns)
            return (self.dt * steps_per_second * 86400 * num_systems) / 1e6
        return 0.0

    def to_csv_row(self) -> dict:
        """Convert to dictionary for CSV row.

        Returns appropriate columns based on whether this is a single-system
        or batched benchmark.
        """
        # Common columns for both single and batched
        row = {
            "backend": self.backend,
            "model_type": self.model_type or "",
            "method": self.name,
            "num_atoms": self.num_atoms,
            "ensemble": self.ensemble,
            "steps": self.num_steps,
            "dt": self.dt,
            "warmup_steps": self.warmup_steps,
            "avg_step_time_ms": self.avg_step_time_ms,
            "total_time_s": self.total_time,
            "throughput_steps_per_s": self.throughput_steps_per_s,
            "throughput_atom_steps_per_s": self.throughput_atom_steps_per_s,
            "ns_per_day": self.ns_per_day,
        }

        # Add batch-specific columns if this is a batched benchmark
        if self.is_batched:
            row["batch_size"] = self.batch_size
            row["total_atoms"] = self.total_atoms
            row["batch_throughput_system_steps_per_s"] = (
                self.batch_throughput_system_steps_per_s
            )

        # Add neighbor list instrumentation when populated
        if self.neighbor_rebuilds is not None:
            row["neighbor_rebuilds"] = self.neighbor_rebuilds
        if self.cells_per_dimension is not None:
            row["cells_per_dimension"] = self.cells_per_dimension
        if self.total_cells is not None:
            row["total_cells"] = self.total_cells

        return row

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization (backward compatibility)."""
        d = {
            "name": self.name,
            "backend": self.backend,
            "ensemble": self.ensemble,
            "num_atoms": self.num_atoms,
            "batch_size": self.batch_size,
            "num_steps": self.num_steps,
            "total_time_s": self.total_time,
            "steps_per_second": self.throughput_steps_per_s,
            "atom_steps_per_second": self.throughput_atom_steps_per_s,
            "avg_step_time_ms": self.avg_step_time_ms,
        }
        if self.is_batched:
            d["batch_size"] = self.batch_size
            d["total_atoms"] = self.total_atoms
            d["batch_throughput_system_steps_per_s"] = (
                self.batch_throughput_system_steps_per_s
            )
        return d


def write_results_csv(
    results: list[BenchmarkResult],
    output_path: str | Path,
) -> None:
    """Write benchmark results to CSV file.

    Automatically detects whether results are single-system or batched and uses
    the appropriate schema.

    Parameters
    ----------
    results : list[BenchmarkResult]
        List of benchmark results to write.
    output_path : str or Path
        Output CSV file path.

    Raises
    ------
    ValueError
        If results list is empty or contains mixed single/batch results.
    """
    if not results:
        raise ValueError("Results list is empty")

    # Check if all results are consistently batched or single-system
    is_batched = results[0].is_batched
    if not all(r.is_batched == is_batched for r in results):
        raise ValueError("Cannot mix single-system and batched results in same CSV")

    # Define column order based on benchmark type
    if is_batched:
        # Batched benchmark schema (16 columns)
        fieldnames = [
            "backend",
            "model_type",
            "method",
            "num_atoms",
            "ensemble",
            "batch_size",
            "total_atoms",
            "steps",
            "dt",
            "warmup_steps",
            "avg_step_time_ms",
            "total_time_s",
            "throughput_steps_per_s",
            "throughput_atom_steps_per_s",
            "ns_per_day",
            "batch_throughput_system_steps_per_s",
        ]
    else:
        # Single-system benchmark schema (13 columns)
        fieldnames = [
            "backend",
            "model_type",
            "method",
            "num_atoms",
            "ensemble",
            "steps",
            "dt",
            "warmup_steps",
            "avg_step_time_ms",
            "total_time_s",
            "throughput_steps_per_s",
            "throughput_atom_steps_per_s",
            "ns_per_day",
        ]

    # Write CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())


def load_config(config_path: str | Path) -> dict:
    """Load benchmark configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_benchmark_header(benchmark_type: str = "MD") -> None:
    """Print a formatted header for benchmark results table.

    Parameters
    ----------
    benchmark_type : str
        Type of benchmark ('MD' or 'Optimization').
    """
    if benchmark_type.upper() == "MD":
        print("\n" + "=" * 110)
        print(
            f"{'Method':<20} {'Atoms':<10} {'Steps':<10} {'Time (s)':<12} "
            f"{'Atom-steps/s':<15} {'ns/day':<12}"
        )
        print("=" * 110)
    else:
        print("\n" + "=" * 100)
        print(
            f"{'Method':<20} {'Atoms':<10} {'Steps':<10} {'Converged':<12} "
            f"{'Time (s)':<12} {'Atom-steps/s':<15}"
        )
        print("=" * 100)


def print_benchmark_result(result: BenchmarkResult, is_md: bool = True) -> None:
    """Print a formatted row for benchmark result.

    Parameters
    ----------
    result : BenchmarkResult
        Benchmark result to print.
    is_md : bool
        Whether this is an MD benchmark (True) or optimization (False).
    """
    oom_note = " [OOM]" if result.total_time == 0 and not result.step_times else ""
    method_str = f"{result.name} ({result.ensemble}){oom_note}"

    if is_md:
        # For MD: show ns/day
        ns_day = result.ns_per_day
        ns_day_str = f"{ns_day:.2f}" if not np.isnan(ns_day) else "N/A"
        print(
            f"{method_str:<20} {result.num_atoms:<10} {result.num_steps:<10} "
            f"{result.total_time:<12.3f} {result.throughput_atom_steps_per_s:<15.2e} "
            f"{ns_day_str:<12}"
        )
    else:
        # For optimization: show convergence instead of ns/day
        converged = "Yes" if result.num_steps < 1000 else "No"
        print(
            f"{method_str:<20} {result.num_atoms:<10} {result.num_steps:<10} "
            f"{converged:<12} {result.total_time:<12.3f} "
            f"{result.throughput_atom_steps_per_s:<15.2e}"
        )


def print_benchmark_footer() -> None:
    """Print a formatted footer for benchmark results table."""
    print("=" * 110)


def print_batch_benchmark_header() -> None:
    """Print a formatted header for batched benchmark results table."""
    print("\n" + "=" * 120)
    print(
        f"{'Backend':<18} {'Method':<18} {'Atoms/sys':<12} {'Batch':<8} {'Total':<10} "
        f"{'Steps':<10} {'Time (s)':<12} {'Atom-steps/s':<15} {'ns/day':<12}"
    )
    print("=" * 120)


def print_batch_benchmark_result(result: BenchmarkResult, is_md: bool = True) -> None:
    """Print a formatted row for batched benchmark result.

    Parameters
    ----------
    result : BenchmarkResult
        Batched benchmark result to print.
    is_md : bool
        Whether this is an MD benchmark (True) or optimization (False).
    """
    oom_note = " [OOM]" if result.total_time == 0 and not result.step_times else ""
    method_str = f"{result.name}{oom_note}"
    batch_size = result.batch_size if result.batch_size else 1
    total_atoms = result.total_atoms

    if is_md:
        ns_day = result.ns_per_day
        ns_day_str = f"{ns_day:.2f}" if not np.isnan(ns_day) else "N/A"
        print(
            f"{result.backend:<18} {method_str:<18} {result.num_atoms:<12} {batch_size:<8} {total_atoms:<10} "
            f"{result.num_steps:<10} {result.total_time:<12.3f} "
            f"{result.throughput_atom_steps_per_s:<15.2e} {ns_day_str:<12}"
        )
    else:
        # For optimization
        print(
            f"{result.backend:<18} {method_str:<18} {result.num_atoms:<12} {batch_size:<8} {total_atoms:<10} "
            f"{result.num_steps:<10} {result.total_time:<12.3f} "
            f"{result.throughput_atom_steps_per_s:<15.2e}"
        )


def print_batch_benchmark_footer() -> None:
    """Print a formatted footer for batched benchmark results table."""
    print("=" * 120)


def get_gpu_sku() -> str:
    """Get GPU model name for benchmark identification.

    Returns
    -------
    str
        GPU model string (e.g., 'rtx4090', 'a100').
        Returns 'cpu' if no CUDA device available.
        Returns 'unknown_gpu' if detection fails.

    Examples
    --------
    >>> get_gpu_sku()
    'rtx4090'
    """
    try:
        if not torch.cuda.is_available():
            return "cpu"

        # Get device name from PyTorch
        device_name = torch.cuda.get_device_name(0)

        # Clean up device name to create SKU string
        # e.g., "NVIDIA GeForce RTX 4090" -> "rtx4090"
        sku = device_name.lower()
        sku = sku.replace("nvidia", "").replace("geforce", "").replace("tesla", "")
        sku = sku.replace("quadro", "").replace("rtx", "rtx")
        sku = "".join(c for c in sku if c.isalnum())

        # Handle common patterns
        if "rtx" in sku:
            # Extract RTX model number
            idx = sku.find("rtx")
            sku = "rtx" + sku[idx + 3 : idx + 7].strip()
        elif "a100" in sku:
            sku = "a100"
        elif "v100" in sku:
            sku = "v100"
        elif "h100" in sku:
            sku = "h100"
        elif sku:
            # Fallback: take first alphanumeric token
            sku = sku.split()[0] if sku.split() else "unknown_gpu"
        else:
            sku = "unknown_gpu"

        return sku

    except Exception:
        return "unknown_gpu"


@wp.kernel
def _pack_virial_flat_to_vec9_kernel(
    virial_flat: wp.array(dtype=Any),
    virial_vec9: wp.array(dtype=Any),
):
    """Pack a 9-element virial into a vec9f/vec9d output array (single system)."""
    sys_id = wp.tid()
    v = virial_vec9[sys_id]
    v0 = virial_vec9[sys_id][0]
    # NOTE ON SIGN CONVENTION:
    # - The LJ kernels accumulate virial with a negative sign convention: W = -Σ r ⊗ F.
    # - The MTK NPT/NPH implementation in `nvalchemiops.dynamics.integrators.npt`
    #   computes pressure as P = (kinetic + virial) / V, which expects virial in the
    #   *positive* convention: +Σ r ⊗ F.
    # Therefore we negate here so the integrators see the expected virial sign.
    virial_vec9[sys_id] = type(v)(
        type(v0)(-virial_flat[0]),
        type(v0)(-virial_flat[1]),
        type(v0)(-virial_flat[2]),
        type(v0)(-virial_flat[3]),
        type(v0)(-virial_flat[4]),
        type(v0)(-virial_flat[5]),
        type(v0)(-virial_flat[6]),
        type(v0)(-virial_flat[7]),
        type(v0)(-virial_flat[8]),
    )


@wp.kernel
def _copy_1d_to_row2d_kernel(src: wp.array(dtype=Any), dst: wp.array2d(dtype=Any)):
    """Copy src[k] -> dst[0, k]. Used to adapt single-system outputs to batched APIs."""
    k = wp.tid()
    if k < src.shape[0]:
        dst[0, k] = src[k]


@wp.kernel
def _virial_to_stress_kernel(
    virial_flat: wp.array(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    external_pressure: wp.float64,
    stress: wp.array(dtype=wp.mat33d),
):
    """Convert virial to stress tensor for cell optimization.

    Computes: stress = P_ext - P_internal

    where P_internal = -virial/V (negated because LJ virial has W = -Σ r ⊗ F).

    At equilibrium: P_internal = P_ext, so stress = 0.
    When P_internal < P_ext: stress > 0, cell contracts (via stress_to_cell_force).
    When P_internal > P_ext: stress < 0, cell expands.
    """
    sys = wp.tid()
    V = volume[sys]
    inv_V = wp.float64(1.0) / V

    # Virial components (row-major: xx, xy, xz, yx, yy, yz, zx, zy, zz)
    # Negate because LJ virial uses convention W = -Σ r ⊗ F
    # After negation: positive = compression (repulsive forces)
    vxx = -virial_flat[0]
    vxy = -virial_flat[1]
    vxz = -virial_flat[2]
    vyx = -virial_flat[3]
    vyy = -virial_flat[4]
    vyz = -virial_flat[5]
    vzx = -virial_flat[6]
    vzy = -virial_flat[7]
    vzz = -virial_flat[8]

    # Internal pressure (diagonal): P_int = virial/V (positive = compression)
    # Stress for optimization: σ = P_ext - P_int
    # This ensures σ = 0 at equilibrium, and correct sign for cell force
    stress[sys] = wp.mat33d(
        external_pressure - vxx * inv_V,
        -vxy * inv_V,
        -vxz * inv_V,
        -vyx * inv_V,
        external_pressure - vyy * inv_V,
        -vyz * inv_V,
        -vzx * inv_V,
        -vzy * inv_V,
        external_pressure - vzz * inv_V,
    )


def virial_to_stress(
    virial_flat: wp.array,
    cell: wp.array,
    external_pressure: float,
    device: str,
) -> wp.array:
    """Convert virial tensor to stress with external pressure.

    Parameters
    ----------
    virial_flat : wp.array, shape (9,)
        Flat virial tensor from LJ computation.
    cell : wp.array, shape (1, 3, 3)
        Cell matrix.
    external_pressure : float
        External pressure in eV/Å³.
    device : str
        Warp device.

    Returns
    -------
    stress : wp.array, shape (1,), dtype=mat33d
        Stress tensor for cell optimization.
    """
    volume = wp.empty(1, dtype=wp.float64, device=device)
    compute_cell_volume(cell, volumes=volume, device=device)
    stress = wp.zeros(1, dtype=wp.mat33d, device=device)
    wp.launch(
        _virial_to_stress_kernel,
        dim=1,
        inputs=[virial_flat, volume, wp.float64(external_pressure)],
        outputs=[stress],
        device=device,
    )
    return stress


# ==============================================================================
# Neighbor List Management
# ==============================================================================


class NeighborListManager:
    """Manages neighbor list construction and updates (warp-native, zero CPU-GPU sync).

    Uses the cell list algorithm for O(N) neighbor finding with periodic boundary
    conditions. All neighbor data lives in pre-allocated warp arrays; the rebuild
    decision is made entirely on the GPU via skin-distance checking.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the system.
    cutoff : float
        Cutoff distance for neighbor detection (Angstrom).
    skin : float
        Neighbor list skin distance (Angstrom). Rebuild when any atom
        moves more than skin/2.
    initial_cell : np.ndarray, shape (3, 3)
        Initial cell matrix used to estimate cell list buffer sizes.
    pbc : list of bool, length 3
        Periodic boundary conditions in each dimension.
    max_neighbors : int, optional
        Maximum number of neighbors per atom.
    half_fill : bool, optional
        If True, only fill half of the neighbor matrix (Newton's 3rd law).
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    wp_vec_dtype : type
        Warp vector dtype (wp.vec3f or wp.vec3d).
    device : str, optional
        Warp device string (e.g., "cuda:0", "cpu").
    """

    def __init__(
        self,
        num_atoms: int,
        cutoff: float,
        skin: float,
        initial_cell: np.ndarray,
        pbc: list,
        max_neighbors: int = 100,
        half_fill: bool = True,
        wp_dtype: type = wp.float64,
        wp_vec_dtype: type = wp.vec3d,
        device: str = "cuda:0",
    ):
        self.num_atoms = num_atoms
        self.cutoff = cutoff
        self.skin = skin
        self.max_neighbors = max_neighbors
        self.half_fill = half_fill
        self.wp_dtype = wp_dtype
        self.wp_vec_dtype = wp_vec_dtype
        self.device = device

        # Store pbc as warp 1D bool array (shape (3,))
        self.wp_pbc = wp.array(pbc, dtype=wp.bool, device=device)

        # Estimate cell list sizes at init (one-time CPU sync is acceptable here)
        torch_dtype = torch.float64 if wp_dtype == wp.float64 else torch.float32
        cell_torch = torch.tensor(
            initial_cell.reshape(1, 3, 3), dtype=torch_dtype, device=str(device)
        )
        pbc_torch = torch.tensor([pbc], dtype=torch.bool, device=str(device))  # (1, 3)
        max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell_torch, pbc_torch, cutoff + skin
        )

        # Cell list internal buffers (pre-allocated once, reused each step)
        self.wp_cells_per_dimension = wp.zeros(3, dtype=wp.int32, device=device)
        self.wp_atom_periodic_shifts = wp.zeros(
            num_atoms, dtype=wp.vec3i, device=device
        )
        self.wp_atom_to_cell_mapping = wp.zeros(
            num_atoms, dtype=wp.vec3i, device=device
        )
        self.wp_atoms_per_cell_count = wp.zeros(
            max_total_cells, dtype=wp.int32, device=device
        )
        self.wp_cell_atom_start_indices = wp.zeros(
            max_total_cells, dtype=wp.int32, device=device
        )
        self.wp_cell_atom_list = wp.zeros(num_atoms, dtype=wp.int32, device=device)
        self.wp_neighbor_search_radius = wp.from_torch(
            neighbor_search_radius, dtype=wp.int32
        )

        # Neighbor output buffers (pre-allocated once)
        self.wp_neighbor_matrix = wp.zeros(
            (num_atoms, max_neighbors), dtype=wp.int32, device=device
        )
        self.wp_neighbor_shifts = wp.zeros(
            (num_atoms, max_neighbors), dtype=wp.vec3i, device=device
        )
        self.wp_num_neighbors = wp.zeros(num_atoms, dtype=wp.int32, device=device)

        # Rebuild detection: reference positions and per-system flag
        self.wp_ref_positions = wp.zeros(num_atoms, dtype=wp_vec_dtype, device=device)
        self.wp_rebuild_flag = wp.zeros(1, dtype=wp.bool, device=device)

        # Instrumentation counters (benchmark-only)
        self.rebuild_count = 0

    def mark_stale(self) -> None:
        """Force a full rebuild on the next update() call.

        Zeros reference positions so the next skin-distance check always exceeds
        the threshold. Use after cell changes (NPT/NPH or variable-cell optimization).
        """
        self.wp_ref_positions.zero_()

    def update(
        self,
        positions_wp: wp.array,
        cell_wp: wp.array,
        cell_inv_wp: wp.array | None = None,
    ) -> None:
        """Check and selectively rebuild the neighbor list (no CPU-GPU sync).

        Parameters
        ----------
        positions_wp : wp.array, shape (N,), dtype=wp.vec3*
            Current atomic positions.
        cell_wp : wp.array, shape (1,), dtype=wp.mat33*
            Current cell matrix.
        cell_inv_wp : wp.array or None, optional
            Precomputed inverse of the cell matrix.  When provided the
            rebuild check uses minimum-image convention (MIC).
        """
        # 1. Zero rebuild flag — kernel only sets True, never clears
        zero_array(self.wp_rebuild_flag, self.device)

        # 2. GPU-side displacement check — writes True if any atom moved > skin/2
        check_neighbor_list_rebuild(
            reference_positions=self.wp_ref_positions,
            current_positions=positions_wp,
            skin_distance_threshold=self.skin / 2.0,
            rebuild_flag=self.wp_rebuild_flag,
            wp_dtype=self.wp_dtype,
            device=self.device,
            update_reference_positions=True,
            cell=cell_wp if cell_inv_wp is not None else None,
            cell_inv=cell_inv_wp,
            pbc=self.wp_pbc if cell_inv_wp is not None else None,
        )

        # 3. Always rebuild cell structure (cheap O(N) spatial binning)
        zero_array(self.wp_atoms_per_cell_count, self.device)
        build_cell_list(
            positions_wp,
            cell_wp,
            self.wp_pbc,
            self.cutoff + self.skin,
            self.wp_cells_per_dimension,
            self.wp_atom_periodic_shifts,
            self.wp_atom_to_cell_mapping,
            self.wp_atoms_per_cell_count,
            self.wp_cell_atom_start_indices,
            self.wp_cell_atom_list,
            self.wp_dtype,
            self.device,
        )

        # 4. Selectively zero num_neighbors and query neighbor matrix
        selective_zero_num_neighbors_single(
            self.wp_num_neighbors, self.wp_rebuild_flag, self.device
        )
        query_cell_list(
            positions_wp,
            cell_wp,
            self.wp_pbc,
            self.cutoff + self.skin,
            self.wp_cells_per_dimension,
            self.wp_neighbor_search_radius,
            self.wp_atom_periodic_shifts,
            self.wp_atom_to_cell_mapping,
            self.wp_atoms_per_cell_count,
            self.wp_cell_atom_start_indices,
            self.wp_cell_atom_list,
            self.wp_neighbor_matrix,
            self.wp_neighbor_shifts,
            self.wp_num_neighbors,
            self.wp_dtype,
            self.device,
            half_fill=self.half_fill,
            rebuild_flags=self.wp_rebuild_flag,
        )

    def total_neighbors(self) -> int:
        """Get total number of neighbors across all atoms."""
        return int(wp.to_torch(self.wp_num_neighbors).sum().item())


class BatchedNeighborListManager:
    """Neighbor list manager for batched systems (warp-native, zero CPU-GPU sync).

    Uses per-system skin-distance rebuild detection entirely on the GPU.
    All neighbor data lives in pre-allocated warp arrays reused each step.

    Parameters
    ----------
    total_atoms : int
        Total number of atoms across all systems.
    cutoff : float
        Cutoff distance for neighbor detection (Angstrom).
    skin : float
        Neighbor list skin distance (Angstrom). Rebuild system when any atom
        in it moves more than skin/2.
    batch_idx : np.ndarray, shape (total_atoms,), dtype=int32
        System index for each atom.
    num_systems : int
        Number of systems in the batch.
    initial_cells : np.ndarray, shape (num_systems, 3, 3)
        Initial cell matrices used to estimate cell list buffer sizes.
    pbc : np.ndarray, shape (num_systems, 3), dtype=bool
        Periodic boundary conditions for each system and dimension.
    max_neighbors : int, optional
        Maximum number of neighbors per atom.
    half_fill : bool, optional
        If True, only fill half of the neighbor matrix (Newton's 3rd law).
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    wp_vec_dtype : type
        Warp vector dtype (wp.vec3f or wp.vec3d).
    device : str, optional
        Warp device string (e.g., "cuda:0", "cpu").
    """

    def __init__(
        self,
        total_atoms: int,
        cutoff: float,
        skin: float,
        batch_idx: np.ndarray,
        num_systems: int,
        initial_cells: np.ndarray,
        pbc: np.ndarray,
        max_neighbors: int = 100,
        half_fill: bool = True,
        wp_dtype: type = wp.float64,
        wp_vec_dtype: type = wp.vec3d,
        device: str = "cuda:0",
    ):
        self.total_atoms = int(total_atoms)
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.num_systems = int(num_systems)
        self.max_neighbors = int(max_neighbors)
        self.half_fill = bool(half_fill)
        self.wp_dtype = wp_dtype
        self.wp_vec_dtype = wp_vec_dtype
        self.device = device

        # batch_idx as warp int32 array
        self.wp_batch_idx = wp.array(
            np.asarray(batch_idx, dtype=np.int32), dtype=wp.int32, device=device
        )

        # pbc as warp 2D bool array (shape (num_systems, 3))
        pbc_np = np.asarray(pbc, dtype=bool)
        pbc_torch = torch.tensor(pbc_np, dtype=torch.bool, device=str(device)).reshape(
            -1, 3
        )
        self.wp_pbc = wp.from_torch(pbc_torch, dtype=wp.bool)

        # Estimate cell list sizes at init (one-time CPU sync is acceptable here)
        torch_dtype = torch.float64 if wp_dtype == wp.float64 else torch.float32
        cells_torch = torch.tensor(
            np.asarray(initial_cells), dtype=torch_dtype, device=str(device)
        )
        max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cells_torch, pbc_torch, cutoff + skin
        )

        # Cell list internal buffers (pre-allocated once, reused each step)
        self.wp_cells_per_dimension = wp.zeros(
            num_systems, dtype=wp.vec3i, device=device
        )
        self.wp_cell_offsets = wp.zeros(num_systems, dtype=wp.int32, device=device)
        self.wp_cells_per_system = wp.zeros(num_systems, dtype=wp.int32, device=device)
        self.wp_atom_periodic_shifts = wp.zeros(
            total_atoms, dtype=wp.vec3i, device=device
        )
        self.wp_atom_to_cell_mapping = wp.zeros(
            total_atoms, dtype=wp.vec3i, device=device
        )
        self.wp_atoms_per_cell_count = wp.zeros(
            max_total_cells, dtype=wp.int32, device=device
        )
        self.wp_cell_atom_start_indices = wp.zeros(
            max_total_cells, dtype=wp.int32, device=device
        )
        self.wp_cell_atom_list = wp.zeros(total_atoms, dtype=wp.int32, device=device)
        self.wp_neighbor_search_radius = wp.from_torch(
            neighbor_search_radius, dtype=wp.vec3i
        )

        # Neighbor output buffers (pre-allocated once)
        self.wp_neighbor_matrix = wp.zeros(
            (total_atoms, max_neighbors), dtype=wp.int32, device=device
        )
        self.wp_neighbor_shifts = wp.zeros(
            (total_atoms, max_neighbors), dtype=wp.vec3i, device=device
        )
        self.wp_num_neighbors = wp.zeros(total_atoms, dtype=wp.int32, device=device)

        # Rebuild detection: reference positions and per-system flags
        self.wp_ref_positions = wp.zeros(total_atoms, dtype=wp_vec_dtype, device=device)
        self.wp_rebuild_flags = wp.zeros(num_systems, dtype=wp.bool, device=device)

        # Instrumentation counters (benchmark-only)
        self.rebuild_count = 0

    def mark_stale(self) -> None:
        """Force a full rebuild of all systems on the next update() call.

        Zeros reference positions so all atoms will exceed the skin threshold.
        Use after cell changes (NPT/NPH or variable-cell optimization).
        """
        self.wp_ref_positions.zero_()

    def update(
        self,
        positions_wp: wp.array,
        cells_wp: wp.array,
        cells_inv_wp: wp.array | None = None,
    ) -> None:
        """Check and selectively rebuild neighbor lists (no CPU-GPU sync).

        Parameters
        ----------
        positions_wp : wp.array, shape (total_atoms,), dtype=wp.vec3*
            Current atomic positions for all systems.
        cells_wp : wp.array, shape (num_systems,), dtype=wp.mat33*
            Current cell matrices.
        cells_inv_wp : wp.array or None, optional
            Precomputed per-system inverse cell matrices.  When provided
            the rebuild check uses minimum-image convention (MIC).
        """
        # 1. Zero per-system flags — kernel only sets True, never clears
        zero_array(self.wp_rebuild_flags, self.device)

        # 2. GPU-side per-system displacement check — no CPU sync
        check_batch_neighbor_list_rebuild(
            reference_positions=self.wp_ref_positions,
            current_positions=positions_wp,
            batch_idx=self.wp_batch_idx,
            skin_distance_threshold=self.skin / 2.0,
            rebuild_flags=self.wp_rebuild_flags,
            wp_dtype=self.wp_dtype,
            device=self.device,
            update_reference_positions=True,
            cell=cells_wp if cells_inv_wp is not None else None,
            cell_inv=cells_inv_wp,
            pbc=self.wp_pbc if cells_inv_wp is not None else None,
        )

        # 3. Always rebuild cell structure (cheap O(N) spatial binning)
        zero_array(self.wp_atoms_per_cell_count, self.device)
        batch_build_cell_list(
            positions_wp,
            cells_wp,
            self.wp_pbc,
            self.cutoff + self.skin,
            self.wp_batch_idx,
            self.wp_cells_per_dimension,
            self.wp_cell_offsets,
            self.wp_cells_per_system,
            self.wp_atom_periodic_shifts,
            self.wp_atom_to_cell_mapping,
            self.wp_atoms_per_cell_count,
            self.wp_cell_atom_start_indices,
            self.wp_cell_atom_list,
            self.wp_dtype,
            self.device,
        )

        # 4. Selectively zero num_neighbors and query — GPU skips unchanged systems
        selective_zero_num_neighbors(
            self.wp_num_neighbors,
            self.wp_batch_idx,
            self.wp_rebuild_flags,
            self.device,
        )
        batch_query_cell_list(
            positions_wp,
            cells_wp,
            self.wp_pbc,
            self.cutoff + self.skin,
            self.wp_batch_idx,
            self.wp_cells_per_dimension,
            self.wp_neighbor_search_radius,
            self.wp_cell_offsets,
            self.wp_atom_periodic_shifts,
            self.wp_atom_to_cell_mapping,
            self.wp_atoms_per_cell_count,
            self.wp_cell_atom_start_indices,
            self.wp_cell_atom_list,
            self.wp_neighbor_matrix,
            self.wp_neighbor_shifts,
            self.wp_num_neighbors,
            self.wp_dtype,
            self.device,
            half_fill=self.half_fill,
            rebuild_flags=self.wp_rebuild_flags,
        )

    def total_neighbors(self) -> int:
        """Get total number of neighbors across all atoms."""
        return int(wp.to_torch(self.wp_num_neighbors).sum().item())


# ==============================================================================
# System Creation
# ==============================================================================


def create_fcc_argon(
    num_unit_cells: int = 4, a: float = 5.26
) -> tuple[np.ndarray, np.ndarray]:
    """Create FCC argon lattice.

    Creates a face-centered cubic (FCC) lattice with 4 atoms per unit cell,
    typical for noble gas crystals near the triple point.

    Parameters
    ----------
    num_unit_cells : int
        Number of unit cells in each dimension. Total atoms = 4 * n^3.
    a : float
        Lattice constant in Angstrom. For argon at 94K: ~5.26 Å

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        Atomic positions in Angstrom.
    cell : np.ndarray, shape (3, 3)
        Unit cell matrix (diagonal).
    """
    # FCC basis (4 atoms per unit cell)
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )

    # Generate all positions
    positions = []
    for i in range(num_unit_cells):
        for j in range(num_unit_cells):
            for k in range(num_unit_cells):
                for b in basis:
                    pos = (np.array([i, j, k]) + b) * a
                    positions.append(pos)

    positions = np.array(positions, dtype=np.float64)

    # Create cell matrix (cubic)
    L = num_unit_cells * a
    cell = np.eye(3, dtype=np.float64) * L

    return positions, cell


def create_fcc_lattice(n_cells: int, a: float) -> tuple[np.ndarray, np.ndarray]:
    """Create FCC lattice with n_cells unit cells per dimension.

    Parameters
    ----------
    n_cells : int
        Number of unit cells in each dimension.
    a : float
        Lattice constant (Å).

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        Atomic positions.
    cell : np.ndarray, shape (3, 3)
        Cell matrix.

    Note
    ----
    This is an alias for :func:`create_fcc_argon` with different parameter name.
    For consistency, consider using :func:`create_fcc_argon` instead.
    """
    return create_fcc_argon(num_unit_cells=n_cells, a=a)


def create_random_cluster(
    num_atoms: int,
    radius: float = 10.0,
    min_dist: float = 3.0,
    center: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Create a random spherical cluster with minimum distance constraint.

    Generates atoms distributed within a sphere, rejecting overlaps to create
    a loose cluster suitable for geometry optimization.

    Parameters
    ----------
    num_atoms : int
        Number of atoms to place.
    radius : float
        Maximum radius of the cluster (Angstrom).
    min_dist : float
        Minimum allowed distance between any two atoms (Angstrom).
        Typically ~0.9 * sigma for LJ systems.
    center : np.ndarray, shape (3,), optional
        Center of the cluster. Defaults to origin.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    positions : np.ndarray, shape (num_atoms, 3)
        Atomic positions in Angstrom.

    Raises
    ------
    RuntimeError
        If unable to place all atoms within max_tries attempts.

    Examples
    --------
    >>> positions = create_random_cluster(32, radius=12.0, min_dist=3.0, seed=42)
    >>> positions.shape
    (32, 3)
    """
    rng = np.random.default_rng(seed)
    positions = np.zeros((num_atoms, 3), dtype=np.float64)
    placed = 0
    max_tries = 100000
    tries = 0

    while placed < num_atoms and tries < max_tries:
        tries += 1
        # Generate point uniformly in sphere using rejection sampling
        v = rng.normal(size=3)
        v /= np.linalg.norm(v) + 1e-12
        r = radius * (rng.random() ** (1.0 / 3.0))  # Uniform in volume
        candidate = r * v

        if placed == 0:
            positions[placed] = candidate
            placed += 1
            continue

        d = np.linalg.norm(positions[:placed] - candidate[None, :], axis=1)
        if np.all(d > min_dist):
            positions[placed] = candidate
            placed += 1

    if placed != num_atoms:
        raise RuntimeError(
            f"Failed to place cluster atoms (placed={placed}/{num_atoms}). "
            f"Try increasing radius or decreasing min_dist."
        )

    # Apply center offset if provided
    if center is not None:
        positions += np.asarray(center, dtype=np.float64)

    return positions


def create_random_box_cluster(
    num_atoms: int,
    box_size: float,
    min_dist: float = 3.0,
    margin: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Create a random cluster in a cubic box with minimum distance constraint.

    Generates atoms uniformly distributed within a box (with margin from edges),
    rejecting overlaps. Suitable for periodic systems.

    Parameters
    ----------
    num_atoms : int
        Number of atoms to place.
    box_size : float
        Size of the cubic box (Angstrom).
    min_dist : float
        Minimum allowed distance between any two atoms (Angstrom).
    margin : float
        Fraction of box size to leave as margin from edges (0 to 0.5).
        Atoms are placed in [margin*L, (1-margin)*L].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    positions : np.ndarray, shape (num_atoms, 3)
        Atomic positions in Angstrom.

    Raises
    ------
    RuntimeError
        If unable to place all atoms within max_tries attempts.

    Examples
    --------
    >>> positions = create_random_box_cluster(32, box_size=30.0, min_dist=3.0, seed=42)
    >>> positions.shape
    (32, 3)
    """
    rng = np.random.default_rng(seed)
    positions = np.zeros((num_atoms, 3), dtype=np.float64)
    placed = 0
    max_tries = 100000
    tries = 0

    lo = margin * box_size
    hi = (1.0 - margin) * box_size

    while placed < num_atoms and tries < max_tries:
        tries += 1
        candidate = rng.uniform(lo, hi, size=3)

        if placed == 0:
            positions[placed] = candidate
            placed += 1
            continue

        d = np.linalg.norm(positions[:placed] - candidate[None, :], axis=1)
        if np.all(d > min_dist):
            positions[placed] = candidate
            placed += 1

    if placed != num_atoms:
        raise RuntimeError(
            f"Failed to place cluster atoms (placed={placed}/{num_atoms}). "
            f"Try increasing box_size or decreasing min_dist."
        )

    return positions


# ==============================================================================
# Core MD System (integrator-agnostic)
# ==============================================================================


class MDSystem:
    """Integrator-agnostic molecular dynamics system.

    This class owns the *state* needed for many MD algorithms:
    - positions / velocities / forces / masses (Warp arrays)
    - cell + inverse (Warp arrays)
    - neighbor list manager (warp-native, zero CPU-GPU sync)
    - LJ force evaluation (via :func:`nvalchemiops.interactions.lj_energy_forces`)

    Integrators (Langevin, Velocity Verlet, NPT, NPH, ...) should be implemented
    as separate "runner" functions operating on this system.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        masses: torch.Tensor | None = None,
        epsilon: float = EPSILON_AR,
        sigma: float = SIGMA_AR,
        cutoff: float = DEFAULT_CUTOFF,
        skin: float = DEFAULT_SKIN,
        switch_width: float = 0.0,
        half_neighbor_list: bool = True,
        device: str = "cuda:0",
        dtype: np.dtype = torch.float64,
    ):
        self.num_atoms = len(positions)
        self.total_atoms = self.num_atoms
        self.num_systems = 1
        self.wp_batch_idx = None
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.switch_width = float(switch_width)
        self.half_neighbor_list = bool(half_neighbor_list)
        self.device = device
        self.wp_device = wp.device_from_torch(device)
        self.num_atoms_per_system = torch.tensor(
            [self.num_atoms], dtype=torch.int32, device=device
        )

        # Determine types
        self.dtype = dtype
        self.wp_dtype = wp.float64 if dtype == torch.float64 else wp.float32
        self.wp_vec_dtype = wp.vec3d if dtype == torch.float64 else wp.vec3f
        self.wp_mat_dtype = wp.mat33d if dtype == torch.float64 else wp.mat33f
        self.torch_dtype = torch.float64 if dtype == torch.float64 else torch.float32

        # Set up masses
        if masses is None:
            masses = torch.full((self.num_atoms,), MASS_AR, dtype=dtype, device=device)
        else:
            masses = masses.astype(dtype)

        # Convert masses to internal MD units (so KE is in eV when v is Å/fs)
        masses = mass_amu_to_internal(masses)

        # Create warp arrays for dynamics
        self.wp_positions = wp.from_torch(positions, dtype=self.wp_vec_dtype)
        self.wp_velocities = wp.zeros(
            self.num_atoms, dtype=self.wp_vec_dtype, device=self.wp_device
        )
        self.wp_forces = wp.zeros(
            self.num_atoms, dtype=self.wp_vec_dtype, device=self.wp_device
        )
        self.wp_masses = wp.from_torch(masses, dtype=self.wp_dtype)
        self.wp_energies = wp.zeros(
            self.num_atoms, dtype=self.wp_dtype, device=self.wp_device
        )
        self.wp_virial_flat = wp.zeros(9, dtype=self.wp_dtype, device=self.wp_device)

        # Cell matrix (shape (1,) for single system)
        cell_reshaped = cell.reshape(1, 3, 3).to(dtype=self.torch_dtype)
        self.wp_cell = wp.from_torch(cell_reshaped, dtype=self.wp_mat_dtype)

        # Compute cell inverse for position wrapping
        self.wp_cell_inv = wp.empty_like(self.wp_cell)
        compute_cell_inverse(self.wp_cell, self.wp_cell_inv, device=self.wp_device)

        # Set up neighbor list manager (initial_cell from torch tensor)
        initial_cell_np = cell.reshape(3, 3).cpu().numpy()
        pbc_list = pbc.cpu().tolist()
        self.neighbor_manager = NeighborListManager(
            num_atoms=self.num_atoms,
            cutoff=cutoff,
            skin=skin,
            initial_cell=initial_cell_np,
            pbc=pbc_list,
            max_neighbors=100,
            half_fill=self.half_neighbor_list,
            wp_dtype=self.wp_dtype,
            wp_vec_dtype=self.wp_vec_dtype,
            device=self.wp_device,
        )

        # Build initial neighbor list (ref_positions=zeros guarantees full rebuild)
        self._update_neighbors()

    def _update_neighbors(self) -> None:
        """Check and selectively rebuild neighbor list (no CPU-GPU sync)."""
        self.neighbor_manager.update(self.wp_positions, self.wp_cell, self.wp_cell_inv)

    def compute_forces(self) -> wp.array:
        """Compute LJ forces and return per-atom potential energies (device array).

        Notes
        -----
        This function intentionally does **not** synchronize or pull data back to
        the host. Host-side reductions (e.g., PE sum) should be done only at
        logging / analysis points.
        """
        # Check and selectively rebuild neighbor list (no CPU-GPU sync)
        self._update_neighbors()

        # Compute LJ energy and forces using the interactions module
        wp_energies, _ = lj_energy_forces(
            positions=self.wp_positions,
            cell=self.wp_cell,
            epsilon=self.epsilon,
            sigma=self.sigma,
            cutoff=self.cutoff,
            switch_width=self.switch_width,
            half_neighbor_list=self.half_neighbor_list,
            neighbor_matrix=self.neighbor_manager.wp_neighbor_matrix,
            neighbor_matrix_shifts=self.neighbor_manager.wp_neighbor_shifts,
            num_neighbors=self.neighbor_manager.wp_num_neighbors,
            fill_value=self.num_atoms,
            device=self.wp_device,
            energies_out=self.wp_energies,
            forces_out=self.wp_forces,
        )
        return wp_energies

    def compute_forces_virial(
        self, virial_tensors: wp.array | None = None
    ) -> wp.array | tuple[wp.array, wp.array, wp.array]:
        """Compute LJ forces and virial.

        Parameters
        ----------
        virial_tensors : wp.array, optional
            If provided (for NPT/NPH), packs virial into vec9 format and returns only energies.
            If None (for variable-cell), returns tuple (energies, forces, virial_flat).

        Returns
        -------
        If virial_tensors is provided:
            wp.array : Per-atom potential energies (shape (num_atoms,))
        If virial_tensors is None:
            tuple[wp.array, wp.array, wp.array] : (energies, forces, virial_flat)
                - energies: Per-atom potential energies (shape (num_atoms,))
                - forces: Forces on atoms (shape (num_atoms,), dtype=vec3d)
                - virial_flat: Flat virial tensor (shape (9,), row-major)

        Notes
        -----
        For barostat integrators (NPT/NPH), provide virial_tensors parameter.
        For variable-cell optimization, call without parameter to get flat virial.
        """
        # Mark stale to force full rebuild (cell may have changed)
        self.neighbor_manager.mark_stale()
        self._update_neighbors()

        wp_energies, wp_forces, wp_virial_flat = lj_energy_forces_virial(
            positions=self.wp_positions,
            cell=self.wp_cell,
            epsilon=self.epsilon,
            sigma=self.sigma,
            cutoff=self.cutoff,
            switch_width=self.switch_width,
            half_neighbor_list=self.half_neighbor_list,
            neighbor_matrix=self.neighbor_manager.wp_neighbor_matrix,
            neighbor_matrix_shifts=self.neighbor_manager.wp_neighbor_shifts,
            num_neighbors=self.neighbor_manager.wp_num_neighbors,
            fill_value=self.num_atoms,
            device=self.wp_device,
            energies_out=self.wp_energies,
            forces_out=self.wp_forces,
            virial_out=self.wp_virial_flat,
        )

        if virial_tensors is not None:
            # Pack float64[9] into vec9[f/d] expected by the MTK integrators
            wp.launch(
                _pack_virial_flat_to_vec9_kernel,
                dim=1,
                inputs=[wp_virial_flat, virial_tensors],
                device=self.wp_device,
            )
            return wp_energies
        else:
            # Return tuple for variable-cell optimization
            return wp_energies, wp_forces, wp_virial_flat

    def update_cell(self, cell: wp.array) -> None:
        """Update cell matrix and recompute cell inverse.

        Parameters
        ----------
        cell : wp.array, shape (1, 3, 3)
            New cell matrix.
        """
        wp.copy(self.wp_cell, cell)
        compute_cell_inverse(self.wp_cell, self.wp_cell_inv, device=self.device)
        # Force full rebuild on next force computation (cell geometry changed)
        self.neighbor_manager.mark_stale()

    def kinetic_energy(self) -> wp.array:
        """Compute kinetic energy on device (shape (1,), in eV)."""
        ke = wp.zeros(1, dtype=self.wp_dtype, device=self.wp_device)
        compute_kinetic_energy(
            velocities=self.wp_velocities,
            masses=self.wp_masses,
            kinetic_energy=ke,
            device=self.wp_device,
        )
        return ke

    def temperature_kT(self) -> wp.array:
        """Compute instantaneous temperature on device (kB*T in eV, shape (1,))."""
        ke = self.kinetic_energy()
        temp = wp.zeros(1, dtype=self.wp_dtype, device=self.wp_device)
        compute_temperature(
            kinetic_energy=ke,
            temperature=temp,
            num_atoms_per_system=wp.array(
                [self.num_atoms], dtype=wp.int32, device=self.wp_device
            ),
        )
        return temp

    def initialize_temperature(self, temperature: float, seed: int = 42) -> None:
        """Initialize velocities to target temperature (single system).

        Parameters
        ----------
        temperature : float
            Target temperature in Kelvin.
        seed : int
            Random seed for reproducibility.
        """
        kT = float(temperature) * KB_EV
        wp_temperature = wp.array([kT], dtype=self.wp_dtype, device=self.wp_device)

        # Scratch arrays for COM removal
        wp_total_momentum = wp.zeros(1, dtype=self.wp_vec_dtype, device=self.wp_device)
        wp_total_mass = wp.zeros(1, dtype=self.wp_dtype, device=self.wp_device)
        wp_com_velocities = wp.zeros(1, dtype=self.wp_vec_dtype, device=self.wp_device)

        initialize_velocities(
            velocities=self.wp_velocities,
            masses=self.wp_masses,
            temperature=wp_temperature,
            total_momentum=wp_total_momentum,
            total_mass=wp_total_mass,
            com_velocities=wp_com_velocities,
            random_seed=seed,
            remove_com=True,
            device=self.wp_device,
        )

        # One-time feedback: host read for user confidence
        actual_kT = float(self.temperature_kT().numpy()[0])
        actual_temp = actual_kT / KB_EV
        print(
            f"Initialized velocities: target={temperature:.1f} K, actual={actual_temp:.1f} K"
        )


class BatchedMDSystem:
    """Batched MD system for multiple independent systems packed into one set of arrays."""

    def __init__(
        self,
        positions: torch.Tensor,  # (N_total, 3)
        cells: torch.Tensor,  # (B, 3, 3)
        pbc: torch.Tensor,  # (B, 3)
        batch_idx: torch.Tensor,  # (N_total,)
        num_systems: int,
        masses: torch.Tensor | None = None,  # (N_total,)
        epsilon: float = EPSILON_AR,
        sigma: float = SIGMA_AR,
        cutoff: float = DEFAULT_CUTOFF,
        skin: float = DEFAULT_SKIN,
        switch_width: float = 0.0,
        half_neighbor_list: bool = True,
        device: str = "cuda:0",
        dtype: np.dtype = np.float64,
    ):
        self.num_systems = int(num_systems)
        self.total_atoms = int(len(positions))
        self.num_atoms = self.total_atoms // self.num_systems
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.switch_width = float(switch_width)
        self.half_neighbor_list = bool(half_neighbor_list)
        self.device = device
        self.wp_device = wp.device_from_torch(device)
        self.dtype = dtype
        self.wp_dtype = wp.float64 if dtype == torch.float64 else wp.float32
        self.wp_vec_dtype = wp.vec3d if dtype == torch.float64 else wp.vec3f
        self.wp_mat_dtype = wp.mat33d if dtype == torch.float64 else wp.mat33f
        self.torch_dtype = torch.float64 if dtype == torch.float64 else torch.float32

        if masses is None:
            masses = torch.full(
                (self.total_atoms,), MASS_AR, dtype=dtype, device=device
            )
        masses = mass_amu_to_internal(masses)

        batch_idx_np = batch_idx.cpu().numpy().astype(np.int32)
        self.num_atoms_per_system = torch.tensor(
            np.bincount(batch_idx_np, minlength=self.num_systems).astype(np.int32),
            dtype=torch.int32,
            device=str(device),
        )

        self.wp_positions = wp.from_torch(positions, dtype=self.wp_vec_dtype)
        self.wp_velocities = wp.zeros(
            self.total_atoms, dtype=self.wp_vec_dtype, device=self.wp_device
        )
        self.wp_forces = wp.zeros(
            self.total_atoms, dtype=self.wp_vec_dtype, device=self.wp_device
        )
        self.wp_masses = wp.from_torch(masses, dtype=self.wp_dtype)
        self.wp_energies = wp.zeros(
            self.total_atoms, dtype=self.wp_dtype, device=self.wp_device
        )
        self.wp_virial_flat = wp.zeros(
            (self.num_systems, 9), dtype=self.wp_dtype, device=self.wp_device
        )

        self.wp_cell = wp.from_torch(cells, dtype=self.wp_mat_dtype)
        self.wp_cell_inv = wp.empty_like(self.wp_cell)
        compute_cell_inverse(self.wp_cell, self.wp_cell_inv, device=self.wp_device)

        self.wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32)

        initial_cells_np = cells.cpu().numpy()
        pbc_np = pbc.cpu().numpy()
        self.neighbor_manager = BatchedNeighborListManager(
            total_atoms=self.total_atoms,
            cutoff=self.cutoff,
            skin=self.skin,
            batch_idx=batch_idx_np,
            num_systems=self.num_systems,
            initial_cells=initial_cells_np,
            pbc=pbc_np,
            max_neighbors=100,
            half_fill=self.half_neighbor_list,
            wp_dtype=self.wp_dtype,
            wp_vec_dtype=self.wp_vec_dtype,
            device=self.wp_device,
        )
        # Initial neighbor list build (ref_positions=zeros guarantees full rebuild)
        self.neighbor_manager.update(self.wp_positions, self.wp_cell, self.wp_cell_inv)

    def initialize_temperature(self, temperatures_K: np.ndarray, seed: int = 0) -> None:
        """Initialize velocities to target temperature (batched mode).

        Parameters
        ----------
        temperatures_K : np.ndarray
            Target temperatures in Kelvin. Shape (num_systems,).
        seed : int
            Random seed for reproducibility.
        """
        from nvalchemiops.dynamics.utils.thermostat_utils import (
            initialize_velocities as init_vel,
        )

        kT = temperatures_K * KB_EV
        wp_temperature = wp.from_torch(kT)
        wp_total_momentum = wp.zeros(
            self.num_systems, dtype=self.wp_vec_dtype, device=self.wp_device
        )
        wp_total_mass = wp.zeros(
            self.num_systems, dtype=self.wp_dtype, device=self.wp_device
        )
        wp_com_velocities = wp.zeros(
            self.num_systems, dtype=self.wp_vec_dtype, device=self.wp_device
        )
        init_vel(
            velocities=self.wp_velocities,
            masses=self.wp_masses,
            temperature=wp_temperature,
            total_momentum=wp_total_momentum,
            total_mass=wp_total_mass,
            com_velocities=wp_com_velocities,
            random_seed=seed,
            remove_com=True,
            batch_idx=self.wp_batch_idx,
            num_systems=self.num_systems,
            device=self.wp_device,
        )

        # One-time feedback: compute achieved per-system temperatures (host read)
        from nvalchemiops.dynamics.utils.thermostat_utils import (
            compute_kinetic_energy as ke_fn,
        )

        wp_ke = wp.zeros(self.num_systems, dtype=self.wp_dtype, device=self.wp_device)
        ke_fn(
            self.wp_velocities,
            self.wp_masses,
            kinetic_energy=wp_ke,
            batch_idx=self.wp_batch_idx,
            num_systems=self.num_systems,
            device=self.wp_device,
        )
        ke = wp_ke.numpy()
        dof = np.maximum(3 * self.num_atoms_per_system - 3, 1).astype(np.float64)
        actual_kT = 2.0 * ke / dof
        actual_T = actual_kT / KB_EV
        print(f"Initialized velocities: target={temperatures_K} K, actual={actual_T} K")

    def compute_forces(self) -> wp.array:
        """Compute LJ forces and return per-atom potential energies (device array).

        Notes
        -----
        This function intentionally does **not** synchronize or pull data back to
        the host. Host-side reductions (e.g., PE sum) should be done only at
        logging / analysis points.
        """
        self.neighbor_manager.update(self.wp_positions, self.wp_cell, self.wp_cell_inv)
        wp_energies, _ = lj_energy_forces(
            positions=self.wp_positions,
            cell=self.wp_cell,
            epsilon=self.epsilon,
            sigma=self.sigma,
            cutoff=self.cutoff,
            switch_width=self.switch_width,
            half_neighbor_list=self.half_neighbor_list,
            neighbor_matrix=self.neighbor_manager.wp_neighbor_matrix,
            neighbor_matrix_shifts=self.neighbor_manager.wp_neighbor_shifts,
            num_neighbors=self.neighbor_manager.wp_num_neighbors,
            fill_value=self.total_atoms,
            batch_idx=self.wp_batch_idx,
            device=self.wp_device,
            energies_out=self.wp_energies,
            forces_out=self.wp_forces,
        )
        return wp_energies

    def compute_forces_virial(
        self, virial_tensors: wp.array | None = None
    ) -> wp.array | tuple[wp.array, wp.array, wp.array]:
        """Compute LJ forces and virial.

        Parameters
        ----------
        virial_tensors : wp.array, optional
            If provided (for NPT/NPH), packs virial into vec9 format and returns only energies.
            If None (for variable-cell), returns tuple (energies, forces, virial_flat).

        Returns
        -------
        If virial_tensors is provided:
            wp.array : Per-atom potential energies (shape (num_atoms,))
        If virial_tensors is None:
            tuple[wp.array, wp.array, wp.array] : (energies, forces, virial_flat)
                - energies: Per-atom potential energies (shape (num_atoms,))
                - forces: Forces on atoms (shape (num_atoms,), dtype=vec3d)
                - virial_flat: Flat virial tensor (shape (9,), row-major)

        Notes
        -----
        For barostat integrators (NPT/NPH), provide virial_tensors parameter.
        For variable-cell optimization, call without parameter to get flat virial.
        """
        # Mark stale to force full rebuild (cell may have changed)
        self.neighbor_manager.mark_stale()
        self.neighbor_manager.update(self.wp_positions, self.wp_cell, self.wp_cell_inv)

        wp_energies, wp_forces, wp_virial_flat = lj_energy_forces_virial(
            positions=self.wp_positions,
            cell=self.wp_cell,
            epsilon=self.epsilon,
            sigma=self.sigma,
            cutoff=self.cutoff,
            switch_width=self.switch_width,
            half_neighbor_list=self.half_neighbor_list,
            neighbor_matrix=self.neighbor_manager.wp_neighbor_matrix,
            neighbor_matrix_shifts=self.neighbor_manager.wp_neighbor_shifts,
            num_neighbors=self.neighbor_manager.wp_num_neighbors,
            fill_value=self.total_atoms,
            device=self.wp_device,
            batch_idx=self.wp_batch_idx,
            energies_out=self.wp_energies,
            forces_out=self.wp_forces,
            virial_out=self.wp_virial_flat,
        )

        if virial_tensors is not None:
            # Copy LJ (num_systems, 9) output into caller's (num_systems,) vec9 buffer.
            num_sys = virial_tensors.shape[0]
            if self.wp_dtype == wp.float64:
                wp.launch(
                    _copy_virial_flat_to_vec9d,
                    dim=num_sys,
                    inputs=[wp_virial_flat, virial_tensors],
                    device=self.wp_device,
                )
            else:
                wp.launch(
                    _copy_virial_flat_to_vec9f,
                    dim=num_sys,
                    inputs=[wp_virial_flat, virial_tensors],
                    device=self.wp_device,
                )
            return wp_energies
        else:
            # Return tuple for variable-cell optimization
            return wp_energies, wp_forces, wp_virial_flat

    def kinetic_energy_per_system(self) -> wp.array:
        """Compute kinetic energy per system (shape (B,), in eV)."""
        ke = wp.zeros(self.num_systems, dtype=self.wp_dtype, device=self.wp_device)
        compute_kinetic_energy(
            velocities=self.wp_velocities,
            masses=self.wp_masses,
            kinetic_energy=ke,
            batch_idx=self.wp_batch_idx,
            num_systems=self.num_systems,
            device=self.wp_device,
        )
        return ke

    def temperature_kT_per_system(self) -> wp.array:
        """Compute temperature per system (kB*T, shape (B,)).

        Note
        ----
        :func:`nvalchemiops.dynamics.utils.compute_temperature` currently takes a
        single `num_atoms` value (per system). For heterogeneous batches this is
        ambiguous, so we require uniform system sizes here.
        """
        ke = self.kinetic_energy_per_system()
        temp = wp.zeros(self.num_systems, dtype=self.wp_dtype, device=self.wp_device)
        compute_temperature(
            kinetic_energy=ke,
            temperature=temp,
            num_atoms_per_system=self.num_atoms_per_system,
        )
        return temp

    def _scalar_pressure_and_volume(
        self,
        virial_tensors: wp.array,
    ) -> tuple[wp.array, wp.array]:
        """Compute scalar pressure (eV/Å^3) and volume (Å^3)."""
        tensor_dtype = vec9f if self.wp_dtype == wp.float32 else vec9d

        # Pre-allocate scratch arrays
        volumes = wp.empty(1, dtype=self.wp_dtype, device=self.wp_device)
        compute_cell_volume(self.wp_cell, volumes=volumes, device=self.wp_device)

        kinetic_tensors = wp.zeros((1, 9), dtype=self.wp_dtype, device=self.wp_device)
        pressure_tensors = wp.zeros(1, dtype=tensor_dtype, device=self.wp_device)
        compute_pressure_tensor(
            velocities=self.wp_velocities,
            masses=self.wp_masses,
            virial_tensors=virial_tensors,
            cells=self.wp_cell,
            kinetic_tensors=kinetic_tensors,
            pressure_tensors=pressure_tensors,
            volumes=volumes,
            device=self.wp_device,
        )

        scalar_pressures = wp.empty(1, dtype=self.wp_dtype, device=self.wp_device)
        compute_scalar_pressure(
            pressure_tensors, scalar_pressures, device=self.wp_device
        )
        return scalar_pressures, volumes


# ==============================================================================
# Unified Benchmark Class
# ==============================================================================


class NvalchemiOpsBenchmark:
    """Unified benchmark class for both single-system and batched simulations.

    This class consolidates MD and optimization benchmarks, automatically detecting
    whether to use single-system or batched mode based on the presence of batch_idx.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions. Shape (N, 3) for single-system or (total_atoms, 3) for batched.
    cell : torch.Tensor
        Unit cell matrix. Shape (1, 3, 3) for single-system or (num_systems, 3, 3) for batched.
    pbc : torch.Tensor
        Periodic boundary conditions, shape (3,).
    masses : torch.Tensor, optional
        Atomic masses. Shape (N,) or (total_atoms,). Default: argon mass for all atoms.
    epsilon : float, optional
        LJ epsilon parameter (eV).
    sigma : float, optional
        LJ sigma parameter (Å).
    cutoff : float, optional
        LJ cutoff distance (Å).
    skin : float, optional
        Neighbor list skin distance (Å). Default 1.0.
    switch_width : float, optional
        Switching function width (Å). Default 0.0.
    half_neighbor_list : bool, optional
        Whether to use half neighbor lists. Default True.
    neighbor_rebuild_interval : int, optional
        Interval for rebuilding neighbor lists (0 = displacement-based). Default 10.
    velocities : torch.Tensor, optional
        Initial velocities. Required for MD, optional for optimization.
    batch_idx : torch.Tensor, optional
        Batch index for each atom (batched mode only). Shape (total_atoms,).

    Notes
    -----
    - Batching is auto-detected: if batch_idx is provided, batched mode is used.
    - For batched mode, positions/velocities/masses should be concatenated across all systems.
    - Supports integrators: VelocityVerlet, Langevin, NoseHoover, NPT, NPH.
    - Supports optimizers: FIRE, FIRE2.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        masses: torch.Tensor | None = None,
        epsilon: float | None = None,
        sigma: float | None = None,
        cutoff: float | None = None,
        skin: float = 1.0,
        switch_width: float = 0.0,
        half_neighbor_list: bool = True,
        neighbor_rebuild_interval: int = 10,
        velocities: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
    ):
        self.is_batched = batch_idx is not None
        if self.is_batched:
            self.system = BatchedMDSystem(
                positions=positions,
                cells=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                num_systems=cell.shape[0],
                masses=masses,
                epsilon=epsilon,
                sigma=sigma,
                cutoff=cutoff,
                skin=skin,
                switch_width=switch_width,
                half_neighbor_list=half_neighbor_list,
                device=positions.device,
                dtype=positions.dtype,
            )
        else:
            self.system = MDSystem(
                positions=positions,
                cell=cell,
                masses=masses,
                pbc=pbc,
                epsilon=epsilon,
                sigma=sigma,
                cutoff=cutoff,
                skin=skin,
                switch_width=switch_width,
                half_neighbor_list=half_neighbor_list,
                device=positions.device,
                dtype=positions.dtype,
            )

        self.neighbor_rebuild_interval = neighbor_rebuild_interval
        self._steps_since_rebuild = 0

        # Device and dtype setup (needed for model creation)
        self.device = positions.device
        self.dtype = positions.dtype
        self.wp_device = str(self.device)

    def __getattr__(self, name: str):
        """Delegate attribute lookups to the underlying system object."""
        system = self.__dict__.get("system")
        if system is not None:
            try:
                return getattr(system, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute {name!r}"
        )

    def _compute_forces(self, wp_positions, compute_virial=False):
        """Update system positions and compute forces (and optionally virial).

        Parameters
        ----------
        wp_positions : wp.array
            Atomic positions.
        compute_virial : bool
            If True, compute and return virial tensor.

        Returns
        -------
        tuple
            (energies, forces, virial) if compute_virial, else (energies, forces).
        """
        self.system.wp_positions = wp_positions
        if compute_virial:
            return self.system.compute_forces_virial()
        energies = self.system.compute_forces()
        return energies, self.system.wp_forces

    def _run_warmup(
        self, run_step_fn, warmup_steps: int, log_message: str = "Warmup"
    ) -> None:
        """Run warmup steps.

        Parameters
        ----------
        run_step_fn : callable
            Function to call for each step (takes no arguments).
        warmup_steps : int
            Number of warmup steps.
        log_message : str, optional
            Message to print during warmup.
        """
        if warmup_steps > 0:
            for _ in range(warmup_steps):
                run_step_fn()
            wp.synchronize()

    def _run_timed_loop(
        self,
        run_step_fn,
        num_steps: int,
        log_interval: int = 100,
        log_fn: callable | None = None,
    ) -> tuple[float, list[float]]:
        """Run timed loop for benchmarking.

        Parameters
        ----------
        run_step_fn : callable
            Function to call for each step (takes step index as argument).
        num_steps : int
            Number of steps to run.
        log_interval : int, optional
            Logging interval.
        log_fn : callable, optional
            Logging function called at intervals. Takes step index as argument.

        Returns
        -------
        total_time : float
            Total time in seconds.
        step_times : list[float]
            Individual step times in seconds.
        """
        import time

        wp.synchronize()
        start_time = time.perf_counter()

        for step in range(num_steps):
            run_step_fn()

        wp.synchronize()
        total_time = time.perf_counter() - start_time
        step_times = [total_time / num_steps] * num_steps
        return total_time, step_times

    def _oom_result(
        self,
        name: str,
        ensemble: str,
        num_steps: int,
        dt: float,
        warmup_steps: int,
    ) -> BenchmarkResult:
        """Build a BenchmarkResult representing an OOM failure.

        Parameters
        ----------
        name : str
            Benchmark name (e.g. 'velocity_verlet').
        ensemble : str
            Ensemble (e.g. 'NVE', 'NVT').
        num_steps : int
            Requested number of steps.
        dt : float
            Timestep in fs.
        warmup_steps : int
            Warmup steps.

        Returns
        -------
        BenchmarkResult
            Result with total_time=0 and empty step_times so throughput is 0.
        """
        return BenchmarkResult(
            name=name,
            backend="nvalchemiops",
            ensemble=ensemble,
            num_atoms=self.system.num_atoms,
            num_steps=num_steps,
            dt=dt,
            warmup_steps=warmup_steps,
            total_time=0.0,
            step_times=[],
            batch_size=self.num_systems if self.is_batched else None,
        )

    # ========================================================================
    # MD Integrators
    # ========================================================================

    def run_velocity_verlet(
        self,
        dt: float,
        num_steps: int,
        warmup_steps: int = 100,
        log_interval: int = 100,
    ) -> BenchmarkResult:
        """Run NVE simulation using velocity Verlet integrator.

        Parameters
        ----------
        dt : float
            Timestep in fs.
        num_steps : int
            Number of simulation steps.
        warmup_steps : int, optional
            Number of warmup steps (excluded from timing).
        log_interval : int, optional
            Logging interval.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.
        """
        try:
            return self._run_velocity_verlet_impl(
                dt=dt,
                num_steps=num_steps,
                warmup_steps=warmup_steps,
                log_interval=log_interval,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._oom_result(
                "velocity_verlet", "NVE", num_steps, dt, warmup_steps
            )

    def _run_velocity_verlet_impl(
        self,
        dt: float,
        num_steps: int,
        warmup_steps: int = 100,
        log_interval: int = 100,
    ) -> BenchmarkResult:
        """Implementation of velocity Verlet (called by run_velocity_verlet)."""

        # Initial forces
        self.system.compute_forces()
        batch_idx = None if not self.is_batched else self.system.wp_batch_idx
        wp_dt = wp.array(
            [dt] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )

        # Warmup
        def step():
            velocity_verlet_position_update(
                self.system.wp_positions,
                self.system.wp_velocities,
                self.system.wp_forces,
                self.system.wp_masses,
                wp_dt,
                batch_idx=batch_idx,
                device=self.system.wp_device,
            )
            wrap_positions_to_cell(
                self.system.wp_positions,
                cells=self.system.wp_cell,
                cells_inv=self.system.wp_cell_inv,
                device=self.system.wp_device,
            )
            self.system.compute_forces()
            velocity_verlet_velocity_finalize(
                self.system.wp_velocities,
                self.system.wp_forces,
                self.system.wp_masses,
                wp_dt,
                batch_idx=batch_idx,
                device=self.system.wp_device,
            )

        self._run_warmup(step, warmup_steps, "Velocity Verlet warmup")

        total_time, step_times = self._run_timed_loop(step, num_steps, log_interval)

        return BenchmarkResult(
            name="velocity_verlet",
            backend="nvalchemiops",
            ensemble="NVE",
            num_atoms=self.num_atoms,
            num_steps=num_steps,
            dt=dt,
            warmup_steps=warmup_steps,
            total_time=total_time,
            step_times=step_times,
            batch_size=self.system.num_systems if self.is_batched else None,
        )

    def run_langevin(
        self,
        dt: float,
        num_steps: int,
        temperature: float,
        friction: float,
        warmup_steps: int = 100,
        log_interval: int = 100,
    ) -> BenchmarkResult:
        """Run NVT simulation using Langevin dynamics (BAOAB integrator).

        Parameters
        ----------
        dt : float
            Timestep in fs.
        num_steps : int
            Number of simulation steps.
        temperature : float
            Target temperature in K.
        friction : float
            Friction coefficient in 1/fs.
        warmup_steps : int, optional
            Number of warmup steps (excluded from timing).
        log_interval : int, optional
            Logging interval.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.
        """
        try:
            return self._run_langevin_impl(
                dt=dt,
                num_steps=num_steps,
                temperature=temperature,
                friction=friction,
                warmup_steps=warmup_steps,
                log_interval=log_interval,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._oom_result("langevin", "NVT", num_steps, dt, warmup_steps)

    def _run_langevin_impl(
        self,
        dt: float,
        num_steps: int,
        temperature: float,
        friction: float,
        warmup_steps: int = 100,
        log_interval: int = 100,
    ) -> BenchmarkResult:
        """Implementation of Langevin (called by run_langevin)."""

        # Convert temperature to kT (eV)
        kT = wp.array(
            [temperature * KB_EV],
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        gamma = wp.array(
            [friction], dtype=self.system.wp_dtype, device=self.system.wp_device
        )
        wp_dt = wp.array(
            [dt] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_temperature = kT
        wp_friction = gamma

        batch_idx = None if not self.is_batched else self.system.wp_batch_idx
        # Initial forces
        self.system.compute_forces()

        wp_ke_scratch = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )

        # Warmup
        def step():
            langevin_baoab_half_step(
                positions=self.system.wp_positions,
                velocities=self.system.wp_velocities,
                forces=self.system.wp_forces,
                masses=self.system.wp_masses,
                dt=wp_dt,
                temperature=wp_temperature,
                friction=wp_friction,
                random_seed=42,
                batch_idx=batch_idx,
                device=self.system.wp_device,
            )
            wrap_positions_to_cell(
                positions=self.system.wp_positions,
                cells=self.system.wp_cell,
                cells_inv=self.system.wp_cell_inv,
                device=self.system.wp_device,
            )
            self.system.compute_forces()
            compute_kinetic_energy(
                velocities=self.system.wp_velocities,
                masses=self.system.wp_masses,
                kinetic_energy=wp_ke_scratch,
                batch_idx=batch_idx,
                num_systems=self.system.num_systems,
                device=self.system.wp_device,
            )
            langevin_baoab_finalize(
                velocities=self.system.wp_velocities,
                forces_new=self.system.wp_forces,
                masses=self.system.wp_masses,
                dt=wp_dt,
                batch_idx=batch_idx,
                device=self.system.wp_device,
            )

        self._run_warmup(step, warmup_steps, "Langevin warmup")

        total_time, step_times = self._run_timed_loop(step, num_steps, log_interval)

        return BenchmarkResult(
            name="langevin",
            backend="nvalchemiops",
            ensemble="NVT",
            num_atoms=self.num_atoms,
            num_steps=num_steps,
            dt=dt,
            warmup_steps=warmup_steps,
            total_time=total_time,
            step_times=step_times,
            batch_size=self.system.num_systems if self.is_batched else None,
        )

    def run_npt(
        self,
        dt: float,
        num_steps: int,
        temperature: float,
        pressure: float = 0.0,
        tau_t: float = 100.0,
        tau_p: float = 1000.0,
        chain_length: int = 3,
        warmup_steps: int = 100,
        log_interval: int = 100,
    ) -> BenchmarkResult:
        """Run NPT simulation with Nosé-Hoover chains + MTK barostat.

        Parameters
        ----------
        dt : float
            Timestep in fs.
        num_steps : int
            Number of simulation steps.
        temperature : float
            Target temperature in K.
        pressure : float, optional
            Target pressure in bar.
        tau_t : float, optional
            Thermostat time constant in fs.
        tau_p : float, optional
            Barostat time constant in fs.
        chain_length : int, optional
            Number of thermostats in chain.
        warmup_steps : int, optional
            Number of warmup steps (excluded from timing).
        log_interval : int, optional
            Logging interval.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.

        Notes
        -----
        - Supports both single-system and batched modes.
        - Virial computation is required for pressure calculation.
        - Neighbor lists are rebuilt every step due to changing cell.
        """
        try:
            return self._run_npt_impl(
                dt=dt,
                num_steps=num_steps,
                target_temperature_K=temperature,
                target_pressure_atm=pressure,
                tdamp_fs=tau_t,
                pdamp_fs=tau_p,
                chain_length=chain_length,
                warmup_steps=warmup_steps,
                log_interval=log_interval,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._oom_result("npt", "NPT", num_steps, dt, warmup_steps)

    def _run_npt_impl(
        self,
        dt: float,
        num_steps: int,
        target_temperature_K: float = 94.4,
        target_pressure_atm: float = 1.0,
        tdamp_fs: float = 500.0,
        pdamp_fs: float = 5000.0,
        chain_length: int = 3,
        warmup_steps: int = 100,
        log_interval: int = 100,
    ) -> BenchmarkResult:
        """Implementation of NPT (called by run_npt)."""
        kT = float(target_temperature_K) * KB_EV
        p_ext = pressure_atm_to_ev_per_a3(target_pressure_atm)

        # Convert units (same convention as examples/dynamics: bar -> eV/Å³)
        # Per-system targets (required by NPT kernels; broadcast same value for all systems).
        wp_target_temperature = wp.array(
            [kT] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_tau = wp.array(
            [float(tdamp_fs)] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_target_pressure = wp.array(
            [p_ext] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        # Thermostat chain masses: always (num_systems, chain_length) so NPT kernel indexing is correct.
        thermostat_masses = wp.empty(
            (self.system.num_systems, int(chain_length)),
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        batch_idx = None if not self.is_batched else self.system.wp_batch_idx
        nhc_compute_masses(
            ndof=3 * self.system.num_atoms_per_system - 3,
            target_temp=wp_target_temperature,
            tau=wp_tau,
            chain_length=int(chain_length),
            masses=thermostat_masses,
            num_systems=self.system.num_systems,
            device=self.system.wp_device,
            dtype=self.system.wp_dtype,
        )
        eta = wp.zeros(
            (self.system.num_systems, chain_length),
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        eta_dot = wp.zeros(
            (self.system.num_systems, chain_length),
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        # Barostat masses/state
        wp_temp_baro = wp.array(
            [kT] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_tau_baro = wp.array(
            [float(pdamp_fs)] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        npt_num_atoms_per_system = wp.array(
            self.system.num_atoms_per_system.to(torch.int32),
            dtype=wp.int32,
            device=self.system.wp_device,
        )
        cell_masses = wp.empty(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        compute_barostat_mass(
            target_temperature=wp_temp_baro,
            tau_p=wp_tau_baro,
            num_atoms=npt_num_atoms_per_system,
            masses_out=cell_masses,
            device=self.system.wp_device,
        )
        cell_velocities = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_mat_dtype,
            device=self.system.wp_device,
        )

        tensor_dtype = vec9f if self.system.wp_dtype == wp.float32 else vec9d
        virial_tensors = wp.zeros(
            self.system.num_systems, dtype=tensor_dtype, device=self.system.wp_device
        )

        # Scratch arrays for NPT step
        npt_pressure_tensors = wp.zeros(
            self.system.num_systems, dtype=tensor_dtype, device=self.system.wp_device
        )
        npt_volumes = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        npt_kinetic_energy = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        npt_cells_inv = wp.empty(
            self.system.num_systems,
            dtype=self.system.wp_mat_dtype,
            device=self.system.wp_device,
        )
        npt_kinetic_tensors = wp.zeros(
            (self.system.num_systems, 9),
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )

        # Initial forces/virial
        wp_energies = self.system.compute_forces_virial(virial_tensors)
        # Pre-compute volumes and kinetic energy for the first step
        compute_cell_volume(
            self.system.wp_cell, volumes=npt_volumes, device=self.system.wp_device
        )
        compute_kinetic_energy(
            self.system.wp_velocities,
            self.system.wp_masses,
            kinetic_energy=npt_kinetic_energy,
            batch_idx=batch_idx,
            num_systems=self.system.num_systems,
            device=self.system.wp_device,
        )

        def _compute_forces_cb(positions, cells, forces, virial_out):
            nonlocal wp_energies
            self.system.wp_positions = positions
            self.system.wp_cell = cells
            compute_cell_inverse(
                self.system.wp_cell,
                self.system.wp_cell_inv,
                device=self.system.wp_device,
            )
            wp_energies = self.system.compute_forces_virial(virial_out)

        dt_array = wp.array(
            [float(dt)] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        num_atom_array = wp.array(
            [self.system.num_atoms] * self.system.num_systems,
            dtype=wp.int32,
            device=self.system.wp_device,
        )

        def step():
            run_npt_step(
                positions=self.system.wp_positions,
                velocities=self.system.wp_velocities,
                forces=self.system.wp_forces,
                masses=self.system.wp_masses,
                cells=self.system.wp_cell,
                cell_velocities=cell_velocities,
                virial_tensors=virial_tensors,
                eta=eta,
                eta_dot=eta_dot,
                thermostat_masses=thermostat_masses,
                cell_masses=cell_masses,
                target_temperature=wp_target_temperature,
                target_pressure=wp_target_pressure,
                num_atoms=num_atom_array,
                chain_length=int(chain_length),
                dt=dt_array,
                pressure_tensors=npt_pressure_tensors,
                volumes=npt_volumes,
                kinetic_energy=npt_kinetic_energy,
                cells_inv=npt_cells_inv,
                kinetic_tensors=npt_kinetic_tensors,
                num_atoms_per_system=npt_num_atoms_per_system,
                compute_forces_fn=_compute_forces_cb,
                batch_idx=batch_idx,
                device=self.system.wp_device,
            )

        self._run_warmup(step, warmup_steps, "NPT warmup")

        total_time, step_times = self._run_timed_loop(step, num_steps, log_interval)

        return BenchmarkResult(
            name="npt",
            backend="nvalchemiops",
            ensemble="NPT",
            num_atoms=self.num_atoms,
            num_steps=num_steps,
            dt=dt,
            warmup_steps=warmup_steps,
            total_time=total_time,
            step_times=step_times,
            batch_size=self.system.num_systems if self.is_batched else None,
        )

    def run_nph(
        self,
        dt: float,
        num_steps: int,
        temperature: float = 94.4,
        pressure: float = 0.0,
        tau_p: float = 1000.0,
        warmup_steps: int = 100,
        log_interval: int = 100,
    ) -> BenchmarkResult:
        """Run NPH simulation with MTK barostat (no thermostat).

        Parameters
        ----------
        dt : float
            Timestep in fs.
        num_steps : int
            Number of simulation steps.
        pressure : float, optional
            Target pressure in bar.
        tau_p : float, optional
            Barostat time constant in fs.
        warmup_steps : int, optional
            Number of warmup steps (excluded from timing).
        log_interval : int, optional
            Logging interval.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.

        Notes
        -----
        - Supports both single-system and batched modes.
        - Virial computation is required for pressure calculation.
        - Neighbor lists are rebuilt every step due to changing cell.
        """
        try:
            return self._run_nph_impl(
                dt=dt,
                num_steps=num_steps,
                target_pressure_atm=pressure,
                pdamp_fs=tau_p,
                reference_temperature_K=temperature,
                warmup_steps=warmup_steps,
                log_interval=log_interval,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._oom_result("nph", "NPH", num_steps, dt, warmup_steps)

    def _run_nph_impl(
        self,
        dt: float,
        num_steps: int,
        target_pressure_atm: float = 1.0,
        pdamp_fs: float = 1000.0,
        reference_temperature_K: float = 94.4,
        warmup_steps: int = 100,
        log_interval: int = 100,
    ) -> BenchmarkResult:
        """Implementation of NPH (called by run_nph)."""
        p_ext = pressure_atm_to_ev_per_a3(target_pressure_atm)
        # Per-system target pressure (kernels expect shape (num_systems,)).
        wp_target_pressure = wp.array(
            [p_ext] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )

        # Barostat mass per system (kT, tau_p, num_atoms_per_system) for correct dynamics in batch.
        kT_ref = float(reference_temperature_K) * KB_EV
        wp_temp_arr = wp.array(
            [kT_ref] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_tau_arr = wp.array(
            [float(pdamp_fs)] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        nph_num_atoms_per_system = wp.from_torch(
            self.system.num_atoms_per_system,
        )
        cell_masses = wp.empty(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        compute_barostat_mass(
            target_temperature=wp_temp_arr,
            tau_p=wp_tau_arr,
            num_atoms=nph_num_atoms_per_system,
            masses_out=cell_masses,
            device=self.system.wp_device,
        )

        cell_velocities = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_mat_dtype,
            device=self.system.wp_device,
        )
        tensor_dtype = vec9f if self.system.wp_dtype == wp.float32 else vec9d
        virial_tensors = wp.zeros(
            self.system.num_systems, dtype=tensor_dtype, device=self.system.wp_device
        )

        # Scratch arrays for NPH step
        nph_pressure_tensors = wp.zeros(
            self.system.num_systems, dtype=tensor_dtype, device=self.system.wp_device
        )
        nph_volumes = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        nph_kinetic_energy = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        nph_cells_inv = wp.empty(
            self.system.num_systems,
            dtype=self.system.wp_mat_dtype,
            device=self.system.wp_device,
        )
        nph_kinetic_tensors = wp.zeros(
            (self.system.num_systems, 9),
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        # nph_num_atoms_per_system already created above for compute_barostat_mass; reuse.

        batch_idx = None if not self.is_batched else self.system.wp_batch_idx

        # Initial forces/virial
        wp_energies = self.system.compute_forces_virial(virial_tensors)

        # Pre-compute volumes and kinetic energy for the first step
        compute_cell_volume(
            self.system.wp_cell, volumes=nph_volumes, device=self.system.wp_device
        )
        compute_kinetic_energy(
            self.system.wp_velocities,
            self.system.wp_masses,
            kinetic_energy=nph_kinetic_energy,
            batch_idx=batch_idx,
            num_systems=self.system.num_systems,
            device=self.system.wp_device,
        )

        # Force computation callback for NPH
        def _compute_forces_cb(positions, cells, forces, virial_out):
            nonlocal wp_energies
            # Ensure the system points at the integrator-updated arrays
            self.system.wp_positions = positions
            self.system.wp_cell = cells
            compute_cell_inverse(
                self.system.wp_cell,
                self.system.wp_cell_inv,
                device=self.system.wp_device,
            )
            wp_energies = self.system.compute_forces_virial(virial_out)

        dt_array = wp.array(
            [float(dt)] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        num_atom_array = wp.array(
            [self.system.num_atoms] * self.system.num_systems,
            dtype=wp.int32,
            device=self.system.wp_device,
        )

        # Warmup
        def step():
            run_nph_step(
                positions=self.system.wp_positions,
                velocities=self.system.wp_velocities,
                forces=self.system.wp_forces,
                masses=self.system.wp_masses,
                cells=self.system.wp_cell,
                cell_velocities=cell_velocities,
                virial_tensors=virial_tensors,
                cell_masses=cell_masses,
                target_pressure=wp_target_pressure,
                num_atoms=num_atom_array,
                dt=dt_array,
                pressure_tensors=nph_pressure_tensors,
                volumes=nph_volumes,
                kinetic_energy=nph_kinetic_energy,
                cells_inv=nph_cells_inv,
                kinetic_tensors=nph_kinetic_tensors,
                num_atoms_per_system=nph_num_atoms_per_system,
                compute_forces_fn=_compute_forces_cb,
                batch_idx=batch_idx,
                device=self.system.wp_device,
            )

        self._run_warmup(step, warmup_steps, "NPH warmup")

        total_time, step_times = self._run_timed_loop(step, num_steps, log_interval)

        return BenchmarkResult(
            name="nph",
            backend="nvalchemiops",
            ensemble="NPH",
            num_atoms=self.num_atoms,
            num_steps=num_steps,
            dt=dt,
            warmup_steps=warmup_steps,
            total_time=total_time,
            step_times=step_times,
            batch_size=self.system.num_systems if self.is_batched else None,
        )

    # ========================================================================
    # Optimization
    # ========================================================================

    def run_fire(
        self,
        max_steps: int = 1000,
        force_tolerance: float = 0.01,
        dt_start: float = 1.0,
        dt_max: float = 10.0,
        dt_min: float = 0.001,
        alpha_start: float = 0.1,
        n_min: int = 5,
        f_inc: float = 1.1,
        f_dec: float = 0.5,
        f_alpha: float = 0.99,
        maxstep: float = 0.2,
        warmup_steps: int = 0,
        log_interval: int = 100,
        check_interval: int = 20,
    ) -> BenchmarkResult:
        """Run FIRE (Fast Inertial Relaxation Engine) geometry optimization.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of optimization steps.
        force_tolerance : float, optional
            Convergence criterion for maximum force magnitude (eV/Å).
        dt_start : float, optional
            Initial timestep in fs.
        dt_max : float, optional
            Maximum timestep in fs.
        dt_min : float, optional
            Minimum timestep in fs.
        alpha_start : float, optional
            Initial mixing parameter.
        n_min : int, optional
            Minimum steps before increasing dt.
        f_inc : float, optional
            Factor to increase dt.
        f_dec : float, optional
            Factor to decrease dt.
        f_alpha : float, optional
            Factor to decrease alpha.
        maxstep : float, optional
            Maximum position change per step (Å).
        warmup_steps : int, optional
            Number of warmup steps (usually 0 for optimization).
        log_interval : int, optional
            Logging interval.
        check_interval : int, optional
            Interval to check convergence.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.

        Notes
        -----
        - FIRE does not require velocities to be initialized.
        - Convergence is checked based on maximum force magnitude.
        - Supports both single-system and batched modes.
        - Uses optimized nvalchemiops FIRE kernels.
        """
        try:
            return self._run_fire_impl(
                max_steps=max_steps,
                force_tolerance=force_tolerance,
                dt_start=dt_start,
                dt_max=dt_max,
                dt_min=dt_min,
                alpha_start=alpha_start,
                n_min=n_min,
                f_inc=f_inc,
                f_dec=f_dec,
                f_alpha=f_alpha,
                maxstep=maxstep,
                warmup_steps=warmup_steps,
                log_interval=log_interval,
                check_interval=check_interval,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._oom_result(
                "fire", "optimization", max_steps, dt_start, warmup_steps
            )

    def _run_fire_impl(
        self,
        max_steps: int = 1000,
        force_tolerance: float = 0.01,
        dt_start: float = 1.0,
        dt_max: float = 10.0,
        dt_min: float = 0.001,
        alpha_start: float = 0.1,
        n_min: int = 5,
        f_inc: float = 1.1,
        f_dec: float = 0.5,
        f_alpha: float = 0.99,
        maxstep: float = 0.2,
        warmup_steps: int = 0,
        log_interval: int = 100,
        check_interval: int = 20,
    ) -> BenchmarkResult:
        """Implementation of FIRE (called by run_fire)."""
        from nvalchemiops.dynamics.optimizers.fire import fire_step
        from nvalchemiops.dynamics.utils import (
            wrap_positions_to_cell,
        )

        # Initialize FIRE control parameters as warp arrays
        wp_alpha = wp.array(
            [alpha_start] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_dt = wp.array(
            [dt_start] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_alpha_start = wp.array(
            [alpha_start] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_f_alpha = wp.array(
            [f_alpha] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_dt_min = wp.array(
            [dt_min] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_dt_max = wp.array(
            [dt_max] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_maxstep = wp.array(
            [maxstep] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_n_steps_positive = wp.zeros(
            self.system.num_systems, dtype=wp.int32, device=self.system.wp_device
        )
        wp_n_min = wp.array(
            [n_min] * self.system.num_systems,
            dtype=wp.int32,
            device=self.system.wp_device,
        )
        wp_f_dec = wp.array(
            [f_dec] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_f_inc = wp.array(
            [f_inc] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )

        # Accumulators (required for single/batch_idx modes)
        wp_vf = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_vv = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_ff = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_uphill_flag = wp.zeros(
            self.system.num_systems, dtype=wp.int32, device=self.system.wp_device
        )

        # Initial forces
        self.system.compute_forces()

        # Warmup (usually not needed for optimization, but included for API consistency)
        if warmup_steps > 0:

            def warmup_step():
                pass  # No warmup needed for FIRE

            self._run_warmup(warmup_step, warmup_steps, "FIRE warmup")

        batch_idx = None if not self.is_batched else self.system.wp_batch_idx
        # Timed loop
        import time

        wp.synchronize()
        start_time = time.perf_counter()

        actual_steps = 0
        for step in range(max_steps):
            actual_steps += 1
            # Check convergence periodically
            if step % check_interval == 0:
                forces_torch = wp.to_torch(self.system.wp_forces)
                max_force = torch.abs(forces_torch).max().item()
                if max_force < force_tolerance:
                    break

            # Zero accumulators before FIRE step
            wp_vf.zero_()
            wp_vv.zero_()
            wp_ff.zero_()

            # FIRE step (includes MD integration + parameter update)
            fire_step(
                positions=self.system.wp_positions,
                velocities=self.system.wp_velocities,
                forces=self.system.wp_forces,
                masses=self.system.wp_masses,
                alpha=wp_alpha,
                dt=wp_dt,
                alpha_start=wp_alpha_start,
                f_alpha=wp_f_alpha,
                dt_min=wp_dt_min,
                dt_max=wp_dt_max,
                maxstep=wp_maxstep,
                n_steps_positive=wp_n_steps_positive,
                n_min=wp_n_min,
                f_dec=wp_f_dec,
                f_inc=wp_f_inc,
                uphill_flag=wp_uphill_flag,
                vf=wp_vf,
                vv=wp_vv,
                ff=wp_ff,
                batch_idx=batch_idx,
            )

            # Wrap positions back into cell
            wrap_positions_to_cell(
                positions=self.system.wp_positions,
                cells=self.system.wp_cell,
                cells_inv=self.system.wp_cell_inv,
                device=self.system.wp_device,
            )

            # Compute new forces
            self.system.compute_forces()

        wp.synchronize()
        total_time = time.perf_counter() - start_time
        step_times = [total_time / actual_steps] * actual_steps

        actual_steps = len(step_times)

        return BenchmarkResult(
            name="fire",
            backend="nvalchemiops",
            ensemble="optimization",
            num_atoms=self.system.num_atoms,
            num_steps=actual_steps,
            dt=dt_start,  # Report initial dt
            warmup_steps=warmup_steps,
            total_time=total_time,
            step_times=step_times,
            batch_size=None if not self.is_batched else self.system.num_systems,
        )

    def run_fire2(
        self,
        max_steps: int = 1000,
        force_tolerance: float = 0.01,
        dt_start: float = 0.045,
        tmax: float = 0.10,
        tmin: float = 0.005,
        delaystep: int = 50,
        dtgrow: float = 1.09,
        dtshrink: float = 0.95,
        alpha0: float = 0.20,
        alphashrink: float = 0.985,
        maxstep: float = 0.25,
        warmup_steps: int = 0,
        log_interval: int = 100,
        check_interval: int = 20,
    ) -> BenchmarkResult:
        """Run FIRE2 geometry optimization (coordinate-only).

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of optimization steps.
        force_tolerance : float, optional
            Convergence criterion for maximum force magnitude (eV/Å).
        dt_start : float, optional
            Initial timestep.
        tmax : float, optional
            Maximum timestep.
        tmin : float, optional
            Minimum timestep.
        delaystep : int, optional
            Minimum positive-power steps before dt growth.
        dtgrow : float, optional
            Timestep growth factor.
        dtshrink : float, optional
            Timestep shrink factor.
        alpha0 : float, optional
            Alpha reset value.
        alphashrink : float, optional
            Alpha decay factor.
        maxstep : float, optional
            Maximum position change per step (Å).
        warmup_steps : int, optional
            Number of warmup steps (usually 0 for optimization).
        log_interval : int, optional
            Logging interval.
        check_interval : int, optional
            Interval to check convergence.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.

        Notes
        -----
        - FIRE2 is a mass-free optimizer; no masses are required.
        - Uses batch_idx batching (not atom_ptr).
        - Convergence is checked based on maximum force magnitude.
        - Supports both single-system and batched modes.
        - Uses 3 fused kernel launches per step for minimal overhead.
        """
        try:
            return self._run_fire2_impl(
                max_steps=max_steps,
                force_tolerance=force_tolerance,
                dt_start=dt_start,
                tmax=tmax,
                tmin=tmin,
                delaystep=delaystep,
                dtgrow=dtgrow,
                dtshrink=dtshrink,
                alpha0=alpha0,
                alphashrink=alphashrink,
                maxstep=maxstep,
                warmup_steps=warmup_steps,
                log_interval=log_interval,
                check_interval=check_interval,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._oom_result(
                "fire2", "optimization", max_steps, dt_start, warmup_steps
            )

    def _run_fire2_impl(
        self,
        max_steps: int = 1000,
        force_tolerance: float = 0.01,
        dt_start: float = 0.045,
        tmax: float = 0.10,
        tmin: float = 0.005,
        delaystep: int = 50,
        dtgrow: float = 1.09,
        dtshrink: float = 0.95,
        alpha0: float = 0.20,
        alphashrink: float = 0.985,
        maxstep: float = 0.25,
        warmup_steps: int = 0,
        log_interval: int = 100,
        check_interval: int = 20,
    ) -> BenchmarkResult:
        """Implementation of FIRE2 (called by run_fire2)."""
        from nvalchemiops.dynamics.optimizers.fire2 import fire2_step
        from nvalchemiops.dynamics.utils import (
            wrap_positions_to_cell,
        )

        # FIRE2 per-system state
        wp_alpha = wp.array(
            [alpha0] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_dt = wp.array(
            [dt_start] * self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_nsteps_inc = wp.zeros(
            self.system.num_systems, dtype=wp.int32, device=self.system.wp_device
        )

        # Scratch buffers
        wp_vf = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_v_sumsq = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_f_sumsq = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )
        wp_max_norm = wp.zeros(
            self.system.num_systems,
            dtype=self.system.wp_dtype,
            device=self.system.wp_device,
        )

        # Initial forces
        self.system.compute_forces()

        # Warmup (usually not needed for optimization, but included for API consistency)
        if warmup_steps > 0:

            def warmup_step():
                pass  # No warmup needed for FIRE2

            self._run_warmup(warmup_step, warmup_steps, "FIRE2 warmup")

        batch_idx = (
            wp.zeros(
                self.system.num_atoms, dtype=wp.int32, device=self.system.wp_device
            )
            if not self.is_batched
            else self.system.wp_batch_idx
        )
        # Timed loop
        import time

        wp.synchronize()
        start_time = time.perf_counter()

        actual_steps = 0
        for step in range(max_steps):
            actual_steps += 1
            # Check convergence periodically
            if step % check_interval == 0:
                forces_torch = wp.to_torch(self.system.wp_forces)
                max_force = torch.abs(forces_torch).max().item()
                if max_force < force_tolerance:
                    break

            # Zero scratch buffers
            wp_vf.zero_()
            wp_v_sumsq.zero_()
            wp_f_sumsq.zero_()
            wp_max_norm.zero_()

            # FIRE2 step
            fire2_step(
                positions=self.system.wp_positions,
                velocities=self.system.wp_velocities,
                forces=self.system.wp_forces,
                batch_idx=batch_idx,
                alpha=wp_alpha,
                dt=wp_dt,
                nsteps_inc=wp_nsteps_inc,
                vf=wp_vf,
                v_sumsq=wp_v_sumsq,
                f_sumsq=wp_f_sumsq,
                max_norm=wp_max_norm,
                delaystep=delaystep,
                dtgrow=dtgrow,
                dtshrink=dtshrink,
                alphashrink=alphashrink,
                alpha0=alpha0,
                tmax=tmax,
                tmin=tmin,
                maxstep=maxstep,
            )

            # Wrap positions back into cell
            wrap_positions_to_cell(
                positions=self.system.wp_positions,
                cells=self.system.wp_cell,
                cells_inv=self.system.wp_cell_inv,
                device=self.system.wp_device,
            )

            # Compute new forces
            self.system.compute_forces()

        wp.synchronize()
        total_time = time.perf_counter() - start_time
        step_times = [total_time / actual_steps] * actual_steps

        actual_steps = len(step_times)

        return BenchmarkResult(
            name="fire2",
            backend="nvalchemiops",
            ensemble="optimization",
            num_atoms=self.system.num_atoms,
            num_steps=actual_steps,
            dt=dt_start,
            warmup_steps=warmup_steps,
            total_time=total_time,
            step_times=step_times,
            batch_size=None if not self.is_batched else self.system.num_systems,
        )

    def _virial_to_cell_force(self, wp_virial, cell_torch):
        """Convert LJ virial (flat) to cell force via stress_to_cell_force.

        Parameters
        ----------
        wp_virial : wp.array
            Flat virial from LJ, shape (9,) or (B*9,) or array2d (B,9).
        cell_torch : torch.Tensor
            Cell matrices, shape (M, 3, 3).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (cell_force, stress) -- cell_force shape (M, 3, 3),
            stress shape (M, 3, 3) in eV/A^3.
        """
        from nvalchemiops.dynamics.utils.cell_filter import stress_to_cell_force
        from nvalchemiops.dynamics.utils.cell_utils import compute_cell_volume

        M = self.system.num_systems
        virial_torch = wp.to_torch(wp_virial)
        virial_mat = virial_torch.reshape(M, 3, 3)
        volume = torch.det(cell_torch).to(self.dtype)
        stress_torch = virial_mat / volume.reshape(-1, 1, 1)
        stress_wp = wp.from_torch(
            stress_torch.contiguous(), dtype=self.system.wp_mat_dtype
        )
        wp_cell_cur = wp.from_torch(
            cell_torch.contiguous(), dtype=self.system.wp_mat_dtype
        )
        wp_volume = wp.empty(M, dtype=self.system.wp_dtype, device=self.wp_device)
        compute_cell_volume(wp_cell_cur, wp_volume, device=self.wp_device)
        cell_force_wp = wp.empty(
            M, dtype=self.system.wp_mat_dtype, device=self.wp_device
        )
        stress_to_cell_force(
            stress_wp,
            wp_cell_cur,
            wp_volume,
            cell_force_wp,
            keep_aligned=True,
            device=self.wp_device,
        )
        return wp.to_torch(cell_force_wp).reshape(M, 3, 3), stress_torch

    def run_fire_cell(
        self,
        max_steps: int = 1000,
        force_tolerance: float = 0.01,
        dt_start: float = 1.0,
        dt_max: float = 10.0,
        dt_min: float = 0.001,
        alpha_start: float = 0.1,
        n_min: int = 5,
        f_inc: float = 1.1,
        f_dec: float = 0.5,
        f_alpha: float = 0.99,
        maxstep: float = 0.2,
        cell_mass: float = 1.0,
        pressure_tolerance: float = 0.3,
        check_interval: int = 20,
    ) -> BenchmarkResult:
        """Run FIRE variable-cell geometry optimization using extended arrays.

        Uses align_cell preprocessing, then packs atomic + cell DOFs into
        extended arrays and runs fire_step on the extended system.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of optimization steps.
        force_tolerance : float, optional
            Convergence criterion for maximum force magnitude (eV/Å).
        dt_start, dt_max, dt_min, alpha_start, n_min, f_inc, f_dec, f_alpha, maxstep
            FIRE1 hyperparameters.
        cell_mass : float, optional
            Mass for cell DOFs (controls cell response speed).
        pressure_tolerance : float, optional
            Convergence criterion for maximum stress (kBar).
        check_interval : int, optional
            Interval to check convergence.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.
        """
        try:
            return self._run_fire_cell_impl(
                max_steps=max_steps,
                force_tolerance=force_tolerance,
                dt_start=dt_start,
                dt_max=dt_max,
                dt_min=dt_min,
                alpha_start=alpha_start,
                n_min=n_min,
                f_inc=f_inc,
                f_dec=f_dec,
                f_alpha=f_alpha,
                maxstep=maxstep,
                cell_mass=cell_mass,
                pressure_tolerance=pressure_tolerance,
                check_interval=check_interval,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._oom_result("fire_cell", "optimization", max_steps, dt_start, 0)

    def _run_fire_cell_impl(
        self,
        max_steps: int = 1000,
        force_tolerance: float = 0.01,
        dt_start: float = 1.0,
        dt_max: float = 10.0,
        dt_min: float = 0.001,
        alpha_start: float = 0.1,
        n_min: int = 5,
        f_inc: float = 1.1,
        f_dec: float = 0.5,
        f_alpha: float = 0.99,
        maxstep: float = 0.2,
        cell_mass: float = 1.0,
        pressure_tolerance: float = 0.3,
        check_interval: int = 20,
    ) -> BenchmarkResult:
        """Implementation of FIRE variable-cell (called by run_fire_cell)."""
        from nvalchemiops.dynamics.optimizers.fire import fire_step
        from nvalchemiops.dynamics.utils import (
            compute_cell_inverse,
            wrap_positions_to_cell,
        )
        from nvalchemiops.dynamics.utils.cell_filter import (
            align_cell,
            extend_batch_idx,
            pack_forces_with_cell,
            pack_positions_with_cell,
            unpack_positions_with_cell,
        )

        M = self.system.num_systems if self.is_batched else 1
        N = self.system.total_atoms

        # Clone state
        wp_positions = self.system.wp_positions
        wp_cell = self.system.wp_cell

        # batch_idx
        if (
            hasattr(self.system, "wp_batch_idx")
            and self.system.wp_batch_idx is not None
        ):
            wp_bidx = self.system.wp_batch_idx
        else:
            wp_bidx = wp.zeros(N, dtype=wp.int32, device=self.wp_device)

        # Align cell (one-time preprocessing; modifies wp_positions and wp_cell in-place)
        wp_transform = wp.empty(
            M, dtype=self.system.wp_mat_dtype, device=self.wp_device
        )
        align_cell(
            wp_positions,
            wp_cell,
            wp_transform,
            batch_idx=wp_bidx,
            device=self.wp_device,
        )
        # Force full neighbor rebuild since cell geometry changed
        self.system.neighbor_manager.mark_stale()
        self.system._update_neighbors()

        # Extended batch_idx: atoms + 2 extra DOFs per system
        N_ext = N + 2 * M
        ext_bidx = wp.empty(N_ext, dtype=wp.int32, device=self.wp_device)
        extend_batch_idx(
            wp_bidx,
            N,
            M,
            ext_bidx,
            device=self.wp_device,
        )

        # Pack initial positions into extended array
        ext_positions = wp.empty(
            N_ext, dtype=self.system.wp_vec_dtype, device=self.wp_device
        )
        pack_positions_with_cell(
            wp_positions,
            wp_cell,
            ext_positions,
            device=self.wp_device,
        )
        ext_velocities = wp.zeros(
            N_ext, dtype=self.system.wp_vec_dtype, device=self.wp_device
        )

        # Extended masses: atomic masses + cell_mass for cell DOFs
        masses_np = wp.to_torch(self.system.wp_masses).cpu().numpy()
        ext_masses_list = list(masses_np)
        for _ in range(M):
            ext_masses_list.extend([cell_mass, cell_mass])
        ext_masses = wp.array(
            ext_masses_list,
            dtype=self.system.wp_dtype,
            device=self.wp_device,
        )

        # FIRE1 per-system state (all as per-system arrays)
        wp_alpha = wp.array(
            [alpha_start] * M,
            dtype=self.system.wp_dtype,
            device=self.wp_device,
        )
        wp_dt = wp.array(
            [dt_start] * M,
            dtype=self.system.wp_dtype,
            device=self.wp_device,
        )
        wp_alpha_start = wp.array(
            [alpha_start] * M,
            dtype=self.system.wp_dtype,
            device=self.wp_device,
        )
        wp_f_alpha = wp.array(
            [f_alpha] * M,
            dtype=self.system.wp_dtype,
            device=self.wp_device,
        )
        wp_dt_min = wp.array(
            [dt_min] * M,
            dtype=self.system.wp_dtype,
            device=self.wp_device,
        )
        wp_dt_max = wp.array(
            [dt_max] * M,
            dtype=self.system.wp_dtype,
            device=self.wp_device,
        )
        wp_maxstep = wp.array(
            [maxstep] * M,
            dtype=self.system.wp_dtype,
            device=self.wp_device,
        )
        wp_n_steps_positive = wp.zeros(M, dtype=wp.int32, device=self.wp_device)
        wp_n_min = wp.array(
            [n_min] * M,
            dtype=wp.int32,
            device=self.wp_device,
        )
        wp_f_dec = wp.array(
            [f_dec] * M,
            dtype=self.system.wp_dtype,
            device=self.wp_device,
        )
        wp_f_inc = wp.array(
            [f_inc] * M,
            dtype=self.system.wp_dtype,
            device=self.wp_device,
        )

        # Accumulators
        wp_vf = wp.zeros(M, dtype=self.system.wp_dtype, device=self.wp_device)
        wp_vv = wp.zeros(M, dtype=self.system.wp_dtype, device=self.wp_device)
        wp_ff = wp.zeros(M, dtype=self.system.wp_dtype, device=self.wp_device)
        wp_uphill_flag = wp.zeros(M, dtype=wp.int32, device=self.wp_device)

        # Initial forces + virial (compute_forces_virial handles mark_stale + rebuild)
        wp_energies, wp_forces, wp_virial = self.system.compute_forces_virial()

        # Cell force from virial
        cell_torch = wp.to_torch(wp_cell).reshape(M, 3, 3)
        cell_force, stress = self._virial_to_cell_force(wp_virial, cell_torch)
        cell_force_wp = wp.from_torch(
            cell_force.contiguous(),
            dtype=self.system.wp_mat_dtype,
        )

        # Pack forces into extended array
        ext_forces = wp.empty(
            N_ext, dtype=self.system.wp_vec_dtype, device=self.wp_device
        )
        pack_forces_with_cell(
            wp_forces,
            cell_force_wp,
            ext_forces,
            device=self.wp_device,
        )

        # Pre-allocate scratch buffers for unpack/repack in the loop
        wp_cell_inv = wp.empty(M, dtype=self.system.wp_mat_dtype, device=self.wp_device)
        wp_positions_scratch = wp.empty(
            N, dtype=self.system.wp_vec_dtype, device=self.wp_device
        )
        wp_cell_scratch = wp.empty(
            M, dtype=self.system.wp_mat_dtype, device=self.wp_device
        )

        # Timed loop
        import time

        wp.synchronize()
        start_time = time.perf_counter()

        actual_steps = 0
        for step_i in range(max_steps):
            actual_steps += 1
            # Check convergence on atomic forces + pressure
            if step_i % check_interval == 0:
                forces_torch = wp.to_torch(wp_forces)
                max_force = torch.abs(forces_torch).max().item()
                stress_kbar = stress * 1602.18
                stress_kbar = 0.5 * (stress_kbar + stress_kbar.transpose(-1, -2))
                p_max = torch.linalg.svdvals(stress_kbar).max().item()
                if max_force < force_tolerance and p_max < pressure_tolerance:
                    break

            # Zero accumulators
            wp_vf.zero_()
            wp_vv.zero_()
            wp_ff.zero_()

            # FIRE step on extended arrays
            fire_step(
                positions=ext_positions,
                velocities=ext_velocities,
                forces=ext_forces,
                masses=ext_masses,
                alpha=wp_alpha,
                dt=wp_dt,
                alpha_start=wp_alpha_start,
                f_alpha=wp_f_alpha,
                dt_min=wp_dt_min,
                dt_max=wp_dt_max,
                maxstep=wp_maxstep,
                n_steps_positive=wp_n_steps_positive,
                n_min=wp_n_min,
                f_dec=wp_f_dec,
                f_inc=wp_f_inc,
                uphill_flag=wp_uphill_flag,
                vf=wp_vf,
                vv=wp_vv,
                ff=wp_ff,
                batch_idx=ext_bidx,
            )

            # Unpack extended positions → atom positions + cell
            unpack_positions_with_cell(
                ext_positions,
                wp_positions_scratch,
                wp_cell_scratch,
                num_atoms=self.system.num_atoms,
                device=self.wp_device,
            )
            # Sync unpacked positions/cell back into the system arrays
            wp.copy(self.system.wp_positions, wp_positions_scratch)
            wp.copy(self.system.wp_cell, wp_cell_scratch)
            wp_positions = self.system.wp_positions
            wp_cell = self.system.wp_cell

            # Wrap positions
            compute_cell_inverse(wp_cell, wp_cell_inv, device=self.wp_device)
            wrap_positions_to_cell(
                wp_positions,
                cells=wp_cell,
                cells_inv=wp_cell_inv,
                device=self.wp_device,
            )

            # Repack wrapped positions into extended array
            pack_positions_with_cell(
                wp_positions,
                wp_cell,
                ext_positions,
                device=self.wp_device,
            )

            # Compute new forces + virial
            # compute_forces_virial calls mark_stale + _update_neighbors internally
            wp_energies, wp_forces, wp_virial = self.system.compute_forces_virial()

            # Cell force from virial
            cell_torch = wp.to_torch(wp_cell).reshape(M, 3, 3)
            cell_force, stress = self._virial_to_cell_force(wp_virial, cell_torch)
            cell_force_wp = wp.from_torch(
                cell_force.contiguous(),
                dtype=self.system.wp_mat_dtype,
            )
            pack_forces_with_cell(
                wp_forces,
                cell_force_wp,
                ext_forces,
                device=self.wp_device,
            )

        wp.synchronize()
        total_time = time.perf_counter() - start_time
        step_times = [total_time / actual_steps] * actual_steps
        actual_steps = len(step_times)

        return BenchmarkResult(
            name="fire_cell",
            backend="nvalchemiops",
            ensemble="optimization",
            num_atoms=self.system.num_atoms,
            num_steps=actual_steps,
            dt=dt_start,
            warmup_steps=0,
            total_time=total_time,
            step_times=step_times,
            batch_size=self.system.num_systems if self.is_batched else None,
        )

    def run_fire2_cell(
        self,
        max_steps: int = 1000,
        force_tolerance: float = 0.01,
        dt_start: float = 0.045,
        tmax: float = 0.10,
        tmin: float = 0.005,
        delaystep: int = 50,
        dtgrow: float = 1.09,
        dtshrink: float = 0.95,
        alpha0: float = 0.20,
        alphashrink: float = 0.985,
        maxstep: float = 0.25,
        pressure_tolerance: float = 0.3,
        check_interval: int = 20,
    ) -> BenchmarkResult:
        """Run FIRE2 variable-cell geometry optimization using extended arrays.

        Uses align_cell preprocessing, then packs atomic + cell DOFs into
        extended arrays and runs fire2_step on the extended system.

        FIRE2 is mass-free: no extended masses are required, unlike FIRE1.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of optimization steps.
        force_tolerance : float, optional
            Convergence criterion for maximum force magnitude (eV/Å).
        dt_start : float, optional
            Initial timestep.
        tmax : float, optional
            Maximum timestep.
        tmin : float, optional
            Minimum timestep.
        delaystep : int, optional
            Minimum positive-power steps before dt growth.
        dtgrow : float, optional
            Timestep growth factor.
        dtshrink : float, optional
            Timestep shrink factor.
        alpha0 : float, optional
            Alpha reset value.
        alphashrink : float, optional
            Alpha decay factor.
        maxstep : float, optional
            Maximum position change per step (Å).
        pressure_tolerance : float, optional
            Convergence criterion for maximum stress (kBar).
        check_interval : int, optional
            Interval to check convergence.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.
        """
        try:
            return self._run_fire2_cell_impl(
                max_steps=max_steps,
                force_tolerance=force_tolerance,
                dt_start=dt_start,
                tmax=tmax,
                tmin=tmin,
                delaystep=delaystep,
                dtgrow=dtgrow,
                dtshrink=dtshrink,
                alpha0=alpha0,
                alphashrink=alphashrink,
                maxstep=maxstep,
                pressure_tolerance=pressure_tolerance,
                check_interval=check_interval,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._oom_result(
                "fire2_cell", "optimization", max_steps, dt_start, 0
            )

    def _run_fire2_cell_impl(
        self,
        max_steps: int = 1000,
        force_tolerance: float = 0.01,
        dt_start: float = 0.045,
        tmax: float = 0.10,
        tmin: float = 0.005,
        delaystep: int = 50,
        dtgrow: float = 1.09,
        dtshrink: float = 0.95,
        alpha0: float = 0.20,
        alphashrink: float = 0.985,
        maxstep: float = 0.25,
        pressure_tolerance: float = 0.3,
        check_interval: int = 20,
    ) -> BenchmarkResult:
        """Implementation of FIRE2 variable-cell (called by run_fire2_cell)."""
        from nvalchemiops.dynamics.optimizers.fire2 import fire2_step
        from nvalchemiops.dynamics.utils import (
            compute_cell_inverse,
            wrap_positions_to_cell,
        )
        from nvalchemiops.dynamics.utils.cell_filter import (
            align_cell,
            extend_batch_idx,
            pack_forces_with_cell,
            pack_positions_with_cell,
            unpack_positions_with_cell,
        )

        M = self.system.num_systems if self.is_batched else 1
        N = self.system.total_atoms

        # Clone state
        wp_positions = self.system.wp_positions
        wp_cell = self.system.wp_cell

        # batch_idx
        if (
            hasattr(self.system, "wp_batch_idx")
            and self.system.wp_batch_idx is not None
        ):
            wp_bidx = self.system.wp_batch_idx
        else:
            wp_bidx = wp.zeros(N, dtype=wp.int32, device=self.wp_device)

        # Align cell (one-time preprocessing; modifies wp_positions and wp_cell in-place)
        wp_transform = wp.empty(
            M, dtype=self.system.wp_mat_dtype, device=self.wp_device
        )
        align_cell(
            wp_positions,
            wp_cell,
            wp_transform,
            batch_idx=wp_bidx,
            device=self.wp_device,
        )
        # Force full neighbor rebuild since cell geometry changed
        self.system.neighbor_manager.mark_stale()
        self.system._update_neighbors()

        # Extended batch_idx: atoms + 2 extra DOFs per system
        N_ext = N + 2 * M
        ext_bidx = wp.empty(N_ext, dtype=wp.int32, device=self.wp_device)
        extend_batch_idx(
            wp_bidx,
            N,
            M,
            ext_bidx,
            device=self.wp_device,
        )

        # Pack initial positions into extended array
        ext_positions = wp.empty(
            N_ext, dtype=self.system.wp_vec_dtype, device=self.wp_device
        )
        pack_positions_with_cell(
            wp_positions,
            wp_cell,
            ext_positions,
            device=self.wp_device,
        )
        ext_velocities = wp.zeros(
            N_ext, dtype=self.system.wp_vec_dtype, device=self.wp_device
        )

        # FIRE2 per-system state (mass-free)
        wp_alpha = wp.array(
            [alpha0] * M,
            dtype=self.system.wp_dtype,
            device=self.wp_device,
        )
        wp_dt = wp.array(
            [dt_start] * M,
            dtype=self.system.wp_dtype,
            device=self.wp_device,
        )
        wp_nsteps_inc = wp.zeros(M, dtype=wp.int32, device=self.wp_device)

        # Scratch buffers
        wp_vf = wp.zeros(M, dtype=self.system.wp_dtype, device=self.wp_device)
        wp_v_sumsq = wp.zeros(M, dtype=self.system.wp_dtype, device=self.wp_device)
        wp_f_sumsq = wp.zeros(M, dtype=self.system.wp_dtype, device=self.wp_device)
        wp_max_norm = wp.zeros(M, dtype=self.system.wp_dtype, device=self.wp_device)

        # Initial forces + virial
        wp_energies, wp_forces, wp_virial = self.system.compute_forces_virial()

        # Cell force from virial
        cell_torch = wp.to_torch(wp_cell).reshape(M, 3, 3)
        cell_force, stress = self._virial_to_cell_force(wp_virial, cell_torch)
        cell_force_wp = wp.from_torch(
            cell_force.contiguous(),
            dtype=self.system.wp_mat_dtype,
        )

        # Pack forces into extended array
        ext_forces = wp.empty(
            N_ext, dtype=self.system.wp_vec_dtype, device=self.wp_device
        )
        pack_forces_with_cell(
            wp_forces,
            cell_force_wp,
            ext_forces,
            device=self.wp_device,
        )

        # Pre-allocate scratch buffers for unpack/repack in the loop
        wp_cell_inv = wp.empty(M, dtype=self.system.wp_mat_dtype, device=self.wp_device)
        wp_positions_scratch = wp.empty(
            N, dtype=self.system.wp_vec_dtype, device=self.wp_device
        )
        wp_cell_scratch = wp.empty(
            M, dtype=self.system.wp_mat_dtype, device=self.wp_device
        )

        # Timed loop
        import time

        wp.synchronize()
        start_time = time.perf_counter()

        actual_steps = 0
        for step_i in range(max_steps):
            actual_steps += 1
            # Check convergence on atomic forces + pressure
            if step_i % check_interval == 0:
                forces_torch = wp.to_torch(wp_forces)
                max_force = torch.abs(forces_torch).max().item()
                stress_kbar = stress * 1602.18
                stress_kbar = 0.5 * (stress_kbar + stress_kbar.transpose(-1, -2))
                p_max = torch.linalg.svdvals(stress_kbar).max().item()
                if max_force < force_tolerance and p_max < pressure_tolerance:
                    break

            # Zero scratch buffers
            wp_vf.zero_()
            wp_v_sumsq.zero_()
            wp_f_sumsq.zero_()
            wp_max_norm.zero_()

            # FIRE2 step on extended arrays (mass-free)
            fire2_step(
                ext_positions,
                ext_velocities,
                ext_forces,
                ext_bidx,
                wp_alpha,
                wp_dt,
                wp_nsteps_inc,
                wp_vf,
                wp_v_sumsq,
                wp_f_sumsq,
                wp_max_norm,
                delaystep=delaystep,
                dtgrow=dtgrow,
                dtshrink=dtshrink,
                alphashrink=alphashrink,
                alpha0=alpha0,
                tmax=tmax,
                tmin=tmin,
                maxstep=maxstep,
            )

            # Unpack extended positions -> atom positions + cell
            unpack_positions_with_cell(
                ext_positions,
                wp_positions_scratch,
                wp_cell_scratch,
                num_atoms=N,
                device=self.wp_device,
            )
            # Sync unpacked positions/cell back into the system arrays
            wp.copy(self.system.wp_positions, wp_positions_scratch)
            wp.copy(self.system.wp_cell, wp_cell_scratch)
            wp_positions = self.system.wp_positions
            wp_cell = self.system.wp_cell

            # Wrap positions
            compute_cell_inverse(wp_cell, wp_cell_inv, device=self.wp_device)
            wrap_positions_to_cell(
                wp_positions,
                cells=wp_cell,
                cells_inv=wp_cell_inv,
                device=self.wp_device,
            )

            # Repack wrapped positions into extended array
            pack_positions_with_cell(
                wp_positions,
                wp_cell,
                ext_positions,
                device=self.wp_device,
            )

            # Compute new forces + virial
            wp_energies, wp_forces, wp_virial = self.system.compute_forces_virial()

            # Cell force from virial
            cell_torch = wp.to_torch(wp_cell).reshape(M, 3, 3)
            cell_force, stress = self._virial_to_cell_force(wp_virial, cell_torch)
            cell_force_wp = wp.from_torch(
                cell_force.contiguous(),
                dtype=self.system.wp_mat_dtype,
            )
            pack_forces_with_cell(
                wp_forces,
                cell_force_wp,
                ext_forces,
                device=self.wp_device,
            )

        wp.synchronize()
        total_time = time.perf_counter() - start_time
        step_times = [total_time / actual_steps] * actual_steps
        actual_steps = len(step_times)

        return BenchmarkResult(
            name="fire2_cell",
            backend="nvalchemiops",
            ensemble="optimization",
            num_atoms=self.system.num_atoms,
            num_steps=actual_steps,
            dt=dt_start,
            warmup_steps=0,
            total_time=total_time,
            step_times=step_times,
            batch_size=self.system.num_systems if self.is_batched else None,
        )
