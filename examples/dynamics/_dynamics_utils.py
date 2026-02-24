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
Utility classes for Langevin dynamics examples.

This module provides helper classes for:
- Neighbor list management with cell list algorithm
- System creation (FCC lattice)
- Statistical analysis utilities
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple

import numpy as np
import torch
import warp as wp

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
from nvalchemiops.torch.neighbors import batch_cell_list, cell_list
from nvalchemiops.torch.neighbors.rebuild_detection import neighbor_list_needs_rebuild

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


# ==============================================================================
# Data Structures
# ==============================================================================


class SimulationStats(NamedTuple):
    """Statistics for a simulation step."""

    step: int
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    temperature: float
    num_neighbors: int
    min_neighbor_distance: float
    max_force: float
    time_per_step_ms: float


class BarostatStats(NamedTuple):
    """Statistics for NPT/NPH simulations (includes pressure/volume)."""

    step: int
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    temperature: float
    pressure: float
    volume: float
    num_neighbors: int
    min_neighbor_distance: float
    max_force: float
    time_per_step_ms: float


@wp.kernel
def _pack_virial_flat_to_vec9_kernel(
    virial_flat: wp.array(dtype=wp.float64),
    virial_vec9: wp.array(dtype=Any),
):
    """Pack a 9-element float64 virial into a vec9f/vec9d output array (single system)."""
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


def neighbor_distance_stats(
    positions: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    fill_value: int,
) -> float:
    """Compute minimum neighbor distance using the current neighbor matrix.

    Notes
    -----
    - Assumes neighbor shifts are integer cell shifts for the *neighbor* atom.
    - Computes distances only over valid neighbor entries (j < fill_value).
    - Uses the same shift convention as the LJ kernel: shift_vec = shift @ cell.
    """

    device = positions.device
    n, max_neighbors = neighbor_matrix.shape

    k = torch.arange(max_neighbors, device=device).view(1, -1).expand(n, -1)
    valid = k < num_neighbors.view(-1, 1)
    j = neighbor_matrix
    valid = valid & (j < fill_value)

    if not torch.any(valid):
        return float("inf")

    ii, kk = torch.where(valid)
    jj = j[ii, kk]

    ri = positions[ii]
    rj = positions[jj]
    shifts = neighbor_matrix_shifts[ii, kk].to(dtype=positions.dtype)

    # shift_vec = shift @ cell  (matches: cell_t * shift in Warp kernel)
    shift_vec = shifts @ cell
    rij = ri - rj - shift_vec
    r = torch.linalg.norm(rij, dim=1)
    return float(r.min().item())


# ==============================================================================
# Neighbor List Management
# ==============================================================================


class NeighborListManager:
    """Manages neighbor list construction and updates.

    Uses the cell list algorithm for O(N) neighbor finding with periodic
    boundary conditions.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the system.
    cutoff : float
        Cutoff distance for neighbor detection (Angstrom).
    skin : float
        Neighbor list skin distance (Angstrom). Rebuild when any atom
        moves more than skin/2.
    max_neighbors : int, optional
        Maximum number of neighbors per atom.
    device : str, optional
        Device for warp arrays (e.g., "cuda:0", "cpu").
    """

    def __init__(
        self,
        num_atoms: int,
        cutoff: float,
        skin: float,
        max_neighbors: int = 100,
        half_fill: bool = True,
        device: str = "cuda:0",
    ):
        self.num_atoms = num_atoms
        self.cutoff = cutoff
        self.skin = skin
        self.max_neighbors = max_neighbors
        self.half_fill = half_fill
        self.device = device

        # Track positions at last rebuild
        self.positions_at_rebuild: torch.Tensor | None = None

        # Torch tensors (from cell_list)
        self.torch_neighbor_matrix: torch.Tensor | None = None
        self.torch_neighbor_shifts: torch.Tensor | None = None
        self.torch_num_neighbors: torch.Tensor | None = None

        # Warp arrays (converted from torch)
        self.wp_neighbor_matrix: wp.array | None = None
        self.wp_neighbor_shifts: wp.array | None = None
        self.wp_num_neighbors: wp.array | None = None

    def needs_rebuild(self, positions: torch.Tensor) -> bool:
        """Check if neighbor list needs rebuilding.

        Parameters
        ----------
        positions : torch.Tensor, shape (N, 3)
            Current atomic positions.

        Returns
        -------
        bool
            True if rebuild needed (any atom moved > skin/2).
        """
        if self.positions_at_rebuild is None:
            return True

        # Use the package's GPU rebuild detection kernel (early-termination).
        # Note: the returned tensor is on-device; calling .item() syncs, but this
        # is still cheaper than a full max-reduction in Python.
        rebuild = neighbor_list_needs_rebuild(
            reference_positions=self.positions_at_rebuild,
            current_positions=positions,
            skin_distance_threshold=float(self.skin / 2.0),
        )
        return bool(rebuild.item())

    def build(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
    ) -> None:
        """Build neighbor list from positions.

        Parameters
        ----------
        positions : torch.Tensor, shape (N, 3)
            Atomic positions.
        cell : torch.Tensor, shape (3, 3)
            Unit cell matrix.
        pbc : torch.Tensor, shape (3,), dtype=bool
            Periodic boundary conditions in each dimension.
        """
        # Use cell_list for efficient neighbor finding
        neighbor_matrix, num_neighbors, neighbor_shifts = cell_list(
            positions=positions,
            cutoff=self.cutoff + self.skin,
            cell=cell,
            pbc=pbc,
            max_neighbors=self.max_neighbors,
            half_fill=self.half_fill,
            fill_value=self.num_atoms,
        )

        self.torch_neighbor_matrix = neighbor_matrix
        self.torch_neighbor_shifts = neighbor_shifts
        self.torch_num_neighbors = num_neighbors
        self.positions_at_rebuild = positions.clone()

        # Convert to warp arrays
        self._convert_to_warp()

    def _convert_to_warp(self) -> None:
        """Convert torch tensors to warp arrays."""
        # Neighbor matrix (int32)
        self.wp_neighbor_matrix = wp.from_torch(
            self.torch_neighbor_matrix, dtype=wp.int32
        )

        # Shifts (vec3i)
        self.wp_neighbor_shifts = wp.from_torch(
            self.torch_neighbor_shifts, dtype=wp.vec3i
        )

        # Num neighbors (int32)
        self.wp_num_neighbors = wp.from_torch(self.torch_num_neighbors, dtype=wp.int32)

    def total_neighbors(self) -> int:
        """Get total number of neighbors across all atoms."""
        if self.torch_num_neighbors is None:
            return 0
        return int(self.torch_num_neighbors.sum().item())


class BatchedNeighborListManager:
    """Neighbor list manager for batched systems (multiple independent cells)."""

    def __init__(
        self,
        total_atoms: int,
        cutoff: float,
        skin: float,
        batch_idx: torch.Tensor,
        num_systems: int,
        max_neighbors: int = 100,
        half_fill: bool = True,
        device: str = "cuda:0",
    ):
        self.total_atoms = int(total_atoms)
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.batch_idx = batch_idx.to(dtype=torch.int32)
        self.num_systems = int(num_systems)
        self.max_neighbors = int(max_neighbors)
        self.half_fill = bool(half_fill)
        self.device = device

        self.positions_at_rebuild: torch.Tensor | None = None

        self.torch_neighbor_matrix: torch.Tensor | None = None
        self.torch_neighbor_shifts: torch.Tensor | None = None
        self.torch_num_neighbors: torch.Tensor | None = None

        self.wp_neighbor_matrix: wp.array | None = None
        self.wp_neighbor_shifts: wp.array | None = None
        self.wp_num_neighbors: wp.array | None = None

    def build(
        self, positions: torch.Tensor, cells: torch.Tensor, pbc: torch.Tensor
    ) -> None:
        neighbor_matrix, num_neighbors, neighbor_shifts = batch_cell_list(
            positions=positions,
            cutoff=self.cutoff + self.skin,
            cell=cells,
            pbc=pbc,
            batch_idx=self.batch_idx,
            max_neighbors=self.max_neighbors,
            half_fill=self.half_fill,
            fill_value=self.total_atoms,
        )
        self.torch_neighbor_matrix = neighbor_matrix
        self.torch_neighbor_shifts = neighbor_shifts
        self.torch_num_neighbors = num_neighbors
        self.positions_at_rebuild = positions.clone()
        self._convert_to_warp()

    def _convert_to_warp(self) -> None:
        self.wp_neighbor_matrix = wp.from_torch(
            self.torch_neighbor_matrix, dtype=wp.int32
        )
        self.wp_neighbor_shifts = wp.from_torch(
            self.torch_neighbor_shifts, dtype=wp.vec3i
        )
        self.wp_num_neighbors = wp.from_torch(self.torch_num_neighbors, dtype=wp.int32)

    def total_neighbors(self) -> int:
        if self.torch_num_neighbors is None:
            return 0
        return int(self.torch_num_neighbors.sum().item())


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
    - neighbor list manager (Torch → Warp conversion)
    - LJ force evaluation (via :func:`nvalchemiops.interactions.lj_energy_forces`)

    Integrators (Langevin, Velocity Verlet, NPT, NPH, ...) should be implemented
    as separate "runner" functions operating on this system.
    """

    def __init__(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        masses: np.ndarray | None = None,
        epsilon: float = EPSILON_AR,
        sigma: float = SIGMA_AR,
        cutoff: float = DEFAULT_CUTOFF,
        skin: float = DEFAULT_SKIN,
        switch_width: float = 0.0,
        half_neighbor_list: bool = True,
        device: str = "cuda:0",
        dtype: np.dtype = np.float64,
    ):
        self.num_atoms = len(positions)
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.switch_width = float(switch_width)
        self.half_neighbor_list = bool(half_neighbor_list)
        self.device = device

        # Determine types
        self.dtype = dtype
        self.wp_dtype = wp.float64 if dtype == np.float64 else wp.float32
        self.wp_vec_dtype = wp.vec3d if dtype == np.float64 else wp.vec3f
        self.wp_mat_dtype = wp.mat33d if dtype == np.float64 else wp.mat33f
        self.torch_dtype = torch.float64 if dtype == np.float64 else torch.float32

        # Set up masses
        if masses is None:
            masses = np.full(self.num_atoms, MASS_AR, dtype=dtype)
        else:
            masses = masses.astype(dtype)

        # Convert masses to internal MD units (so KE is in eV when v is Å/fs)
        masses = mass_amu_to_internal(masses)

        # Create torch tensors (for neighbor list - uses PyTorch interface)
        self.torch_positions = torch.tensor(
            positions, dtype=self.torch_dtype, device=device
        )
        self.torch_cell = torch.tensor(cell, dtype=self.torch_dtype, device=device)
        self.torch_pbc = torch.tensor([True, True, True], device=device)

        # Create warp arrays for dynamics
        self.wp_positions = wp.array(
            positions.astype(dtype), dtype=self.wp_vec_dtype, device=device
        )
        self.wp_velocities = wp.zeros(
            self.num_atoms, dtype=self.wp_vec_dtype, device=device
        )
        self.wp_forces = wp.zeros(
            self.num_atoms, dtype=self.wp_vec_dtype, device=device
        )
        self.wp_masses = wp.array(masses, dtype=self.wp_dtype, device=device)

        # Cell matrix (shape (1,) for single system)
        cell_reshaped = cell.reshape(1, 3, 3).astype(dtype)
        self.wp_cell = wp.array(cell_reshaped, dtype=self.wp_mat_dtype, device=device)

        # Compute cell inverse for position wrapping
        self.wp_cell_inv = wp.empty_like(self.wp_cell)
        compute_cell_inverse(self.wp_cell, self.wp_cell_inv, device=device)

        # Set up neighbor list manager
        self.neighbor_manager = NeighborListManager(
            num_atoms=self.num_atoms,
            cutoff=cutoff,
            skin=skin,
            device=device,
            half_fill=self.half_neighbor_list,
        )

        # Build initial neighbor list
        self._rebuild_neighbors()

        print(f"Initialized MD system with {self.num_atoms} atoms")
        print(f"  Cell: {cell[0, 0]:.2f} x {cell[1, 1]:.2f} x {cell[2, 2]:.2f} Å")
        print(f"  Cutoff: {cutoff:.2f} Å (+ {skin:.2f} Å skin)")
        print(f"  LJ: ε = {epsilon:.4f} eV, σ = {sigma:.2f} Å")
        print(f"  Device: {device}, dtype: {dtype}")
        print("  Units: x [Å], t [fs], E [eV], m [eV·fs²/Å²] (from amu), v [Å/fs]")

    def _rebuild_neighbors(self) -> None:
        """Rebuild neighbor list from current positions."""
        # Sync warp positions to torch
        self.torch_positions = wp.to_torch(self.wp_positions)

        # Sync warp cell to torch (cell may change for NPT/NPH)
        self.torch_cell = wp.to_torch(self.wp_cell).squeeze(0)

        self.neighbor_manager.build(
            positions=self.torch_positions,
            cell=self.torch_cell,
            pbc=self.torch_pbc,
        )

    def _check_rebuild(self) -> bool:
        """Check if neighbor list needs rebuilding and rebuild if so."""
        self.torch_positions = wp.to_torch(self.wp_positions)

        if self.neighbor_manager.needs_rebuild(self.torch_positions):
            self._rebuild_neighbors()
            return True
        return False

    def compute_forces(self) -> wp.array:
        """Compute LJ forces and return per-atom potential energies (device array).

        Notes
        -----
        This function intentionally does **not** synchronize or pull data back to
        the host. Host-side reductions (e.g., PE sum) should be done only at
        logging / analysis points.
        """
        # Check neighbor list
        self._check_rebuild()

        # Compute LJ energy and forces using the interactions module
        wp_energies, wp_forces = lj_energy_forces(
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
            device=self.device,
        )

        # Copy computed forces to our force array
        wp.copy(self.wp_forces, wp_forces)
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
        # Always rebuild neighbors (cell may have changed)
        self._rebuild_neighbors()

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
            device=self.device,
        )

        wp.copy(self.wp_forces, wp_forces)

        if virial_tensors is not None:
            # Pack float64[9] into vec9[f/d] expected by the MTK integrators
            wp.launch(
                _pack_virial_flat_to_vec9_kernel,
                dim=1,
                inputs=[wp_virial_flat, virial_tensors],
                device=self.device,
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
        # Update torch cell for neighbor list
        self.torch_cell = wp.to_torch(self.wp_cell).squeeze(0)
        # Force neighbor rebuild on next force computation
        self.neighbor_manager.positions_at_rebuild = None

    def kinetic_energy(self) -> wp.array:
        """Compute kinetic energy on device (shape (1,), in eV)."""
        ke = wp.zeros(1, dtype=self.wp_dtype, device=self.device)
        compute_kinetic_energy(
            velocities=self.wp_velocities,
            masses=self.wp_masses,
            kinetic_energy=ke,
            device=self.device,
        )
        return ke

    def temperature_kT(self) -> wp.array:
        """Compute instantaneous temperature on device (kB*T in eV, shape (1,))."""
        ke = self.kinetic_energy()
        temp = wp.zeros(1, dtype=self.wp_dtype, device=self.device)
        compute_temperature(
            kinetic_energy=ke,
            temperature=temp,
            num_atoms=self.num_atoms,
            device=self.device,
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
        wp_temperature = wp.array([kT], dtype=self.wp_dtype, device=self.device)

        # Scratch arrays for COM removal
        wp_total_momentum = wp.zeros(1, dtype=self.wp_vec_dtype, device=self.device)
        wp_total_mass = wp.zeros(1, dtype=self.wp_dtype, device=self.device)
        wp_com_velocities = wp.zeros(1, dtype=self.wp_vec_dtype, device=self.device)

        initialize_velocities(
            velocities=self.wp_velocities,
            masses=self.wp_masses,
            temperature=wp_temperature,
            total_momentum=wp_total_momentum,
            total_mass=wp_total_mass,
            com_velocities=wp_com_velocities,
            random_seed=seed,
            remove_com=True,
            device=self.device,
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
        positions: np.ndarray,  # (N_total, 3)
        cells: np.ndarray,  # (B, 3, 3)
        batch_idx: np.ndarray,  # (N_total,)
        num_systems: int,
        masses: np.ndarray | None = None,  # (N_total,)
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
        self.num_atoms = self.total_atoms
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.switch_width = float(switch_width)
        self.half_neighbor_list = bool(half_neighbor_list)
        self.device = device

        self.dtype = dtype
        self.wp_dtype = wp.float64 if dtype == np.float64 else wp.float32
        self.wp_vec_dtype = wp.vec3d if dtype == np.float64 else wp.vec3f
        self.wp_mat_dtype = wp.mat33d if dtype == np.float64 else wp.mat33f
        self.torch_dtype = torch.float64 if dtype == np.float64 else torch.float32

        if masses is None:
            masses = np.full(self.total_atoms, MASS_AR, dtype=dtype)
        masses = mass_amu_to_internal(masses.astype(dtype))

        self.torch_positions = torch.tensor(
            positions, dtype=self.torch_dtype, device=device
        )
        self.torch_cells = torch.tensor(cells, dtype=self.torch_dtype, device=device)
        self.torch_pbc = torch.tensor(
            np.tile([True, True, True], (self.num_systems, 1)),
            dtype=torch.bool,
            device=device,
        )
        self.torch_batch_idx = torch.tensor(batch_idx, dtype=torch.int32, device=device)
        self.torch_num_atoms_per_system = torch.bincount(
            self.torch_batch_idx, minlength=self.num_systems
        ).to(dtype=torch.int32)
        self.num_atoms_per_system = (
            self.torch_num_atoms_per_system.detach().cpu().numpy()
        )

        self.wp_positions = wp.array(
            positions.astype(dtype), dtype=self.wp_vec_dtype, device=device
        )
        self.wp_velocities = wp.zeros(
            self.total_atoms, dtype=self.wp_vec_dtype, device=device
        )
        self.wp_forces = wp.zeros(
            self.total_atoms, dtype=self.wp_vec_dtype, device=device
        )
        self.wp_masses = wp.array(masses, dtype=self.wp_dtype, device=device)

        self.wp_cells = wp.array(
            cells.astype(dtype), dtype=self.wp_mat_dtype, device=device
        )
        self.wp_cells_inv = wp.empty_like(self.wp_cells)
        compute_cell_inverse(self.wp_cells, self.wp_cells_inv, device=device)

        self.wp_batch_idx = wp.array(
            batch_idx.astype(np.int32), dtype=wp.int32, device=device
        )

        self.neighbor_manager = BatchedNeighborListManager(
            total_atoms=self.total_atoms,
            cutoff=self.cutoff,
            skin=self.skin,
            batch_idx=self.torch_batch_idx,
            num_systems=self.num_systems,
            max_neighbors=100,
            half_fill=self.half_neighbor_list,
            device=device,
        )
        self._rebuild_neighbors()

    def _rebuild_neighbors(self) -> None:
        self.torch_positions = wp.to_torch(self.wp_positions)
        self.torch_cells = wp.to_torch(self.wp_cells)
        self.neighbor_manager.build(
            self.torch_positions, self.torch_cells, self.torch_pbc
        )

    def initialize_temperature(self, temperatures_K: np.ndarray, seed: int = 0) -> None:
        from nvalchemiops.dynamics.utils.thermostat_utils import (
            initialize_velocities as init_vel,
        )

        kT = np.asarray(temperatures_K, dtype=np.float64) * KB_EV
        wp_temperature = wp.array(
            kT.astype(self.dtype), dtype=self.wp_dtype, device=self.device
        )
        B = self.num_systems
        wp_total_momentum = wp.zeros(B, dtype=self.wp_vec_dtype, device=self.device)
        wp_total_mass = wp.zeros(B, dtype=self.wp_dtype, device=self.device)
        wp_com_velocities = wp.zeros(B, dtype=self.wp_vec_dtype, device=self.device)
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
            num_systems=B,
            device=self.device,
        )

        # One-time feedback: compute achieved per-system temperatures (host read)
        from nvalchemiops.dynamics.utils.thermostat_utils import (
            compute_kinetic_energy as ke_fn,
        )

        wp_ke = wp.zeros(B, dtype=self.wp_dtype, device=self.device)
        ke_fn(
            self.wp_velocities,
            self.wp_masses,
            kinetic_energy=wp_ke,
            batch_idx=self.wp_batch_idx,
            num_systems=B,
            device=self.device,
        )
        ke = wp_ke.numpy()
        dof = np.maximum(3 * self.num_atoms_per_system - 3, 1).astype(np.float64)
        actual_kT = 2.0 * ke / dof
        actual_T = actual_kT / KB_EV
        print(f"Initialized velocities: target={temperatures_K} K, actual={actual_T} K")

    def compute_forces(self) -> wp.array:
        self._rebuild_neighbors()
        wp_energies, wp_forces = lj_energy_forces(
            positions=self.wp_positions,
            cell=self.wp_cells,
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
            device=self.device,
        )
        wp.copy(self.wp_forces, wp_forces)
        return wp_energies

    def kinetic_energy_per_system(self) -> wp.array:
        """Compute kinetic energy per system (shape (B,), in eV)."""
        B = self.num_systems
        ke = wp.zeros(B, dtype=self.wp_dtype, device=self.device)
        compute_kinetic_energy(
            velocities=self.wp_velocities,
            masses=self.wp_masses,
            kinetic_energy=ke,
            batch_idx=self.wp_batch_idx,
            num_systems=B,
            device=self.device,
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
        n = np.asarray(self.num_atoms_per_system, dtype=np.int64)
        if not np.all(n == n[0]):
            raise ValueError(
                "temperature_kT_per_system requires uniform num_atoms per system; "
                f"got {n}."
            )
        B = self.num_systems
        ke = self.kinetic_energy_per_system()
        temp = wp.zeros(B, dtype=self.wp_dtype, device=self.device)
        compute_temperature(
            kinetic_energy=ke,
            temperature=temp,
            num_atoms=int(n[0]),
            batch_idx=self.wp_batch_idx,
            num_systems=B,
            device=self.device,
        )
        return temp


def run_langevin_baoab(
    system: MDSystem,
    num_steps: int,
    dt_fs: float,
    temperature_K: float,
    friction_per_fs: float,
    log_interval: int = 100,
    seed: int = 42,
) -> list[SimulationStats]:
    """Run Langevin (BAOAB) dynamics on an :class:`MDSystem`."""
    kT = temperature_K * KB_EV

    wp_dt = wp.array([dt_fs], dtype=system.wp_dtype, device=system.device)
    wp_temperature = wp.array([kT], dtype=system.wp_dtype, device=system.device)
    wp_friction = wp.array(
        [friction_per_fs], dtype=system.wp_dtype, device=system.device
    )

    wp_energies = system.compute_forces()
    stats_history: list[SimulationStats] = []

    print(f"\nRunning {num_steps} Langevin steps at T={temperature_K:.1f} K")
    print(f"  dt = {dt_fs:.3f} fs, friction = {friction_per_fs:.4f} 1/fs")
    print_header()

    step_start = time.perf_counter()

    for step in range(num_steps):
        langevin_baoab_half_step(
            positions=system.wp_positions,
            velocities=system.wp_velocities,
            forces=system.wp_forces,
            masses=system.wp_masses,
            dt=wp_dt,
            temperature=wp_temperature,
            friction=wp_friction,
            random_seed=seed + step,
            device=system.device,
        )

        wrap_positions_to_cell(
            positions=system.wp_positions,
            cells=system.wp_cell,
            cells_inv=system.wp_cell_inv,
            device=system.device,
        )

        wp_energies = system.compute_forces()
        ke_arr = system.kinetic_energy()

        langevin_baoab_finalize(
            velocities=system.wp_velocities,
            forces_new=system.wp_forces,
            masses=system.wp_masses,
            dt=wp_dt,
            device=system.device,
        )

        if step % log_interval == 0 or step == num_steps - 1:
            elapsed = time.perf_counter() - step_start
            ms_per_step = elapsed * 1000 / max(1, log_interval)

            temp_arr = system.temperature_kT()

            temp_K = float(temp_arr.numpy()[0]) / KB_EV
            pe = float(wp_energies.numpy().sum())
            ke = float(ke_arr.numpy()[0])

            max_force = float(
                torch.linalg.norm(wp.to_torch(system.wp_forces), dim=1).max().item()
            )
            min_r = neighbor_distance_stats(
                positions=wp.to_torch(system.wp_positions),
                cell=system.torch_cell,
                neighbor_matrix=system.neighbor_manager.torch_neighbor_matrix,
                neighbor_matrix_shifts=system.neighbor_manager.torch_neighbor_shifts,
                num_neighbors=system.neighbor_manager.torch_num_neighbors,
                fill_value=system.num_atoms,
            )

            stats = SimulationStats(
                step=step,
                kinetic_energy=ke,
                potential_energy=pe,
                total_energy=ke + pe,
                temperature=temp_K,
                num_neighbors=system.neighbor_manager.total_neighbors(),
                min_neighbor_distance=min_r,
                max_force=max_force,
                time_per_step_ms=ms_per_step,
            )
            stats_history.append(stats)
            print_stats(stats)
            step_start = time.perf_counter()

    return stats_history


def run_batched_langevin_baoab(
    system: BatchedMDSystem,
    num_steps: int,
    dt_fs: float,
    temperatures_K: np.ndarray,
    frictions_per_fs: np.ndarray,
    log_interval: int = 100,
    seed: int = 0,
) -> dict[int, list[SimulationStats]]:
    """Run batched Langevin (BAOAB) for multiple independent systems."""
    kT = np.asarray(temperatures_K, dtype=np.float64) * KB_EV
    gamma = np.asarray(frictions_per_fs, dtype=np.float64)

    wp_dt = wp.array(
        np.full(system.num_systems, dt_fs, dtype=system.dtype),
        dtype=system.wp_dtype,
        device=system.device,
    )
    wp_temperature = wp.array(
        kT.astype(system.dtype), dtype=system.wp_dtype, device=system.device
    )
    wp_friction = wp.array(
        gamma.astype(system.dtype), dtype=system.wp_dtype, device=system.device
    )

    energies = system.compute_forces()

    history: dict[int, list[SimulationStats]] = {
        i: [] for i in range(system.num_systems)
    }

    print(
        f"\nRunning batched Langevin (BAOAB): {system.num_systems} systems, {system.total_atoms} atoms total"
    )
    print(
        f"  dt = {dt_fs:.3f} fs; temperatures={temperatures_K} K; frictions={frictions_per_fs} 1/fs"
    )
    # We'll print per-system rows at each log point.

    from nvalchemiops.dynamics.utils.thermostat_utils import (
        compute_kinetic_energy as ke_fn,
    )

    wp_ke_scratch = wp.zeros(
        system.num_systems, dtype=system.wp_dtype, device=system.device
    )

    dof = np.maximum(
        3 * np.asarray(system.num_atoms_per_system, dtype=np.int64) - 3, 1
    ).astype(np.float64)

    step_start = time.perf_counter()
    for step in range(num_steps):
        langevin_baoab_half_step(
            positions=system.wp_positions,
            velocities=system.wp_velocities,
            forces=system.wp_forces,
            masses=system.wp_masses,
            dt=wp_dt,
            temperature=wp_temperature,
            friction=wp_friction,
            random_seed=seed + step,
            batch_idx=system.wp_batch_idx,
            device=system.device,
        )

        wrap_positions_to_cell(
            positions=system.wp_positions,
            batch_idx=system.wp_batch_idx,
            cells=system.wp_cells,
            cells_inv=system.wp_cells_inv,
            device=system.device,
        )

        energies = system.compute_forces()

        ke_fn(
            system.wp_velocities,
            system.wp_masses,
            kinetic_energy=wp_ke_scratch,
            batch_idx=system.wp_batch_idx,
            num_systems=system.num_systems,
            device=system.device,
        )
        ke = wp_ke_scratch
        langevin_baoab_finalize(
            velocities=system.wp_velocities,
            forces_new=system.wp_forces,
            masses=system.wp_masses,
            dt=wp_dt,
            batch_idx=system.wp_batch_idx,
            device=system.device,
        )

        if step % log_interval == 0 or step == num_steps - 1:
            elapsed = time.perf_counter() - step_start
            ms_per_step = elapsed * 1000 / max(1, log_interval)

            # Host reads at log points only
            ke_np = ke.numpy()
            kT_np = 2.0 * ke_np / dof
            e_np = wp.to_torch(energies).cpu()
            b_np = wp.to_torch(system.wp_batch_idx).cpu()
            pe_per_sys = torch.bincount(
                b_np, weights=e_np.double(), minlength=system.num_systems
            ).numpy()

            # Neighbor counts per system (sum of per-atom neighbor counts)
            if system.neighbor_manager.torch_num_neighbors is not None:
                nn = (
                    system.neighbor_manager.torch_num_neighbors.detach()
                    .cpu()
                    .to(torch.float64)
                )
                neighbors_per_sys = (
                    torch.bincount(b_np, weights=nn, minlength=system.num_systems)
                    .numpy()
                    .astype(np.int64)
                )
            else:
                neighbors_per_sys = np.full(system.num_systems, -1, dtype=np.int64)

            # Max |F| per system
            f_norm = (
                torch.linalg.norm(wp.to_torch(system.wp_forces), dim=1).detach().cpu()
            )
            max_force_per_sys = np.zeros(system.num_systems, dtype=np.float64)
            for sys_id in range(system.num_systems):
                mask = b_np == sys_id
                max_force_per_sys[sys_id] = (
                    float(f_norm[mask].max().item())
                    if torch.any(mask)
                    else float("nan")
                )

            print("=" * 120)
            print(
                f"{'Step':>8} {'Sys':>5} {'KE (eV)':>12} {'PE (eV)':>12} {'Total (eV)':>12} "
                f"{'T (K)':>10} {'Neighbors':>10} {'max|F|':>10} {'ms/step':>9}"
            )
            print("=" * 120)
            for sys_id in range(system.num_systems):
                temp_K = float(kT_np[sys_id]) / KB_EV
                ke_sys = float(ke_np[sys_id])
                pe_sys = float(pe_per_sys[sys_id])
                stats = SimulationStats(
                    step=step,
                    kinetic_energy=ke_sys,
                    potential_energy=pe_sys,
                    total_energy=ke_sys + pe_sys,
                    temperature=temp_K,
                    num_neighbors=int(neighbors_per_sys[sys_id]),
                    min_neighbor_distance=float("nan"),
                    max_force=float(max_force_per_sys[sys_id]),
                    time_per_step_ms=ms_per_step,
                )
                history[sys_id].append(stats)
                print(
                    f"{step:>8d} {sys_id:>5d} {ke_sys:>12.4f} {pe_sys:>12.4f} {ke_sys + pe_sys:>12.4f} "
                    f"{temp_K:>10.2f} {int(neighbors_per_sys[sys_id]):>10d} {float(max_force_per_sys[sys_id]):>10.3e} {ms_per_step:>9.3f}"
                )
            step_start = time.perf_counter()

    return history


def run_velocity_verlet(
    system: MDSystem,
    num_steps: int,
    dt_fs: float,
    log_interval: int = 100,
) -> list[SimulationStats]:
    """Run Velocity Verlet (NVE) dynamics on an :class:`MDSystem`.

    Notes
    -----
    This is a **microcanonical (NVE)** integrator with no thermostat.
    It is therefore sensitive to:
    - timestep stability,
    - neighbor-list correctness,
    - force smoothness at the cutoff (consider using `switch_width > 0`).
    """
    wp_dt = wp.array([dt_fs], dtype=system.wp_dtype, device=system.device)

    # Ensure initial forces are available
    wp_energies = system.compute_forces()
    stats_history: list[SimulationStats] = []

    print(f"\nRunning {num_steps} Velocity Verlet steps (NVE)")
    print(f"  dt = {dt_fs:.3f} fs")
    print_header()

    step_start = time.perf_counter()

    for step in range(num_steps):
        # Pass 1: update positions and half-step velocities
        velocity_verlet_position_update(
            positions=system.wp_positions,
            velocities=system.wp_velocities,
            forces=system.wp_forces,
            masses=system.wp_masses,
            dt=wp_dt,
            device=system.device,
        )

        # Enforce periodicity
        wrap_positions_to_cell(
            positions=system.wp_positions,
            cells=system.wp_cell,
            cells_inv=system.wp_cell_inv,
            device=system.device,
        )

        # Recompute forces at new positions (also rebuilds neighbor list as needed)
        wp_energies = system.compute_forces()
        ke_arr = system.kinetic_energy()

        # Pass 2: finalize velocities using new forces
        velocity_verlet_velocity_finalize(
            velocities=system.wp_velocities,
            forces_new=system.wp_forces,
            masses=system.wp_masses,
            dt=wp_dt,
            device=system.device,
        )

        if step % log_interval == 0 or step == num_steps - 1:
            elapsed = time.perf_counter() - step_start
            ms_per_step = elapsed * 1000 / max(1, log_interval)

            temp_arr = system.temperature_kT()
            temp_K = float(temp_arr.numpy()[0]) / KB_EV
            pe = float(wp_energies.numpy().sum())
            ke = float(ke_arr.numpy()[0])

            max_force = float(
                torch.linalg.norm(wp.to_torch(system.wp_forces), dim=1).max().item()
            )
            min_r = neighbor_distance_stats(
                positions=wp.to_torch(system.wp_positions),
                cell=system.torch_cell,
                neighbor_matrix=system.neighbor_manager.torch_neighbor_matrix,
                neighbor_matrix_shifts=system.neighbor_manager.torch_neighbor_shifts,
                num_neighbors=system.neighbor_manager.torch_num_neighbors,
                fill_value=system.num_atoms,
            )

            stats = SimulationStats(
                step=step,
                kinetic_energy=ke,
                potential_energy=pe,
                total_energy=ke + pe,
                temperature=temp_K,
                num_neighbors=system.neighbor_manager.total_neighbors(),
                min_neighbor_distance=min_r,
                max_force=max_force,
                time_per_step_ms=ms_per_step,
            )
            stats_history.append(stats)
            print_stats(stats)
            step_start = time.perf_counter()

    return stats_history


def _scalar_pressure_and_volume(
    system: MDSystem,
    virial_tensors: wp.array,
) -> tuple[float, float]:
    """Compute scalar pressure (eV/Å^3) and volume (Å^3)."""
    tensor_dtype = vec9f if system.wp_dtype == wp.float32 else vec9d

    # Pre-allocate scratch arrays
    volumes = wp.empty(1, dtype=system.wp_dtype, device=system.device)
    compute_cell_volume(system.wp_cell, volumes=volumes, device=system.device)

    kinetic_tensors = wp.zeros((1, 9), dtype=system.wp_dtype, device=system.device)
    pressure_tensors = wp.zeros(1, dtype=tensor_dtype, device=system.device)
    compute_pressure_tensor(
        velocities=system.wp_velocities,
        masses=system.wp_masses,
        virial_tensors=virial_tensors,
        cells=system.wp_cell,
        kinetic_tensors=kinetic_tensors,
        pressure_tensors=pressure_tensors,
        volumes=volumes,
        device=system.device,
    )

    scalar_pressures = wp.empty(1, dtype=system.wp_dtype, device=system.device)
    compute_scalar_pressure(pressure_tensors, scalar_pressures, device=system.device)
    p = float(scalar_pressures.numpy()[0])

    V = float(volumes.numpy()[0])
    return p, V


def run_nph_mtk(
    system: MDSystem,
    num_steps: int,
    dt_fs: float,
    target_pressure_atm: float = 1.0,
    pdamp_fs: float = 1000.0,
    reference_temperature_K: float = 94.4,
    log_interval: int = 100,
) -> list[BarostatStats]:
    """Run NPH (MTK) dynamics on an :class:`MDSystem` (isotropic pressure control)."""
    # Target pressure in internal units (eV/Å^3)
    p_ext = pressure_atm_to_ev_per_a3(target_pressure_atm)
    wp_target_pressure = wp.array([p_ext], dtype=system.wp_dtype, device=system.device)

    # Barostat mass uses kT and tau_p
    kT_ref = float(reference_temperature_K) * KB_EV
    wp_temp_arr = wp.array([kT_ref], dtype=system.wp_dtype, device=system.device)
    wp_tau_arr = wp.array(
        [float(pdamp_fs)], dtype=system.wp_dtype, device=system.device
    )
    wp_natoms_arr = wp.array([system.num_atoms], dtype=wp.int32, device=system.device)
    cell_masses = wp.empty(1, dtype=system.wp_dtype, device=system.device)
    compute_barostat_mass(
        target_temperature=wp_temp_arr,
        tau_p=wp_tau_arr,
        num_atoms=wp_natoms_arr,
        masses_out=cell_masses,
        device=system.device,
    )

    cell_velocities = wp.zeros(1, dtype=system.wp_mat_dtype, device=system.device)
    tensor_dtype = vec9f if system.wp_dtype == wp.float32 else vec9d
    virial_tensors = wp.zeros(1, dtype=tensor_dtype, device=system.device)

    # Scratch arrays for NPH step
    nph_pressure_tensors = wp.zeros(1, dtype=tensor_dtype, device=system.device)
    nph_volumes = wp.zeros(1, dtype=system.wp_dtype, device=system.device)
    nph_kinetic_energy = wp.zeros(1, dtype=system.wp_dtype, device=system.device)
    nph_cells_inv = wp.empty(1, dtype=system.wp_mat_dtype, device=system.device)
    nph_kinetic_tensors = wp.zeros((1, 9), dtype=system.wp_dtype, device=system.device)
    nph_num_atoms_per_system = wp.array(
        [system.num_atoms], dtype=wp.int32, device=system.device
    )

    # Initial forces/virial
    wp_energies = system.compute_forces_virial(virial_tensors)

    # Pre-compute volumes and kinetic energy for the first step
    compute_cell_volume(system.wp_cell, volumes=nph_volumes, device=system.device)
    compute_kinetic_energy(
        system.wp_velocities,
        system.wp_masses,
        kinetic_energy=nph_kinetic_energy,
        device=system.device,
    )

    stats_history: list[BarostatStats] = []

    print(f"\nRunning {num_steps} NPH (MTK) steps at P={target_pressure_atm:.3f} atm")
    print(f"  dt = {dt_fs:.3f} fs, pdamp = {pdamp_fs:.1f} fs (barostat timescale)")
    print_header_barostat()

    step_start = time.perf_counter()

    def _compute_forces_cb(positions, cells, forces, virial_out):
        nonlocal wp_energies
        # Ensure the system points at the integrator-updated arrays
        system.wp_positions = positions
        system.wp_cell = cells
        wp_energies = system.compute_forces_virial(virial_out)
        wp.copy(forces, system.wp_forces)

    for step in range(num_steps):
        run_nph_step(
            positions=system.wp_positions,
            velocities=system.wp_velocities,
            forces=system.wp_forces,
            masses=system.wp_masses,
            cells=system.wp_cell,
            cell_velocities=cell_velocities,
            virial_tensors=virial_tensors,
            cell_masses=cell_masses,
            target_pressure=wp_target_pressure,
            num_atoms=system.num_atoms,
            dt=float(dt_fs),
            pressure_tensors=nph_pressure_tensors,
            volumes=nph_volumes,
            kinetic_energy=nph_kinetic_energy,
            cells_inv=nph_cells_inv,
            kinetic_tensors=nph_kinetic_tensors,
            num_atoms_per_system=nph_num_atoms_per_system,
            compute_forces_fn=_compute_forces_cb,
            device=system.device,
        )

        ke_arr = system.kinetic_energy()

        if step % log_interval == 0 or step == num_steps - 1:
            elapsed = time.perf_counter() - step_start
            ms_per_step = elapsed * 1000 / max(1, log_interval)

            temp_K = float(system.temperature_kT().numpy()[0]) / KB_EV
            pe = float(wp_energies.numpy().sum())
            ke = float(ke_arr.numpy()[0])

            pressure, volume = _scalar_pressure_and_volume(system, virial_tensors)
            pressure_atm = pressure_ev_per_a3_to_atm(pressure)

            max_force = float(
                torch.linalg.norm(wp.to_torch(system.wp_forces), dim=1).max().item()
            )
            min_r = neighbor_distance_stats(
                positions=wp.to_torch(system.wp_positions),
                cell=wp.to_torch(system.wp_cell).squeeze(0),
                neighbor_matrix=system.neighbor_manager.torch_neighbor_matrix,
                neighbor_matrix_shifts=system.neighbor_manager.torch_neighbor_shifts,
                num_neighbors=system.neighbor_manager.torch_num_neighbors,
                fill_value=system.num_atoms,
            )

            stats = BarostatStats(
                step=step,
                kinetic_energy=ke,
                potential_energy=pe,
                total_energy=ke + pe,
                temperature=temp_K,
                pressure=pressure_atm,
                volume=volume,
                num_neighbors=system.neighbor_manager.total_neighbors(),
                min_neighbor_distance=min_r,
                max_force=max_force,
                time_per_step_ms=ms_per_step,
            )
            stats_history.append(stats)
            print_stats_barostat(stats)
            step_start = time.perf_counter()

    return stats_history


def run_npt_mtk(
    system: MDSystem,
    num_steps: int,
    dt_fs: float,
    target_temperature_K: float = 94.4,
    target_pressure_atm: float = 1.0,
    tdamp_fs: float = 500.0,
    pdamp_fs: float = 5000.0,
    chain_length: int = 3,
    log_interval: int = 100,
) -> list[BarostatStats]:
    """Run NPT (MTK + NHC) dynamics on an :class:`MDSystem` (isotropic pressure)."""
    kT = float(target_temperature_K) * KB_EV
    p_ext = pressure_atm_to_ev_per_a3(target_pressure_atm)

    wp_target_temperature = wp.array([kT], dtype=system.wp_dtype, device=system.device)
    wp_target_pressure = wp.array([p_ext], dtype=system.wp_dtype, device=system.device)

    # Thermostat chain masses/state (expects kT, tau in same time units as dt)
    thermostat_masses = wp.empty(
        int(chain_length), dtype=system.wp_dtype, device=system.device
    )
    nhc_compute_masses(
        ndof=3 * system.num_atoms,
        target_temp=kT,
        tau=float(tdamp_fs),
        chain_length=int(chain_length),
        masses=thermostat_masses,
        num_systems=1,
        device=system.device,
        dtype=system.wp_dtype,
    )
    # nhc_compute_masses returns shape (chain_length,) for single-system; NPT expects (B, chain_length)
    if thermostat_masses.ndim == 1:
        thermostat_masses_2d = wp.zeros(
            (1, chain_length), dtype=system.wp_dtype, device=system.device
        )
        wp.launch(
            _copy_1d_to_row2d_kernel,
            dim=chain_length,
            inputs=[thermostat_masses, thermostat_masses_2d],
            device=system.device,
        )
        thermostat_masses = thermostat_masses_2d
    eta = wp.zeros((1, chain_length), dtype=system.wp_dtype, device=system.device)
    eta_dot = wp.zeros((1, chain_length), dtype=system.wp_dtype, device=system.device)

    # Barostat masses/state
    wp_temp_baro = wp.array([kT], dtype=system.wp_dtype, device=system.device)
    wp_tau_baro = wp.array(
        [float(pdamp_fs)], dtype=system.wp_dtype, device=system.device
    )
    wp_natoms_baro = wp.array([system.num_atoms], dtype=wp.int32, device=system.device)
    cell_masses = wp.empty(1, dtype=system.wp_dtype, device=system.device)
    compute_barostat_mass(
        target_temperature=wp_temp_baro,
        tau_p=wp_tau_baro,
        num_atoms=wp_natoms_baro,
        masses_out=cell_masses,
        device=system.device,
    )
    cell_velocities = wp.zeros(1, dtype=system.wp_mat_dtype, device=system.device)

    tensor_dtype = vec9f if system.wp_dtype == wp.float32 else vec9d
    virial_tensors = wp.zeros(1, dtype=tensor_dtype, device=system.device)

    # Scratch arrays for NPT step
    npt_pressure_tensors = wp.zeros(1, dtype=tensor_dtype, device=system.device)
    npt_volumes = wp.zeros(1, dtype=system.wp_dtype, device=system.device)
    npt_kinetic_energy = wp.zeros(1, dtype=system.wp_dtype, device=system.device)
    npt_cells_inv = wp.empty(1, dtype=system.wp_mat_dtype, device=system.device)
    npt_kinetic_tensors = wp.zeros((1, 9), dtype=system.wp_dtype, device=system.device)
    npt_num_atoms_per_system = wp.array(
        [system.num_atoms], dtype=wp.int32, device=system.device
    )

    # Initial forces/virial
    wp_energies = system.compute_forces_virial(virial_tensors)

    # Pre-compute volumes and kinetic energy for the first step
    compute_cell_volume(system.wp_cell, volumes=npt_volumes, device=system.device)
    compute_kinetic_energy(
        system.wp_velocities,
        system.wp_masses,
        kinetic_energy=npt_kinetic_energy,
        device=system.device,
    )

    stats_history: list[BarostatStats] = []

    print(
        f"\nRunning {num_steps} NPT (MTK) steps at T={target_temperature_K:.1f} K, P={target_pressure_atm:.3f} atm"
    )
    print(
        f"  dt = {dt_fs:.3f} fs, tdamp = {tdamp_fs:.1f} fs, pdamp = {pdamp_fs:.1f} fs, chain_length = {chain_length}"
    )
    print_header_barostat()

    step_start = time.perf_counter()

    def _compute_forces_cb(positions, cells, forces, virial_out):
        nonlocal wp_energies
        system.wp_positions = positions
        system.wp_cell = cells
        wp_energies = system.compute_forces_virial(virial_out)
        wp.copy(forces, system.wp_forces)

    for step in range(num_steps):
        run_npt_step(
            positions=system.wp_positions,
            velocities=system.wp_velocities,
            forces=system.wp_forces,
            masses=system.wp_masses,
            cells=system.wp_cell,
            cell_velocities=cell_velocities,
            virial_tensors=virial_tensors,
            eta=eta,
            eta_dot=eta_dot,
            thermostat_masses=thermostat_masses,
            cell_masses=cell_masses,
            target_temperature=wp_target_temperature,
            target_pressure=wp_target_pressure,
            num_atoms=system.num_atoms,
            chain_length=int(chain_length),
            dt=float(dt_fs),
            pressure_tensors=npt_pressure_tensors,
            volumes=npt_volumes,
            kinetic_energy=npt_kinetic_energy,
            cells_inv=npt_cells_inv,
            kinetic_tensors=npt_kinetic_tensors,
            num_atoms_per_system=npt_num_atoms_per_system,
            compute_forces_fn=_compute_forces_cb,
            device=system.device,
        )

        ke_arr = system.kinetic_energy()

        if step % log_interval == 0 or step == num_steps - 1:
            elapsed = time.perf_counter() - step_start
            ms_per_step = elapsed * 1000 / max(1, log_interval)

            temp_K = float(system.temperature_kT().numpy()[0]) / KB_EV
            pe = float(wp_energies.numpy().sum())
            ke = float(ke_arr.numpy()[0])

            pressure, volume = _scalar_pressure_and_volume(system, virial_tensors)
            pressure_atm = pressure_ev_per_a3_to_atm(pressure)

            max_force = float(
                torch.linalg.norm(wp.to_torch(system.wp_forces), dim=1).max().item()
            )
            min_r = neighbor_distance_stats(
                positions=wp.to_torch(system.wp_positions),
                cell=wp.to_torch(system.wp_cell).squeeze(0),
                neighbor_matrix=system.neighbor_manager.torch_neighbor_matrix,
                neighbor_matrix_shifts=system.neighbor_manager.torch_neighbor_shifts,
                num_neighbors=system.neighbor_manager.torch_num_neighbors,
                fill_value=system.num_atoms,
            )

            stats = BarostatStats(
                step=step,
                kinetic_energy=ke,
                potential_energy=pe,
                total_energy=ke + pe,
                temperature=temp_K,
                pressure=pressure_atm,
                volume=volume,
                num_neighbors=system.neighbor_manager.total_neighbors(),
                min_neighbor_distance=min_r,
                max_force=max_force,
                time_per_step_ms=ms_per_step,
            )
            stats_history.append(stats)
            print_stats_barostat(stats)
            step_start = time.perf_counter()

    return stats_history


# ==============================================================================
# Printing Utilities
# ==============================================================================


def print_header() -> None:
    """Print simulation statistics header."""
    print("=" * 95)
    print(
        f"{'Step':>8} {'KE (eV)':>12} {'PE (eV)':>12} {'Total (eV)':>12} "
        f"{'T (K)':>10} {'Neighbors':>10} {'min r (Å)':>10} {'max|F|':>10}"
    )
    print("=" * 95)


def print_stats(stats: SimulationStats) -> None:
    """Print statistics for a simulation step."""
    print(
        f"{stats.step:>8d} {stats.kinetic_energy:>12.4f} {stats.potential_energy:>12.4f} "
        f"{stats.total_energy:>12.4f} {stats.temperature:>10.2f} "
        f"{stats.num_neighbors:>10d} {stats.min_neighbor_distance:>10.3f} {stats.max_force:>10.3e}"
    )


def print_header_barostat() -> None:
    """Print simulation statistics header for NPT/NPH runs (includes pressure/volume)."""
    print("=" * 120)
    print(
        f"{'Step':>8} {'KE (eV)':>12} {'PE (eV)':>12} {'Total (eV)':>12} "
        f"{'T (K)':>10} {'P (atm)':>10} {'V (Å^3)':>10} "
        f"{'Neighbors':>10} {'min r (Å)':>10} {'max|F|':>10}"
    )
    print("=" * 120)


def print_stats_barostat(stats: BarostatStats) -> None:
    """Print statistics for a barostat simulation step."""
    print(
        f"{stats.step:>8d} {stats.kinetic_energy:>12.4f} {stats.potential_energy:>12.4f} "
        f"{stats.total_energy:>12.4f} {stats.temperature:>10.2f} "
        f"{stats.pressure:>10.3f} {stats.volume:>10.2f} "
        f"{stats.num_neighbors:>10d} {stats.min_neighbor_distance:>10.3f} {stats.max_force:>10.3e}"
    )
