# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Crystal Structure Generation Utilities for Electrostatics Tests
===============================================================

This module provides functions to generate common crystal structures
for testing electrostatic calculations. Structures are returned as
dictionaries with positions, cell, and charges that can be easily
converted to torch tensors.

Supported structures:
- CsCl (BCC-like ionic crystal)
- Wurtzite (hexagonal ZnS)
- Zincblende (cubic ZnS)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from nvalchemiops.torch.neighbors import cell_list


class CrystalSystem(NamedTuple):
    """Container for crystal structure data.

    Attributes
    ----------
    positions : np.ndarray, shape (N, 3)
        Cartesian atomic positions in Angstroms.
    cell : np.ndarray, shape (3, 3)
        Unit cell matrix (row vectors).
    charges : np.ndarray, shape (N,)
        Atomic charges.
    symbols : list[str]
        Atomic symbols.
    """

    positions: np.ndarray
    cell: np.ndarray
    charges: np.ndarray
    symbols: list[str]


def create_cscl_supercell(size: int) -> CrystalSystem:
    """
    Create CsCl supercell of given size.

    CsCl has a BCC-like structure with:
    - Cs at corners (0, 0, 0)
    - Cl at body center (0.5, 0.5, 0.5)

    Parameters
    ----------
    size : int
        Linear supercell size (total atoms = 2 * size³)

    Returns
    -------
    CrystalSystem
        Crystal structure with positions, cell, charges, and symbols.

    Notes
    -----
    Base CsCl unit cell has 2 atoms with a = 4.14 Å.
    A size=n supercell has 2n³ atoms.
    """
    a = 4.14  # Lattice constant in Angstroms

    # Base unit cell (cubic)
    base_cell = np.eye(3) * a

    # Fractional positions in unit cell
    base_frac_positions = np.array(
        [
            [0.0, 0.0, 0.0],  # Cs
            [0.5, 0.5, 0.5],  # Cl
        ]
    )
    base_charges = np.array([1.0, -1.0])
    base_symbols = ["Cs", "Cl"]

    # Generate supercell
    positions_list = []
    charges_list = []
    symbols_list = []

    for i in range(size):
        for j in range(size):
            for k in range(size):
                offset = np.array([i, j, k])
                for atom_idx, frac_pos in enumerate(base_frac_positions):
                    # Convert fractional to Cartesian and add supercell offset
                    cart_pos = (frac_pos + offset) @ base_cell
                    positions_list.append(cart_pos)
                    charges_list.append(base_charges[atom_idx])
                    symbols_list.append(base_symbols[atom_idx])

    # Supercell lattice vectors
    supercell = base_cell * size

    return CrystalSystem(
        positions=np.array(positions_list),
        cell=supercell,
        charges=np.array(charges_list),
        symbols=symbols_list,
    )


def create_wurtzite_system(size: int) -> CrystalSystem:
    """
    Create Wurtzite supercell of given size.

    Wurtzite (ZnS) has a hexagonal structure with 4 atoms per unit cell:
    - Zn at (0, 0, 0) and (1/3, 2/3, 1/2)
    - S at (0, 0, u) and (1/3, 2/3, 1/2 + u) where u ≈ 3/8

    Parameters
    ----------
    size : int
        Linear supercell size (total atoms = 4 * size³)

    Returns
    -------
    CrystalSystem
        Crystal structure with positions, cell, charges, and symbols.

    Notes
    -----
    Uses lattice parameters a = 3.21 Å, c = 5.21 Å (typical for ZnS).
    """
    a = 3.21  # Lattice constant a in Angstroms
    c = 5.21  # Lattice constant c in Angstroms
    u = 0.375  # Internal parameter (ideal wurtzite: 3/8)

    # Hexagonal cell vectors (row vectors)
    # a1 = (a, 0, 0)
    # a2 = (-a/2, a*sqrt(3)/2, 0)
    # a3 = (0, 0, c)
    sqrt3_2 = np.sqrt(3.0) / 2.0
    base_cell = np.array(
        [
            [a, 0.0, 0.0],
            [-a / 2.0, a * sqrt3_2, 0.0],
            [0.0, 0.0, c],
        ]
    )

    # Fractional positions in unit cell (4 atoms)
    base_frac_positions = np.array(
        [
            [0.0, 0.0, 0.0],  # Zn
            [1.0 / 3.0, 2.0 / 3.0, 0.5],  # Zn
            [0.0, 0.0, u],  # S
            [1.0 / 3.0, 2.0 / 3.0, 0.5 + u],  # S
        ]
    )
    base_charges = np.array([1.0, 1.0, -1.0, -1.0])
    base_symbols = ["Zn", "Zn", "S", "S"]

    # Generate supercell
    positions_list = []
    charges_list = []
    symbols_list = []

    for i in range(size):
        for j in range(size):
            for k in range(size):
                offset = np.array([i, j, k])
                for atom_idx, frac_pos in enumerate(base_frac_positions):
                    # Convert fractional to Cartesian and add supercell offset
                    cart_pos = (frac_pos + offset) @ base_cell
                    positions_list.append(cart_pos)
                    charges_list.append(base_charges[atom_idx])
                    symbols_list.append(base_symbols[atom_idx])

    # Supercell lattice vectors
    supercell = base_cell * size

    return CrystalSystem(
        positions=np.array(positions_list),
        cell=supercell,
        charges=np.array(charges_list),
        symbols=symbols_list,
    )


def create_zincblende_system(size: int) -> CrystalSystem:
    """
    Create Zincblende supercell of given size.

    Zincblende (ZnS) has an FCC-like cubic structure with 2 atoms per
    primitive cell (8 atoms per conventional cubic cell):
    - Zn at FCC positions
    - S at FCC positions shifted by (1/4, 1/4, 1/4)

    For the primitive (2-atom) cell:
    - Zn at (0, 0, 0)
    - S at (1/4, 1/4, 1/4)

    Parameters
    ----------
    size : int
        Linear supercell size (total atoms = 2 * size³)

    Returns
    -------
    CrystalSystem
        Crystal structure with positions, cell, charges, and symbols.

    Notes
    -----
    Uses conventional cubic cell with a = 5.41 Å (typical for ZnS zincblende).
    Each conventional cell contains 8 atoms (4 Zn + 4 S).

    We use the primitive FCC cell with 2 atoms for efficiency.
    """
    # Conventional cubic lattice constant
    a_conv = 5.41  # Angstroms

    # Primitive FCC cell vectors (row vectors)
    # These span the primitive cell with 2 atoms
    a = a_conv / 2.0
    base_cell = np.array(
        [
            [0.0, a, a],
            [a, 0.0, a],
            [a, a, 0.0],
        ]
    )

    # Fractional positions in primitive FCC cell (2 atoms)
    base_frac_positions = np.array(
        [
            [0.0, 0.0, 0.0],  # Zn
            [0.25, 0.25, 0.25],  # S
        ]
    )
    base_charges = np.array([2.0, -2.0])
    base_symbols = ["Zn", "S"]

    # Generate supercell
    positions_list = []
    charges_list = []
    symbols_list = []

    for i in range(size):
        for j in range(size):
            for k in range(size):
                offset = np.array([i, j, k])
                for atom_idx, frac_pos in enumerate(base_frac_positions):
                    # Convert fractional to Cartesian and add supercell offset
                    cart_pos = (frac_pos + offset) @ base_cell
                    positions_list.append(cart_pos)
                    charges_list.append(base_charges[atom_idx])
                    symbols_list.append(base_symbols[atom_idx])

    # Supercell lattice vectors
    supercell = base_cell * size

    return CrystalSystem(
        positions=np.array(positions_list),
        cell=supercell,
        charges=np.array(charges_list),
        symbols=symbols_list,
    )


def create_nacl_system(size: int) -> CrystalSystem:
    """
    Create NaCl (rock salt) supercell of given size.

    NaCl has an FCC-like structure with:
    - Na at FCC positions
    - Cl at FCC positions shifted by (0.5, 0, 0)

    Parameters
    ----------
    size : int
        Linear supercell size (total atoms = 2 * size³ for primitive cell)

    Returns
    -------
    CrystalSystem
        Crystal structure with positions, cell, charges, and symbols.

    Notes
    -----
    Uses conventional cubic cell with a = 5.64 Å.
    We use the primitive FCC cell with 2 atoms (1 Na + 1 Cl).
    """
    # Conventional cubic lattice constant
    a_conv = 5.64  # Angstroms

    # Primitive FCC cell vectors (row vectors)
    a = a_conv / 2.0
    base_cell = np.array(
        [
            [0.0, a, a],
            [a, 0.0, a],
            [a, a, 0.0],
        ]
    )

    # Fractional positions in primitive FCC cell (2 atoms)
    base_frac_positions = np.array(
        [
            [0.0, 0.0, 0.0],  # Na
            [0.5, 0.5, 0.5],  # Cl
        ]
    )
    base_charges = np.array([1.0, -1.0])
    base_symbols = ["Na", "Cl"]

    # Generate supercell
    positions_list = []
    charges_list = []
    symbols_list = []

    for i in range(size):
        for j in range(size):
            for k in range(size):
                offset = np.array([i, j, k])
                for atom_idx, frac_pos in enumerate(base_frac_positions):
                    # Convert fractional to Cartesian and add supercell offset
                    cart_pos = (frac_pos + offset) @ base_cell
                    positions_list.append(cart_pos)
                    charges_list.append(base_charges[atom_idx])
                    symbols_list.append(base_symbols[atom_idx])

    # Supercell lattice vectors
    supercell = base_cell * size

    return CrystalSystem(
        positions=np.array(positions_list),
        cell=supercell,
        charges=np.array(charges_list),
        symbols=symbols_list,
    )


def create_simple_cubic_system(
    size: int, lattice_constant: float = 3.0, charge: float = 1.0
) -> CrystalSystem:
    """
    Create simple cubic lattice with alternating charges.

    Parameters
    ----------
    size : int
        Linear supercell size (total atoms = size³)
    lattice_constant : float, default=3.0
        Lattice constant in Angstroms.
    charge : float, default=1.0
        Magnitude of charge (alternates +/- based on position parity).

    Returns
    -------
    CrystalSystem
        Crystal structure with positions, cell, charges, and symbols.
    """
    a = lattice_constant
    base_cell = np.eye(3) * a

    # Generate supercell with alternating charges
    positions_list = []
    charges_list = []
    symbols_list = []

    for i in range(size):
        for j in range(size):
            for k in range(size):
                pos = np.array([i, j, k]) @ base_cell
                positions_list.append(pos)
                # Alternating charges based on position parity
                parity = (i + j + k) % 2
                q = charge if parity == 0 else -charge
                charges_list.append(q)
                symbols_list.append("X+" if parity == 0 else "X-")

    supercell = base_cell * size

    return CrystalSystem(
        positions=np.array(positions_list),
        cell=supercell,
        charges=np.array(charges_list),
        symbols=symbols_list,
    )


# =============================================================================
# Virial Test Utilities
# =============================================================================

VIRIAL_DTYPE = torch.float64  # Need double precision for FD virial tests


def make_virial_cscl_system(
    size: int = 2,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
):
    """Create a CsCl test system for virial tests.

    Parameters
    ----------
    size : int
        Supercell linear size.
    dtype : torch.dtype, optional
        Tensor dtype. Defaults to ``VIRIAL_DTYPE`` (float64).
    device : torch.device, optional
        Device. Defaults to CPU.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (positions, charges, cell) where cell has shape (1, 3, 3).
    """
    if dtype is None:
        dtype = VIRIAL_DTYPE
    if device is None:
        device = torch.device("cpu")
    crystal = create_cscl_supercell(size)
    positions = torch.tensor(crystal.positions, dtype=dtype, device=device)
    charges = torch.tensor(crystal.charges, dtype=dtype, device=device)
    cell = torch.tensor(crystal.cell, dtype=dtype, device=device).unsqueeze(0)
    return positions, charges, cell


def get_virial_neighbor_data(positions, cell, cutoff=6.0):
    """Compute neighbor list for virial test systems.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (neighbor_list, neighbor_ptr, unit_shifts).
    """
    pbc = torch.tensor([True, True, True], dtype=torch.bool, device=positions.device)
    neighbor_list, neighbor_ptr, unit_shifts = cell_list(
        positions,
        cutoff,
        cell.squeeze(0),
        pbc,
        return_neighbor_list=True,
    )
    return neighbor_list, neighbor_ptr, unit_shifts


def make_virial_batch_cscl_system(size: int = 1, device: torch.device | None = None):
    """Create a 2-system batch from identical CsCl supercells.

    Returns
    -------
    tuple
        (positions, charges, cell, alpha, batch_idx,
         pos_single, q_single, cell_single, alpha_single, n_atoms)
    """
    if device is None:
        device = torch.device("cpu")
    crystal = create_cscl_supercell(size)
    n_atoms = len(crystal.charges)

    pos_single = torch.tensor(crystal.positions, dtype=VIRIAL_DTYPE, device=device)
    q_single = torch.tensor(crystal.charges, dtype=VIRIAL_DTYPE, device=device)
    cell_single = torch.tensor(
        crystal.cell, dtype=VIRIAL_DTYPE, device=device
    ).unsqueeze(0)
    alpha_single = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)

    positions = torch.cat([pos_single, pos_single], dim=0)
    charges = torch.cat([q_single, q_single], dim=0)
    cell_batch = torch.cat([cell_single, cell_single], dim=0)
    alpha = torch.tensor([0.3, 0.3], dtype=VIRIAL_DTYPE, device=device)
    batch_idx = torch.cat(
        [
            torch.zeros(n_atoms, dtype=torch.int32, device=device),
            torch.ones(n_atoms, dtype=torch.int32, device=device),
        ]
    )

    return (
        positions,
        charges,
        cell_batch,
        alpha,
        batch_idx,
        pos_single,
        q_single,
        cell_single,
        alpha_single,
        n_atoms,
    )


def make_virial_crystal_system(system_fn, size=1, device: torch.device | None = None):
    """Create crystal system from a factory function for virial tests.

    Parameters
    ----------
    system_fn : callable
        One of ``create_cscl_supercell``, ``create_wurtzite_system``, etc.
    """
    if device is None:
        device = torch.device("cpu")
    crystal = system_fn(size)
    positions = torch.tensor(crystal.positions, dtype=VIRIAL_DTYPE, device=device)
    charges = torch.tensor(crystal.charges, dtype=VIRIAL_DTYPE, device=device)
    cell = torch.tensor(crystal.cell, dtype=VIRIAL_DTYPE, device=device).unsqueeze(0)
    return positions, charges, cell


def make_non_neutral_system(device: torch.device):
    """Create a non-neutral 2-atom system (Q_total = +0.5) in a cubic cell."""
    cell = torch.tensor(
        [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
        dtype=VIRIAL_DTYPE,
        device=device,
    )
    positions = torch.tensor(
        [[2.5, 5.0, 5.0], [7.5, 5.0, 5.0]],
        dtype=VIRIAL_DTYPE,
        device=device,
    )
    charges = torch.tensor([1.0, -0.5], dtype=VIRIAL_DTYPE, device=device)
    return positions, charges, cell


def apply_strain(positions, cell, epsilon, device):
    """Apply infinitesimal strain: x' = (I + eps) @ x, cell' = (I + eps) @ cell."""
    I_plus_eps = torch.eye(3, dtype=VIRIAL_DTYPE, device=device) + epsilon
    new_positions = positions @ I_plus_eps.T
    new_cell = cell @ I_plus_eps.T
    return new_positions, new_cell


def fd_virial_component(energy_fn, positions, cell, a, b, device, h=1e-5):
    """Finite-difference virial for component (a, b).

    virial_ab = -dE/d(epsilon_ab) ≈ -[E(+h) - E(-h)] / (2h)
    """
    eps_plus = torch.zeros(3, 3, dtype=VIRIAL_DTYPE, device=device)
    eps_plus[a, b] = h
    pos_p, cell_p = apply_strain(positions, cell, eps_plus, device)
    E_plus = energy_fn(pos_p, cell_p)

    eps_minus = torch.zeros(3, 3, dtype=VIRIAL_DTYPE, device=device)
    eps_minus[a, b] = -h
    pos_m, cell_m = apply_strain(positions, cell, eps_minus, device)
    E_minus = energy_fn(pos_m, cell_m)

    return -(E_plus - E_minus) / (2.0 * h)


def fd_virial_full(energy_fn, positions, cell, device, h=1e-5):
    """Compute full 3x3 virial tensor by finite differences."""
    virial = torch.zeros(3, 3, dtype=VIRIAL_DTYPE, device=device)
    for a in range(3):
        for b in range(3):
            virial[a, b] = fd_virial_component(
                energy_fn, positions, cell, a, b, device, h
            )
    return virial


# Convenience exports
__all__ = [
    "CrystalSystem",
    "create_cscl_supercell",
    "create_wurtzite_system",
    "create_zincblende_system",
    "create_nacl_system",
    "create_simple_cubic_system",
    "VIRIAL_DTYPE",
    "make_virial_cscl_system",
    "get_virial_neighbor_data",
    "make_virial_batch_cscl_system",
    "make_virial_crystal_system",
    "make_non_neutral_system",
    "apply_strain",
    "fd_virial_component",
    "fd_virial_full",
]
