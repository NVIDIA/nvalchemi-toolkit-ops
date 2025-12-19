# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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


# Convenience exports
__all__ = [
    "CrystalSystem",
    "create_cscl_supercell",
    "create_wurtzite_system",
    "create_zincblende_system",
    "create_nacl_system",
    "create_simple_cubic_system",
]
