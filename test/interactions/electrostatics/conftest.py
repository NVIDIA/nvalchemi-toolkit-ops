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

"""Pytest configuration, fixtures, and crystal structure utilities for electrostatics tests.

This module provides:
- Pytest configuration hooks and markers
- Device fixtures for CPU/GPU testing
- Crystal structure generators that return numpy arrays, usable by all
  binding layers (warp, torch, jax)

Supported crystal structures:
- CsCl (BCC-like ionic crystal)
- Wurtzite (hexagonal ZnS)
- Zincblende (cubic ZnS)
- NaCl (rock salt)
- Simple cubic with alternating charges
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
import warp as wp

# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Configure pytest for electrostatics tests."""
    config.addinivalue_line("markers", "slow: marks tests as slow (performance tests)")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "warp: marks tests that require Warp")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        if "cuda" in item.name.lower() or "gpu" in item.name.lower():
            item.add_marker(pytest.mark.gpu)

        if "performance" in item.name.lower() or "stress" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# ---------------------------------------------------------------------------
# Device / warp fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return wp.is_cuda_available()


@pytest.fixture(scope="session", autouse=True)
def setup_warp():
    """Initialize Warp if available."""
    wp.init()

    if wp.is_cuda_available():
        wp.set_device("cuda:0")

    yield


@pytest.fixture(params=["cpu", "cuda:0"], ids=["cpu", "gpu"])
def device(request):
    """Fixture providing both CPU and GPU devices.

    GPU tests are skipped if CUDA is not available.

    Returns
    -------
    str
        Device name ("cpu" or "cuda:0")

    Notes
    -----
    This fixture can be used for both warp and PyTorch tests.
    For PyTorch tensors, convert "cuda:0" to "cuda" when needed.
    """
    device_name = request.param
    if device_name == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA not available")
    return device_name


# ---------------------------------------------------------------------------
# Crystal structure data container
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Crystal structure generators
# ---------------------------------------------------------------------------


def create_cscl_supercell(size: int) -> CrystalSystem:
    """Create CsCl supercell of given size.

    CsCl has a BCC-like structure with:
    - Cs at corners (0, 0, 0)
    - Cl at body center (0.5, 0.5, 0.5)

    Parameters
    ----------
    size : int
        Linear supercell size (total atoms = 2 * size^3)

    Returns
    -------
    CrystalSystem
        Crystal structure with positions, cell, charges, and symbols.

    Notes
    -----
    Base CsCl unit cell has 2 atoms with a = 4.14 Angstrom.
    A size=n supercell has 2n^3 atoms.
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
    """Create Wurtzite supercell of given size.

    Wurtzite (ZnS) has a hexagonal structure with 4 atoms per unit cell:
    - Zn at (0, 0, 0) and (1/3, 2/3, 1/2)
    - S at (0, 0, u) and (1/3, 2/3, 1/2 + u) where u ~ 3/8

    Parameters
    ----------
    size : int
        Linear supercell size (total atoms = 4 * size^3)

    Returns
    -------
    CrystalSystem
        Crystal structure with positions, cell, charges, and symbols.

    Notes
    -----
    Uses lattice parameters a = 3.21 Angstrom, c = 5.21 Angstrom (typical for ZnS).
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
    """Create Zincblende supercell of given size.

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
        Linear supercell size (total atoms = 2 * size^3)

    Returns
    -------
    CrystalSystem
        Crystal structure with positions, cell, charges, and symbols.

    Notes
    -----
    Uses conventional cubic cell with a = 5.41 Angstrom (typical for ZnS zincblende).
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
    """Create NaCl (rock salt) supercell of given size.

    NaCl has an FCC-like structure with:
    - Na at FCC positions
    - Cl at FCC positions shifted by (0.5, 0, 0)

    Parameters
    ----------
    size : int
        Linear supercell size (total atoms = 2 * size^3 for primitive cell)

    Returns
    -------
    CrystalSystem
        Crystal structure with positions, cell, charges, and symbols.

    Notes
    -----
    Uses conventional cubic cell with a = 5.64 Angstrom.
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
    """Create simple cubic lattice with alternating charges.

    Parameters
    ----------
    size : int
        Linear supercell size (total atoms = size^3)
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


# ---------------------------------------------------------------------------
# Crystal structure fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=["cscl", "wurtzite", "zincblende"],
    ids=["cscl", "wurtzite", "zincblende"],
)
def crystal_system_fn(request):
    """Fixture providing crystal structure generator functions.

    Returns a callable that takes a ``size`` parameter and returns a
    `CrystalSystem` NamedTuple with numpy arrays.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object with crystal type parameter.

    Returns
    -------
    callable
        Crystal structure generator function that accepts a ``size`` int.

    Notes
    -----
    Parametrized over the three main crystal types used in Ewald/PME tests.
    Tests using this fixture will run once per crystal type.  The returned
    function produces numpy arrays; each test is responsible for converting
    to its framework (torch, jax, etc.).
    """
    generators = {
        "cscl": create_cscl_supercell,
        "wurtzite": create_wurtzite_system,
        "zincblende": create_zincblende_system,
    }
    return generators[request.param]


@pytest.fixture()
def crystal_generators():
    """Fixture providing all crystal structure generator functions as a dict.

    Returns
    -------
    dict[str, callable]
        Mapping from crystal type name to generator function.

    Notes
    -----
    Available crystal types:

    - ``"cscl"``: CsCl supercell (BCC-like structure)
    - ``"wurtzite"``: Wurtzite structure (hexagonal ZnS)
    - ``"zincblende"``: Zincblende structure (cubic ZnS)
    - ``"nacl"``: NaCl (rocksalt) structure
    - ``"simple_cubic"``: Simple cubic lattice with customizable parameters

    Each generator returns numpy arrays; the test is responsible for
    framework conversion.

    Examples
    --------
    >>> def test_example(crystal_generators):
    ...     system = crystal_generators["cscl"](size=2)
    ...     positions = system.positions  # numpy array
    """
    return {
        "cscl": create_cscl_supercell,
        "wurtzite": create_wurtzite_system,
        "zincblende": create_zincblende_system,
        "nacl": create_nacl_system,
        "simple_cubic": create_simple_cubic_system,
    }
