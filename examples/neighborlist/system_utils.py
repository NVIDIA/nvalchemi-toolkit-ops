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
import numpy as np
from pymatgen.core import Lattice, Structure

# ==============================================================================
# Pymatgen Structure Utilities
# ==============================================================================


def create_bulk_structure(
    symbol: str, crystal_type: str, a: float, cubic: bool = False
) -> Structure:
    """Create a bulk crystal structure using pymatgen.

    Creates standard crystal structures with common lattice types.

    Parameters
    ----------
    symbol : str
        Chemical symbol of the element (e.g., "Al", "Fe").
    crystal_type : str
        Crystal structure type. Supported: "fcc", "bcc", "sc" (simple cubic).
    a : float
        Lattice constant in Angstroms.
    cubic : bool, default=False
        If True, create a cubic supercell for non-cubic structures.

    Returns
    -------
    Structure
        pymatgen Structure object representing the bulk crystal.

    Examples
    --------
    >>> fcc_al = create_bulk_structure("Al", "fcc", a=4.05)
    >>> bcc_fe = create_bulk_structure("Fe", "bcc", a=2.87, cubic=True)
    """
    lattice = Lattice.cubic(a)

    if crystal_type.lower() == "fcc":
        # Face-centered cubic: atoms at corners and face centers
        coords = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
        )
        species = [symbol] * 4
    elif crystal_type.lower() == "bcc":
        # Body-centered cubic: atoms at corners and body center
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        species = [symbol] * 2
    elif crystal_type.lower() in ["sc", "simple_cubic"]:
        # Simple cubic: atom at corner only
        coords = np.array([[0.0, 0.0, 0.0]])
        species = [symbol]
    else:
        raise ValueError(
            f"Unsupported crystal type: {crystal_type}. "
            "Supported types: 'fcc', 'bcc', 'sc' (simple cubic)"
        )

    structure = Structure(lattice, species, coords, coords_are_cartesian=False)

    return structure


def create_molecule_structure(name: str, box_size: float = 10.0) -> Structure:
    """Create a simple molecular structure with predefined coordinates.

    Provides common small molecules with approximate equilibrium geometries.

    Parameters
    ----------
    name : str
        Name of the molecule. Supported: "H2O", "CO2", "CH4".
    box_size : float, default=10.0
        Size of the cubic box in Angstroms (used for periodic boundary conditions).

    Returns
    -------
    Structure
        pymatgen Structure object with the molecule centered in a box.

    Raises
    ------
    ValueError
        If the molecule name is not supported.

    Notes
    -----
    Coordinates are approximate equilibrium geometries and not optimized.
    The molecules are placed in a cubic box with periodic boundary conditions.

    Examples
    --------
    >>> water = create_molecule_structure("H2O", box_size=15.0)
    >>> co2 = create_molecule_structure("CO2")
    """
    # Define molecule coordinates (Cartesian, in Angstroms)
    # Centered around origin, will be shifted to box center
    molecules = {
        "H2O": {
            "species": ["O", "H", "H"],
            "coords": np.array(
                [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
            ),
        },
        "CO2": {
            "species": ["C", "O", "O"],
            "coords": np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]),
        },
        "CH4": {
            "species": ["C", "H", "H", "H", "H"],
            "coords": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.629, 0.629, 0.629],
                    [-0.629, -0.629, 0.629],
                    [-0.629, 0.629, -0.629],
                    [0.629, -0.629, -0.629],
                ]
            ),
        },
    }

    if name not in molecules:
        raise ValueError(
            f"Unsupported molecule: {name}. "
            f"Supported molecules: {list(molecules.keys())}"
        )

    mol_data = molecules[name]
    species = mol_data["species"]
    coords = mol_data["coords"]

    # Center the molecule in the box
    coords_centered = coords + box_size / 2.0

    # Create cubic lattice
    lattice = Lattice.cubic(box_size)

    # Create structure with Cartesian coordinates
    structure = Structure(lattice, species, coords_centered, coords_are_cartesian=True)

    return structure
