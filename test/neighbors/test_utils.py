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

"""Utility functions for neighbor list tests."""

from importlib import import_module

import numpy as np
import pytest
import torch

try:
    primitive_neighbor_list = getattr(import_module("vesin"), "NeighborList", None)
    run_primitive_nl = True
except ModuleNotFoundError:
    primitive_neighbor_list = None
    run_primitive_nl = False


def create_simple_cubic_system(
    num_atoms: int = 8,
    cell_size: float = 2.0,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a simple cubic system for testing.

    Parameters
    ----------
    num_atoms : int
        Number of atoms. Should be a perfect cube.
    cell_size : float
        Size of the unit cell
    dtype : torch.dtype
        Data type for positions and cell
    device : str
        Device to place tensors on

    Returns
    -------
    positions : torch.Tensor
        Atomic coordinates (num_atoms, 3)
    cell : torch.Tensor
        Unit cell matrix (3, 3)
    pbc : torch.Tensor
        Periodic boundary conditions (3,)
    """
    # Create cubic lattice points
    n_side = int(round(num_atoms ** (1 / 3)))
    if n_side**3 != num_atoms:
        n_side = int(np.ceil(num_atoms ** (1 / 3)))

    # Generate grid coordinates
    coords = []
    spacing = cell_size / n_side
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                if len(coords) < num_atoms:
                    coords.append([i * spacing, j * spacing, k * spacing])

    positions = torch.tensor(coords[:num_atoms], dtype=dtype, device=device)

    # Create unit cell matrix
    cell = (torch.eye(3, dtype=dtype, device=device) * cell_size).reshape(1, 3, 3)

    # All periodic boundary conditions
    pbc = torch.tensor([True, True, True], device=device).reshape(1, 3)

    return positions, cell, pbc


def create_random_system(
    num_atoms: int = 50,
    cell_size: float = 5.0,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    seed: int | None = 42,
    pbc_flag: bool | list[bool] = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a random system for testing.

    Parameters
    ----------
    num_atoms : int
        Number of atoms
    cell_size : float
        Size of the unit cell
    dtype : torch.dtype
        Data type for coordinates and cell
    device : str
        Device to place tensors on
    seed : int, optional
        Random seed for reproducibility
    pbc : bool
        Periodic boundary conditions
    Returns
    -------
    positions : torch.Tensor
        Atomic coordinates (num_atoms, 3)
    cell : torch.Tensor
        Unit cell matrix (3, 3)
    pbc : torch.Tensor
        Periodic boundary conditions (3,)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Random positions within the cell
    positions = torch.rand(num_atoms, 3, dtype=dtype, device=device) * cell_size

    # Create unit cell matrix
    cell = (torch.eye(3, dtype=dtype, device=device) * cell_size).reshape(1, 3, 3)

    # All periodic boundary conditions
    pbc = torch.tensor(
        pbc_flag if isinstance(pbc_flag, list) else [pbc_flag, pbc_flag, pbc_flag],
        device=device,
    ).reshape(1, 3)

    return positions, cell, pbc


def create_nonorthorhombic_system(
    num_atoms: int = 50,
    a: float = 5.0,
    b: float = 5.0,
    c: float = 5.0,
    alpha: float = 60.0,
    beta: float = 60.0,
    gamma: float = 60.0,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    seed: int | None = 42,
    pbc_flag: bool | list[bool] = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a system with non-orthorhombic cell for testing.

    The cell is defined by lattice parameters (a, b, c, alpha, beta, gamma)
    and converted to a cell matrix following standard crystallographic conventions.

    Parameters
    ----------
    num_atoms : int
        Number of atoms
    a, b, c : float
        Lattice parameters (cell edge lengths)
    alpha, beta, gamma : float
        Lattice angles in degrees (alpha = angle between b and c,
        beta = angle between a and c, gamma = angle between a and b)
    dtype : torch.dtype
        Data type for coordinates and cell
    device : str
        Device to place tensors on
    seed : int, optional
        Random seed for reproducibility
    pbc_flag : bool or list[bool]
        Periodic boundary conditions

    Returns
    -------
    positions : torch.Tensor
        Atomic coordinates (num_atoms, 3)
    cell : torch.Tensor
        Unit cell matrix (1, 3, 3)
    pbc : torch.Tensor
        Periodic boundary conditions (1, 3)

    Notes
    -----
    Common non-orthorhombic cells:
    - Triclinic: all parameters different
    - Monoclinic: alpha = gamma = 90, beta != 90
    - Hexagonal: a = b, alpha = beta = 90, gamma = 120
    - Rhombohedral: a = b = c, alpha = beta = gamma != 90
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Convert angles to radians
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)

    # Construct cell matrix following standard crystallographic convention
    # a-axis along x, b-axis in xy plane, c-axis positioned by angles
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)

    # Cell matrix columns are lattice vectors
    # a = [ax, 0, 0]
    # b = [bx, by, 0]
    # c = [cx, cy, cz]

    ax = a
    ay = 0.0
    az = 0.0

    bx = b * cos_gamma
    by = b * sin_gamma
    bz = 0.0

    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = (
        c
        * np.sqrt(
            1.0
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
            + 2.0 * cos_alpha * cos_beta * cos_gamma
        )
        / sin_gamma
    )

    # Create cell matrix (row vectors are lattice vectors)
    cell_np = np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]], dtype=np.float64)

    cell = torch.tensor(cell_np, dtype=dtype, device=device).reshape(1, 3, 3)

    # Generate random fractional coordinates [0, 1) in each direction
    frac_coords = torch.rand(num_atoms, 3, dtype=dtype, device=device)

    # Convert fractional to Cartesian coordinates
    # positions = frac_coords @ cell_matrix
    positions = torch.matmul(frac_coords, cell.squeeze(0))

    # Set periodic boundary conditions
    pbc = torch.tensor(
        pbc_flag if isinstance(pbc_flag, list) else [pbc_flag, pbc_flag, pbc_flag],
        device=device,
    ).reshape(1, 3)

    return positions, cell, pbc


def create_structure_HoTlPd(dtype, device):
    """Return the structure of HoTlPd."""
    positions = torch.tensor(
        [
            [4.64882481e00, 0.00000000e00, 1.87730266e00],
            [1.56295308e00, 2.70711418e00, 1.87730266e00],
            [-2.32441241e00, 4.02600045e00, 1.87730266e00],
            [2.08725046e00, 0.00000000e00, 0.00000000e00],
            [2.84374025e00, 4.92550268e00, 0.00000000e00],
            [-1.04362523e00, 1.80761195e00, 0.00000000e00],
            [-3.88994531e-06, 4.48874533e00, 0.00000000e00],
            [3.88736937e00, 2.24436930e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 1.87730266e00],
        ],
        dtype=dtype,
        device=device,
    )
    cell = torch.tensor(
        [
            [7.77473097, 0.0, 0.0],
            [-3.88736549, 6.73311463, 0.0],
            [0.0, 0.0, 3.75460533],
        ],
        dtype=dtype,
        device=device,
    )
    pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
    return positions, cell, pbc


def create_structure_SiCu(dtype, device):
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    cell = torch.tensor(
        [
            [0.0, 3.0, 3.0],
            [3.0, 0.0, 3.0],
            [3.0, 3.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
    return positions, cell, pbc


def create_batch_systems(
    num_systems: int = 3,
    atoms_per_system: list = None,
    cell_sizes: list = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    seed: int | None = 42,
    pbc_flag: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create multiple systems for batch testing.

    Parameters
    ----------
    num_systems : int
        Number of systems in the batch
    atoms_per_system : list, optional
        Number of atoms in each system
    cell_sizes : list, optional
        Cell size for each system
    dtype : torch.dtype
        Data type for coordinates and cell
    device : str
        Device to place tensors on
    seed : int, optional
        Random seed for reproducibility
    pbc_flag : bool
        Periodic boundary conditions
    Returns
    -------
    positions : torch.Tensor
        Concatenated atomic positions (total_atoms, 3)
    cell : torch.Tensor
        Cell matrices for each system (num_systems, 3, 3)
    pbc : torch.Tensor
        PBC for each system (num_systems, 3)
    ptr : torch.Tensor
        Pointer to start of each system (num_systems + 1,)
    """
    if atoms_per_system is None:
        atoms_per_system = [10, 15, 12][:num_systems]
    if cell_sizes is None:
        cell_sizes = [3.0, 4.0, 3.5][:num_systems]

    if seed is not None:
        torch.manual_seed(seed)

    coords = []
    cells = []
    pbcs = []
    ptr = [0]

    for i in range(num_systems):
        positions, cell, pbc = create_random_system(
            atoms_per_system[i],
            cell_sizes[i],
            dtype,
            device,
            seed=(seed + i) if seed else None,
            pbc_flag=pbc_flag,
        )
        coords.append(positions)
        cells.append(cell)
        pbcs.append(pbc)
        ptr.append(ptr[-1] + atoms_per_system[i])

    # Concatenate positions
    positions_batch = torch.cat(coords, dim=0)

    # Stack cells and pbc
    cell_batch = torch.stack(cells, dim=0).reshape(num_systems, 3, 3)
    pbc_batch = torch.stack(pbcs, dim=0).reshape(num_systems, 3)

    # Convert ptr to tensor
    ptr_tensor = torch.tensor(ptr, dtype=torch.int64, device=device)

    return positions_batch, cell_batch, pbc_batch, ptr_tensor


@pytest.mark.skipif(not run_primitive_nl, reason="Consistency check needs `vesin`.")
def brute_force_neighbors(
    positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, cutoff: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Brute force neighbor calculation for reference.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic coordinates (num_atoms, 3)
    cell : torch.Tensor
        Unit cell matrix (3, 3)
    pbc : torch.Tensor
        Periodic boundary conditions (3,)
    cutoff : float
        Cutoff distance

    Returns
    -------
    i : torch.Tensor
        Source atom indices
    j : torch.Tensor
        Target atom indices
    u : torch.Tensor
        Unit cell shift indices (int)
    S : torch.Tensor
        Cartesian shifts
    """
    positions_dtype = positions.dtype
    device = positions.device
    positions = positions.cpu().numpy()
    if cell is not None:
        cell = cell.squeeze().cpu().numpy()
    else:
        cell = np.eye(3)
    if pbc is not None:
        pbc = pbc.squeeze().cpu().numpy()
    else:
        pbc = np.array([False, False, False])

    calculator = primitive_neighbor_list(cutoff=cutoff, full_list=True, sorted=True)
    i, j, u = calculator.compute(
        points=positions, box=cell, periodic=pbc, quantities="ijS"
    )
    S = u.dot(cell)
    return (
        torch.as_tensor(i, dtype=torch.int32, device=device),
        torch.as_tensor(j, dtype=torch.int32, device=device),
        torch.as_tensor(u, dtype=torch.int32, device=device),
        torch.as_tensor(S, dtype=positions_dtype, device=device),
    )


def count_neighbors_reference(
    positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Count neighbors using brute force for reference.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic coordinates (num_atoms, 3)
    cell : torch.Tensor
        Unit cell matrix (3, 3)
    pbc : torch.Tensor
        Periodic boundary conditions (3,)
    cutoff : float
        Cutoff distance

    Returns
    -------
    nneigh : torch.Tensor
        Number of neighbors for each atom
    """
    i, _, _, _ = brute_force_neighbors(positions, cell, pbc, cutoff)

    num_atoms = positions.shape[0]
    nneigh = torch.zeros(num_atoms, dtype=torch.int32, device=positions.device)

    if len(i) > 0:
        # Count neighbors for each atom
        for atom_i in range(num_atoms):
            nneigh[atom_i] = torch.sum(i == atom_i).item()

    return nneigh


def assert_neighbor_lists_equal(
    result1: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    result2: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Assert that two neighbor lists are equivalent.

    Parameters
    ----------
    result1, result2 : tuple
        Neighbor list results (i, j, u)
    """
    i1, j1, u1 = result1
    i2, j2, u2 = result2

    # Check lengths match
    assert len(i1) == len(i2), f"Neighbor list lengths differ: {len(i1)} vs {len(i2)}"

    if len(i1) == 0:
        return  # Both empty, that's fine

    # Sort both lists by (i, j, u) for comparison
    def sort_key(i, j, u):
        return np.lexsort(
            [u[:, 2].cpu(), u[:, 1].cpu(), u[:, 0].cpu(), j.cpu(), i.cpu()]
        )

    idx1 = sort_key(i1, j1, u1)
    idx2 = sort_key(i2, j2, u2)

    # Compare sorted lists
    torch.testing.assert_close(i1[idx1], i2[idx2])
    torch.testing.assert_close(j1[idx1], j2[idx2])
    torch.testing.assert_close(u1[idx1], u2[idx2])


def assert_neighbor_matrix_equal(
    result1: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    result2: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Assert that two neighbor matrices are equivalent.

    Parameters
    ----------
    result1, result2 : tuple
        Neighbor matrix results (neighbor_matrix, num_neighbors, shifts - optional)
    """
    if len(result1) == 2:
        neighbor_matrix1, num_neighbors1 = result1
        neighbor_matrix2, num_neighbors2 = result2
        shifts1 = None
        shifts2 = None
    elif len(result1) == 3:
        neighbor_matrix1, num_neighbors1, shifts1 = result1
        neighbor_matrix2, num_neighbors2, shifts2 = result2
    else:
        raise ValueError(f"Invalid result length: {len(result1)}")

    # Check lengths match
    assert neighbor_matrix1.shape == neighbor_matrix2.shape, (
        f"Neighbor matrix shapes differ: {neighbor_matrix1.shape} vs {neighbor_matrix2.shape}"
    )
    assert num_neighbors1.shape == num_neighbors2.shape, (
        f"Number of neighbors shapes differ: {num_neighbors1.shape} vs {num_neighbors2.shape}"
    )
    if shifts1 is not None:
        assert shifts1.shape == shifts2.shape, (
            f"Shifts shapes differ: {shifts1.shape} vs {shifts2.shape}"
        )

    # Compare neighbor matrices
    torch.testing.assert_close(num_neighbors1, num_neighbors2)
    # test neighbor matricies by row
    for i in range(neighbor_matrix1.shape[0]):
        # sort the rows
        row1 = neighbor_matrix1[i]
        row1_sorted, indices1 = torch.sort(row1, dim=0)
        row2 = neighbor_matrix2[i]
        row2_sorted, indices2 = torch.sort(row2, dim=0)
        assert torch.equal(row1_sorted, row2_sorted), f"Row {i} mismatch"

        if shifts1 is not None:
            shifts1_sorted = shifts1[i][indices1]
            shifts2_sorted = shifts2[i][indices2]
            assert torch.equal(shifts1_sorted, shifts2_sorted), (
                f"Row {i} shifts mismatch"
            )
