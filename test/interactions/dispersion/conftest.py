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
Pytest fixtures and utilities for DFT-D3 kernel testing.

This module contains:
- Dummy parameter tables for testing (NOT physically accurate)
- Test system geometries (H2, CH4, edge cases)
- Pytest fixtures for devices and systems
- Utility functions for array conversion and allocation
"""

from __future__ import annotations

import pytest
import torch
import warp as wp

from nvalchemiops.interactions.dispersion.dftd3 import D3Parameters

# ==============================================================================
# Parameter Tables (Dummy values for testing only)
# ==============================================================================


def get_element_tables(z_max: int = 17) -> dict:
    """
    Get dummy element parameter tables for testing.

    These are NOT physically accurate parameters - they're made-up values designed
    for numerical stability and testing purposes only. Do not use for production.

    Parameters
    ----------
    z_max : int
        Maximum atomic number to include (default: 17 for Cl)

    Returns
    -------
    dict
        Dictionary with keys:
        - rcov: Covalent radii [Zmax+1] in Bohr
        - r4r2: <r⁴>/<r²> expectation values [z_max+1]
        - c6ref: C6 reference values [(z_max+1)*(z_max+1)*25] flattened
        - cnref_i: CN reference for atom i [(z_max+1)*(z_max+1)*25] flattened
        - cnref_j: CN reference for atom j [(z_max+1)*(z_max+1)*25] flattened
        - z_max_inc: z_max + 1
    """
    # incremented maximum atomic number
    z_max_inc = z_max + 1

    # Covalent radii in Bohr (made up but reasonable scale)
    # Index by atomic number: [0, H, He, ..., Ne, ..., Cl]
    rcov = torch.zeros(z_max_inc, dtype=torch.float32)
    rcov[0:10] = torch.tensor(
        [
            0.0,  # Z=0 (padding)  # NOSONAR (S125) "chemical formula"
            0.6,  # H  (Z=1)
            0.8,  # He (Z=2)
            2.8,  # Li (Z=3)
            2.0,  # Be (Z=4)
            1.6,  # B  (Z=5)
            1.4,  # C  (Z=6)
            1.3,  # N  (Z=7)
            1.2,  # O  (Z=8)
            1.5,  # F  (Z=9)
        ],
        dtype=torch.float32,
    )
    rcov[10] = 1.5  # Ne (Z=10)
    rcov[17] = 1.8  # Cl (Z=17)

    # Maximum coordination numbers (physically motivated but simplified)
    cnmax = torch.zeros(z_max_inc, dtype=torch.float32)
    cnmax[0:10] = torch.tensor(
        [
            0.0,  # Z=0  # NOSONAR (S125) "chemical formula"
            1.5,  # H
            1.0,  # He
            6.0,  # Li
            4.0,  # Be
            4.0,  # B
            4.0,  # C
            4.0,  # N
            2.5,  # O
            1.5,  # F
        ],
        dtype=torch.float32,
    )
    cnmax[10] = 1.0  # Ne (noble gas, essentially 0 but avoid division issues)
    cnmax[17] = 2.0  # Cl

    # <r⁴>/<r²> expectation values (made up, positive values)
    # Larger for more polarizable species
    r4r2 = torch.zeros(z_max_inc, dtype=torch.float32)
    r4r2[0:10] = torch.tensor(
        [
            0.0,  # Z=0  # NOSONAR (S125) "chemical formula"
            2.0,  # H
            1.5,  # He
            10.0,  # Li
            6.0,  # Be
            5.0,  # B
            4.5,  # C
            4.0,  # N
            3.5,  # O
            3.0,  # F
        ],
        dtype=torch.float32,
    )
    r4r2[10] = 4.5  # Ne (moderately polarizable)
    r4r2[17] = 8.0  # Cl (more polarizable)

    # C6 reference grid: 5x5 grid for each element pair
    # Total size: z_max_inc * z_max_inc * 25
    # We'll fill with simple positive values scaled by atomic numbers
    c6ref = torch.zeros(z_max_inc * z_max_inc * 25, dtype=torch.float32)
    cnref_i = torch.zeros(z_max_inc * z_max_inc * 25, dtype=torch.float32)
    cnref_j = torch.zeros(z_max_inc * z_max_inc * 25, dtype=torch.float32)

    # Fill C6 and CN reference grids
    for zi in range(z_max_inc):
        for zj in range(z_max_inc):
            base = (zi * z_max_inc + zj) * 25
            for p in range(5):
                for q in range(5):
                    idx = base + p * 5 + q
                    # CN reference grid: evenly spaced from 0 to cnmax
                    if zi > 0:
                        cnref_i[idx] = (p / 4.0) * cnmax[zi]
                    if zj > 0:
                        cnref_j[idx] = (q / 4.0) * cnmax[zj]

                    # C6 values: scale with zi*zj and vary with CN grid point
                    # Use a simple formula that gives positive, reasonable values
                    if zi > 0 and zj > 0:
                        c6ref[idx] = 10.0 * float(zi * zj) * (1.0 + 0.1 * p + 0.1 * q)

    result = {
        "rcov": rcov,
        "r4r2": r4r2,
        "c6ref": c6ref,
        "cnref_i": cnref_i,
        "cnref_j": cnref_j,
        "z_max_inc": z_max_inc,
    }
    return result


def make_d3_parameters(element_tables: dict) -> D3Parameters:
    """
    Create a D3Parameters instance from element_tables dictionary.

    Parameters
    ----------
    element_tables : dict
        Dictionary from get_element_tables() containing test parameters

    Returns
    -------
    D3Parameters
        Validated D3Parameters instance for testing
    """
    max_z_inc = element_tables["z_max_inc"]
    c6_reference = element_tables["c6ref"].reshape(max_z_inc, max_z_inc, 5, 5)
    coord_num_ref = element_tables["cnref_i"].reshape(max_z_inc, max_z_inc, 5, 5)

    return D3Parameters(
        rcov=element_tables["rcov"],
        r4r2=element_tables["r4r2"],
        c6ab=c6_reference,
        cn_ref=coord_num_ref,
    )


def get_functional_params() -> dict:
    """
    Get dummy functional parameters (PBE-like) for testing.

    Returns
    -------
    dict
        Dictionary with DFT-D3 functional parameters:
        - a1, a2: BJ damping parameters
        - s6, s8: Scaling factors
        - k1: CN counting steepness
        - k3: C6 interpolation steepness
    """
    return {
        "a1": 0.4,
        "a2": 4.0,  # Bohr
        "s6": 1.0,
        "s8": 0.8,
        "k1": 16.0,  # Bohr^-1
        "k3": -4.0,
    }


# ==============================================================================
# Test System Geometries
# ==============================================================================


def get_h2_system(separation: float = 1.4) -> dict:
    """
    Get simple H2 molecule geometry for testing.

    Parameters
    ----------
    separation : float
        H-H distance in Bohr (default: 1.4)

    Returns
    -------
    dict
        Dictionary with:
        - coord: [6] flattened coordinates in Bohr
        - numbers: [2] atomic numbers
        - nbmat: [2, 5] neighbor matrix 2D array
        - B: number of atoms (2)
        - M: max neighbors (5)
    """
    # H2 molecule along x-axis
    coord = torch.tensor(
        [
            0.0,
            0.0,
            0.0,  # H1 at origin
            separation,
            0.0,
            0.0,  # H2 at (r, 0, 0)
        ],
        dtype=torch.float32,
    )

    numbers = torch.tensor([1, 1], dtype=torch.int32)  # Both hydrogen

    # Neighbor matrix: each atom has the other as neighbor, rest padding
    # For atom 0: neighbor is atom 1, then padding (use B=2 as sentinel)
    # For atom 1: neighbor is atom 0, then padding
    B, M = 2, 5
    nbmat = torch.tensor(
        [
            [1, 2, 2, 2, 2],  # Atom 0's neighbors: [1, padding, padding, ...]
            [0, 2, 2, 2, 2],  # Atom 1's neighbors: [0, padding, padding, ...]
        ],
        dtype=torch.int32,
    )

    return {
        "coord": coord,
        "numbers": numbers,
        "nbmat": nbmat,
        "B": B,
        "M": M,
    }


def get_ch4_like_system() -> dict:
    """
    Get simple CH4-like molecule geometry for testing.

    Returns
    -------
    dict
        Dictionary with:
        - coord: [15] flattened coordinates in Bohr
        - numbers: [5] atomic numbers (C + 4H)
        - nbmat: [5, 10] neighbor matrix 2D array
        - B: number of atoms (5)
        - M: max neighbors (10)
    """
    # Simplified CH4: C at origin, 4 H in tetrahedral-ish positions
    r_CH = 2.0  # C-H distance in Bohr # NOSONAR (S117) "chemical formula"
    coord = torch.tensor(
        [
            0.0,
            0.0,
            0.0,  # C at origin
            r_CH,
            0.0,
            0.0,  # H1
            -r_CH,
            0.0,
            0.0,  # H2
            0.0,
            r_CH,
            0.0,  # H3
            0.0,
            -r_CH,
            0.0,  # H4
        ],
        dtype=torch.float32,
    )

    numbers = torch.tensor([6, 1, 1, 1, 1], dtype=torch.int32)  # C + 4H

    # Neighbor matrix: C has 4 H neighbors, each H has C as neighbor
    B, M = 5, 10
    nbmat = torch.full((B, M), B, dtype=torch.int32)  # Fill with padding (B=5)

    # C (atom 0) has neighbors 1,2,3,4
    nbmat[0, 0:4] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)

    # Each H has C (atom 0) as neighbor
    for i in range(1, 5):
        nbmat[i, 0] = 0

    return {
        "coord": coord,
        "numbers": numbers,
        "nbmat": nbmat,
        "B": B,
        "M": M,
    }


def get_single_atom_system() -> dict:
    """
    Get single atom system (edge case for testing).

    Returns
    -------
    dict
        Dictionary with single H atom, no neighbors
    """
    coord = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    numbers = torch.tensor([1], dtype=torch.int32)
    B, M = 1, 5
    nbmat = torch.full((B, M), B, dtype=torch.int32)  # All padding

    return {
        "coord": coord,
        "numbers": numbers,
        "nbmat": nbmat,
        "B": B,
        "M": M,
    }


def get_empty_neighbors_system() -> dict:
    """
    Get system with atoms but no neighbors (all padding).

    Returns
    -------
    dict
        Dictionary with 3 atoms but no neighbors
    """
    coord = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,  # Far away
            20.0,
            0.0,
            0.0,  # Even farther
        ],
        dtype=torch.float32,
    )

    numbers = torch.tensor([1, 1, 1], dtype=torch.int32)
    B, M = 3, 5
    nbmat = torch.full((B, M), B, dtype=torch.int32)  # All padding

    return {
        "coord": coord,
        "numbers": numbers,
        "nbmat": nbmat,
        "B": B,
        "M": M,
    }


def get_ne2_system(separation: float = 5.8) -> dict:
    """
    Get Ne2 dimer for testing dispersion in noble gases.

    Noble gases are ideal for testing dispersion since their interactions
    are purely dispersive (no covalent bonding, minimal electrostatics).

    Parameters
    ----------
    separation : float
        Ne-Ne distance in Bohr (default: 5.8, near equilibrium)

    Returns
    -------
    dict
        Dictionary with Ne2 dimer geometry
    """
    # Ne2 along x-axis
    coord = torch.tensor(
        [
            0.0,
            0.0,
            0.0,  # Ne1 at origin
            separation,
            0.0,
            0.0,  # Ne2
        ],
        dtype=torch.float32,
    )

    numbers = torch.tensor([10, 10], dtype=torch.int32)  # Both neon

    # Neighbor matrix
    B, M = 2, 5
    nbmat = torch.tensor(
        [
            [1, 2, 2, 2, 2],  # Atom 0's neighbors: [1, padding, ...]
            [0, 2, 2, 2, 2],  # Atom 1's neighbors: [0, padding, ...]
        ],
        dtype=torch.int32,
    )

    return {
        "coord": coord,
        "numbers": numbers,
        "nbmat": nbmat,
        "B": B,
        "M": M,
    }


def get_hcl_dimer_system() -> dict:
    """
    Get HCl dimer for testing realistic molecular dispersion.

    HCl dimer tests heteronuclear dispersion with different element types
    and more realistic polarizabilities. Configuration is parallel displaced.

    Returns
    -------
    dict
        Dictionary with HCl dimer geometry (4 atoms total)
    """
    # HCl bond length ~2.4 Bohr, dimer separation ~7 Bohr
    # Parallel configuration:
    # HCl molecule 1: H at origin, Cl along +x
    # HCl molecule 2: H at (0, 7, 0), Cl along +x

    r_HCl = 2.4  # H-Cl bond length in Bohr # NOSONAR (S117) "chemical formula"
    sep = 7.0  # Separation between molecules

    coord = torch.tensor(
        [
            # Molecule 1
            0.0,
            0.0,
            0.0,  # H1
            r_HCl,
            0.0,
            0.0,  # Cl1
            # Molecule 2
            0.0,
            sep,
            0.0,  # H2
            r_HCl,
            sep,
            0.0,  # Cl2
        ],
        dtype=torch.float32,
    )

    numbers = torch.tensor([1, 17, 1, 17], dtype=torch.int32)  # H, Cl, H, Cl

    # Neighbor matrix: each atom sees all others as potential neighbors
    B, M = 4, 5
    nbmat = torch.full((B, M), B, dtype=torch.int32)  # Fill with padding

    # Atom 0 (H1): neighbors are Cl1, H2, Cl2
    nbmat[0, 0:3] = torch.tensor([1, 2, 3], dtype=torch.int32)
    # Atom 1 (Cl1): neighbors are H1, H2, Cl2
    nbmat[1, 0:3] = torch.tensor([0, 2, 3], dtype=torch.int32)
    # Atom 2 (H2): neighbors are H1, Cl1, Cl2
    nbmat[2, 0:3] = torch.tensor([0, 1, 3], dtype=torch.int32)
    # Atom 3 (Cl2): neighbors are H1, Cl1, H2
    nbmat[3, 0:3] = torch.tensor([0, 1, 2], dtype=torch.int32)

    return {
        "coord": coord,
        "numbers": numbers,
        "nbmat": nbmat,
        "B": B,
        "M": M,
    }


# ==============================================================================
# Pytest Fixtures
# ==============================================================================


@pytest.fixture
def device_cpu():
    """Fixture for CPU-only tests."""
    return "cpu"


@pytest.fixture
def device_gpu():
    """Fixture for GPU-only tests (skip if unavailable)."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA not available")
    return "cuda:0"


@pytest.fixture(params=["cpu", "cuda:0"], ids=["cpu", "gpu"])
def device(request):
    """
    Fixture providing both CPU and GPU devices.

    GPU tests are skipped if CUDA is not available.

    Returns
    -------
    str
        Device name ("cpu" or "cuda:0")
    """
    device_name = request.param
    if device_name == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA not available")
    return device_name


@pytest.fixture(
    params=[
        pytest.param(
            (wp.float16, wp.vec3h, "float16"),
            marks=pytest.mark.xfail(
                reason="float16 has severe numerical instability for DFT-D3 calculations, "
                "producing NaN values in intermediate results due to limited precision "
                "(~3 decimal digits) in exponential and division operations"
            ),
        ),
        (wp.float32, wp.vec3f, "float32"),
        (wp.float64, wp.vec3d, "float64"),
    ],
    ids=["float16", "float32", "float64"],
)
def precision(request):
    """
    Fixture providing (scalar_dtype, vec_dtype, name) for different precisions.

    Returns
    -------
    tuple
        (scalar_dtype, vec_dtype, precision_name)

    Notes
    -----
    float16 tests are marked as expected to fail due to severe numerical instability
    in the dispersion calculations. The limited precision of float16 (~3 decimal digits)
    is insufficient for the exponential and division operations in the DFT-D3
    algorithm, leading to NaN values in intermediate calculations.
    """
    return request.param


@pytest.fixture
def element_tables():
    """Fixture providing dummy element parameter tables (up to Cl, Z=17)."""
    return get_element_tables(z_max=17)


@pytest.fixture
def functional_params():
    """Fixture providing dummy functional parameters."""
    return get_functional_params()


@pytest.fixture
def d3_parameters(element_tables):
    """Fixture providing D3Parameters instance from element_tables."""
    return make_d3_parameters(element_tables)


@pytest.fixture
def h2_system():
    """Fixture providing H2 molecule geometry."""
    return get_h2_system(separation=1.4)


@pytest.fixture
def h2_close():
    """Fixture providing H2 with very small separation (edge case)."""
    return get_h2_system(separation=0.1)


@pytest.fixture
def ch4_like_system():
    """Fixture providing CH4-like molecule geometry."""
    return get_ch4_like_system()


@pytest.fixture
def single_atom_system():
    """Fixture providing single atom system (edge case)."""
    return get_single_atom_system()


@pytest.fixture
def empty_neighbors_system():
    """Fixture providing system with no neighbors."""
    return get_empty_neighbors_system()


@pytest.fixture
def ne2_system():
    """Fixture providing Ne2 dimer (noble gas dispersion test)."""
    return get_ne2_system(separation=5.8)


@pytest.fixture
def hcl_dimer_system():
    """Fixture providing HCl dimer (realistic molecular dispersion test)."""
    return get_hcl_dimer_system()


# ==============================================================================
# Reference Output Fixtures
# ==============================================================================


@pytest.fixture
def ne2_reference_cpu():
    """Reference outputs for Ne2 system on CPU."""
    return {
        "cn": torch.tensor([4.4183229329e-04, 4.4183229329e-04], dtype=torch.float32),
        "inv_r": torch.tensor(
            [
                [1.7241378129e-01, 0.0, 0.0, 0.0, 0.0],
                [1.7241378129e-01, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "df_dr": torch.tensor(
            [
                [6.3015980413e-04, 0.0, 0.0, 0.0, 0.0],
                [6.3015980413e-04, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "energy_per_atom": torch.tensor(
            [-7.0807463489e-03, -7.0807463489e-03], dtype=torch.float32
        ),
        "total_energy": torch.tensor([-1.4161492698e-02], dtype=torch.float32),
        "dE_dCN": torch.tensor(
            [-2.0259325393e-03, -2.0259325393e-03], dtype=torch.float32
        ),
        "force": torch.tensor(
            [
                [3.2497653738e-03, 0.0, 0.0],
                [-3.2497653738e-03, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    }


@pytest.fixture
def hcl_dimer_reference_cpu():
    """Reference outputs for HCl dimer system on CPU."""
    return {
        "cn": torch.tensor(
            [5.0002193451e-01, 5.0044161081e-01, 5.0002193451e-01, 5.0044161081e-01],
            dtype=torch.float32,
        ),
        "inv_r": torch.tensor(
            [
                [4.1666665673e-01, 1.4285714924e-01, 1.3513512909e-01, 0.0, 0.0],
                [4.1666665673e-01, 1.3513512909e-01, 1.4285714924e-01, 0.0, 0.0],
                [1.4285714924e-01, 1.3513512909e-01, 4.1666665673e-01, 0.0, 0.0],
                [1.3513512909e-01, 1.4285714924e-01, 4.1666665673e-01, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "df_dr": torch.tensor(
            [
                [1.6666666269, 6.8485655902e-07, 1.4150593415e-05, 0.0, 0.0],
                [1.6666666269, 1.4150593415e-05, 4.9519009190e-04, 0.0, 0.0],
                [6.8485655902e-07, 1.4150593415e-05, 1.6666666269, 0.0, 0.0],
                [1.4150593415e-05, 4.9519009190e-04, 1.6666666269, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "energy_per_atom": torch.tensor(
            [
                -2.7685246896e-03,
                -8.2953069359e-03,
                -2.7685246896e-03,
                -8.2953069359e-03,
            ],
            dtype=torch.float32,
        ),
        "total_energy": torch.tensor([-2.2127663717e-02], dtype=torch.float32),
        "dE_dCN": torch.tensor(
            [
                -1.0476986645e-03,
                -2.5185216218e-03,
                -1.0476985481e-03,
                -2.5185216218e-03,
            ],
            dtype=torch.float32,
        ),
        "force": torch.tensor(
            [
                [6.2320637517e-03, 8.8818743825e-04, 0.0],
                [-6.2320632860e-03, 1.9026985392e-03, 0.0],
                [6.2320632860e-03, -8.8818743825e-04, 0.0],
                [-6.2320632860e-03, -1.9026985392e-03, 0.0],
            ],
            dtype=torch.float32,
        ),
    }


# Utility functions


def adjust_neighbor_matrix_for_subsystem(
    nbmat: torch.Tensor,
    atom_start: int,
    atom_end: int,
    max_neighbors: int,
    n_atoms_subsystem: int,
) -> torch.Tensor:
    """
    Adjust neighbor matrix indices from batch to subsystem coordinates.

    When extracting a subsystem from a batch, neighbor indices need to be
    adjusted to be relative to the subsystem rather than the batch.

    Parameters
    ----------
    nbmat : torch.Tensor
        Neighbor matrix from batch system, shape [n_atoms, max_neighbors]
    atom_start : int
        Starting atom index in batch (inclusive)
    atom_end : int
        Ending atom index in batch (exclusive)
    max_neighbors : int
        Maximum neighbors per atom
    n_atoms_subsystem : int
        Number of atoms in the subsystem (used as padding value)

    Returns
    -------
    torch.Tensor
        Adjusted neighbor matrix with indices relative to subsystem
    """
    nbmat_adjusted = nbmat.clone()
    n_atoms = atom_end - atom_start

    for i in range(n_atoms):
        for k in range(max_neighbors):
            neighbor = nbmat_adjusted[i, k]
            # Check if neighbor is within the subsystem range
            if atom_start <= neighbor < atom_end:
                # Adjust to subsystem-relative index
                nbmat_adjusted[i, k] = neighbor - atom_start
            else:
                # Outside subsystem or padding - mark as padding
                nbmat_adjusted[i, k] = n_atoms_subsystem

    return nbmat_adjusted


def to_warp(array: torch.Tensor, dtype=None, device: str = "cpu") -> wp.array:
    """
    Convert PyTorch tensor to warp array.

    Parameters
    ----------
    array : torch.Tensor
        Input PyTorch tensor
    dtype : warp dtype, optional
        Target dtype (inferred from PyTorch if None)
    device : str
        Device name ("cpu" or "cuda:0")

    Returns
    -------
    wp.array
        Warp array on specified device
    """
    # Map warp dtypes to torch dtypes
    warp_to_torch_dtype = {
        wp.float16: torch.float16,
        wp.float32: torch.float32,
        wp.float64: torch.float64,
        wp.int32: torch.int32,
        wp.int64: torch.int64,
        # Vec3 types map to their underlying scalar type
        wp.vec3h: torch.float16,
        wp.vec3f: torch.float32,
        wp.vec3d: torch.float64,
    }

    if dtype is None:
        # Infer dtype from PyTorch tensor
        if array.dtype == torch.float32:
            dtype = wp.float32
        elif array.dtype == torch.float64:
            dtype = wp.float64
        elif array.dtype == torch.float16:
            dtype = wp.float16
        elif array.dtype == torch.int32:
            dtype = wp.int32
        elif array.dtype == torch.int64:
            dtype = wp.int64
        else:
            raise ValueError(f"Unsupported dtype: {array.dtype}")

    # Convert torch tensor to appropriate dtype if needed
    target_torch_dtype = warp_to_torch_dtype.get(dtype, array.dtype)
    array_converted = array.to(dtype=target_torch_dtype, device=device)

    return wp.from_torch(array_converted, dtype=dtype)


def from_warp(wp_array: wp.array) -> torch.Tensor:
    """
    Convert warp array to PyTorch tensor.

    Parameters
    ----------
    wp_array : wp.array
        Input warp array

    Returns
    -------
    torch.Tensor
        PyTorch tensor
    """
    return wp.to_torch(wp_array)


def allocate_outputs(
    n_atoms: int,
    max_neighbors: int,
    device: str = "cpu",
    scalar_dtype=wp.float32,
    vec_dtype=wp.vec3f,
    num_systems: int = 1,
) -> dict:
    """
    Allocate zero-initialized output arrays for DFT-D3 kernels.

    Parameters
    ----------
    n_atoms : int
        Number of atoms
    max_neighbors : int
        Maximum neighbors per atom
    device : str
        Device name
    scalar_dtype : warp dtype
        Scalar floating point type (default: wp.float32)
    vec_dtype : warp dtype
        Vector type (default: wp.vec3f)

    num_systems : int
        Number of independent systems (default: 1). Used to size total_energy array.

    Returns
    -------
    dict
        Dictionary with allocated warp arrays:
        - cn: [n_atoms]
        - inv_r: [n_atoms, max_neighbors]
        - df_dr: [n_atoms, max_neighbors]
        - energy_contributions: [n_atoms, max_neighbors]
        - force_contributions: [n_atoms, max_neighbors] vec3
        - energy_per_atom: [n_atoms]
        - total_energy: [num_systems]
        - dE_dCN: [n_atoms]
        - force: [n_atoms] vec3
    """
    return {
        "cn": wp.zeros(n_atoms, dtype=scalar_dtype, device=device),
        "inv_r": wp.zeros((n_atoms, max_neighbors), dtype=scalar_dtype, device=device),
        "df_dr": wp.zeros((n_atoms, max_neighbors), dtype=scalar_dtype, device=device),
        "energy_contributions": wp.zeros(
            (n_atoms, max_neighbors), dtype=scalar_dtype, device=device
        ),
        "force_contributions": wp.zeros(
            (n_atoms, max_neighbors), dtype=vec_dtype, device=device
        ),
        "energy_per_atom": wp.zeros(n_atoms, dtype=scalar_dtype, device=device),
        "total_energy": wp.zeros(num_systems, dtype=scalar_dtype, device=device),
        "dE_dCN": wp.zeros(n_atoms, dtype=scalar_dtype, device=device),
        "force": wp.zeros(n_atoms, dtype=vec_dtype, device=device),
    }


def prepare_inputs(
    system: dict,
    element_tables: dict,
    device: str = "cpu",
    scalar_dtype=wp.float32,
    vec_dtype=wp.vec3f,
) -> dict:
    """
    Prepare input arrays for DFT-D3 kernels.

    Parameters
    ----------
    system : dict
        System geometry (from fixtures)
    element_tables : dict
        Element parameter tables
    device : str
        Device name
    scalar_dtype : warp dtype
        Scalar floating point type (default: wp.float32)
    vec_dtype : warp dtype
        Vector type (default: wp.vec3f)

    Returns
    -------
    dict
        Dictionary with warp arrays ready for kernel launch
    """
    # Reshape coord from [B*3] to [B, 3] for vec3 format
    B = system["B"]
    coord_flat = system["coord"]
    coord_reshaped = coord_flat.reshape(B, 3)

    return {
        "coord": to_warp(coord_reshaped, vec_dtype, device),
        "numbers": to_warp(system["numbers"], wp.int32, device),
        "nbmat": to_warp(system["nbmat"], wp.int32, device),
        "rcov": to_warp(element_tables["rcov"], scalar_dtype, device),
        "r4r2": to_warp(element_tables["r4r2"], scalar_dtype, device),
        "c6ref": to_warp(element_tables["c6ref"], scalar_dtype, device),
        "cnref_i": to_warp(element_tables["cnref_i"], scalar_dtype, device),
        "cnref_j": to_warp(element_tables["cnref_j"], scalar_dtype, device),
    }


@pytest.fixture
def batch_four_systems():
    """Fixture providing 4 independent H2 systems in a batch.

    Returns a tuple of (combined_system, batch_indices) where:
    - combined_system: Dict with concatenated geometries for 4 H2 molecules
    - batch_indices: Tensor mapping atoms to their system index [0,0,1,1,2,2,3,3]

    Each H2 has 2 atoms with different separations.
    """
    # Create 4 independent H2 systems with different separations
    separations = [1.4, 1.5, 1.3, 1.6]

    all_coords = []
    all_numbers = []
    all_nbmat_rows = []
    total_atoms_so_far = 0

    # Total batch dimensions
    B = len(separations) * 2  # 4 systems × 2 atoms each = 8 atoms
    M = 5  # Max neighbors

    for sep in separations:
        # Each H2: atoms at (x_offset, 0, 0) and (x_offset+sep, 0, 0)
        coord = torch.tensor(
            [
                float(total_atoms_so_far),
                0.0,
                0.0,  # H1
                float(total_atoms_so_far) + sep,
                0.0,
                0.0,  # H2
            ],
            dtype=torch.float32,
        )
        all_coords.append(coord)
        all_numbers.append(torch.tensor([1, 1], dtype=torch.int32))

        # Neighbor matrix for this H2: each H sees the other
        # Adjust neighbor indices relative to concatenated array
        nbmat_h1 = torch.full((M,), B, dtype=torch.int32)  # Padding value = B
        nbmat_h1[0] = total_atoms_so_far + 1  # H1's neighbor is H2

        nbmat_h2 = torch.full((M,), B, dtype=torch.int32)  # Padding value = B
        nbmat_h2[0] = total_atoms_so_far  # H2's neighbor is H1

        all_nbmat_rows.append(nbmat_h1)
        all_nbmat_rows.append(nbmat_h2)
        total_atoms_so_far += 2

    # Concatenate all

    coord_combined = torch.cat(all_coords, dim=0)
    numbers_combined = torch.cat(all_numbers, dim=0)
    nbmat_combined = torch.stack(all_nbmat_rows, dim=0)

    # Batch indices: [0, 0, 1, 1, 2, 2, 3, 3]
    batch_indices = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int32)

    system = {
        "coord": coord_combined,
        "numbers": numbers_combined,
        "nbmat": nbmat_combined,
        "B": B,
        "M": M,
    }

    return system, batch_indices
