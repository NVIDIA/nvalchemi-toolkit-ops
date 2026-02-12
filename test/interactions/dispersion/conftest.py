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
Pytest fixtures and utilities for DFT-D3 kernel testing.

This module contains:
- Dummy parameter tables for testing (NOT physically accurate)
- Test system geometries (H2, CH4, edge cases)
- Pytest fixtures for devices and systems
- Utility functions for array conversion and allocation
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

# ==============================================================================
# Parameter Tables (Dummy values for testing only)
# ==============================================================================


@pytest.fixture(scope="session")
def element_tables():
    """
    Session-scoped fixture providing dummy element parameter tables for testing.

    These are NOT physically accurate parameters - they're made-up values designed
    for numerical stability and testing purposes only. Do not use for production.

    Returns
    -------
    dict
        Dictionary with keys:
        - rcov: Covalent radii [Zmax+1] in Bohr (numpy array)
        - r4r2: <r⁴>/<r²> expectation values [z_max+1] (numpy array)
        - c6ref: C6 reference values [(z_max+1)*(z_max+1)*25] flattened (numpy array)
        - cnref_i: CN reference for atom i [(z_max+1)*(z_max+1)*25] flattened (numpy array)
        - cnref_j: CN reference for atom j [(z_max+1)*(z_max+1)*25] flattened (numpy array)
        - z_max_inc: z_max + 1
    """
    z_max = 17  # Maximum atomic number (Cl)
    # incremented maximum atomic number
    z_max_inc = z_max + 1

    # Covalent radii in Bohr (made up but reasonable scale)
    # Index by atomic number: [0, H, He, ..., Ne, ..., Cl]
    rcov = np.zeros(z_max_inc, dtype=np.float32)
    rcov[0:10] = np.array(
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
        dtype=np.float32,
    )
    rcov[10] = 1.5  # Ne (Z=10)
    rcov[17] = 1.8  # Cl (Z=17)

    # Maximum coordination numbers (physically motivated but simplified)
    cnmax = np.zeros(z_max_inc, dtype=np.float32)
    cnmax[0:10] = np.array(
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
        dtype=np.float32,
    )
    cnmax[10] = 1.0  # Ne (noble gas, essentially 0 but avoid division issues)
    cnmax[17] = 2.0  # Cl

    # <r⁴>/<r²> expectation values (made up, positive values)
    # Larger for more polarizable species
    r4r2 = np.zeros(z_max_inc, dtype=np.float32)
    r4r2[0:10] = np.array(
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
        dtype=np.float32,
    )
    r4r2[10] = 4.5  # Ne (moderately polarizable)
    r4r2[17] = 8.0  # Cl (more polarizable)

    # C6 reference grid: 5x5 grid for each element pair
    # Total size: z_max_inc * z_max_inc * 25
    # We'll fill with simple positive values scaled by atomic numbers
    c6ref = np.zeros(z_max_inc * z_max_inc * 25, dtype=np.float32)
    cnref_i = np.zeros(z_max_inc * z_max_inc * 25, dtype=np.float32)
    cnref_j = np.zeros(z_max_inc * z_max_inc * 25, dtype=np.float32)

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


# D3Parameters removed - it's PyTorch-specific and lives in nvalchemiops.torch.interactions.dispersion
# Core tests should use warp arrays directly


@pytest.fixture(scope="session")
def functional_params():
    """
    Session-scoped fixture providing dummy functional parameters (PBE-like) for testing.

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


@pytest.fixture(scope="session")
def h2_system():
    """
    Session-scoped fixture providing simple H2 molecule geometry for testing.

    Returns
    -------
    dict
        Dictionary with:
        - coord: [6] flattened coordinates in Bohr (numpy array)
        - numbers: [2] atomic numbers (numpy array)
        - nbmat: [2, 5] neighbor matrix 2D array (numpy array)
        - B: number of atoms (2)
        - M: max neighbors (5)
    """
    separation = 1.4  # H-H distance in Bohr
    # H2 molecule along x-axis
    coord = np.array(
        [
            0.0,
            0.0,
            0.0,  # H1 at origin
            separation,
            0.0,
            0.0,  # H2 at (r, 0, 0)
        ],
        dtype=np.float32,
    )

    numbers = np.array([1, 1], dtype=np.int32)  # Both hydrogen

    # Neighbor matrix: each atom has the other as neighbor, rest padding
    # For atom 0: neighbor is atom 1, then padding (use B=2 as sentinel)
    # For atom 1: neighbor is atom 0, then padding
    B, M = 2, 5
    nbmat = np.array(
        [
            [1, 2, 2, 2, 2],  # Atom 0's neighbors: [1, padding, padding, ...]
            [0, 2, 2, 2, 2],  # Atom 1's neighbors: [0, padding, padding, ...]
        ],
        dtype=np.int32,
    )

    return {
        "coord": coord,
        "numbers": numbers,
        "nbmat": nbmat,
        "B": B,
        "M": M,
    }


@pytest.fixture(scope="session")
def h2_close():
    """
    Session-scoped fixture providing H2 with very small separation (edge case).

    Returns
    -------
    dict
        Dictionary with H2 system at separation 0.1 Bohr (numpy arrays)
    """
    separation = 0.1  # Very small H-H distance in Bohr
    # H2 molecule along x-axis
    coord = np.array(
        [
            0.0,
            0.0,
            0.0,  # H1 at origin
            separation,
            0.0,
            0.0,  # H2 at (r, 0, 0)
        ],
        dtype=np.float32,
    )

    numbers = np.array([1, 1], dtype=np.int32)  # Both hydrogen

    # Neighbor matrix: each atom has the other as neighbor, rest padding
    B, M = 2, 5
    nbmat = np.array(
        [
            [1, 2, 2, 2, 2],  # Atom 0's neighbors: [1, padding, padding, ...]
            [0, 2, 2, 2, 2],  # Atom 1's neighbors: [0, padding, padding, ...]
        ],
        dtype=np.int32,
    )

    return {
        "coord": coord,
        "numbers": numbers,
        "nbmat": nbmat,
        "B": B,
        "M": M,
    }


@pytest.fixture(scope="session")
def ch4_like_system():
    """
    Session-scoped fixture providing simple CH4-like molecule geometry for testing.

    Returns
    -------
    dict
        Dictionary with:
        - coord: [15] flattened coordinates in Bohr (numpy array)
        - numbers: [5] atomic numbers (C + 4H) (numpy array)
        - nbmat: [5, 10] neighbor matrix 2D array (numpy array)
        - B: number of atoms (5)
        - M: max neighbors (10)
    """
    # Simplified CH4: C at origin, 4 H in tetrahedral-ish positions
    r_CH = 2.0  # C-H distance in Bohr # NOSONAR (S117) "chemical formula"
    coord = np.array(
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
        dtype=np.float32,
    )

    numbers = np.array([6, 1, 1, 1, 1], dtype=np.int32)  # C + 4H

    # Neighbor matrix: C has 4 H neighbors, each H has C as neighbor
    B, M = 5, 10
    nbmat = np.full((B, M), B, dtype=np.int32)  # Fill with padding (B=5)

    # C (atom 0) has neighbors 1,2,3,4
    nbmat[0, 0:4] = np.array([1, 2, 3, 4], dtype=np.int32)

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


@pytest.fixture(scope="session")
def single_atom_system():
    """
    Session-scoped fixture providing single atom system (edge case for testing).

    Returns
    -------
    dict
        Dictionary with single H atom, no neighbors (numpy arrays)
    """
    coord = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    numbers = np.array([1], dtype=np.int32)
    B, M = 1, 5
    nbmat = np.full((B, M), B, dtype=np.int32)  # All padding

    return {
        "coord": coord,
        "numbers": numbers,
        "nbmat": nbmat,
        "B": B,
        "M": M,
    }


@pytest.fixture(scope="session")
def empty_neighbors_system():
    """
    Session-scoped fixture providing system with atoms but no neighbors (all padding).

    Returns
    -------
    dict
        Dictionary with 3 atoms but no neighbors (numpy arrays)
    """
    coord = np.array(
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
        dtype=np.float32,
    )

    numbers = np.array([1, 1, 1], dtype=np.int32)
    B, M = 3, 5
    nbmat = np.full((B, M), B, dtype=np.int32)  # All padding

    return {
        "coord": coord,
        "numbers": numbers,
        "nbmat": nbmat,
        "B": B,
        "M": M,
    }


@pytest.fixture(scope="session")
def ne2_system():
    """
    Session-scoped fixture providing Ne2 dimer for testing dispersion in noble gases.

    Noble gases are ideal for testing dispersion since their interactions
    are purely dispersive (no covalent bonding, minimal electrostatics).

    Returns
    -------
    dict
        Dictionary with Ne2 dimer geometry (numpy arrays)
    """
    separation = 5.8  # Ne-Ne distance in Bohr (near equilibrium)
    # Ne2 along x-axis
    coord = np.array(
        [
            0.0,
            0.0,
            0.0,  # Ne1 at origin
            separation,
            0.0,
            0.0,  # Ne2
        ],
        dtype=np.float32,
    )

    numbers = np.array([10, 10], dtype=np.int32)  # Both neon

    # Neighbor matrix
    B, M = 2, 5
    nbmat = np.array(
        [
            [1, 2, 2, 2, 2],  # Atom 0's neighbors: [1, padding, ...]
            [0, 2, 2, 2, 2],  # Atom 1's neighbors: [0, padding, ...]
        ],
        dtype=np.int32,
    )

    return {
        "coord": coord,
        "numbers": numbers,
        "nbmat": nbmat,
        "B": B,
        "M": M,
    }


@pytest.fixture(scope="session")
def hcl_dimer_system():
    """
    Session-scoped fixture providing HCl dimer for testing realistic molecular dispersion.

    HCl dimer tests heteronuclear dispersion with different element types
    and more realistic polarizabilities. Configuration is parallel displaced.

    Returns
    -------
    dict
        Dictionary with HCl dimer geometry (4 atoms total, numpy arrays)
    """
    # HCl bond length ~2.4 Bohr, dimer separation ~7 Bohr
    # Parallel configuration:
    # HCl molecule 1: H at origin, Cl along +x
    # HCl molecule 2: H at (0, 7, 0), Cl along +x

    r_HCl = 2.4  # H-Cl bond length in Bohr # NOSONAR (S117) "chemical formula"
    sep = 7.0  # Separation between molecules

    coord = np.array(
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
        dtype=np.float32,
    )

    numbers = np.array([1, 17, 1, 17], dtype=np.int32)  # H, Cl, H, Cl

    # Neighbor matrix: each atom sees all others as potential neighbors
    B, M = 4, 5
    nbmat = np.full((B, M), B, dtype=np.int32)  # Fill with padding

    # Atom 0 (H1): neighbors are Cl1, H2, Cl2
    nbmat[0, 0:3] = np.array([1, 2, 3], dtype=np.int32)
    # Atom 1 (Cl1): neighbors are H1, H2, Cl2
    nbmat[1, 0:3] = np.array([0, 2, 3], dtype=np.int32)
    # Atom 2 (H2): neighbors are H1, Cl1, Cl2
    nbmat[2, 0:3] = np.array([0, 1, 3], dtype=np.int32)
    # Atom 3 (Cl2): neighbors are H1, Cl1, H2
    nbmat[3, 0:3] = np.array([0, 1, 2], dtype=np.int32)

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
@pytest.fixture(params=["cpu", "cuda:0"], ids=["cpu", "gpu"])
def device(request):
    """
    Fixture providing both CPU and GPU devices.

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


# d3_parameters fixture removed - D3Parameters is PyTorch-specific
# PyTorch binding tests should import D3Parameters from nvalchemiops.torch.interactions.dispersion


# ==============================================================================
# Reference Output Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def ne2_reference_cpu():
    """
    Session-scoped reference outputs for Ne2 system on CPU (numpy arrays).

    To regenerate: python test/interactions/dispersion/generate_reference_outputs.py
    """
    return {
        "cn": np.array([4.4183229329e-04, 4.4183229329e-04], dtype=np.float32),
        "total_energy": np.array([-1.4161492698e-02], dtype=np.float32),
        "force": np.array(
            [
                [3.2497653738e-03, 0.0, 0.0],
                [-3.2497653738e-03, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }


@pytest.fixture(scope="session")
def hcl_dimer_reference_cpu():
    """
    Session-scoped reference outputs for HCl dimer system on CPU (numpy arrays).

    To regenerate: python test/interactions/dispersion/generate_reference_outputs.py
    """
    return {
        "cn": np.array(
            [5.0002193451e-01, 5.0044161081e-01, 5.0002193451e-01, 5.0044161081e-01],
            dtype=np.float32,
        ),
        "total_energy": np.array([-2.2127663717e-02], dtype=np.float32),
        "force": np.array(
            [
                [6.2320637517e-03, 8.8818743825e-04, 0.0],
                [-6.2320632860e-03, 1.9026985392e-03, 0.0],
                [6.2320632860e-03, -8.8818743825e-04, 0.0],
                [-6.2320632860e-03, -1.9026985392e-03, 0.0],
            ],
            dtype=np.float32,
        ),
    }


# Utility functions


def adjust_neighbor_matrix_for_subsystem(
    nbmat: np.ndarray,
    atom_start: int,
    atom_end: int,
    max_neighbors: int,
    n_atoms_subsystem: int,
) -> np.ndarray:
    """
    Adjust neighbor matrix indices from batch to subsystem coordinates.

    When extracting a subsystem from a batch, neighbor indices need to be
    adjusted to be relative to the subsystem rather than the batch.

    Parameters
    ----------
    nbmat : np.ndarray
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
    np.ndarray
        Adjusted neighbor matrix with indices relative to subsystem
    """
    nbmat_adjusted = nbmat.copy()
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


def to_warp(array: np.ndarray, dtype=None, device: str = "cpu") -> wp.array:
    """
    Convert numpy array to warp array.

    Parameters
    ----------
    array : np.ndarray
        Input numpy array
    dtype : warp dtype, optional
        Target dtype (inferred from numpy if None)
    device : str
        Device name ("cpu" or "cuda:0")

    Returns
    -------
    wp.array
        Warp array on specified device
    """
    # Map warp dtypes to numpy dtypes
    warp_to_numpy_dtype = {
        wp.float16: np.float16,
        wp.float32: np.float32,
        wp.float64: np.float64,
        wp.int32: np.int32,
        wp.int64: np.int64,
        # Vec3 types map to their underlying scalar type
        wp.vec3h: np.float16,
        wp.vec3f: np.float32,
        wp.vec3d: np.float64,
    }

    if dtype is None:
        # Infer dtype from numpy array
        if array.dtype == np.float32:
            dtype = wp.float32
        elif array.dtype == np.float64:
            dtype = wp.float64
        elif array.dtype == np.float16:
            dtype = wp.float16
        elif array.dtype == np.int32:
            dtype = wp.int32
        elif array.dtype == np.int64:
            dtype = wp.int64
        else:
            raise ValueError(f"Unsupported dtype: {array.dtype}")

    # Convert numpy array to appropriate dtype if needed
    target_numpy_dtype = warp_to_numpy_dtype.get(dtype, array.dtype)
    if array.dtype != target_numpy_dtype:
        array = array.astype(target_numpy_dtype)

    return wp.from_numpy(array, dtype=dtype, device=device)


def from_warp(wp_array: wp.array) -> np.ndarray:
    """
    Convert warp array to numpy array.

    Parameters
    ----------
    wp_array : wp.array
        Input warp array

    Returns
    -------
    np.ndarray
        Numpy array
    """
    return wp_array.numpy()


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


@pytest.fixture(scope="session")
def batch_four_systems():
    """Session-scoped fixture providing 4 independent H2 systems in a batch.

    Returns a tuple of (combined_system, batch_indices) where:
    - combined_system: Dict with concatenated geometries for 4 H2 molecules (numpy arrays)
    - batch_indices: Array mapping atoms to their system index [0,0,1,1,2,2,3,3]

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
        coord = np.array(
            [
                float(total_atoms_so_far),
                0.0,
                0.0,  # H1
                float(total_atoms_so_far) + sep,
                0.0,
                0.0,  # H2
            ],
            dtype=np.float32,
        )
        all_coords.append(coord)
        all_numbers.append(np.array([1, 1], dtype=np.int32))

        # Neighbor matrix for this H2: each H sees the other
        # Adjust neighbor indices relative to concatenated array
        nbmat_h1 = np.full((M,), B, dtype=np.int32)  # Padding value = B
        nbmat_h1[0] = total_atoms_so_far + 1  # H1's neighbor is H2

        nbmat_h2 = np.full((M,), B, dtype=np.int32)  # Padding value = B
        nbmat_h2[0] = total_atoms_so_far  # H2's neighbor is H1

        all_nbmat_rows.append(nbmat_h1)
        all_nbmat_rows.append(nbmat_h2)
        total_atoms_so_far += 2

    # Concatenate all
    coord_combined = np.concatenate(all_coords, axis=0)
    numbers_combined = np.concatenate(all_numbers, axis=0)
    nbmat_combined = np.stack(all_nbmat_rows, axis=0)

    # Batch indices: [0, 0, 1, 1, 2, 2, 3, 3]
    batch_indices = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)

    system = {
        "coord": coord_combined,
        "numbers": numbers_combined,
        "nbmat": nbmat_combined,
        "B": B,
        "M": M,
    }

    return system, batch_indices
