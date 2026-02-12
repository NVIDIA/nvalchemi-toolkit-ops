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
Script to generate reference outputs for DFT-D3 tests using the new API.

This script uses the dftd3_nm high-level launcher to compute DFT-D3 dispersion
corrections and formats the outputs for copy-paste into conftest.py fixtures.

Usage
-----
Run from the repository root:
    python test/interactions/dispersion/generate_reference_outputs.py

The output can be copy-pasted directly into conftest.py to update the
ne2_reference_cpu and hcl_dimer_reference_cpu fixtures.

Outputs Generated
-----------------
The new API exposes only final physics outputs:
- cn: Coordination numbers [num_atoms]
- total_energy: Total dispersion energy [num_systems]
- force: Atomic forces [num_atoms, 3]

Intermediate values (inv_r, df_dr, energy_per_atom, dE_dCN) are internal to
the kernels and no longer exposed.

See conftest_fixtures_update.md for details on updating test fixtures.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from nvalchemiops.interactions.dispersion._dftd3 import dftd3_nm

# ==============================================================================
# Helper Functions (extracted from pytest fixtures)
# ==============================================================================


def get_element_tables(z_max: int = 17) -> dict:
    """
    Create dummy element parameter tables for testing.

    These are NOT physically accurate parameters - they're made-up values designed
    for numerical stability and testing purposes only. Do not use for production.

    Parameters
    ----------
    z_max : int
        Maximum atomic number (default: 17 for Cl)

    Returns
    -------
    dict
        Dictionary with element parameter arrays
    """
    z_max_inc = z_max + 1

    # Covalent radii in Bohr (made up but reasonable scale)
    rcov = np.zeros(z_max_inc, dtype=np.float32)
    rcov[0:10] = np.array(
        [
            0.0,  # Z=0 (padding)
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

    # Maximum coordination numbers
    cnmax = np.zeros(z_max_inc, dtype=np.float32)
    cnmax[0:10] = np.array(
        [
            0.0,  # Z=0
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
    cnmax[10] = 1.0  # Ne
    cnmax[17] = 2.0  # Cl

    # <r⁴>/<r²> expectation values
    r4r2 = np.zeros(z_max_inc, dtype=np.float32)
    r4r2[0:10] = np.array(
        [
            0.0,  # Z=0
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
    r4r2[10] = 4.5  # Ne
    r4r2[17] = 8.0  # Cl

    # C6 reference grid: 5x5 grid for each element pair
    # Shape will be [z_max_inc, z_max_inc, 5, 5]
    c6ref = np.zeros((z_max_inc, z_max_inc, 5, 5), dtype=np.float32)
    cnref_i = np.zeros((z_max_inc, z_max_inc, 5, 5), dtype=np.float32)
    cnref_j = np.zeros((z_max_inc, z_max_inc, 5, 5), dtype=np.float32)

    # Fill C6 and CN reference grids
    for zi in range(z_max_inc):
        for zj in range(z_max_inc):
            for p in range(5):
                for q in range(5):
                    # CN reference grid: evenly spaced from 0 to cnmax
                    if zi > 0:
                        cnref_i[zi, zj, p, q] = (p / 4.0) * cnmax[zi]
                    if zj > 0:
                        cnref_j[zi, zj, p, q] = (q / 4.0) * cnmax[zj]

                    # C6 values: scale with zi*zj and vary with CN grid point
                    if zi > 0 and zj > 0:
                        c6ref[zi, zj, p, q] = (
                            10.0 * float(zi * zj) * (1.0 + 0.1 * p + 0.1 * q)
                        )

    return {
        "rcov": rcov,
        "r4r2": r4r2,
        "c6ref": c6ref,
        "cnref_i": cnref_i,
        "cnref_j": cnref_j,
        "z_max_inc": z_max_inc,
    }


def get_functional_params() -> dict:
    """
    Get dummy functional parameters (PBE-like) for testing.

    Returns
    -------
    dict
        Dictionary with DFT-D3 functional parameters
    """
    return {
        "a1": 0.4,
        "a2": 4.0,  # Bohr
        "s6": 1.0,
        "s8": 0.8,
        "k1": 16.0,  # Bohr^-1
        "k3": -4.0,
    }


def get_ne2_system(separation: float = 5.8) -> dict:
    """
    Get Ne2 dimer geometry for testing.

    Parameters
    ----------
    separation : float
        Ne-Ne distance in Bohr

    Returns
    -------
    dict
        Dictionary with system geometry
    """
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


def get_hcl_dimer_system() -> dict:
    """
    Get HCl dimer geometry for testing.

    Returns
    -------
    dict
        Dictionary with HCl dimer geometry (4 atoms total)
    """
    r_HCl = 2.4  # H-Cl bond length in Bohr
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
    warp_to_numpy_dtype = {
        wp.float32: np.float32,
        wp.float64: np.float64,
        wp.int32: np.int32,
        wp.vec3f: np.float32,
        wp.vec3d: np.float64,
    }

    if dtype is None:
        # Infer dtype from numpy array
        if array.dtype == np.float32:
            dtype = wp.float32
        elif array.dtype == np.float64:
            dtype = wp.float64
        elif array.dtype == np.int32:
            dtype = wp.int32
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


# ==============================================================================
# Main Pipeline Function
# ==============================================================================


def run_full_pipeline(
    system: dict, element_tables: dict, functional_params: dict, device: str = "cpu"
):
    """
    Run full DFT-D3 pipeline and return all outputs.

    Parameters
    ----------
    system : dict
        System geometry
    element_tables : dict
        Element parameter tables
    functional_params : dict
        Functional parameters
    device : str
        Device to run on

    Returns
    -------
    dict
        All outputs from the pipeline
    """
    B = system["B"]

    # Reshape coord from [B*3] to [B, 3] for vec3 format
    coord_flat = system["coord"]
    coord_reshaped = coord_flat.reshape(B, 3)

    # Convert inputs to warp arrays
    positions = to_warp(coord_reshaped, wp.vec3f, device)
    numbers = to_warp(system["numbers"], wp.int32, device)
    neighbor_matrix = to_warp(system["nbmat"], wp.int32, device)
    covalent_radii = to_warp(element_tables["rcov"], wp.float32, device)
    r4r2 = to_warp(element_tables["r4r2"], wp.float32, device)
    c6_reference = to_warp(element_tables["c6ref"], wp.float32, device)

    # Create coord_num_ref array with proper structure
    # The launcher expects coord_num_ref[zi, zj] to give CN references for both atoms
    # We'll use cnref_i as the primary reference (symmetric for homonuclear)
    coord_num_ref = to_warp(element_tables["cnref_i"], wp.float32, device)

    # Allocate output arrays
    coord_num = wp.zeros(B, dtype=wp.float32, device=device)
    forces = wp.zeros(B, dtype=wp.vec3f, device=device)
    energy = wp.zeros(1, dtype=wp.float32, device=device)  # Single system
    virial = wp.zeros(1, dtype=wp.mat33f, device=device)

    # Create batch indices (all atoms in system 0)
    batch_indices = np.zeros(B, dtype=np.int32)
    batch_idx = to_warp(batch_indices, wp.int32, device)

    # Extract functional params
    a1 = functional_params["a1"]
    a2 = functional_params["a2"]
    s6 = functional_params["s6"]
    s8 = functional_params["s8"]
    k1 = functional_params["k1"]
    k3 = functional_params["k3"]

    # Run DFT-D3 calculation using high-level launcher
    dftd3_nm(
        positions=positions,
        numbers=numbers,
        neighbor_matrix=neighbor_matrix,
        covalent_radii=covalent_radii,
        r4r2=r4r2,
        c6_reference=c6_reference,
        coord_num_ref=coord_num_ref,
        a1=a1,
        a2=a2,
        s8=s8,
        coord_num=coord_num,
        forces=forces,
        energy=energy,
        virial=virial,
        vec_dtype=wp.vec3f,
        k1=k1,
        k3=k3,
        s6=s6,
        fill_value=B,
        batch_idx=batch_idx,
        device=device,
    )

    # Convert outputs to numpy
    return {
        "cn": from_warp(coord_num),
        "total_energy": from_warp(energy),
        "force": from_warp(forces),
    }


def format_array(arr: np.ndarray, name: str) -> str:
    """Format numpy array as Python code for copy-paste into conftest.py."""
    # Handle 1D arrays (cn, total_energy)
    if arr.ndim == 1:
        if arr.size == 1:
            # Single value (like total_energy)
            return f'        "{name}": np.array([{arr[0]:.10e}], dtype=np.float32),'
        elif arr.size <= 4:
            # Short array (like 2-4 atoms)
            values = ", ".join(f"{x:.10e}" for x in arr)
            return f'        "{name}": np.array([{values}], dtype=np.float32),'
        else:
            # Longer arrays - multi-line format
            lines = [f'        "{name}": np.array(']
            values = ", ".join(f"{x:.10e}" for x in arr)
            lines.append(f"            [{values}],")
            lines.append("            dtype=np.float32,")
            lines.append("        ),")
            return "\n".join(lines)

    # Handle 2D arrays (force vectors [N, 3])
    elif arr.ndim == 2:
        lines = [f'        "{name}": np.array(']
        lines.append("            [")
        for i, row in enumerate(arr):
            values = ", ".join(f"{x:.10e}" for x in row)
            comma = "," if i < len(arr) - 1 else ""
            lines.append(f"                [{values}]{comma}")
        lines.append("            ],")
        lines.append("            dtype=np.float32,")
        lines.append("        ),")
        return "\n".join(lines)

    # Fallback for other dimensions
    else:
        return f'        "{name}": np.array({arr.tolist()!r}, dtype=np.float32),'


def main():
    """Generate and print reference outputs."""
    print("=" * 80)
    print("GENERATING REFERENCE OUTPUTS FOR DFT-D3 KERNEL TESTS")
    print("=" * 80)
    print()

    # Get parameters (Zmax=17 to include Cl)
    element_tables = get_element_tables(z_max=17)
    functional_params = get_functional_params()

    output_keys = [
        "cn",
        "total_energy",
        "force",
    ]

    # =========================================================================
    # Test Ne2 system (noble gas dispersion)
    # =========================================================================
    print("Ne2 SYSTEM (separation=5.8 Bohr) on CPU:")
    print("-" * 80)
    print("Noble gas dimer - pure dispersion interaction")
    print()
    ne2_system = get_ne2_system(separation=5.8)
    ne2_cpu = run_full_pipeline(
        ne2_system, element_tables, functional_params, device="cpu"
    )

    print("Copy-paste this into conftest.py (ne2_reference_cpu fixture):")
    print()
    print("    return {")
    for key in output_keys:
        if key in ne2_cpu:
            print(format_array(ne2_cpu[key], key))
    print("    }")
    print()
    print(f"Ne2 total energy: {ne2_cpu['total_energy'][0]:.6e} Hartree")
    print(f"Ne2 atoms: {len(ne2_cpu['cn'])}")
    print()

    # =========================================================================
    # Test HCl dimer system (realistic molecular dispersion)
    # =========================================================================
    print()
    print("HCl DIMER SYSTEM (parallel, sep=7 Bohr) on CPU:")
    print("-" * 80)
    print("Realistic molecular dimer - heteronuclear dispersion")
    print()
    hcl_system = get_hcl_dimer_system()
    hcl_cpu = run_full_pipeline(
        hcl_system, element_tables, functional_params, device="cpu"
    )

    print("Copy-paste this into conftest.py (hcl_dimer_reference_cpu fixture):")
    print()
    print("    return {")
    for key in output_keys:
        if key in hcl_cpu:
            print(format_array(hcl_cpu[key], key))
    print("    }")
    print()
    print(f"HCl dimer total energy: {hcl_cpu['total_energy'][0]:.6e} Hartree")
    print(f"HCl dimer atoms: {len(hcl_cpu['cn'])}")
    print()

    # =========================================================================
    # GPU tests if available
    # =========================================================================
    if wp.is_cuda_available():
        print()
        print("=" * 80)
        print("GPU VERIFICATION")
        print("=" * 80)
        print()

        # Ne2 on GPU
        print("Ne2 on GPU:")
        print("-" * 80)
        ne2_gpu = run_full_pipeline(
            ne2_system, element_tables, functional_params, device="cuda:0"
        )
        print("CPU/GPU Differences:")
        for key in ne2_cpu.keys():
            if key in ne2_gpu:
                diff = np.abs(ne2_cpu[key] - ne2_gpu[key]).max()
                print(f"  {key:20s}: max diff = {diff:.2e}")
        print()

        # HCl dimer on GPU
        print("HCl dimer on GPU:")
        print("-" * 80)
        hcl_gpu = run_full_pipeline(
            hcl_system, element_tables, functional_params, device="cuda:0"
        )
        print("CPU/GPU Differences:")
        for key in hcl_cpu.keys():
            if key in hcl_gpu:
                diff = np.abs(hcl_cpu[key] - hcl_gpu[key]).max()
                print(f"  {key:20s}: max diff = {diff:.2e}")
    else:
        print()
        print("GPU not available, skipping GPU reference generation.")

    print()
    print("=" * 80)
    print("DONE! Copy the reference data above into your test file.")
    print("=" * 80)


if __name__ == "__main__":
    main()
