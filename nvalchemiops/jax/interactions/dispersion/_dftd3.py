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

"""JAX DFT-D3 Dispersion Correction Implementation.

This module implements JAX bindings for DFT-D3(BJ) dispersion corrections using
Warp kernels. It mirrors the PyTorch implementation while using JAX arrays and
functional programming patterns.

The module provides:
- `D3Parameters`: Dataclass for organizing DFT-D3 parameters
- `dftd3()`: High-level JAX function for computing dispersion energy and forces

Support for both neighbor matrix and neighbor list formats, with optional
periodic boundary conditions.

Examples
--------
Using D3Parameters dataclass:

>>> import jax.numpy as jnp
>>> from nvalchemiops.jax.interactions.dispersion import dftd3, D3Parameters
>>>
>>> # Create parameters
>>> params = D3Parameters(
...     rcov=jnp.array([...]),  # [max_Z+1] float32
...     r4r2=jnp.array([...]),
...     c6ab=jnp.array([...]),  # [max_Z+1, max_Z+1, 5, 5]
...     cn_ref=jnp.array([...]),
... )
>>>
>>> # Compute dispersion
>>> energy, forces, coord_num = dftd3(
...     positions, numbers,
...     neighbor_matrix=neighbor_matrix,
...     a1=0.3981, a2=4.4211, s8=1.9889,
...     d3_params=params,
... )

Using neighbor list format:

>>> energy, forces, coord_num = dftd3(
...     positions, numbers,
...     neighbor_list=neighbor_list,
...     neighbor_ptr=neighbor_ptr,
...     a1=0.3981, a2=4.4211, s8=1.9889,
...     d3_params=params,
... )
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel

from nvalchemiops.interactions.dispersion._dftd3 import (
    _cn_forces_contrib_kernel_matrix_overload as wp_cn_forces_contrib_nm,
)
from nvalchemiops.interactions.dispersion._dftd3 import (
    _cn_forces_contrib_kernel_overload as wp_cn_forces_contrib_nl,
)
from nvalchemiops.interactions.dispersion._dftd3 import (
    _cn_kernel_matrix_overload as wp_cn_kernel_nm,
)
from nvalchemiops.interactions.dispersion._dftd3 import (
    _cn_kernel_overload as wp_cn_kernel_nl,
)
from nvalchemiops.interactions.dispersion._dftd3 import (
    _compute_cartesian_shifts_matrix_overload as wp_compute_cartesian_shifts_nm,
)
from nvalchemiops.interactions.dispersion._dftd3 import (
    _compute_cartesian_shifts_overload as wp_compute_cartesian_shifts_nl,
)
from nvalchemiops.interactions.dispersion._dftd3 import (
    _direct_forces_and_dE_dCN_kernel_matrix_overload as wp_direct_forces_kernel_nm,
)
from nvalchemiops.interactions.dispersion._dftd3 import (
    _direct_forces_and_dE_dCN_kernel_overload as wp_direct_forces_kernel_nl,
)

# ==============================================================================
# JAX Kernel Wrappers (jax_kernel around Warp kernel overloads)
# ==============================================================================

# --- Pass 0: Cartesian Shift Computation ---

compute_cartesian_shifts_nm = jax_kernel(
    wp_compute_cartesian_shifts_nm[wp.float32], num_outputs=1
)
compute_cartesian_shifts_nl = jax_kernel(
    wp_compute_cartesian_shifts_nl[wp.float32], num_outputs=1
)
# --- Pass 1: Coordination Number Computation ---

cn_kernel_nm = jax_kernel(wp_cn_kernel_nm[wp.float32], num_outputs=1)
cn_kernel_nl = jax_kernel(wp_cn_kernel_nl[wp.float32], num_outputs=1)
# --- Pass 2: Direct Forces and dE/dCN Computation ---

direct_forces_kernel_nm = jax_kernel(
    wp_direct_forces_kernel_nm[wp.float32],
    num_outputs=4,
    in_out_argnames=["dE_dCN", "forces", "energy", "virial"],
)
direct_forces_kernel_nl = jax_kernel(
    wp_direct_forces_kernel_nl[wp.float32],
    num_outputs=4,
    in_out_argnames=["dE_dCN", "forces", "energy", "virial"],
)

# --- Pass 3: CN-Dependent Force Contribution ---

cn_forces_contrib_nm = jax_kernel(
    wp_cn_forces_contrib_nm[wp.float32],
    num_outputs=2,
    in_out_argnames=["forces", "virial"],
)
cn_forces_contrib_nl = jax_kernel(
    wp_cn_forces_contrib_nl[wp.float32],
    num_outputs=2,
    in_out_argnames=["forces", "virial"],
)

__all__ = [
    "D3Parameters",
    "cn_forces_contrib_nl",
    "cn_forces_contrib_nm",
    "cn_kernel_nl",
    "cn_kernel_nm",
    "compute_cartesian_shifts_nl",
    "compute_cartesian_shifts_nm",
    "dftd3",
    "direct_forces_kernel_nl",
    "direct_forces_kernel_nm",
]


# ==============================================================================
# Parameter Dataclass
# ==============================================================================


@dataclass
class D3Parameters:
    """
    DFT-D3 reference parameters for dispersion correction calculations.

    This dataclass encapsulates all element-specific parameters required for
    DFT-D3 dispersion corrections. The main purpose for this structure is to
    provide validation, ensuring the correct shapes, dtypes, and keys are
    present and complete. These parameters are used by :func:`dftd3`.

    Parameters
    ----------
    rcov : jax.Array
        Covalent radii [max_Z+1] as float32. Units should be consistent
        with position coordinates. Index 0 is reserved for
        padding; valid atomic numbers are 1 to max_Z.
    r4r2 : jax.Array
        <r⁴>/<r²> expectation values [max_Z+1] as float32.
        Dimensionless ratio used for computing C8 coefficients from C6 values.
    c6ab : jax.Array
        C6 reference coefficients [max_Z+1, max_Z+1, interp_mesh, interp_mesh]
        as float32. Units are energy x distance^6. Indexed by atomic numbers and
        coordination number reference indices.
    cn_ref : jax.Array
        Coordination number reference grid [max_Z+1, max_Z+1, interp_mesh, interp_mesh]
        as float32. Dimensionless CN values for Gaussian interpolation.
    interp_mesh : int, optional
        Size of the coordination number interpolation mesh. Default: 5
        (standard DFT-D3 uses a 5x5 grid)

    Raises
    ------
    ValueError
        If parameter shapes are inconsistent or invalid
    TypeError
        If parameters are not jax.Array or have invalid dtypes

    Notes
    -----
    - Parameters should use consistent units matching your coordinate system.
      Standard D3 parameters from the Grimme group use atomic units (Bohr for
      distances, Hartree x Bohr^6 for C6 coefficients).
    - Index 0 in all arrays is reserved for padding atoms (atomic number 0)
    - Valid atomic numbers range from 1 to max_z
    - The standard DFT-D3 implementation supports elements 1-94 (H to Pu)
    - Parameters should be float32 for efficiency

    Examples
    --------
    Create parameters from individual arrays:

    >>> params = D3Parameters(
    ...     rcov=jnp.array([...]),
    ...     r4r2=jnp.array([...]),
    ...     c6ab=jnp.array([...]),
    ...     cn_ref=jnp.array([...]),
    ... )
    """

    rcov: jax.Array
    r4r2: jax.Array
    c6ab: jax.Array
    cn_ref: jax.Array
    interp_mesh: int = 5

    def __post_init__(self) -> None:
        """Validate parameter shapes, dtypes, and physical constraints."""
        # Type validation
        for name, arr in [
            ("rcov", self.rcov),
            ("r4r2", self.r4r2),
            ("c6ab", self.c6ab),
            ("cn_ref", self.cn_ref),
        ]:
            if not hasattr(arr, "shape"):
                raise TypeError(
                    f"Parameter '{name}' must be a jax.Array, got {type(arr)}"
                )
            if arr.dtype not in (jnp.float32, jnp.float64):
                raise TypeError(
                    f"Parameter '{name}' must be float32 or float64, got {arr.dtype}"
                )

        # Shape validation
        if self.rcov.ndim != 1:
            raise ValueError(
                f"rcov must be 1D array [max_Z+1], got shape {self.rcov.shape}"
            )

        max_z = self.rcov.shape[0] - 1
        if max_z < 1:
            raise ValueError(
                f"rcov must have at least 2 elements (padding + 1 element), got {self.rcov.shape[0]}"
            )

        if self.r4r2.shape != (max_z + 1,):
            raise ValueError(
                f"r4r2 must have shape [{max_z + 1}] to match rcov, got {self.r4r2.shape}"
            )

        expected_c6_shape = (max_z + 1, max_z + 1, self.interp_mesh, self.interp_mesh)
        if self.c6ab.shape != expected_c6_shape:
            raise ValueError(
                f"c6ab must have shape {expected_c6_shape}, got {self.c6ab.shape}"
            )

        expected_cn_shape = (max_z + 1, max_z + 1, self.interp_mesh, self.interp_mesh)
        if self.cn_ref.shape != expected_cn_shape:
            raise ValueError(
                f"cn_ref must have shape {expected_cn_shape}, got {self.cn_ref.shape}"
            )

    @property
    def max_z(self) -> int:
        """Maximum atomic number supported by these parameters."""
        return self.rcov.shape[0] - 1


# ==============================================================================
# JAX Wrapper Functions
# ==============================================================================


def _dftd3_nm_impl(
    positions: jax.Array,
    numbers: jax.Array,
    neighbor_matrix: jax.Array,
    covalent_radii: jax.Array,
    r4r2: jax.Array,
    c6_reference: jax.Array,
    coord_num_ref: jax.Array,
    a1: float,
    a2: float,
    s8: float,
    k1: float = 16.0,
    k3: float = -4.0,
    s6: float = 1.0,
    s5_smoothing_on: float = 1e10,
    s5_smoothing_off: float = 1e10,
    fill_value: int | None = None,
    batch_idx: jax.Array | None = None,
    cell: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    compute_virial: bool = False,
    num_systems: int | None = None,
) -> (
    tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Internal implementation for neighbor matrix format using jax_kernel wrappers."""
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1] if num_atoms > 0 else 0

    # Set fill_value if not provided
    if fill_value is None:
        fill_value = num_atoms

    # Handle empty case
    if num_atoms == 0:
        if num_systems is None:
            num_systems = 1
            if batch_idx is not None:
                try:
                    num_systems = int(jnp.max(batch_idx)) + 1
                except (
                    jax.errors.ConcretizationTypeError,
                    jax.errors.TracerIntegerConversionError,
                ):
                    raise ValueError(
                        "Cannot infer num_systems inside jax.jit. "
                        "Please provide num_systems explicitly when using jax.jit."
                    ) from None
        empty_energy = jnp.zeros(num_systems, dtype=jnp.float32)
        empty_forces = jnp.zeros((0, 3), dtype=jnp.float32)
        empty_cn = jnp.zeros((0,), dtype=jnp.float32)
        if compute_virial:
            empty_virial = jnp.zeros((num_systems, 3, 3), dtype=jnp.float32)
            return empty_energy, empty_forces, empty_cn, empty_virial
        return empty_energy, empty_forces, empty_cn

    # Determine number of systems
    if num_systems is None:
        if cell is not None:
            num_systems = cell.shape[0]
        elif batch_idx is not None:
            try:
                num_systems = int(jnp.max(batch_idx)) + 1
            except (
                jax.errors.ConcretizationTypeError,
                jax.errors.TracerIntegerConversionError,
            ):
                raise ValueError(
                    "Cannot infer num_systems inside jax.jit. "
                    "Please provide num_systems explicitly when using jax.jit."
                ) from None
        else:
            num_systems = 1

    # Create batch indices if not provided
    if batch_idx is None:
        batch_idx = jnp.zeros(num_atoms, dtype=jnp.int32)

    # Ensure arrays have correct dtypes for kernels (float32 for now)
    positions_f32 = positions.astype(jnp.float32)
    numbers_i32 = numbers.astype(jnp.int32)
    neighbor_matrix_i32 = neighbor_matrix.astype(jnp.int32)
    batch_idx_i32 = batch_idx.astype(jnp.int32)
    covalent_radii_f32 = covalent_radii.astype(jnp.float32)
    r4r2_f32 = r4r2.astype(jnp.float32)
    c6_reference_f32 = c6_reference.astype(jnp.float32)
    coord_num_ref_f32 = coord_num_ref.astype(jnp.float32)

    # Precompute inv_w for S5 switching
    if s5_smoothing_off > s5_smoothing_on:
        inv_w = 1.0 / (s5_smoothing_off - s5_smoothing_on)
    else:
        inv_w = 0.0

    # Pass 0: Handle PBC - determine if periodic and compute cartesian shifts
    if cell is not None and neighbor_matrix_shifts is not None:
        periodic = True
        cell_f32 = cell.astype(jnp.float32)
        neighbor_matrix_shifts_i32 = neighbor_matrix_shifts.astype(jnp.int32)

        # compute_cartesian_shifts returns a tuple with 1 output
        # Launch dim is derived from first array argument, but we need 2D launch (num_atoms, max_neighbors)
        # Use launch_dims parameter to specify
        (cartesian_shifts,) = compute_cartesian_shifts_nm(
            cell_f32,
            neighbor_matrix_shifts_i32,
            neighbor_matrix_i32,
            batch_idx_i32,
            int(fill_value),
            launch_dims=(num_atoms, max_neighbors),
        )
    else:
        periodic = False
        # Create zero shifts array (not used but need correct shape for kernel)
        cartesian_shifts = jnp.zeros((num_atoms, max_neighbors, 3), dtype=jnp.float32)

    # Pass 1: Compute coordination numbers
    # cn_kernel_nm returns a tuple with 1 output (coord_num)
    # Inputs: positions, numbers, neighbor_matrix, cartesian_shifts, covalent_radii, k1, fill_value, periodic
    # Launch dim is inferred from the first array argument (positions_f32)
    (coord_num,) = cn_kernel_nm(
        positions_f32,
        numbers_i32,
        neighbor_matrix_i32,
        cartesian_shifts,
        covalent_radii_f32,
        float(k1),
        int(fill_value),
        periodic,
    )

    # Pass 2: Compute direct forces, energy, and accumulate dE/dCN
    # direct_forces_kernel_nm returns a tuple with 4 outputs (dE_dCN, forces, energy, virial)
    # Inputs (20): positions, numbers, neighbor_matrix, cartesian_shifts, coord_num, r4r2,
    #              c6_reference, coord_num_ref, k3, a1, a2, s6, s8, s5_on, s5_off, inv_w,
    #              fill_value, periodic, batch_idx, compute_virial
    # Inputs (4 in_out): dE_dCN, forces, energy, virial (pre-allocated, zeroed)
    # Outputs (4): dE_dCN, forces, energy, virial (modified versions returned)
    # Output dims: dE_dCN [num_atoms], forces [num_atoms, 3], energy [num_systems], virial [num_systems, 3, 3]
    # Note: Pre-allocating zeroed arrays is required because jax_kernel does not zero-initialize
    #       and the kernel uses atomic_add for energy/virial
    dE_dCN_init = jnp.zeros(num_atoms, dtype=jnp.float32)
    forces_init = jnp.zeros((num_atoms, 3), dtype=jnp.float32)
    energy_init = jnp.zeros(num_systems, dtype=jnp.float32)
    virial_init = jnp.zeros((num_systems, 3, 3), dtype=jnp.float32)

    dE_dCN, forces, energy, virial = direct_forces_kernel_nm(
        positions_f32,
        numbers_i32,
        neighbor_matrix_i32,
        cartesian_shifts,
        coord_num,
        r4r2_f32,
        c6_reference_f32,
        coord_num_ref_f32,
        float(k3),
        float(a1),
        float(a2),
        float(s6),
        float(s8),
        float(s5_smoothing_on),
        float(s5_smoothing_off),
        float(inv_w),
        int(fill_value),
        periodic,
        batch_idx_i32,
        compute_virial,
        dE_dCN_init,
        forces_init,
        energy_init,
        virial_init,
    )

    # Pass 3: Add CN-dependent force contribution
    # cn_forces_contrib_nm returns a tuple with 2 outputs (forces, virial)
    # Inputs (11): positions, numbers, neighbor_matrix, cartesian_shifts, covalent_radii,
    #              dE_dCN, k1, fill_value, periodic, batch_idx, compute_virial
    # Inputs (2 in_out): forces, virial (pre-allocated, zeroed)
    # Outputs (2): forces, virial (modified versions returned)
    # Note: These are NEW forces/virial arrays - they will be added to existing ones after kernel
    # Note: Pre-allocating zeroed arrays is required because jax_kernel does not zero-initialize
    #       and the kernel reads from forces[atom_i] and uses atomic_add for virial
    forces_cn_init = jnp.zeros((num_atoms, 3), dtype=jnp.float32)
    virial_cn_init = jnp.zeros((num_systems, 3, 3), dtype=jnp.float32)

    forces_cn, virial_cn = cn_forces_contrib_nm(
        positions_f32,
        numbers_i32,
        neighbor_matrix_i32,
        cartesian_shifts,
        covalent_radii_f32,
        dE_dCN,
        float(k1),
        int(fill_value),
        periodic,
        batch_idx_i32,
        compute_virial,
        forces_cn_init,
        virial_cn_init,
    )

    # Add CN force contribution to direct forces
    forces = forces + forces_cn
    virial = virial + virial_cn

    # Return JAX arrays
    if compute_virial:
        return energy, forces, coord_num, virial
    else:
        return energy, forces, coord_num


def _dftd3_nl_impl(
    positions: jax.Array,
    numbers: jax.Array,
    idx_j: jax.Array,
    neighbor_ptr: jax.Array,
    covalent_radii: jax.Array,
    r4r2: jax.Array,
    c6_reference: jax.Array,
    coord_num_ref: jax.Array,
    a1: float,
    a2: float,
    s8: float,
    k1: float = 16.0,
    k3: float = -4.0,
    s6: float = 1.0,
    s5_smoothing_on: float = 1e10,
    s5_smoothing_off: float = 1e10,
    batch_idx: jax.Array | None = None,
    cell: jax.Array | None = None,
    unit_shifts: jax.Array | None = None,
    compute_virial: bool = False,
    num_systems: int | None = None,
) -> (
    tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Internal implementation for neighbor list format using jax_kernel wrappers."""
    num_atoms = positions.shape[0]
    num_edges = idx_j.shape[0]

    # Handle empty case
    if num_atoms == 0 or num_edges == 0:
        if num_systems is None:
            num_systems = 1
            if batch_idx is not None:
                try:
                    num_systems = int(jnp.max(batch_idx)) + 1
                except (
                    jax.errors.ConcretizationTypeError,
                    jax.errors.TracerIntegerConversionError,
                ):
                    raise ValueError(
                        "Cannot infer num_systems inside jax.jit. "
                        "Please provide num_systems explicitly when using jax.jit."
                    ) from None
        empty_energy = jnp.zeros(num_systems, dtype=jnp.float32)
        empty_forces = jnp.zeros((0, 3), dtype=jnp.float32)
        empty_cn = jnp.zeros((0,), dtype=jnp.float32)
        if compute_virial:
            empty_virial = jnp.zeros((num_systems, 3, 3), dtype=jnp.float32)
            return empty_energy, empty_forces, empty_cn, empty_virial
        return empty_energy, empty_forces, empty_cn

    # Determine number of systems
    if num_systems is None:
        if cell is not None:
            num_systems = cell.shape[0]
        elif batch_idx is not None:
            try:
                num_systems = int(jnp.max(batch_idx)) + 1
            except (
                jax.errors.ConcretizationTypeError,
                jax.errors.TracerIntegerConversionError,
            ):
                raise ValueError(
                    "Cannot infer num_systems inside jax.jit. "
                    "Please provide num_systems explicitly when using jax.jit."
                ) from None
        else:
            num_systems = 1

    # Create batch indices if not provided
    if batch_idx is None:
        batch_idx = jnp.zeros(num_atoms, dtype=jnp.int32)

    # Ensure arrays have correct dtypes for kernels (float32 for now)
    positions_f32 = positions.astype(jnp.float32)
    numbers_i32 = numbers.astype(jnp.int32)
    idx_j_i32 = idx_j.astype(jnp.int32)
    neighbor_ptr_i32 = neighbor_ptr.astype(jnp.int32)
    batch_idx_i32 = batch_idx.astype(jnp.int32)
    covalent_radii_f32 = covalent_radii.astype(jnp.float32)
    r4r2_f32 = r4r2.astype(jnp.float32)
    c6_reference_f32 = c6_reference.astype(jnp.float32)
    coord_num_ref_f32 = coord_num_ref.astype(jnp.float32)

    # Precompute inv_w for S5 switching
    if s5_smoothing_off > s5_smoothing_on:
        inv_w = 1.0 / (s5_smoothing_off - s5_smoothing_on)
    else:
        inv_w = 0.0

    # Pass 0: Handle PBC - determine if periodic and compute cartesian shifts
    if unit_shifts is not None and cell is not None:
        periodic = True
        cell_f32 = cell.astype(jnp.float32)
        unit_shifts_i32 = unit_shifts.astype(jnp.int32)

        # compute_cartesian_shifts_nl returns a tuple with 1 output
        (cartesian_shifts,) = compute_cartesian_shifts_nl(
            cell_f32,
            unit_shifts_i32,
            neighbor_ptr_i32,
            batch_idx_i32,
        )
    else:
        periodic = False
        # Create zero shifts array (not used but need correct shape for kernel)
        cartesian_shifts = jnp.zeros((num_edges, 3), dtype=jnp.float32)

    # Pass 1: Compute coordination numbers
    # cn_kernel_nl returns a tuple with 1 output (coord_num)
    # Inputs: positions, numbers, idx_j, neighbor_ptr, cartesian_shifts, covalent_radii, k1, periodic
    # Launch dim is inferred from the first array argument (positions_f32)
    (coord_num,) = cn_kernel_nl(
        positions_f32,
        numbers_i32,
        idx_j_i32,
        neighbor_ptr_i32,
        cartesian_shifts,
        covalent_radii_f32,
        float(k1),
        periodic,
    )

    # Pass 2: Compute direct forces, energy, and accumulate dE/dCN
    # direct_forces_kernel_nl returns a tuple with 4 outputs (dE_dCN, forces, energy, virial)
    # Inputs (17): positions, numbers, idx_j, neighbor_ptr, cartesian_shifts, coord_num, r4r2,
    #              c6_reference, coord_num_ref, k3, a1, a2, s6, s8, s5_on, s5_off, inv_w,
    #              periodic, batch_idx, compute_virial
    # Inputs (4 in_out): dE_dCN, forces, energy, virial (pre-allocated, zeroed)
    # Outputs (4): dE_dCN, forces, energy, virial (modified versions returned)
    # Note: Pre-allocating zeroed arrays is required because jax_kernel does not zero-initialize
    #       and the kernel uses atomic_add for energy/virial
    dE_dCN_init = jnp.zeros(num_atoms, dtype=jnp.float32)
    forces_init = jnp.zeros((num_atoms, 3), dtype=jnp.float32)
    energy_init = jnp.zeros(num_systems, dtype=jnp.float32)
    virial_init = jnp.zeros((num_systems, 3, 3), dtype=jnp.float32)

    dE_dCN, forces, energy, virial = direct_forces_kernel_nl(
        positions_f32,
        numbers_i32,
        idx_j_i32,
        neighbor_ptr_i32,
        cartesian_shifts,
        coord_num,
        r4r2_f32,
        c6_reference_f32,
        coord_num_ref_f32,
        float(k3),
        float(a1),
        float(a2),
        float(s6),
        float(s8),
        float(s5_smoothing_on),
        float(s5_smoothing_off),
        float(inv_w),
        periodic,
        batch_idx_i32,
        compute_virial,
        dE_dCN_init,
        forces_init,
        energy_init,
        virial_init,
    )

    # Pass 3: Add CN-dependent force contribution
    # cn_forces_contrib_nl returns a tuple with 2 outputs (forces, virial)
    # Inputs (9): positions, numbers, idx_j, neighbor_ptr, cartesian_shifts, covalent_radii,
    #              dE_dCN, k1, periodic, batch_idx, compute_virial
    # Inputs (2 in_out): forces, virial (pre-allocated, zeroed)
    # Outputs (2): forces, virial (modified versions returned)
    # Note: These are NEW forces/virial arrays - they will be added to existing ones after kernel
    # Note: Pre-allocating zeroed arrays is required because jax_kernel does not zero-initialize
    #       and the kernel reads from forces[atom_i] and uses atomic_add for virial
    forces_cn_init = jnp.zeros((num_atoms, 3), dtype=jnp.float32)
    virial_cn_init = jnp.zeros((num_systems, 3, 3), dtype=jnp.float32)

    forces_cn, virial_cn = cn_forces_contrib_nl(
        positions_f32,
        numbers_i32,
        idx_j_i32,
        neighbor_ptr_i32,
        cartesian_shifts,
        covalent_radii_f32,
        dE_dCN,
        float(k1),
        periodic,
        batch_idx_i32,
        compute_virial,
        forces_cn_init,
        virial_cn_init,
    )

    # Add CN force contribution to direct forces
    forces = forces + forces_cn
    virial = virial + virial_cn

    # Return JAX arrays
    if compute_virial:
        return energy, forces, coord_num, virial
    else:
        return energy, forces, coord_num


def dftd3(
    positions: jax.Array,
    numbers: jax.Array,
    a1: float,
    a2: float,
    s8: float,
    k1: float = 16.0,
    k3: float = -4.0,
    s6: float = 1.0,
    s5_smoothing_on: float = 1e10,
    s5_smoothing_off: float = 1e10,
    fill_value: int | None = None,
    d3_params: D3Parameters | dict[str, jax.Array] | None = None,
    covalent_radii: jax.Array | None = None,
    r4r2: jax.Array | None = None,
    c6_reference: jax.Array | None = None,
    coord_num_ref: jax.Array | None = None,
    batch_idx: jax.Array | None = None,
    cell: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    neighbor_list: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    unit_shifts: jax.Array | None = None,
    compute_virial: bool = False,
    num_systems: int | None = None,
) -> (
    tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """
    Compute DFT-D3(BJ) dispersion energy and forces using Warp with JAX arrays.

    **DFT-D3 parameters must be explicitly provided** using one of three methods:

    1. **D3Parameters dataclass**: Supply a :class:`D3Parameters` instance (recommended).
       Individual parameters can override dataclass values if both are provided.

    2. **Explicit parameters**: Supply all four parameters individually:
       ``covalent_radii``, ``r4r2``, ``c6_reference``, and ``coord_num_ref``.

    3. **Dictionary**: Provide a ``d3_params`` dictionary with keys:
       ``"rcov"``, ``"r4r2"``, ``"c6ab"``, and ``"cn_ref"``.
       Individual parameters can override dictionary values if both are provided.

    See ``examples/dispersion/utils.py`` for parameter generation utilities.

    Parameters
    ----------
    positions : jax.Array
        Atomic coordinates [num_atoms, 3] as float32 or float64, in consistent distance
        units (conventionally Bohr when using standard D3 parameters)
    numbers : jax.Array
        Atomic numbers [num_atoms] as int32
    a1 : float
        Becke-Johnson damping parameter 1 (functional-dependent, dimensionless)
    a2 : float
        Becke-Johnson damping parameter 2 (functional-dependent), in same units as positions
    s8 : float
        C8 term scaling factor (functional-dependent, dimensionless)
    k1 : float, optional
        CN counting function steepness parameter, in inverse distance units
        (typically 16.0 1/Bohr for atomic units). Default: 16.0
    k3 : float, optional
        CN interpolation Gaussian width parameter (typically -4.0, dimensionless).
        Default: -4.0
    s6 : float, optional
        C6 term scaling factor (typically 1.0, dimensionless). Default: 1.0
    s5_smoothing_on : float, optional
        Distance where S5 switching begins, in same units as positions. Default: 1e10
    s5_smoothing_off : float, optional
        Distance where S5 switching completes, in same units as positions.
        Default: 1e10 (effectively no cutoff)
    fill_value : int | None, optional
        Value indicating padding in neighbor_matrix. If None, defaults to num_atoms.
        Default: None
    d3_params : D3Parameters | dict[str, jax.Array] | None, optional
        DFT-D3 parameters provided as either:
        - :class:`D3Parameters` dataclass instance (recommended)
        - Dictionary with keys: "rcov", "r4r2", "c6ab", "cn_ref"
        Individual parameters below can override values from d3_params.
    covalent_radii : jax.Array | None, optional
        Covalent radii [max_Z+1] as float32, indexed by atomic number, in same units
        as positions. If provided, overrides the value in d3_params.
    r4r2 : jax.Array | None, optional
        <r4>/<r2> expectation values [max_Z+1] as float32 for C8 computation (dimensionless).
        If provided, overrides the value in d3_params.
    c6_reference : jax.Array | None, optional
        C6 reference values [max_Z+1, max_Z+1, 5, 5] as float32 in energy × distance^6 units.
        If provided, overrides the value in d3_params.
    coord_num_ref : jax.Array | None, optional
        CN reference grid [max_Z+1, max_Z+1, 5, 5] as float32 (dimensionless).
        If provided, overrides the value in d3_params.
    batch_idx : jax.Array or None, optional
        Batch indices [num_atoms] as int32. If None, all atoms are assumed
        to be in a single system (batch 0). Default: None
    cell : jax.Array or None, optional
        Unit cell lattice vectors [num_systems, 3, 3] for PBC, in same dtype and units as positions.
        Convention: cell[s, i, :] is i-th lattice vector for system s.
        If None, non-periodic calculation. Default: None
    neighbor_matrix : jax.Array | None, optional
        Neighbor indices [num_atoms, max_neighbors] as int32. Each row i contains
        indices of atom i's neighbors, padded with ``fill_value`` for unused slots.
        Mutually exclusive with ``neighbor_list``. Default: None
    neighbor_matrix_shifts : jax.Array or None, optional
        Integer unit cell shifts [num_atoms, max_neighbors, 3] as int32 for PBC with
        neighbor_matrix format. If None, non-periodic calculation. Mutually exclusive
        with unit_shifts. Default: None
    neighbor_list : jax.Array or None, optional
        Neighbor pairs [2, num_pairs] as int32 in COO format, where row 0 contains
        source atom indices and row 1 contains target atom indices. Alternative to
        neighbor_matrix for sparse neighbor representations. Mutually exclusive with
        neighbor_matrix. Must be used together with `neighbor_ptr`. Default: None
    neighbor_ptr : jax.Array or None, optional
        CSR row pointers [num_atoms+1] as int32. Required when using `neighbor_list`.
        Indicates that `neighbor_list[1, :]` contains destination atoms in CSR format.
        Default: None
    unit_shifts : jax.Array or None, optional
        Integer unit cell shifts [num_pairs, 3] as int32 for PBC with neighbor_list
        format. If None, non-periodic calculation. Mutually exclusive with
        neighbor_matrix_shifts. Default: None
    compute_virial : bool, optional
        If True, compute and return virial tensor. Default: False
    num_systems : int, optional
        Number of systems in batch. If None, inferred from ``cell``
        or from ``batch_idx`` (introduces device synchronization overhead). Default: None

    Returns
    -------
    energy : jax.Array
        Total dispersion energy [num_systems] as float32. Units are energy
        (Hartree when using standard D3 parameters).
    forces : jax.Array
        Atomic forces [num_atoms, 3] as float32. Units are energy/distance
        (Hartree/Bohr when using standard D3 parameters).
    coord_num : jax.Array
        Coordination numbers [num_atoms] as float32 (dimensionless)
    virial : jax.Array, optional
        Virial tensor [num_systems, 3, 3] as float32. Only returned
        if compute_virial=True.

    Notes
    -----
    - **Unit consistency**: All inputs must use consistent units. Standard D3 parameters
      from the Grimme group use atomic units (Bohr for distances, Hartree for energy).
    - Float32 or float64 precision for positions and cell; outputs always float32
    - **Neighbor formats**: Supports both neighbor_matrix (dense) and neighbor_list (sparse)
      formats. Choose neighbor_list for sparse systems or when memory efficiency is important.
    - Padding atoms indicated by numbers[i] == 0
    - Requires symmetric neighbor representation (each pair appears twice)
    - **Two-body only**: Computes pairwise C6 and C8 dispersion terms; three-body
      Axilrod-Teller-Muto (ATM/C9) terms are not included
    - Virial computation requires periodic boundary conditions.

    Raises
    ------
    ValueError
        If neighbor format is invalid or PBC requirements are not met
    RuntimeError
        If DFT-D3 parameters are not provided

    Examples
    --------
    Using neighbor matrix format:

    >>> energy, forces, coord_num = dftd3(
    ...     positions, numbers,
    ...     neighbor_matrix=neighbor_matrix,
    ...     a1=0.3981, a2=4.4211, s8=1.9889,
    ...     d3_params=params,
    ... )

    Using neighbor list format with PBC:

    >>> energy, forces, coord_num, virial = dftd3(
    ...     positions, numbers,
    ...     neighbor_list=neighbor_list,
    ...     neighbor_ptr=neighbor_ptr,
    ...     a1=0.3981, a2=4.4211, s8=1.9889,
    ...     d3_params=params,
    ...     cell=cell,
    ...     unit_shifts=unit_shifts,
    ...     compute_virial=True,
    ... )
    """
    # Validate neighbor format inputs
    matrix_provided = neighbor_matrix is not None
    list_provided = neighbor_list is not None

    if matrix_provided and list_provided:
        raise ValueError(
            "Cannot provide both neighbor_matrix and neighbor_list. "
            "Please provide only one neighbor representation format."
        )
    if not matrix_provided and not list_provided:
        raise ValueError("Must provide either neighbor_matrix or neighbor_list.")

    # Validate PBC shift inputs match neighbor format
    if matrix_provided and unit_shifts is not None:
        raise ValueError(
            "unit_shifts is for neighbor_list format. "
            "Use neighbor_matrix_shifts for neighbor_matrix format."
        )
    if list_provided and neighbor_matrix_shifts is not None:
        raise ValueError(
            "neighbor_matrix_shifts is for neighbor_matrix format. "
            "Use unit_shifts for neighbor_list format."
        )

    # Validate neighbor_ptr is provided when using neighbor_list format
    if list_provided and neighbor_ptr is None:
        raise ValueError(
            "neighbor_ptr must be provided when using neighbor_list format."
        )

    # Validate functional parameters
    if a1 is None or a2 is None or s8 is None:
        raise ValueError(
            "Functional parameters a1, a2, and s8 must be provided. "
            "These are functional-dependent parameters required for DFT-D3(BJ) calculations."
        )

    # Validate virial computation requires PBC
    if compute_virial:
        if cell is None:
            raise ValueError(
                "Virial computation requires periodic boundary conditions. "
                "Please provide unit cell parameters (cell) and shifts."
            )
        if matrix_provided and neighbor_matrix_shifts is None:
            raise ValueError(
                "Virial computation requires neighbor_matrix_shifts for neighbor_matrix format."
            )
        if list_provided and unit_shifts is None:
            raise ValueError(
                "Virial computation requires unit_shifts for neighbor_list format."
            )

    # Determine how parameters are being supplied
    if all(
        param is not None
        for param in [covalent_radii, r4r2, c6_reference, coord_num_ref]
    ):
        # Use explicit parameters directly
        pass
    elif d3_params is not None:
        # Convert D3Parameters to dictionary for consistent access
        if isinstance(d3_params, D3Parameters):
            d3_dict = {
                "rcov": d3_params.rcov,
                "r4r2": d3_params.r4r2,
                "c6ab": d3_params.c6ab,
                "cn_ref": d3_params.cn_ref,
            }
        else:
            d3_dict = d3_params

        # Set parameters from dictionary if not already set
        if covalent_radii is None:
            covalent_radii = d3_dict["rcov"]
        if r4r2 is None:
            r4r2 = d3_dict["r4r2"]
        if c6_reference is None:
            c6_reference = d3_dict["c6ab"]
        if coord_num_ref is None:
            coord_num_ref = d3_dict["cn_ref"]
    else:
        raise RuntimeError(
            "DFT-D3 parameters must be explicitly provided. "
            "Either supply all individual parameters (covalent_radii, r4r2, "
            "c6_reference, coord_num_ref), provide a D3Parameters instance, "
            "or provide a d3_params dictionary."
        )

    # Determine number of systems for energy allocation
    if num_systems is None:
        if batch_idx is None:
            num_systems = 1
        elif cell is not None:
            num_systems = cell.shape[0]
        else:
            try:
                num_systems = int(jnp.max(batch_idx)) + 1
            except (
                jax.errors.ConcretizationTypeError,
                jax.errors.TracerIntegerConversionError,
            ):
                raise ValueError(
                    "Cannot infer num_systems inside jax.jit. "
                    "Please provide num_systems explicitly when using jax.jit."
                ) from None

    # Dispatch to appropriate implementation based on neighbor format
    if neighbor_matrix is not None:
        return _dftd3_nm_impl(
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
            k1=k1,
            k3=k3,
            s6=s6,
            s5_smoothing_on=s5_smoothing_on,
            s5_smoothing_off=s5_smoothing_off,
            fill_value=fill_value,
            batch_idx=batch_idx,
            cell=cell,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_virial=compute_virial,
            num_systems=num_systems,
        )
    else:
        # Extract idx_j from neighbor_list (row 1 contains destination atoms)
        idx_j_csr = neighbor_list[1]

        return _dftd3_nl_impl(
            positions=positions,
            numbers=numbers,
            idx_j=idx_j_csr,
            neighbor_ptr=neighbor_ptr,
            covalent_radii=covalent_radii,
            r4r2=r4r2,
            c6_reference=c6_reference,
            coord_num_ref=coord_num_ref,
            a1=a1,
            a2=a2,
            s8=s8,
            k1=k1,
            k3=k3,
            s6=s6,
            s5_smoothing_on=s5_smoothing_on,
            s5_smoothing_off=s5_smoothing_off,
            batch_idx=batch_idx,
            cell=cell,
            unit_shifts=unit_shifts,
            compute_virial=compute_virial,
            num_systems=num_systems,
        )
