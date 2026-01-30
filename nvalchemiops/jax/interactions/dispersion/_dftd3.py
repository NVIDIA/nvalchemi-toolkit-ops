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

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import warp as wp

from nvalchemiops.interactions.dispersion._dftd3 import (
    dftd3_nl as wp_dftd3_nl,
)
from nvalchemiops.interactions.dispersion._dftd3 import (
    dftd3_nm as wp_dftd3_nm,
)

__all__ = [
    "D3Parameters",
    "dftd3",
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
    device: str | None = None,
) -> (
    tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Internal implementation for neighbor matrix format using Warp kernels."""
    num_atoms = positions.shape[0]

    # Set fill_value if not provided
    if fill_value is None:
        fill_value = num_atoms

    # Handle empty case
    if num_atoms == 0:
        num_systems = 1
        if batch_idx is not None:
            num_systems = int(jnp.max(batch_idx)) + 1
        empty_energy = jnp.zeros(num_systems, dtype=jnp.float32)
        empty_forces = jnp.zeros((0, 3), dtype=jnp.float32)
        empty_cn = jnp.zeros((0,), dtype=jnp.float32)
        if compute_virial:
            empty_virial = jnp.zeros((num_systems, 3, 3), dtype=jnp.float32)
            return empty_energy, empty_forces, empty_cn, empty_virial
        return empty_energy, empty_forces, empty_cn

    # Determine number of systems
    if cell is not None:
        num_systems = cell.shape[0]
    elif batch_idx is not None:
        num_systems = int(jnp.max(batch_idx)) + 1
    else:
        num_systems = 1

    # Determine vector/matrix dtype based on positions
    if positions.dtype == jnp.float64:
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d
    else:
        vec_dtype = wp.vec3f
        mat_dtype = wp.mat33f

    # Infer device from JAX array placement if not provided
    # Also get JAX device for output array allocation
    if device is None:
        # Check if JAX array is on GPU
        try:
            jax_device = positions.devices().pop()
            device_kind = jax_device.platform
            if device_kind == "gpu":
                device = "cuda:0"
            else:
                device = "cpu"
        except (AttributeError, KeyError):
            jax_device = jax.devices("cpu")[0]
            device = "cpu"
    else:
        # Map warp device string to JAX device
        if device.startswith("cuda"):
            jax_device = jax.devices("gpu")[0]
        else:
            jax_device = jax.devices("cpu")[0]

    # Create batch indices if not provided
    if batch_idx is None:
        batch_idx = jax.device_put(jnp.zeros(num_atoms, dtype=jnp.int32), jax_device)

    # Convert JAX input arrays to Warp using dlpack (zero-copy when possible)
    # JAX arrays are already contiguous by design
    positions_wp = wp.from_dlpack(positions, dtype=vec_dtype)
    numbers_wp = wp.from_dlpack(numbers.astype(jnp.int32), dtype=wp.int32)
    neighbor_matrix_wp = wp.from_dlpack(
        neighbor_matrix.astype(jnp.int32), dtype=wp.int32
    )
    batch_idx_wp = wp.from_dlpack(batch_idx.astype(jnp.int32), dtype=wp.int32)

    # Convert parameters to float32 and to Warp arrays using dlpack
    covalent_radii_wp = wp.from_dlpack(
        covalent_radii.astype(jnp.float32), dtype=wp.float32
    )
    r4r2_wp = wp.from_dlpack(r4r2.astype(jnp.float32), dtype=wp.float32)
    c6_reference_wp = wp.from_dlpack(c6_reference.astype(jnp.float32), dtype=wp.float32)
    coord_num_ref_wp = wp.from_dlpack(
        coord_num_ref.astype(jnp.float32), dtype=wp.float32
    )

    # Handle cell and shifts for PBC
    if cell is not None and neighbor_matrix_shifts is not None:
        cell_wp = wp.from_dlpack(cell.astype(positions.dtype), dtype=mat_dtype)
        neighbor_matrix_shifts_wp = wp.from_dlpack(
            neighbor_matrix_shifts.astype(jnp.int32), dtype=wp.vec3i
        )
    else:
        cell_wp = None
        neighbor_matrix_shifts_wp = None

    # Allocate output arrays in JAX on the same device as inputs
    # Warp kernel writes directly to JAX array's memory (zero-copy)
    energy = jax.device_put(jnp.zeros(num_systems, dtype=jnp.float32), jax_device)
    forces = jax.device_put(jnp.zeros((num_atoms, 3), dtype=jnp.float32), jax_device)
    coord_num = jax.device_put(jnp.zeros(num_atoms, dtype=jnp.float32), jax_device)
    if compute_virial:
        virial = jax.device_put(
            jnp.zeros((num_systems, 3, 3), dtype=jnp.float32), jax_device
        )
    else:
        virial = jax.device_put(jnp.zeros((0, 3, 3), dtype=jnp.float32), jax_device)

    # Convert JAX output arrays to Warp via dlpack (zero-copy view)
    energy_wp = wp.from_dlpack(energy, dtype=wp.float32)
    forces_wp = wp.from_dlpack(forces, dtype=wp.vec3f)
    coord_num_wp = wp.from_dlpack(coord_num, dtype=wp.float32)
    virial_wp = wp.from_dlpack(virial, dtype=wp.mat33f)

    # Call Warp launcher
    wp_dftd3_nm(
        positions=positions_wp,
        numbers=numbers_wp,
        neighbor_matrix=neighbor_matrix_wp,
        covalent_radii=covalent_radii_wp,
        r4r2=r4r2_wp,
        c6_reference=c6_reference_wp,
        coord_num_ref=coord_num_ref_wp,
        a1=a1,
        a2=a2,
        s8=s8,
        coord_num=coord_num_wp,
        forces=forces_wp,
        energy=energy_wp,
        virial=virial_wp,
        vec_dtype=vec_dtype,
        k1=k1,
        k3=k3,
        s6=s6,
        s5_smoothing_on=s5_smoothing_on,
        s5_smoothing_off=s5_smoothing_off,
        fill_value=fill_value,
        batch_idx=batch_idx_wp,
        cell=cell_wp,
        neighbor_matrix_shifts=neighbor_matrix_shifts_wp,
        compute_virial=compute_virial,
        device=device,
    )

    # Synchronize device to ensure Warp kernel writes are visible to JAX
    wp.synchronize_device(device)

    # Return JAX arrays (which now contain computed values via shared memory)
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
    device: str | None = None,
) -> (
    tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Internal implementation for neighbor list format using Warp kernels."""
    num_atoms = positions.shape[0]
    num_edges = idx_j.shape[0]

    # Handle empty case
    if num_atoms == 0 or num_edges == 0:
        num_systems = 1
        if batch_idx is not None:
            num_systems = int(jnp.max(batch_idx)) + 1
        empty_energy = jnp.zeros(num_systems, dtype=jnp.float32)
        empty_forces = jnp.zeros((0, 3), dtype=jnp.float32)
        empty_cn = jnp.zeros((0,), dtype=jnp.float32)
        if compute_virial:
            empty_virial = jnp.zeros((num_systems, 3, 3), dtype=jnp.float32)
            return empty_energy, empty_forces, empty_cn, empty_virial
        return empty_energy, empty_forces, empty_cn

    # Determine number of systems
    if cell is not None:
        num_systems = cell.shape[0]
    elif batch_idx is not None:
        num_systems = int(jnp.max(batch_idx)) + 1
    else:
        num_systems = 1

    # Determine vector/matrix dtype based on positions
    if positions.dtype == jnp.float64:
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d
    else:
        vec_dtype = wp.vec3f
        mat_dtype = wp.mat33f

    # Infer device from JAX array placement if not provided
    # Also get JAX device for output array allocation
    if device is None:
        # Check if JAX array is on GPU
        try:
            jax_device = positions.devices().pop()
            device_kind = jax_device.platform
            if device_kind == "gpu":
                device = "cuda:0"
            else:
                device = "cpu"
        except (AttributeError, KeyError):
            jax_device = jax.devices("cpu")[0]
            device = "cpu"
    else:
        # Map warp device string to JAX device
        if device.startswith("cuda"):
            jax_device = jax.devices("gpu")[0]
        else:
            jax_device = jax.devices("cpu")[0]

    # Create batch indices if not provided
    if batch_idx is None:
        batch_idx = jax.device_put(jnp.zeros(num_atoms, dtype=jnp.int32), jax_device)

    # Convert JAX input arrays to Warp using dlpack (zero-copy when possible)
    # JAX arrays are already contiguous by design
    positions_wp = wp.from_dlpack(positions, dtype=vec_dtype)
    numbers_wp = wp.from_dlpack(numbers.astype(jnp.int32), dtype=wp.int32)
    idx_j_wp = wp.from_dlpack(idx_j.astype(jnp.int32), dtype=wp.int32)
    neighbor_ptr_wp = wp.from_dlpack(neighbor_ptr.astype(jnp.int32), dtype=wp.int32)
    batch_idx_wp = wp.from_dlpack(batch_idx.astype(jnp.int32), dtype=wp.int32)

    # Convert parameters to float32 and to Warp arrays using dlpack
    covalent_radii_wp = wp.from_dlpack(
        covalent_radii.astype(jnp.float32), dtype=wp.float32
    )
    r4r2_wp = wp.from_dlpack(r4r2.astype(jnp.float32), dtype=wp.float32)
    c6_reference_wp = wp.from_dlpack(c6_reference.astype(jnp.float32), dtype=wp.float32)
    coord_num_ref_wp = wp.from_dlpack(
        coord_num_ref.astype(jnp.float32), dtype=wp.float32
    )

    # Handle cell and shifts for PBC
    if unit_shifts is not None and cell is not None:
        cell_wp = wp.from_dlpack(cell.astype(positions.dtype), dtype=mat_dtype)
        unit_shifts_wp = wp.from_dlpack(unit_shifts.astype(jnp.int32), dtype=wp.vec3i)
    else:
        cell_wp = None
        unit_shifts_wp = None

    # Allocate output arrays in JAX on the same device as inputs
    # Warp kernel writes directly to JAX array's memory (zero-copy)
    energy = jax.device_put(jnp.zeros(num_systems, dtype=jnp.float32), jax_device)
    forces = jax.device_put(jnp.zeros((num_atoms, 3), dtype=jnp.float32), jax_device)
    coord_num = jax.device_put(jnp.zeros(num_atoms, dtype=jnp.float32), jax_device)
    if compute_virial:
        virial = jax.device_put(
            jnp.zeros((num_systems, 3, 3), dtype=jnp.float32), jax_device
        )
    else:
        virial = jax.device_put(jnp.zeros((0, 3, 3), dtype=jnp.float32), jax_device)

    # Convert JAX output arrays to Warp via dlpack (zero-copy view)
    energy_wp = wp.from_dlpack(energy, dtype=wp.float32)
    forces_wp = wp.from_dlpack(forces, dtype=wp.vec3f)
    coord_num_wp = wp.from_dlpack(coord_num, dtype=wp.float32)
    virial_wp = wp.from_dlpack(virial, dtype=wp.mat33f)

    # Call Warp launcher
    wp_dftd3_nl(
        positions=positions_wp,
        numbers=numbers_wp,
        idx_j=idx_j_wp,
        neighbor_ptr=neighbor_ptr_wp,
        covalent_radii=covalent_radii_wp,
        r4r2=r4r2_wp,
        c6_reference=c6_reference_wp,
        coord_num_ref=coord_num_ref_wp,
        a1=a1,
        a2=a2,
        s8=s8,
        coord_num=coord_num_wp,
        forces=forces_wp,
        energy=energy_wp,
        virial=virial_wp,
        vec_dtype=vec_dtype,
        k1=k1,
        k3=k3,
        s6=s6,
        s5_smoothing_on=s5_smoothing_on,
        s5_smoothing_off=s5_smoothing_off,
        batch_idx=batch_idx_wp,
        cell=cell_wp,
        unit_shifts=unit_shifts_wp,
        compute_virial=compute_virial,
        device=device,
    )

    # Synchronize device to ensure Warp kernel writes are visible to JAX
    wp.synchronize_device(device)

    # Return JAX arrays (which now contain computed values via shared memory)
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
    device: str | None = None,
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
        Neighbor indices [num_atoms, max_neighbors] as int32. See PyTorch version docstring
        for details on the format. Mutually exclusive with neighbor_list. Default: None
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
        Number of systems in batch. If none provided, inferred from cell
        or from batch_idx. Default: None
    device : str or None, optional
        Warp device string (e.g., 'cuda:0', 'cpu'). If None, inferred from
        positions array placement. Default: None

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
            num_systems = int(jnp.max(batch_idx)) + 1

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
            device=device,
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
            device=device,
        )
