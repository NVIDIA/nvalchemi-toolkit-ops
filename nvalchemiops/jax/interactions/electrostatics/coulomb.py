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

"""JAX Coulomb electrostatics implementation.

Wraps the framework-agnostic Warp kernels from
``nvalchemiops.interactions.electrostatics.coulomb`` with JAX bindings.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from warp.jax_experimental import jax_kernel

from nvalchemiops.interactions.electrostatics.coulomb import (
    _batch_coulomb_energy_forces_kernel,
    _batch_coulomb_energy_forces_matrix_kernel,
    _batch_coulomb_energy_kernel,
    _batch_coulomb_energy_matrix_kernel,
    _coulomb_energy_forces_kernel,
    _coulomb_energy_forces_matrix_kernel,
    _coulomb_energy_kernel,
    _coulomb_energy_matrix_kernel,
)

__all__ = [
    "coulomb_energy",
    "coulomb_forces",
    "coulomb_energy_forces",
]

# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# --- Neighbor List (CSR) Format ---

_jax_coulomb_energy_list = jax_kernel(
    _coulomb_energy_kernel,
    num_outputs=1,
    in_out_argnames=["energies"],
    enable_backward=False,
)

_jax_coulomb_energy_forces_list = jax_kernel(
    _coulomb_energy_forces_kernel,
    num_outputs=2,
    in_out_argnames=["energies", "forces"],
    enable_backward=False,
)

_jax_batch_coulomb_energy_list = jax_kernel(
    _batch_coulomb_energy_kernel,
    num_outputs=1,
    in_out_argnames=["energies"],
    enable_backward=False,
)

_jax_batch_coulomb_energy_forces_list = jax_kernel(
    _batch_coulomb_energy_forces_kernel,
    num_outputs=2,
    in_out_argnames=["energies", "forces"],
    enable_backward=False,
)

# --- Neighbor Matrix Format ---

_jax_coulomb_energy_matrix = jax_kernel(
    _coulomb_energy_matrix_kernel,
    num_outputs=1,
    in_out_argnames=["atomic_energies"],
    enable_backward=False,
)

_jax_coulomb_energy_forces_matrix = jax_kernel(
    _coulomb_energy_forces_matrix_kernel,
    num_outputs=2,
    in_out_argnames=["atomic_energies", "atomic_forces"],
    enable_backward=False,
)

_jax_batch_coulomb_energy_matrix = jax_kernel(
    _batch_coulomb_energy_matrix_kernel,
    num_outputs=1,
    in_out_argnames=["atomic_energies"],
    enable_backward=False,
)

_jax_batch_coulomb_energy_forces_matrix = jax_kernel(
    _batch_coulomb_energy_forces_matrix_kernel,
    num_outputs=2,
    in_out_argnames=["atomic_energies", "atomic_forces"],
    enable_backward=False,
)


# ==============================================================================
# Public API
# ==============================================================================


def coulomb_energy(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    cutoff: float,
    alpha: float = 0.0,
    neighbor_list: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    neighbor_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    fill_value: int | None = None,
    batch_idx: jax.Array | None = None,
) -> jax.Array:
    """Compute Coulomb electrostatic energies.

    Computes pairwise electrostatic energies using the Coulomb law,
    with optional erfc damping for Ewald/PME real-space calculations.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    charges : jax.Array, shape (N,)
        Atomic charges.
    cell : jax.Array, shape (1, 3, 3) or (B, 3, 3)
        Unit cell matrix. Shape (B, 3, 3) for batched calculations.
    cutoff : float
        Cutoff distance for interactions.
    alpha : float, default=0.0
        Ewald splitting parameter. Use 0.0 for undamped Coulomb.
    neighbor_list : jax.Array | None, shape (2, num_pairs)
        Neighbor pairs in COO format. Row 0 = source, Row 1 = target.
    neighbor_ptr : jax.Array | None, shape (N+1,)
        CSR row pointers for neighbor list. Required with neighbor_list.
        Provided by neighborlist module.
    neighbor_shifts : jax.Array | None, shape (num_pairs, 3)
        Integer unit cell shifts for neighbor list format.
    neighbor_matrix : jax.Array | None, shape (N, max_neighbors)
        Neighbor indices in matrix format.
    neighbor_matrix_shifts : jax.Array | None, shape (N, max_neighbors, 3)
        Integer unit cell shifts for matrix format.
    fill_value : int | None
        Fill value for neighbor matrix padding.
    batch_idx : jax.Array | None, shape (N,)
        Batch indices for each atom.

    Returns
    -------
    energies : jax.Array, shape (N,)
        Per-atom energies. Sum to get total energy.

    Examples
    --------
    >>> # Direct Coulomb (undamped)
    >>> energies = coulomb_energy(
    ...     positions, charges, cell, cutoff=10.0, alpha=0.0,
    ...     neighbor_list=neighbor_list, neighbor_ptr=neighbor_ptr,
    ...     neighbor_shifts=neighbor_shifts
    ... )
    >>> total_energy = energies.sum()

    >>> # Ewald/PME real-space (damped)
    >>> energies = coulomb_energy(
    ...     positions, charges, cell, cutoff=10.0, alpha=0.3,
    ...     neighbor_list=neighbor_list, neighbor_ptr=neighbor_ptr,
    ...     neighbor_shifts=neighbor_shifts
    ... )
    """
    # Validate inputs
    use_list = neighbor_list is not None and neighbor_shifts is not None
    use_matrix = neighbor_matrix is not None and neighbor_matrix_shifts is not None

    if not use_list and not use_matrix:
        raise ValueError(
            "Must provide either neighbor_list/neighbor_shifts or "
            "neighbor_matrix/neighbor_matrix_shifts"
        )

    if use_list and use_matrix:
        raise ValueError(
            "Cannot provide both neighbor list and neighbor matrix formats"
        )

    # Store original dtype for output
    original_dtype = positions.dtype

    # Convert to float64 for numerical stability
    positions_f64 = positions.astype(jnp.float64)
    charges_f64 = charges.astype(jnp.float64)
    cell_f64 = cell.astype(jnp.float64)

    # Ensure cell is (B, 3, 3)
    if cell_f64.ndim == 2:
        cell_f64 = cell_f64[jnp.newaxis, :, :]

    num_atoms = positions_f64.shape[0]
    is_batched = batch_idx is not None

    # Allocate output
    energies = jnp.zeros(num_atoms, dtype=jnp.float64)

    if use_list:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr is required when using neighbor_list format")

        # Extract idx_j (target indices) from neighbor_list
        idx_j = neighbor_list[1].astype(jnp.int32)
        neighbor_ptr_i32 = neighbor_ptr.astype(jnp.int32)
        neighbor_shifts_i32 = neighbor_shifts.astype(jnp.int32)

        if is_batched:
            batch_idx_i32 = batch_idx.astype(jnp.int32)
            (energies,) = _jax_batch_coulomb_energy_list(
                positions_f64,
                charges_f64,
                cell_f64,
                idx_j,
                neighbor_ptr_i32,
                neighbor_shifts_i32,
                batch_idx_i32,
                float(cutoff),
                float(alpha),
                energies,
                launch_dims=(num_atoms,),
            )
        else:
            (energies,) = _jax_coulomb_energy_list(
                positions_f64,
                charges_f64,
                cell_f64,
                idx_j,
                neighbor_ptr_i32,
                neighbor_shifts_i32,
                float(cutoff),
                float(alpha),
                energies,
                launch_dims=(num_atoms,),
            )
    else:
        neighbor_matrix_i32 = neighbor_matrix.astype(jnp.int32)
        neighbor_matrix_shifts_i32 = neighbor_matrix_shifts.astype(jnp.int32)

        if fill_value is None:
            fill_value = num_atoms

        if is_batched:
            batch_idx_i32 = batch_idx.astype(jnp.int32)
            (energies,) = _jax_batch_coulomb_energy_matrix(
                positions_f64,
                charges_f64,
                cell_f64,
                neighbor_matrix_i32,
                neighbor_matrix_shifts_i32,
                batch_idx_i32,
                float(cutoff),
                float(alpha),
                int(fill_value),
                energies,
                launch_dims=(num_atoms,),
            )
        else:
            (energies,) = _jax_coulomb_energy_matrix(
                positions_f64,
                charges_f64,
                cell_f64,
                neighbor_matrix_i32,
                neighbor_matrix_shifts_i32,
                float(cutoff),
                float(alpha),
                int(fill_value),
                energies,
                launch_dims=(num_atoms,),
            )

    return energies.astype(original_dtype)


def coulomb_forces(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    cutoff: float,
    alpha: float = 0.0,
    neighbor_list: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    neighbor_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    fill_value: int | None = None,
    batch_idx: jax.Array | None = None,
) -> jax.Array:
    """Compute Coulomb electrostatic forces.

    Convenience wrapper that returns only forces (no energies).

    Parameters
    ----------
    See coulomb_energy for parameter descriptions.

    Returns
    -------
    forces : jax.Array, shape (N, 3)
        Forces on each atom.

    See Also
    --------
    coulomb_energy_forces : Compute both energies and forces
    """
    _, forces = coulomb_energy_forces(
        positions=positions,
        charges=charges,
        cell=cell,
        cutoff=cutoff,
        alpha=alpha,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        fill_value=fill_value,
        batch_idx=batch_idx,
    )
    return forces


def coulomb_energy_forces(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    cutoff: float,
    alpha: float = 0.0,
    neighbor_list: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    neighbor_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    fill_value: int | None = None,
    batch_idx: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute Coulomb electrostatic energies and forces.

    Computes pairwise electrostatic energies and forces using the Coulomb law,
    with optional erfc damping for Ewald/PME real-space calculations.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    charges : jax.Array, shape (N,)
        Atomic charges.
    cell : jax.Array, shape (1, 3, 3) or (B, 3, 3)
        Unit cell matrix. Shape (B, 3, 3) for batched calculations.
    cutoff : float
        Cutoff distance for interactions.
    alpha : float, default=0.0
        Ewald splitting parameter. Use 0.0 for undamped Coulomb.
    neighbor_list : jax.Array | None, shape (2, num_pairs)
        Neighbor pairs in COO format.
    neighbor_ptr : jax.Array | None, shape (N+1,)
        CSR row pointers for neighbor list. Required with neighbor_list.
        Provided by neighborlist module.
    neighbor_shifts : jax.Array | None, shape (num_pairs, 3)
        Integer unit cell shifts for neighbor list format.
    neighbor_matrix : jax.Array | None, shape (N, max_neighbors)
        Neighbor indices in matrix format.
    neighbor_matrix_shifts : jax.Array | None, shape (N, max_neighbors, 3)
        Integer unit cell shifts for matrix format.
    fill_value : int | None
        Fill value for neighbor matrix padding.
    batch_idx : jax.Array | None, shape (N,)
        Batch indices for each atom.

    Returns
    -------
    energies : jax.Array, shape (N,)
        Per-atom energies.
    forces : jax.Array, shape (N, 3)
        Forces on each atom.

    Examples
    --------
    >>> # Direct Coulomb
    >>> energies, forces = coulomb_energy_forces(
    ...     positions, charges, cell, cutoff=10.0, alpha=0.0,
    ...     neighbor_list=neighbor_list, neighbor_ptr=neighbor_ptr,
    ...     neighbor_shifts=neighbor_shifts
    ... )

    >>> # Ewald/PME real-space
    >>> energies, forces = coulomb_energy_forces(
    ...     positions, charges, cell, cutoff=10.0, alpha=0.3,
    ...     neighbor_matrix=neighbor_matrix, neighbor_matrix_shifts=neighbor_matrix_shifts,
    ...     fill_value=num_atoms
    ... )
    """
    # Validate inputs
    use_list = neighbor_list is not None and neighbor_shifts is not None
    use_matrix = neighbor_matrix is not None and neighbor_matrix_shifts is not None

    if not use_list and not use_matrix:
        raise ValueError(
            "Must provide either neighbor_list/neighbor_shifts or "
            "neighbor_matrix/neighbor_matrix_shifts"
        )

    if use_list and use_matrix:
        raise ValueError(
            "Cannot provide both neighbor list and neighbor matrix formats"
        )

    # Store original dtype for output
    original_dtype = positions.dtype

    # Convert to float64 for numerical stability
    positions_f64 = positions.astype(jnp.float64)
    charges_f64 = charges.astype(jnp.float64)
    cell_f64 = cell.astype(jnp.float64)

    # Ensure cell is (B, 3, 3)
    if cell_f64.ndim == 2:
        cell_f64 = cell_f64[jnp.newaxis, :, :]

    num_atoms = positions_f64.shape[0]
    is_batched = batch_idx is not None

    # Allocate outputs
    energies = jnp.zeros(num_atoms, dtype=jnp.float64)
    forces = jnp.zeros((num_atoms, 3), dtype=jnp.float64)

    if use_list:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr is required when using neighbor_list format")

        # Extract idx_j (target indices) from neighbor_list
        idx_j = neighbor_list[1].astype(jnp.int32)
        neighbor_ptr_i32 = neighbor_ptr.astype(jnp.int32)
        neighbor_shifts_i32 = neighbor_shifts.astype(jnp.int32)

        if is_batched:
            batch_idx_i32 = batch_idx.astype(jnp.int32)
            energies, forces = _jax_batch_coulomb_energy_forces_list(
                positions_f64,
                charges_f64,
                cell_f64,
                idx_j,
                neighbor_ptr_i32,
                neighbor_shifts_i32,
                batch_idx_i32,
                float(cutoff),
                float(alpha),
                energies,
                forces,
                launch_dims=(num_atoms,),
            )
        else:
            energies, forces = _jax_coulomb_energy_forces_list(
                positions_f64,
                charges_f64,
                cell_f64,
                idx_j,
                neighbor_ptr_i32,
                neighbor_shifts_i32,
                float(cutoff),
                float(alpha),
                energies,
                forces,
                launch_dims=(num_atoms,),
            )
    else:
        neighbor_matrix_i32 = neighbor_matrix.astype(jnp.int32)
        neighbor_matrix_shifts_i32 = neighbor_matrix_shifts.astype(jnp.int32)

        if fill_value is None:
            fill_value = num_atoms

        if is_batched:
            batch_idx_i32 = batch_idx.astype(jnp.int32)
            energies, forces = _jax_batch_coulomb_energy_forces_matrix(
                positions_f64,
                charges_f64,
                cell_f64,
                neighbor_matrix_i32,
                neighbor_matrix_shifts_i32,
                batch_idx_i32,
                float(cutoff),
                float(alpha),
                int(fill_value),
                energies,
                forces,
                launch_dims=(num_atoms,),
            )
        else:
            energies, forces = _jax_coulomb_energy_forces_matrix(
                positions_f64,
                charges_f64,
                cell_f64,
                neighbor_matrix_i32,
                neighbor_matrix_shifts_i32,
                float(cutoff),
                float(alpha),
                int(fill_value),
                energies,
                forces,
                launch_dims=(num_atoms,),
            )
    return energies.astype(original_dtype), forces.astype(original_dtype)
