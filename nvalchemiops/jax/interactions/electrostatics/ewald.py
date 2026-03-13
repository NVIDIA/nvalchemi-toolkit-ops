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

"""JAX Ewald summation implementation.

Wraps the framework-agnostic Warp kernels from
``nvalchemiops.interactions.electrostatics.ewald_kernels`` with JAX bindings.

The Ewald method splits long-range Coulomb interactions into:
    E_total = E_real + E_reciprocal - E_self - E_background
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel

from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    BATCH_BLOCK_SIZE,
    _batch_ewald_real_space_energy_forces_charge_grad_kernel_overload,
    _batch_ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload,
    _batch_ewald_real_space_energy_forces_kernel_overload,
    _batch_ewald_real_space_energy_forces_neighbor_matrix_kernel_overload,
    _batch_ewald_real_space_energy_kernel_overload,
    _batch_ewald_real_space_energy_neighbor_matrix_kernel_overload,
    _batch_ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload,
    _batch_ewald_reciprocal_space_energy_forces_kernel_overload,
    _batch_ewald_reciprocal_space_energy_kernel_compute_energy_overload,
    _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload,
    _batch_ewald_reciprocal_space_virial_kernel_overload,
    _batch_ewald_subtract_self_energy_kernel_overload,
    _ewald_real_space_energy_forces_charge_grad_kernel_overload,
    _ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload,
    _ewald_real_space_energy_forces_kernel_overload,
    _ewald_real_space_energy_forces_neighbor_matrix_kernel_overload,
    _ewald_real_space_energy_kernel_overload,
    _ewald_real_space_energy_neighbor_matrix_kernel_overload,
    _ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload,
    _ewald_reciprocal_space_energy_forces_kernel_overload,
    _ewald_reciprocal_space_energy_kernel_compute_energy_overload,
    _ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload,
    _ewald_reciprocal_space_virial_kernel_overload,
    _ewald_subtract_self_energy_kernel_overload,
)
from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.jax.interactions.electrostatics.parameters import (
    estimate_ewald_parameters,
)

__all__ = [
    "ewald_real_space",
    "ewald_reciprocal_space",
    "ewald_summation",
]

PI = math.pi

# ==============================================================================
# Helper for Creating JAX Kernel Dictionaries
# ==============================================================================

# Dtype normalization for kernel lookup
_DTYPE_MAP = {
    jnp.float32: jnp.float32,
    jnp.float64: jnp.float64,
}


def _normalize_dtype(dtype):
    """Normalize dtype for kernel dictionary lookup.

    Parameters
    ----------
    dtype : dtype-like
        Input dtype from a JAX array.

    Returns
    -------
    jnp.float32 or jnp.float64
        Normalized JAX dtype for kernel lookup.
    """
    # Convert to JAX dtype if needed
    if dtype == jnp.float32 or str(dtype) == "float32":
        return jnp.float32
    elif dtype == jnp.float64 or str(dtype) == "float64":
        return jnp.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _make_jax_kernels(
    wp_overload_dict: dict,
    num_outputs: int,
    in_out_argnames: list[str],
) -> dict:
    """Maps a ``jax`` data type to ``warp``.

    Parameters
    ----------
    wp_overload_dict : dict
        Warp kernel overload dictionary keyed by wp.float32/wp.float64.
    num_outputs : int
        Number of output arrays returned by the kernel.
    in_out_argnames : list of str
        Names of in-place output arguments.

    Returns
    -------
    dict
        Dictionary mapping jnp.float32/jnp.float64 to jax_kernel instances.
    """
    _JAX_TO_WP = {jnp.float32: wp.float32, jnp.float64: wp.float64}
    return {
        jax_dtype: jax_kernel(
            wp_overload_dict[wp_dtype],
            num_outputs=num_outputs,
            in_out_argnames=in_out_argnames,
            enable_backward=False,
        )
        for jax_dtype, wp_dtype in _JAX_TO_WP.items()
    }


# ==============================================================================
# JAX Kernel Wrappers - Real Space
# ==============================================================================

# --- Neighbor List (CSR) Format ---

_jax_ewald_real_space_energy_list = _make_jax_kernels(
    _ewald_real_space_energy_kernel_overload, 1, ["pair_energies"]
)

_jax_ewald_real_space_energy_forces_list = _make_jax_kernels(
    _ewald_real_space_energy_forces_kernel_overload,
    3,
    ["pair_energies", "atomic_forces", "virial"],
)

_jax_ewald_real_space_energy_forces_charge_grad_list = _make_jax_kernels(
    _ewald_real_space_energy_forces_charge_grad_kernel_overload,
    4,
    ["pair_energies", "atomic_forces", "charge_gradients", "virial"],
)

_jax_batch_ewald_real_space_energy_list = _make_jax_kernels(
    _batch_ewald_real_space_energy_kernel_overload, 1, ["pair_energies"]
)

_jax_batch_ewald_real_space_energy_forces_list = _make_jax_kernels(
    _batch_ewald_real_space_energy_forces_kernel_overload,
    3,
    ["pair_energies", "atomic_forces", "virial"],
)

_jax_batch_ewald_real_space_energy_forces_charge_grad_list = _make_jax_kernels(
    _batch_ewald_real_space_energy_forces_charge_grad_kernel_overload,
    4,
    ["pair_energies", "atomic_forces", "charge_gradients", "virial"],
)

# --- Neighbor Matrix Format ---

_jax_ewald_real_space_energy_matrix = _make_jax_kernels(
    _ewald_real_space_energy_neighbor_matrix_kernel_overload, 1, ["pair_energies"]
)

_jax_ewald_real_space_energy_forces_matrix = _make_jax_kernels(
    _ewald_real_space_energy_forces_neighbor_matrix_kernel_overload,
    3,
    ["pair_energies", "atomic_forces", "virial"],
)

_jax_ewald_real_space_energy_forces_charge_grad_matrix = _make_jax_kernels(
    _ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload,
    4,
    ["pair_energies", "atomic_forces", "charge_gradients", "virial"],
)

_jax_batch_ewald_real_space_energy_matrix = _make_jax_kernels(
    _batch_ewald_real_space_energy_neighbor_matrix_kernel_overload, 1, ["pair_energies"]
)

_jax_batch_ewald_real_space_energy_forces_matrix = _make_jax_kernels(
    _batch_ewald_real_space_energy_forces_neighbor_matrix_kernel_overload,
    3,
    ["pair_energies", "atomic_forces", "virial"],
)

_jax_batch_ewald_real_space_energy_forces_charge_grad_matrix = _make_jax_kernels(
    _batch_ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload,
    4,
    ["pair_energies", "atomic_forces", "charge_gradients", "virial"],
)

# ==============================================================================
# JAX Kernel Wrappers - Reciprocal Space
# ==============================================================================

# --- Structure Factor Computation ---

_jax_ewald_reciprocal_fill_structure_factors = _make_jax_kernels(
    _ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload,
    5,
    [
        "total_charge",
        "cos_k_dot_r",
        "sin_k_dot_r",
        "real_structure_factors",
        "imag_structure_factors",
    ],
)

_jax_batch_ewald_reciprocal_fill_structure_factors = _make_jax_kernels(
    _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload,
    5,
    [
        "total_charges",
        "cos_k_dot_r",
        "sin_k_dot_r",
        "real_structure_factors",
        "imag_structure_factors",
    ],
)

# --- Energy Computation ---

_jax_ewald_reciprocal_compute_energy = _make_jax_kernels(
    _ewald_reciprocal_space_energy_kernel_compute_energy_overload,
    1,
    ["reciprocal_energies"],
)

_jax_batch_ewald_reciprocal_compute_energy = _make_jax_kernels(
    _batch_ewald_reciprocal_space_energy_kernel_compute_energy_overload,
    1,
    ["reciprocal_energies"],
)

# --- Energy + Forces ---

_jax_ewald_reciprocal_energy_forces = _make_jax_kernels(
    _ewald_reciprocal_space_energy_forces_kernel_overload,
    2,
    ["reciprocal_energies", "atomic_forces"],
)

_jax_batch_ewald_reciprocal_energy_forces = _make_jax_kernels(
    _batch_ewald_reciprocal_space_energy_forces_kernel_overload,
    2,
    ["reciprocal_energies", "atomic_forces"],
)

# --- Energy + Forces + Charge Gradients ---

_jax_ewald_reciprocal_energy_forces_charge_grad = _make_jax_kernels(
    _ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload,
    3,
    ["reciprocal_energies", "atomic_forces", "charge_gradients"],
)

_jax_batch_ewald_reciprocal_energy_forces_charge_grad = _make_jax_kernels(
    _batch_ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload,
    3,
    ["reciprocal_energies", "atomic_forces", "charge_gradients"],
)

# --- Self-Energy Correction ---

_jax_ewald_subtract_self_energy = _make_jax_kernels(
    _ewald_subtract_self_energy_kernel_overload, 1, ["energy_out"]
)

_jax_batch_ewald_subtract_self_energy = _make_jax_kernels(
    _batch_ewald_subtract_self_energy_kernel_overload, 1, ["energy_out"]
)

# --- Reciprocal-Space Virial ---

_jax_ewald_reciprocal_virial = _make_jax_kernels(
    _ewald_reciprocal_space_virial_kernel_overload,
    1,
    ["virial"],
)

_jax_batch_ewald_reciprocal_virial = _make_jax_kernels(
    _batch_ewald_reciprocal_space_virial_kernel_overload,
    1,
    ["virial"],
)


# ==============================================================================
# Helper Functions
# ==============================================================================


def _prepare_alpha_array(
    alpha: float | jax.Array,
    num_systems: int,
    dtype: jnp.dtype = jnp.float64,
) -> jax.Array:
    """Convert alpha to a per-system array of shape (B,) or (1,).

    Parameters
    ----------
    alpha : float or jax.Array
        Ewald splitting parameter.
    num_systems : int
        Number of systems.
    dtype : jnp.dtype, optional
        Data type for the output array. Defaults to jnp.float64.

    Returns
    -------
    jax.Array
        Alpha array of shape (B,) or (1,).
    """
    if isinstance(alpha, (int, float)):
        return jnp.full(num_systems, float(alpha), dtype=dtype)
    elif isinstance(alpha, jax.Array):
        # generate elements from scalar
        if alpha.ndim == 0:
            return jnp.full(num_systems, alpha[0], dtype=dtype)
        elif len(alpha) != num_systems:
            raise ValueError(
                f"alpha has {alpha.shape[0]} values but there are {num_systems} systems"
            )
        else:
            return alpha.astype(dtype)
    else:
        raise TypeError(f"alpha must be float or jax.Array, got {type(alpha)}")


def _compute_total_charge(
    charges: jax.Array, batch_idx: jax.Array | None, num_systems: int = 1
) -> jax.Array:
    """Compute total charge (per system if batched).

    Parameters
    ----------
    charges : jax.Array, shape (N,)
        Atomic charges.
    batch_idx : jax.Array | None, shape (N,)
        Batch indices.
    num_systems : int, optional
        Number of systems in the batch. Only used when batch_idx is not None.
        Default is 1.

    Returns
    -------
    jax.Array
        Total charge, shape (1,) for single system or (B,) for batch.
    """
    if batch_idx is None:
        return jnp.array([charges.sum()], dtype=jnp.float64)
    else:
        total_charges = jnp.zeros(num_systems, dtype=jnp.float64)
        total_charges = total_charges.at[batch_idx].add(charges)
        return total_charges


# ==============================================================================
# Public API
# ==============================================================================


def ewald_real_space(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: float | jax.Array,
    neighbor_list: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    neighbor_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    mask_value: int | None = None,
    batch_idx: jax.Array | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
) -> jax.Array | tuple[jax.Array, ...]:
    """Compute real-space Ewald energy and optionally forces, charge gradients, and virial.

    Computes the damped Coulomb interactions for atom pairs within the real-space
    cutoff. The complementary error function (erfc) damping ensures rapid
    convergence in real space.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    charges : jax.Array, shape (N,)
        Atomic partial charges.
    cell : jax.Array, shape (1, 3, 3) or (B, 3, 3)
        Unit cell matrices.
    alpha : float or jax.Array
        Ewald splitting parameter. Can be a float or array of shape (1,) or (B,).
    neighbor_list : jax.Array | None, shape (2, M)
        Neighbor list in COO format.
    neighbor_ptr : jax.Array | None, shape (N+1,)
        CSR row pointers for neighbor list.
    neighbor_shifts : jax.Array | None, shape (M, 3)
        Periodic image shifts for neighbor list.
    neighbor_matrix : jax.Array | None, shape (N, max_neighbors)
        Dense neighbor matrix format.
    neighbor_matrix_shifts : jax.Array | None, shape (N, max_neighbors, 3)
        Periodic image shifts for neighbor_matrix.
    mask_value : int | None, optional
        Value indicating invalid entries in neighbor_matrix.
        If None (default), uses num_atoms as the mask value.
    batch_idx : jax.Array | None, shape (N,)
        System index for each atom.
    compute_forces : bool, default=False
        Whether to compute explicit forces.
    compute_charge_gradients : bool, default=False
        Whether to compute charge gradients.
    compute_virial : bool, default=False
        Whether to compute the virial tensor.

    Returns
    -------
    energies : jax.Array, shape (N,)
        Per-atom real-space energy.
    forces : jax.Array, shape (N, 3), optional
        Forces (if compute_forces=True or compute_charge_gradients=True).
    charge_gradients : jax.Array, shape (N,), optional
        Charge gradients (if compute_charge_gradients=True).
    virial : jax.Array, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor (if compute_virial=True). Always last in the return tuple.
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

    # Store input dtype for kernel dispatch and outputs
    dtype = _normalize_dtype(positions.dtype)

    # Cast inputs to consistent dtype
    positions_cast = positions.astype(dtype)
    charges_cast = charges.astype(dtype)
    cell_cast = cell.astype(dtype)

    # Ensure cell is (B, 3, 3)
    if cell_cast.ndim == 2:
        cell_cast = cell_cast[jnp.newaxis, :, :]

    num_atoms = positions_cast.shape[0]
    is_batched = batch_idx is not None

    # Default mask_value to num_atoms (matches cell_list fill_value convention)
    if mask_value is None:
        mask_value = num_atoms

    # Derive num_systems from cell shape (cell is always (B, 3, 3) by caller convention)
    if is_batched:
        num_systems = cell_cast.shape[0]

    # Prepare alpha
    alpha_arr = _prepare_alpha_array(alpha, cell_cast.shape[0], dtype=dtype)

    # Allocate outputs (energies always float64, forces match input dtype)
    energies = jnp.zeros(num_atoms, dtype=jnp.float64)

    if use_list:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr is required when using neighbor_list format")
        if neighbor_list is None or neighbor_shifts is None:
            raise ValueError("neighbor_list and neighbor_shifts are required")

        # Extract idx_j from neighbor_list
        idx_j = neighbor_list[1].astype(jnp.int32)
        neighbor_ptr_i32 = neighbor_ptr.astype(jnp.int32)
        neighbor_shifts_i32 = neighbor_shifts.astype(jnp.int32)

        if is_batched:
            batch_idx_i32 = batch_idx.astype(jnp.int32)

            # Determine if we need the force kernel (for forces or virial)
            need_force_kernel = compute_forces or compute_virial

            if compute_charge_gradients:
                forces = jnp.zeros((num_atoms, 3), dtype=dtype)
                charge_grads = jnp.zeros(num_atoms, dtype=jnp.float64)
                virial = jnp.zeros((num_systems, 3, 3), dtype=dtype)
                (energies, forces, charge_grads, virial) = (
                    _jax_batch_ewald_real_space_energy_forces_charge_grad_list[dtype](
                        positions_cast,
                        charges_cast,
                        cell_cast,
                        batch_idx_i32,
                        idx_j,
                        neighbor_ptr_i32,
                        neighbor_shifts_i32,
                        alpha_arr,
                        int(compute_virial),
                        energies,
                        forces,
                        charge_grads,
                        virial,
                        launch_dims=(num_atoms,),
                    )
                )
            elif need_force_kernel:
                forces = jnp.zeros((num_atoms, 3), dtype=dtype)
                virial = jnp.zeros((num_systems, 3, 3), dtype=dtype)
                (energies, forces, virial) = (
                    _jax_batch_ewald_real_space_energy_forces_list[dtype](
                        positions_cast,
                        charges_cast,
                        cell_cast,
                        batch_idx_i32,
                        idx_j,
                        neighbor_ptr_i32,
                        neighbor_shifts_i32,
                        alpha_arr,
                        int(compute_virial),
                        energies,
                        forces,
                        virial,
                        launch_dims=(num_atoms,),
                    )
                )
            else:
                (energies,) = _jax_batch_ewald_real_space_energy_list[dtype](
                    positions_cast,
                    charges_cast,
                    cell_cast,
                    batch_idx_i32,
                    idx_j,
                    neighbor_ptr_i32,
                    neighbor_shifts_i32,
                    alpha_arr,
                    energies,
                    launch_dims=(num_atoms,),
                )
        else:
            # Determine if we need the force kernel (for forces or virial)
            need_force_kernel = compute_forces or compute_virial

            if compute_charge_gradients:
                forces = jnp.zeros((num_atoms, 3), dtype=dtype)
                charge_grads = jnp.zeros(num_atoms, dtype=jnp.float64)
                virial = jnp.zeros((1, 3, 3), dtype=dtype)
                (energies, forces, charge_grads, virial) = (
                    _jax_ewald_real_space_energy_forces_charge_grad_list[dtype](
                        positions_cast,
                        charges_cast,
                        cell_cast,
                        idx_j,
                        neighbor_ptr_i32,
                        neighbor_shifts_i32,
                        alpha_arr,
                        int(compute_virial),
                        energies,
                        forces,
                        charge_grads,
                        virial,
                        launch_dims=(num_atoms,),
                    )
                )
            elif need_force_kernel:
                forces = jnp.zeros((num_atoms, 3), dtype=dtype)
                virial = jnp.zeros((1, 3, 3), dtype=dtype)
                (energies, forces, virial) = _jax_ewald_real_space_energy_forces_list[
                    dtype
                ](
                    positions_cast,
                    charges_cast,
                    cell_cast,
                    idx_j,
                    neighbor_ptr_i32,
                    neighbor_shifts_i32,
                    alpha_arr,
                    int(compute_virial),
                    energies,
                    forces,
                    virial,
                    launch_dims=(num_atoms,),
                )
            else:
                (energies,) = _jax_ewald_real_space_energy_list[dtype](
                    positions_cast,
                    charges_cast,
                    cell_cast,
                    idx_j,
                    neighbor_ptr_i32,
                    neighbor_shifts_i32,
                    alpha_arr,
                    energies,
                    launch_dims=(num_atoms,),
                )
    else:
        # Matrix format
        if neighbor_matrix is None or neighbor_matrix_shifts is None:
            raise ValueError("neighbor_matrix and neighbor_matrix_shifts are required")

        neighbor_matrix_i32 = neighbor_matrix.astype(jnp.int32)
        neighbor_matrix_shifts_i32 = neighbor_matrix_shifts.astype(jnp.int32)

        if is_batched:
            batch_idx_i32 = batch_idx.astype(jnp.int32)

            # Determine if we need the force kernel (for forces or virial)
            need_force_kernel = compute_forces or compute_virial

            if compute_charge_gradients:
                forces = jnp.zeros((num_atoms, 3), dtype=dtype)
                charge_grads = jnp.zeros(num_atoms, dtype=jnp.float64)
                virial = jnp.zeros((num_systems, 3, 3), dtype=dtype)
                (energies, forces, charge_grads, virial) = (
                    _jax_batch_ewald_real_space_energy_forces_charge_grad_matrix[dtype](
                        positions_cast,
                        charges_cast,
                        cell_cast,
                        batch_idx_i32,
                        neighbor_matrix_i32,
                        neighbor_matrix_shifts_i32,
                        int(mask_value),
                        alpha_arr,
                        int(compute_virial),
                        energies,
                        forces,
                        charge_grads,
                        virial,
                        launch_dims=(num_atoms,),
                    )
                )
            elif need_force_kernel:
                forces = jnp.zeros((num_atoms, 3), dtype=dtype)
                virial = jnp.zeros((num_systems, 3, 3), dtype=dtype)
                (energies, forces, virial) = (
                    _jax_batch_ewald_real_space_energy_forces_matrix[dtype](
                        positions_cast,
                        charges_cast,
                        cell_cast,
                        batch_idx_i32,
                        neighbor_matrix_i32,
                        neighbor_matrix_shifts_i32,
                        int(mask_value),
                        alpha_arr,
                        int(compute_virial),
                        energies,
                        forces,
                        virial,
                        launch_dims=(num_atoms,),
                    )
                )
            else:
                (energies,) = _jax_batch_ewald_real_space_energy_matrix[dtype](
                    positions_cast,
                    charges_cast,
                    cell_cast,
                    batch_idx_i32,
                    neighbor_matrix_i32,
                    neighbor_matrix_shifts_i32,
                    int(mask_value),
                    alpha_arr,
                    energies,
                    launch_dims=(num_atoms,),
                )
        else:
            # Determine if we need the force kernel (for forces or virial)
            need_force_kernel = compute_forces or compute_virial

            if compute_charge_gradients:
                forces = jnp.zeros((num_atoms, 3), dtype=dtype)
                charge_grads = jnp.zeros(num_atoms, dtype=jnp.float64)
                virial = jnp.zeros((1, 3, 3), dtype=dtype)
                (energies, forces, charge_grads, virial) = (
                    _jax_ewald_real_space_energy_forces_charge_grad_matrix[dtype](
                        positions_cast,
                        charges_cast,
                        cell_cast,
                        neighbor_matrix_i32,
                        neighbor_matrix_shifts_i32,
                        int(mask_value),
                        alpha_arr,
                        int(compute_virial),
                        energies,
                        forces,
                        charge_grads,
                        virial,
                        launch_dims=(num_atoms,),
                    )
                )
            elif need_force_kernel:
                forces = jnp.zeros((num_atoms, 3), dtype=dtype)
                virial = jnp.zeros((1, 3, 3), dtype=dtype)
                (energies, forces, virial) = _jax_ewald_real_space_energy_forces_matrix[
                    dtype
                ](
                    positions_cast,
                    charges_cast,
                    cell_cast,
                    neighbor_matrix_i32,
                    neighbor_matrix_shifts_i32,
                    int(mask_value),
                    alpha_arr,
                    int(compute_virial),
                    energies,
                    forces,
                    virial,
                    launch_dims=(num_atoms,),
                )
            else:
                (energies,) = _jax_ewald_real_space_energy_matrix[dtype](
                    positions_cast,
                    charges_cast,
                    cell_cast,
                    neighbor_matrix_i32,
                    neighbor_matrix_shifts_i32,
                    int(mask_value),
                    alpha_arr,
                    energies,
                    launch_dims=(num_atoms,),
                )

    # Return results (energies and charge_grads are float64, forces match input dtype)
    # Virial is always last in the return tuple when requested
    def _build_result():
        result = [energies]
        if compute_forces and forces is not None:
            result.append(forces)
        if compute_charge_gradients and charge_grads is not None:
            result.append(charge_grads)
        if compute_virial and virial is not None:
            result.append(virial)
        return tuple(result) if len(result) > 1 else result[0]

    # Initialize optional variables to None if not computed
    if not (compute_forces or compute_virial or compute_charge_gradients):
        forces = None
        charge_grads = None
        virial = None
    elif not compute_charge_gradients:
        charge_grads = None
    if not compute_virial:
        virial = None

    return _build_result()


def ewald_reciprocal_space(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    k_vectors: jax.Array,
    alpha: float | jax.Array,
    batch_idx: jax.Array | None = None,
    max_atoms_per_system: int | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
) -> jax.Array | tuple[jax.Array, ...]:
    """Compute reciprocal-space Ewald energy and optionally forces, charge gradients, and virial.

    Computes the smooth long-range electrostatic contribution using structure
    factors in reciprocal space. Includes self-energy and background corrections.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    charges : jax.Array, shape (N,)
        Atomic partial charges.
    cell : jax.Array, shape (1, 3, 3) or (B, 3, 3)
        Unit cell matrices.
    k_vectors : jax.Array
        Reciprocal lattice vectors. Shape (K, 3) for single system, (B, K, 3) for batch.
    alpha : float or jax.Array
        Ewald splitting parameter. Can be a float or array of shape (1,) or (B,).
    batch_idx : jax.Array | None, shape (N,)
        System index for each atom.
    max_atoms_per_system : int | None, optional
        Maximum number of atoms in any single system in the batch.
        Required when using ``jax.jit`` with batched inputs. If None,
        inferred from data (fails under JIT).
    compute_forces : bool, default=False
        Whether to compute explicit forces.
    compute_charge_gradients : bool, default=False
        Whether to compute charge gradients.
    compute_virial : bool, default=False
        Whether to compute the virial tensor.

    Returns
    -------
    energies : jax.Array, shape (N,)
        Per-atom reciprocal-space energy (with corrections applied).
    forces : jax.Array, shape (N, 3), optional
        Forces (if compute_forces=True or compute_charge_gradients=True).
    charge_gradients : jax.Array, shape (N,), optional
        Charge gradients (if compute_charge_gradients=True).
    virial : jax.Array, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor (if compute_virial=True). Always last in the return tuple.
    """
    # Store input dtype for kernel dispatch and outputs
    dtype = _normalize_dtype(positions.dtype)

    # Cast inputs to consistent dtype
    positions_cast = positions.astype(dtype)
    charges_cast = charges.astype(dtype)
    cell_cast = cell.astype(dtype)
    k_vectors_cast = k_vectors.astype(dtype)

    # Ensure cell is (B, 3, 3)
    if cell_cast.ndim == 2:
        cell_cast = cell_cast[jnp.newaxis, :, :]

    num_atoms = positions_cast.shape[0]
    is_batched = batch_idx is not None

    # Prepare alpha
    alpha_arr = _prepare_alpha_array(alpha, cell_cast.shape[0], dtype=dtype)

    # Compute total charge (always float64)
    # num_systems is derived from cell shape (cell is always (B, 3, 3) by caller convention)
    total_charge = _compute_total_charge(
        charges_cast, batch_idx, num_systems=cell_cast.shape[0]
    )

    # Determine k-vector dimensions
    if is_batched:
        # k_vectors should be (B, K, 3); expand from (K, 3) if necessary
        if k_vectors_cast.ndim == 2:
            k_vectors_cast = jnp.tile(
                k_vectors_cast[jnp.newaxis, :, :],
                (cell_cast.shape[0], 1, 1),
            )
        num_k = k_vectors_cast.shape[1]
        num_systems = k_vectors_cast.shape[0]
    else:
        # k_vectors: (K, 3)
        num_k = k_vectors_cast.shape[0]
        num_systems = 1

    # Allocate intermediate arrays for structure factors (always float64)
    if is_batched:
        cos_k_dot_r = jnp.zeros((num_k, num_atoms), dtype=jnp.float64)
        sin_k_dot_r = jnp.zeros((num_k, num_atoms), dtype=jnp.float64)
        real_sf = jnp.zeros((num_systems, num_k), dtype=jnp.float64)
        imag_sf = jnp.zeros((num_systems, num_k), dtype=jnp.float64)
    else:
        cos_k_dot_r = jnp.zeros((num_k, num_atoms), dtype=jnp.float64)
        sin_k_dot_r = jnp.zeros((num_k, num_atoms), dtype=jnp.float64)
        real_sf = jnp.zeros(num_k, dtype=jnp.float64)
        imag_sf = jnp.zeros(num_k, dtype=jnp.float64)

    # Allocate output arrays (energies always float64)
    raw_energies = jnp.zeros(num_atoms, dtype=jnp.float64)
    energies = jnp.zeros(num_atoms, dtype=jnp.float64)

    # Step 1: Fill structure factors
    if is_batched:
        batch_idx_i32 = batch_idx.astype(jnp.int32)

        # Compute atom_start, atom_end, and max_blocks_per_system for batch kernels
        atom_counts = jnp.bincount(batch_idx_i32, length=num_systems)
        atom_end = jnp.cumsum(atom_counts).astype(jnp.int32)
        atom_start = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), atom_end[:-1]])
        if max_atoms_per_system is None:
            try:
                max_atoms_per_system = int(atom_counts.max())
            except (
                jax.errors.ConcretizationTypeError,
                jax.errors.TracerIntegerConversionError,
            ):
                raise ValueError(
                    "Cannot infer max_atoms_per_system inside jax.jit. "
                    "Please provide max_atoms_per_system explicitly when "
                    "using jax.jit."
                ) from None
        max_blocks_per_system = (
            max_atoms_per_system + BATCH_BLOCK_SIZE - 1
        ) // BATCH_BLOCK_SIZE

        (total_charge, cos_k_dot_r, sin_k_dot_r, real_sf, imag_sf) = (
            _jax_batch_ewald_reciprocal_fill_structure_factors[dtype](
                positions_cast,
                charges_cast,
                k_vectors_cast,
                cell_cast,
                alpha_arr,
                atom_start,
                atom_end,
                total_charge,
                cos_k_dot_r,
                sin_k_dot_r,
                real_sf,
                imag_sf,
                launch_dims=(num_k, num_systems, max_blocks_per_system),
            )
        )
    else:
        (total_charge, cos_k_dot_r, sin_k_dot_r, real_sf, imag_sf) = (
            _jax_ewald_reciprocal_fill_structure_factors[dtype](
                positions_cast,
                charges_cast,
                k_vectors_cast,
                cell_cast,
                alpha_arr,
                total_charge,
                cos_k_dot_r,
                sin_k_dot_r,
                real_sf,
                imag_sf,
                launch_dims=(num_k,),
            )
        )

    # Step 2: Compute energy (and forces/charge_grads if requested)
    if is_batched:
        batch_idx_i32 = batch_idx.astype(jnp.int32)

        if compute_charge_gradients:
            forces = jnp.zeros((num_atoms, 3), dtype=dtype)
            charge_grads = jnp.zeros(num_atoms, dtype=jnp.float64)
            (raw_energies, forces, charge_grads) = (
                _jax_batch_ewald_reciprocal_energy_forces_charge_grad[dtype](
                    charges_cast,
                    batch_idx_i32,
                    k_vectors_cast,
                    cos_k_dot_r,
                    sin_k_dot_r,
                    real_sf,
                    imag_sf,
                    raw_energies,
                    forces,
                    charge_grads,
                    launch_dims=(num_atoms,),
                )
            )
        elif compute_forces:
            forces = jnp.zeros((num_atoms, 3), dtype=dtype)
            (raw_energies, forces) = _jax_batch_ewald_reciprocal_energy_forces[dtype](
                charges_cast,
                batch_idx_i32,
                k_vectors_cast,
                cos_k_dot_r,
                sin_k_dot_r,
                real_sf,
                imag_sf,
                raw_energies,
                forces,
                launch_dims=(num_atoms,),
            )
        else:
            (raw_energies,) = _jax_batch_ewald_reciprocal_compute_energy[dtype](
                charges_cast,
                batch_idx_i32,
                cos_k_dot_r,
                sin_k_dot_r,
                real_sf,
                imag_sf,
                raw_energies,
                launch_dims=(num_atoms,),
            )
    else:
        if compute_charge_gradients:
            forces = jnp.zeros((num_atoms, 3), dtype=dtype)
            charge_grads = jnp.zeros(num_atoms, dtype=jnp.float64)
            (raw_energies, forces, charge_grads) = (
                _jax_ewald_reciprocal_energy_forces_charge_grad[dtype](
                    charges_cast,
                    k_vectors_cast,
                    cos_k_dot_r,
                    sin_k_dot_r,
                    real_sf,
                    imag_sf,
                    raw_energies,
                    forces,
                    charge_grads,
                    launch_dims=(num_atoms,),
                )
            )
        elif compute_forces:
            forces = jnp.zeros((num_atoms, 3), dtype=dtype)
            (raw_energies, forces) = _jax_ewald_reciprocal_energy_forces[dtype](
                charges_cast,
                k_vectors_cast,
                cos_k_dot_r,
                sin_k_dot_r,
                real_sf,
                imag_sf,
                raw_energies,
                forces,
                launch_dims=(num_atoms,),
            )
        else:
            (raw_energies,) = _jax_ewald_reciprocal_compute_energy[dtype](
                charges_cast,
                cos_k_dot_r,
                sin_k_dot_r,
                real_sf,
                imag_sf,
                raw_energies,
                launch_dims=(num_atoms,),
            )

    # Step 3: Apply self-energy and background corrections
    if is_batched:
        batch_idx_i32 = batch_idx.astype(jnp.int32)
        (energies,) = _jax_batch_ewald_subtract_self_energy[dtype](
            charges_cast,
            batch_idx_i32,
            alpha_arr,
            total_charge,
            raw_energies,
            energies,
            launch_dims=(num_atoms,),
        )
    else:
        (energies,) = _jax_ewald_subtract_self_energy[dtype](
            charges_cast,
            alpha_arr,
            total_charge,
            raw_energies,
            energies,
            launch_dims=(num_atoms,),
        )

    # Step 4: Compute virial if requested
    virial = None
    if compute_virial:
        volume = jnp.abs(jnp.linalg.det(cell_cast)).astype(jnp.float64)
        if is_batched:
            virial = jnp.zeros((num_systems, 3, 3), dtype=dtype)
            (virial,) = _jax_batch_ewald_reciprocal_virial[dtype](
                k_vectors_cast,  # (B, K, 3)
                alpha_arr,
                volume,
                real_sf,  # (B, K)
                imag_sf,  # (B, K)
                virial,
                launch_dims=(num_k, num_systems),
            )
        else:
            virial = jnp.zeros((1, 3, 3), dtype=dtype)
            (virial,) = _jax_ewald_reciprocal_virial[dtype](
                k_vectors_cast,  # (K, 3)
                alpha_arr,
                volume,
                real_sf,  # (K,)
                imag_sf,  # (K,)
                virial,
                launch_dims=(num_k,),
            )

    # Apply corrections to charge gradients if requested
    if compute_charge_gradients:
        # Self-energy gradient: 2 * alpha / sqrt(pi) * q
        alpha_val = alpha_arr[0] if not is_batched else alpha_arr[batch_idx]
        self_energy_grad = 2.0 * alpha_val / jnp.sqrt(PI) * charges_cast

        # Background gradient: pi / (2 * alpha^2 * V) * Q_total
        volume = jnp.abs(jnp.linalg.det(cell_cast)).astype(jnp.float64)
        if is_batched:
            total_charge_per_atom = total_charge[batch_idx]
            volume_per_atom = volume[batch_idx]
        else:
            total_charge_per_atom = total_charge[0]
            volume_per_atom = volume[0]

        background_grad = (
            PI / (2.0 * alpha_val * alpha_val * volume_per_atom) * total_charge_per_atom
        )

        charge_grads = charge_grads - self_energy_grad - background_grad

    # Return results (energies and charge_grads are float64, forces match input dtype)
    # Virial is always last in the return tuple when requested
    def _build_result():
        result = [energies]
        if compute_forces and forces is not None:
            result.append(forces)
        if compute_charge_gradients and charge_grads is not None:
            result.append(charge_grads)
        if compute_virial and virial is not None:
            result.append(virial)
        return tuple(result) if len(result) > 1 else result[0]

    # Initialize optional variables to None if not computed
    if not (compute_forces or compute_charge_gradients):
        forces = None
        charge_grads = None
    elif not compute_charge_gradients:
        charge_grads = None

    return _build_result()


def ewald_summation(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: float | jax.Array | None = None,
    k_vectors: jax.Array | None = None,
    k_cutoff: float | jax.Array | None = None,
    miller_bounds: tuple[int, int, int] | None = None,
    batch_idx: jax.Array | None = None,
    max_atoms_per_system: int | None = None,
    neighbor_list: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    neighbor_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    mask_value: int | None = None,
    compute_forces: bool = False,
    compute_virial: bool = False,
    accuracy: float = 1e-6,
) -> jax.Array | tuple[jax.Array, ...]:
    """Compute complete Ewald summation (real + reciprocal space).

    The Ewald method splits long-range Coulomb into:
        E_total = E_real + E_reciprocal - E_self - E_background

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    charges : jax.Array, shape (N,)
        Atomic partial charges.
    cell : jax.Array, shape (1, 3, 3) or (B, 3, 3)
        Unit cell matrices.
    alpha : float or jax.Array or None
        Ewald splitting parameter. If None, estimated automatically.
    k_vectors : jax.Array | None
        Reciprocal lattice vectors. If None, generated automatically.
        Shape (K, 3) for single system, (B, K, 3) for batch.
    k_cutoff : float | None
        K-space cutoff. Used only if k_vectors is None.
    miller_bounds : tuple[int, int, int] | None, optional
        Precomputed maximum Miller indices (M_h, M_k, M_l). Forwarded to
        :func:`generate_k_vectors_ewald_summation` when ``k_vectors`` is ``None``.
        When provided, makes k-vector generation compatible with ``jax.jit``.
        Use :func:`generate_miller_indices` to precompute. Ignored when
        ``k_vectors`` is explicitly provided.
    batch_idx : jax.Array | None, shape (N,)
        System index for each atom.
    max_atoms_per_system : int | None, optional
        Maximum number of atoms in any single system in the batch.
        Required when using ``jax.jit`` with batched inputs. If None,
        inferred from data (fails under JIT).
    neighbor_list : jax.Array | None, shape (2, M)
        Neighbor list in COO format.
    neighbor_ptr : jax.Array | None, shape (N+1,)
        CSR row pointers for neighbor list.
    neighbor_shifts : jax.Array | None, shape (M, 3)
        Periodic image shifts for neighbor list.
    neighbor_matrix : jax.Array | None, shape (N, max_neighbors)
        Dense neighbor matrix format.
    neighbor_matrix_shifts : jax.Array | None, shape (N, max_neighbors, 3)
        Periodic image shifts for neighbor_matrix.
    mask_value : int | None
        Value indicating invalid entries in neighbor_matrix.
    compute_forces : bool, default=False
        Whether to compute forces.
    compute_virial : bool, default=False
        Whether to compute the virial tensor.
    accuracy : float, default=1e-6
        Target accuracy for automatic parameter estimation.

    Returns
    -------
    energies : jax.Array, shape (N,)
        Per-atom total Ewald energy.
    forces : jax.Array, shape (N, 3), optional
        Forces (if compute_forces=True).
    virial : jax.Array, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor (if compute_virial=True). Always last in the return tuple.

    Examples
    --------
    >>> # Complete Ewald summation with automatic parameters
    >>> energies, forces = ewald_summation(
    ...     positions, charges, cell,
    ...     neighbor_list=nl, neighbor_ptr=neighbor_ptr, neighbor_shifts=shifts,
    ...     accuracy=1e-6,
    ...     compute_forces=True,
    ... )
    """
    # Auto-estimate alpha and k_cutoff if not provided
    if alpha is None or k_cutoff is None:
        # Ensure cell is (B, 3, 3) for parameter estimation
        cell_3d = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]

        params = estimate_ewald_parameters(
            positions=positions,
            cell=cell_3d,
            batch_idx=batch_idx,
            accuracy=accuracy,
        )

        if alpha is None:
            alpha = params.alpha
        if k_cutoff is None:
            k_cutoff = params.reciprocal_space_cutoff

    # Generate k_vectors if not provided
    if k_vectors is None:
        # Ensure cell is (B, 3, 3)
        cell_3d = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]

        # Ensure k_cutoff is defined
        if k_cutoff is None:
            raise ValueError("k_cutoff must be provided if k_vectors is None")

        k_vectors = generate_k_vectors_ewald_summation(
            cell=cell_3d,
            k_cutoff=k_cutoff,
            miller_bounds=miller_bounds,
        )

    # Compute real-space component
    real_result = ewald_real_space(
        positions=positions,
        charges=charges,
        cell=cell,
        alpha=alpha,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        mask_value=mask_value,
        batch_idx=batch_idx,
        compute_forces=compute_forces,
        compute_charge_gradients=False,
        compute_virial=compute_virial,
    )

    # Compute reciprocal-space component
    recip_result = ewald_reciprocal_space(
        positions=positions,
        charges=charges,
        cell=cell,
        k_vectors=k_vectors,
        alpha=alpha,
        batch_idx=batch_idx,
        max_atoms_per_system=max_atoms_per_system,
        compute_forces=compute_forces,
        compute_charge_gradients=False,
        compute_virial=compute_virial,
    )

    # Sum contributions
    # Both real_result and recip_result have matching tuple structure based on flags
    # The order is: (energies, [forces], [virial]) - virial always last when present
    if compute_forces and compute_virial:
        real_energies, real_forces, real_virial = real_result  # type: ignore[misc]
        recip_energies, recip_forces, recip_virial = recip_result  # type: ignore[misc]
        total_energies = real_energies + recip_energies
        total_forces = real_forces + recip_forces
        total_virial = real_virial + recip_virial
        return total_energies, total_forces, total_virial
    elif compute_forces:
        real_energies, real_forces = real_result  # type: ignore[misc]
        recip_energies, recip_forces = recip_result  # type: ignore[misc]
        total_energies = real_energies + recip_energies
        total_forces = real_forces + recip_forces
        return total_energies, total_forces
    elif compute_virial:
        real_energies, real_virial = real_result  # type: ignore[misc]
        recip_energies, recip_virial = recip_result  # type: ignore[misc]
        total_energies = real_energies + recip_energies
        total_virial = real_virial + recip_virial
        return total_energies, total_virial
    else:
        real_energies = real_result  # type: ignore[assignment]
        recip_energies = recip_result  # type: ignore[assignment]
        total_energies = real_energies + recip_energies
        return total_energies
