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

import functools
import math
import warnings

import jax
import jax.numpy as jnp
import warp as wp
from jax.interpreters import ad as jax_ad
from jax.scipy.special import erfc
from warp.jax_experimental import GraphMode, jax_callable

from nvalchemiops.interactions.electrostatics._factory_common import _DerivState
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    BATCH_BLOCK_SIZE,
    batch_ewald_reciprocal_space_fill_structure_factors,
    should_tile_ewald_recip_fill,
)
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    ewald_reciprocal_space_fill_structure_factors as _wp_ewald_recip_fill,
)
from nvalchemiops.interactions.electrostatics.ewald_real_factory import (
    get_ewald_real_kernel,
)
from nvalchemiops.interactions.electrostatics.ewald_recip_factory import (
    get_ewald_recip_component_kernel,
    get_ewald_recip_kernel,
)
from nvalchemiops.jax.interactions.electrostatics._autograd import (
    _cell_grad_from_strain_virial,
    _inject_charge_grad,
)
from nvalchemiops.jax.interactions.electrostatics._lazy_jax_kernels import (
    _make_jax_kernel_factory,
)
from nvalchemiops.jax.interactions.electrostatics._utils import (
    _build_electrostatic_result,
    _component_direct_output_deprecation_msg,
    _direct_output_deprecation_msg,
    _normalize_dtype,
    _prepare_cell,
)
from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.jax.interactions.electrostatics.parameters import (
    estimate_ewald_parameters,
)
from nvalchemiops.jax.interactions.electrostatics.slab import (
    _prepare_pbc_for_slab,
    _slab_correction_energy_autodiff,
)
from nvalchemiops.jax.interactions.electrostatics.slab import (
    compute_slab_correction as _compute_slab_correction,
)

__all__ = [
    "ewald_real_space",
    "ewald_reciprocal_space",
    "ewald_summation",
]

PI = math.pi

# ``_make_jax_kernel_factory`` returns lazy dtype mappings whose entries
# materialize their ``jax_kernel`` wrappers on first ``__getitem__``. Module import
# is therefore free of FFI work; warp NVRTC compile defers to first launch.


def _jax_can_tile_ewald_recip() -> bool:
    """Return whether JAX reciprocal tiled callbacks should be used."""
    # ``jax_callable`` + nested ``wp.launch_tiled`` is not stable under the
    # current JAX/Warp stack for jitted reciprocal calls. Keep JAX on the
    # existing ``jax_kernel`` path until Warp exposes tiled launch metadata
    # through that wrapper.
    return False


# ==============================================================================
# JAX Kernel Wrappers - Real Space
# ==============================================================================


@functools.cache
def _jax_ewald_real_forward(
    batched: bool,
    neighbor_input: str,
    deriv_state: _DerivState,
    cell_grad: bool,
):
    """Return the lazy JAX wrapper for a factory-backed Ewald real forward kernel."""
    return _make_jax_kernel_factory(
        lambda wp_dtype: get_ewald_real_kernel(
            wp_dtype,
            batched=batched,
            neighbor_input=neighbor_input,
            deriv_state=deriv_state,
            cell_grad=cell_grad,
            order="forward",
        ),
        4,
        ["pair_energies", "atomic_forces", "charge_gradients", "virial"],
    )


@functools.cache
def _jax_ewald_real_double_backward(
    batched: bool,
    neighbor_input: str,
    deriv_state: _DerivState,
    cell_grad: bool,
):
    """Return the lazy JAX wrapper for a factory-backed Ewald real HVP kernel."""
    output_names = ["grad_grad_energy", "grad_positions", "grad_charges"]
    if cell_grad:
        output_names.append("grad_cell")
    return _make_jax_kernel_factory(
        lambda wp_dtype: get_ewald_real_kernel(
            wp_dtype,
            batched=batched,
            neighbor_input=neighbor_input,
            deriv_state=deriv_state,
            cell_grad=cell_grad,
            order="double_backward",
        ),
        len(output_names),
        output_names,
    )


def _jax_ewald_recip_component(
    component: str,
    output_names: list[str],
    *,
    batched: bool = False,
):
    """Return a lazy JAX wrapper for a factory-backed Ewald reciprocal component."""
    return _make_jax_kernel_factory(
        lambda wp_dtype: get_ewald_recip_component_kernel(
            wp_dtype,
            component=component,
            batched=batched,
        ),
        len(output_names),
        output_names,
    )


# ==============================================================================
# JAX Kernel Wrappers - Reciprocal Space
# ==============================================================================

# --- Structure Factor Computation ---

_jax_ewald_reciprocal_fill_structure_factors = _jax_ewald_recip_component(
    "fill",
    [
        "total_charge",
        "cos_k_dot_r",
        "sin_k_dot_r",
        "real_structure_factors",
        "imag_structure_factors",
    ],
)

_jax_batch_ewald_reciprocal_fill_structure_factors = _jax_ewald_recip_component(
    "fill",
    [
        "total_charges",
        "cos_k_dot_r",
        "sin_k_dot_r",
        "real_structure_factors",
        "imag_structure_factors",
    ],
    batched=True,
)


def _ewald_recip_fill_tiled_f32(
    positions: wp.array(dtype=wp.vec3f),
    charges: wp.array(dtype=wp.float32),
    k_vectors: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    alpha: wp.array(dtype=wp.float32),
    total_charge: wp.array(dtype=wp.float64),
    cos_k_dot_r: wp.array(dtype=wp.float64, ndim=2),
    sin_k_dot_r: wp.array(dtype=wp.float64, ndim=2),
    real_structure_factors: wp.array(dtype=wp.float64),
    imag_structure_factors: wp.array(dtype=wp.float64),
) -> None:
    _wp_ewald_recip_fill(
        positions,
        charges,
        k_vectors,
        cell,
        alpha,
        total_charge,
        cos_k_dot_r,
        sin_k_dot_r,
        real_structure_factors,
        imag_structure_factors,
        wp.float32,
        str(positions.device),
    )


def _ewald_recip_fill_tiled_f64(
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    alpha: wp.array(dtype=wp.float64),
    total_charge: wp.array(dtype=wp.float64),
    cos_k_dot_r: wp.array(dtype=wp.float64, ndim=2),
    sin_k_dot_r: wp.array(dtype=wp.float64, ndim=2),
    real_structure_factors: wp.array(dtype=wp.float64),
    imag_structure_factors: wp.array(dtype=wp.float64),
) -> None:
    _wp_ewald_recip_fill(
        positions,
        charges,
        k_vectors,
        cell,
        alpha,
        total_charge,
        cos_k_dot_r,
        sin_k_dot_r,
        real_structure_factors,
        imag_structure_factors,
        wp.float64,
        str(positions.device),
    )


def _batch_ewald_recip_fill_tiled_f32(
    positions: wp.array(dtype=wp.vec3f),
    charges: wp.array(dtype=wp.float32),
    k_vectors: wp.array(dtype=wp.vec3f, ndim=2),
    cell: wp.array(dtype=wp.mat33f),
    alpha: wp.array(dtype=wp.float32),
    atom_start: wp.array(dtype=wp.int32),
    atom_end: wp.array(dtype=wp.int32),
    total_charges: wp.array(dtype=wp.float64),
    cos_k_dot_r: wp.array(dtype=wp.float64, ndim=2),
    sin_k_dot_r: wp.array(dtype=wp.float64, ndim=2),
    real_structure_factors: wp.array(dtype=wp.float64, ndim=2),
    imag_structure_factors: wp.array(dtype=wp.float64, ndim=2),
    max_blocks_per_system: wp.int32,
) -> None:
    batch_ewald_reciprocal_space_fill_structure_factors(
        positions,
        charges,
        k_vectors,
        cell,
        alpha,
        atom_start,
        atom_end,
        total_charges,
        cos_k_dot_r,
        sin_k_dot_r,
        real_structure_factors,
        imag_structure_factors,
        int(k_vectors.shape[1]),
        int(cell.shape[0]),
        int(max_blocks_per_system),
        wp.float32,
        str(positions.device),
    )


def _batch_ewald_recip_fill_tiled_f64(
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d, ndim=2),
    cell: wp.array(dtype=wp.mat33d),
    alpha: wp.array(dtype=wp.float64),
    atom_start: wp.array(dtype=wp.int32),
    atom_end: wp.array(dtype=wp.int32),
    total_charges: wp.array(dtype=wp.float64),
    cos_k_dot_r: wp.array(dtype=wp.float64, ndim=2),
    sin_k_dot_r: wp.array(dtype=wp.float64, ndim=2),
    real_structure_factors: wp.array(dtype=wp.float64, ndim=2),
    imag_structure_factors: wp.array(dtype=wp.float64, ndim=2),
    max_blocks_per_system: wp.int32,
) -> None:
    batch_ewald_reciprocal_space_fill_structure_factors(
        positions,
        charges,
        k_vectors,
        cell,
        alpha,
        atom_start,
        atom_end,
        total_charges,
        cos_k_dot_r,
        sin_k_dot_r,
        real_structure_factors,
        imag_structure_factors,
        int(k_vectors.shape[1]),
        int(cell.shape[0]),
        int(max_blocks_per_system),
        wp.float64,
        str(positions.device),
    )


_JAX_EWALD_RECIP_FILL_TILED = {
    jnp.dtype(jnp.float32): jax_callable(
        _ewald_recip_fill_tiled_f32,
        num_outputs=5,
        in_out_argnames=[
            "total_charge",
            "cos_k_dot_r",
            "sin_k_dot_r",
            "real_structure_factors",
            "imag_structure_factors",
        ],
        graph_mode=GraphMode.NONE,
    ),
    jnp.dtype(jnp.float64): jax_callable(
        _ewald_recip_fill_tiled_f64,
        num_outputs=5,
        in_out_argnames=[
            "total_charge",
            "cos_k_dot_r",
            "sin_k_dot_r",
            "real_structure_factors",
            "imag_structure_factors",
        ],
        graph_mode=GraphMode.NONE,
    ),
}


_JAX_BATCH_EWALD_RECIP_FILL_TILED = {
    jnp.dtype(jnp.float32): jax_callable(
        _batch_ewald_recip_fill_tiled_f32,
        num_outputs=5,
        in_out_argnames=[
            "total_charges",
            "cos_k_dot_r",
            "sin_k_dot_r",
            "real_structure_factors",
            "imag_structure_factors",
        ],
        graph_mode=GraphMode.NONE,
    ),
    jnp.dtype(jnp.float64): jax_callable(
        _batch_ewald_recip_fill_tiled_f64,
        num_outputs=5,
        in_out_argnames=[
            "total_charges",
            "cos_k_dot_r",
            "sin_k_dot_r",
            "real_structure_factors",
            "imag_structure_factors",
        ],
        graph_mode=GraphMode.NONE,
    ),
}

# --- Energy Computation ---

_jax_ewald_reciprocal_compute_energy = _jax_ewald_recip_component(
    "compute_energy",
    ["reciprocal_energies"],
)

_jax_batch_ewald_reciprocal_compute_energy = _jax_ewald_recip_component(
    "compute_energy",
    ["reciprocal_energies"],
    batched=True,
)

# --- Energy + Forces ---

_jax_ewald_reciprocal_energy_forces = _jax_ewald_recip_component(
    "compute_energy_forces",
    ["reciprocal_energies", "atomic_forces"],
)

_jax_batch_ewald_reciprocal_energy_forces = _jax_ewald_recip_component(
    "compute_energy_forces",
    ["reciprocal_energies", "atomic_forces"],
    batched=True,
)

# --- Energy + Forces + Charge Gradients ---

_jax_ewald_reciprocal_energy_forces_charge_grad = _jax_ewald_recip_component(
    "compute_energy_forces_charge_grad",
    ["reciprocal_energies", "atomic_forces", "charge_gradients"],
)

_jax_batch_ewald_reciprocal_energy_forces_charge_grad = _jax_ewald_recip_component(
    "compute_energy_forces_charge_grad",
    ["reciprocal_energies", "atomic_forces", "charge_gradients"],
    batched=True,
)

# --- Self-Energy Correction ---

_jax_ewald_subtract_self_energy = _jax_ewald_recip_component(
    "subtract_self",
    ["energy_out"],
)

_jax_batch_ewald_subtract_self_energy = _jax_ewald_recip_component(
    "subtract_self",
    ["energy_out"],
    batched=True,
)

# --- Reciprocal-Space Virial ---

_jax_ewald_reciprocal_virial = _jax_ewald_recip_component(
    "virial",
    ["virial"],
)

_jax_batch_ewald_reciprocal_virial = _jax_ewald_recip_component(
    "virial",
    ["virial"],
    batched=True,
)

_jax_ewald_reciprocal_double_backward_reduce = _make_jax_kernel_factory(
    lambda wp_dtype: get_ewald_recip_kernel(
        wp_dtype,
        batched=False,
        deriv_state=_DerivState.E_F_dQ,
        cell_grad=False,
        order="double_backward",
    ).fill,
    7,
    [
        "gA",
        "gB",
        "gC",
        "gD",
        "gP",
        "gQ",
        "grad_grad_energy",
    ],
)

_jax_ewald_reciprocal_double_backward_compute = _make_jax_kernel_factory(
    lambda wp_dtype: get_ewald_recip_kernel(
        wp_dtype,
        batched=False,
        deriv_state=_DerivState.E_F_dQ,
        cell_grad=False,
        order="double_backward",
    ).compute,
    2,
    ["grad_positions", "grad_charges"],
)

_jax_batch_ewald_reciprocal_double_backward_reduce = _make_jax_kernel_factory(
    lambda wp_dtype: get_ewald_recip_kernel(
        wp_dtype,
        batched=True,
        deriv_state=_DerivState.E_F_dQ,
        cell_grad=False,
        order="double_backward",
    ).fill,
    7,
    [
        "gA",
        "gB",
        "gC",
        "gD",
        "gP",
        "gQ",
        "grad_grad_energy",
    ],
)

_jax_batch_ewald_reciprocal_double_backward_compute = _make_jax_kernel_factory(
    lambda wp_dtype: get_ewald_recip_kernel(
        wp_dtype,
        batched=True,
        deriv_state=_DerivState.E_F_dQ,
        cell_grad=False,
        order="double_backward",
    ).compute,
    2,
    ["grad_positions", "grad_charges"],
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
            return jnp.full(num_systems, alpha, dtype=dtype)
        elif len(alpha) != num_systems:
            raise ValueError(
                f"alpha has {alpha.shape[0]} values but there are {num_systems} systems"
            )
        else:
            return alpha.astype(dtype)
    else:
        raise TypeError(f"alpha must be float or jax.Array, got {type(alpha)}")


def _launch_ewald_real_forward_factory(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    energies: jax.Array,
    dtype,
    *,
    is_batched: bool,
    use_matrix: bool,
    batch_idx: jax.Array | None = None,
    idx_j: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    unit_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    unit_shifts_matrix: jax.Array | None = None,
    mask_value: int = 0,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, jax.Array | None]:
    """Launch the factory-backed JAX Ewald real forward kernel."""
    num_atoms = positions.shape[0]
    num_systems = cell.shape[0] if is_batched else 1
    need_forces = compute_forces or compute_charge_gradients or compute_virial

    if compute_charge_gradients:
        deriv_state = _DerivState.E_F_dQ
    elif need_forces:
        deriv_state = _DerivState.E_F
    else:
        deriv_state = _DerivState.E

    batch_arg = batch_idx.astype(jnp.int32) if batch_idx is not None else _empty_i32()
    if use_matrix:
        if neighbor_matrix is None or unit_shifts_matrix is None:
            raise ValueError("neighbor_matrix and unit_shifts_matrix are required")
        idx_arg = _empty_i32()
        ptr_arg = _empty_i32()
        shifts_arg = _empty_vec(jnp.int32)
        matrix_arg = neighbor_matrix.astype(jnp.int32)
        matrix_shifts_arg = unit_shifts_matrix.astype(jnp.int32)
        neighbor_input = "matrix"
    else:
        if idx_j is None or neighbor_ptr is None or unit_shifts is None:
            raise ValueError("idx_j, neighbor_ptr, and unit_shifts are required")
        idx_arg = idx_j.astype(jnp.int32)
        ptr_arg = neighbor_ptr.astype(jnp.int32)
        shifts_arg = unit_shifts.astype(jnp.int32)
        matrix_arg = _empty_matrix_i32()
        matrix_shifts_arg = _empty_shift_matrix()
        neighbor_input = "list"

    forces_arg = (
        jnp.zeros((num_atoms, 3), dtype=dtype) if need_forces else _empty_vec(dtype)
    )
    charge_arg = (
        jnp.zeros(num_atoms, dtype=jnp.float64)
        if compute_charge_gradients
        else jnp.zeros((0,), dtype=jnp.float64)
    )
    virial_arg = (
        jnp.zeros((num_systems, 3, 3), dtype=dtype)
        if compute_virial
        else _empty_mat(dtype)
    )

    kernel = _jax_ewald_real_forward(
        is_batched,
        neighbor_input,
        deriv_state,
        compute_virial,
    )[dtype]
    energies, forces, charge_grads, virial = kernel(
        positions,
        charges,
        cell,
        batch_arg,
        idx_arg,
        ptr_arg,
        shifts_arg,
        matrix_arg,
        matrix_shifts_arg,
        int(mask_value),
        alpha,
        energies,
        forces_arg,
        charge_arg,
        virial_arg,
        launch_dims=(num_atoms,),
    )

    return (
        energies,
        forces if need_forces else None,
        charge_grads if compute_charge_gradients else None,
        virial if compute_virial else None,
    )


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


def _ewald_real_space_impl(
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
        System index for each atom. When provided, atoms must be grouped by
        system: ``batch_idx`` must be contiguous, nondecreasing, and use system
        IDs ``0..B-1``.
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

    # Prepare alpha
    alpha_arr = _prepare_alpha_array(alpha, cell_cast.shape[0], dtype=dtype)

    # Allocate outputs (energies always float64, forces match input dtype)
    energies = jnp.zeros(num_atoms, dtype=jnp.float64)

    if use_list:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr is required when using neighbor_list format")
        if neighbor_list is None or neighbor_shifts is None:
            raise ValueError("neighbor_list and neighbor_shifts are required")
        idx_j = neighbor_list[1]
        matrix_arg = None
        matrix_shifts_arg = None
    else:
        if neighbor_matrix is None or neighbor_matrix_shifts is None:
            raise ValueError("neighbor_matrix and neighbor_matrix_shifts are required")
        idx_j = None
        matrix_arg = neighbor_matrix
        matrix_shifts_arg = neighbor_matrix_shifts

    energies, forces, charge_grads, virial = _launch_ewald_real_forward_factory(
        positions_cast,
        charges_cast,
        cell_cast,
        alpha_arr,
        energies,
        dtype,
        is_batched=is_batched,
        use_matrix=use_matrix,
        batch_idx=batch_idx,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=neighbor_shifts,
        neighbor_matrix=matrix_arg,
        unit_shifts_matrix=matrix_shifts_arg,
        mask_value=int(mask_value),
        compute_forces=compute_forces,
        compute_charge_gradients=compute_charge_gradients,
        compute_virial=compute_virial,
    )

    return _build_electrostatic_result(
        energies,
        forces,
        charge_grads,
        virial,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )


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
    """Compute real-space Ewald energy and optional direct derivative outputs.

    Energy-only calls participate in JAX autodiff through a private custom-JVP
    wrapper. ``compute_forces=True`` remains a forward/direct escape hatch for
    no-autograd MD/inference loops; charge-gradient and virial direct outputs
    are deprecated training-style outputs and warn.
    """
    component_deprecated_flags = tuple(
        name
        for name, enabled in (
            ("compute_charge_gradients", compute_charge_gradients),
            ("compute_virial", compute_virial),
        )
        if enabled
    )
    if component_deprecated_flags:
        warnings.warn(
            _component_direct_output_deprecation_msg(
                "ewald_real_space", component_deprecated_flags
            ),
            DeprecationWarning,
            stacklevel=2,
        )

    if compute_forces or compute_charge_gradients or compute_virial:
        return _ewald_real_space_impl(
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
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )

    if mask_value is None:
        mask_value = positions.shape[0]
    use_matrix = neighbor_matrix is not None and neighbor_matrix_shifts is not None
    return _ewald_real_space_energy_jvp(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        neighbor_list,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
        use_matrix,
    )


def _ewald_reciprocal_space_impl(
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
        System index for each atom. When provided, atoms must be grouped by
        system: ``batch_idx`` must be contiguous, nondecreasing, and use system
        IDs ``0..B-1``.
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

    # Compute total charge and Q/V correction factor (always float64).
    # The Warp fill kernel also has a historical total_charge output slot; pass
    # it a zero scratch buffer and use this explicit JAX value for corrections so
    # the buffer cannot become Q + Q/V.
    total_charge = _compute_total_charge(
        charges_cast, batch_idx, num_systems=cell_cast.shape[0]
    )
    volumes_for_background = jnp.abs(jnp.linalg.det(cell_cast)).astype(jnp.float64)
    total_charge_over_volume = total_charge / volumes_for_background
    fill_total_charge = jnp.zeros_like(total_charge)

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

        if _jax_can_tile_ewald_recip() and should_tile_ewald_recip_fill(
            int(max_atoms_per_system)
        ):
            (_fill_total_charge, cos_k_dot_r, sin_k_dot_r, real_sf, imag_sf) = (
                _JAX_BATCH_EWALD_RECIP_FILL_TILED[jnp.dtype(dtype)](
                    positions_cast,
                    charges_cast,
                    k_vectors_cast,
                    cell_cast,
                    alpha_arr,
                    atom_start,
                    atom_end,
                    fill_total_charge,
                    cos_k_dot_r,
                    sin_k_dot_r,
                    real_sf,
                    imag_sf,
                    max_blocks_per_system,
                )
            )
        else:
            (_fill_total_charge, cos_k_dot_r, sin_k_dot_r, real_sf, imag_sf) = (
                _jax_batch_ewald_reciprocal_fill_structure_factors[dtype](
                    positions_cast,
                    charges_cast,
                    k_vectors_cast,
                    cell_cast,
                    alpha_arr,
                    atom_start,
                    atom_end,
                    fill_total_charge,
                    cos_k_dot_r,
                    sin_k_dot_r,
                    real_sf,
                    imag_sf,
                    launch_dims=(num_k, num_systems, max_blocks_per_system),
                )
            )
    else:
        if _jax_can_tile_ewald_recip() and should_tile_ewald_recip_fill(int(num_atoms)):
            (_fill_total_charge, cos_k_dot_r, sin_k_dot_r, real_sf, imag_sf) = (
                _JAX_EWALD_RECIP_FILL_TILED[jnp.dtype(dtype)](
                    positions_cast,
                    charges_cast,
                    k_vectors_cast,
                    cell_cast,
                    alpha_arr,
                    fill_total_charge,
                    cos_k_dot_r,
                    sin_k_dot_r,
                    real_sf,
                    imag_sf,
                )
            )
        else:
            (_fill_total_charge, cos_k_dot_r, sin_k_dot_r, real_sf, imag_sf) = (
                _jax_ewald_reciprocal_fill_structure_factors[dtype](
                    positions_cast,
                    charges_cast,
                    k_vectors_cast,
                    cell_cast,
                    alpha_arr,
                    fill_total_charge,
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
            total_charge_over_volume,
            raw_energies,
            energies,
            launch_dims=(num_atoms,),
        )
    else:
        (energies,) = _jax_ewald_subtract_self_energy[dtype](
            charges_cast,
            alpha_arr,
            total_charge_over_volume,
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

            total_charges_v = (
                jnp.zeros(
                    num_systems,
                    dtype=dtype,
                )
                .at[batch_idx.astype(jnp.int32)]
                .add(charges_cast)
            )
            volumes_v = jnp.abs(jnp.linalg.det(cell_cast)).astype(dtype)
            alpha_v = alpha_arr.astype(dtype)
            e_bg = PI * total_charges_v**2 / (2.0 * alpha_v**2 * volumes_v)
            eye = jnp.eye(3, dtype=dtype)
            virial = virial - e_bg[:, jnp.newaxis, jnp.newaxis] * eye
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

            q_total = charges_cast.sum().astype(dtype)
            vol_v = jnp.abs(jnp.linalg.det(cell_cast.squeeze(0))).astype(dtype)
            alpha_val_v = alpha_arr.astype(dtype).squeeze()
            e_bg = PI * q_total**2 / (2.0 * alpha_val_v**2 * vol_v)
            eye = jnp.eye(3, dtype=dtype)
            virial = virial - e_bg * eye

    # Apply corrections to charge gradients if requested
    if compute_charge_gradients:
        # Self-energy gradient: 2 * alpha / sqrt(pi) * q
        alpha_val = alpha_arr[0] if not is_batched else alpha_arr[batch_idx]
        self_energy_grad = 2.0 * alpha_val / jnp.sqrt(PI) * charges_cast

        if is_batched:
            total_charge_over_volume_per_atom = total_charge_over_volume[batch_idx]
        else:
            total_charge_over_volume_per_atom = total_charge_over_volume[0]
        background_grad = (
            PI / (alpha_val * alpha_val) * total_charge_over_volume_per_atom
        )

        charge_grads = charge_grads - self_energy_grad - background_grad

    # Initialize optional variables to None if not computed
    if not (compute_forces or compute_charge_gradients):
        forces = None
        charge_grads = None
    elif not compute_charge_gradients:
        charge_grads = None

    return _build_electrostatic_result(
        energies,
        forces,
        charge_grads,
        virial,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )


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
    """Compute reciprocal-space Ewald energy and optional direct outputs.

    Energy-only calls participate in JAX autodiff through a private custom-JVP
    wrapper. ``compute_forces=True`` remains a forward/direct escape hatch for
    no-autograd MD/inference loops; charge-gradient and virial direct outputs
    are deprecated training-style outputs and warn.
    """
    component_deprecated_flags = tuple(
        name
        for name, enabled in (
            ("compute_charge_gradients", compute_charge_gradients),
            ("compute_virial", compute_virial),
        )
        if enabled
    )
    if component_deprecated_flags:
        warnings.warn(
            _component_direct_output_deprecation_msg(
                "ewald_reciprocal_space", component_deprecated_flags
            ),
            DeprecationWarning,
            stacklevel=2,
        )

    k_vectors = jax.lax.stop_gradient(k_vectors)

    if compute_forces or compute_charge_gradients or compute_virial:
        return _ewald_reciprocal_space_impl(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
            batch_idx=batch_idx,
            max_atoms_per_system=max_atoms_per_system,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )

    return _ewald_reciprocal_space_energy_jvp(
        positions,
        charges,
        cell,
        k_vectors,
        alpha,
        batch_idx,
        max_atoms_per_system,
    )


def _tangent_or_zeros(tangent, primal: jax.Array, dtype=None) -> jax.Array:
    """Materialize a custom-JVP tangent, replacing symbolic zeros with arrays."""
    out_dtype = primal.dtype if dtype is None else dtype
    if _is_symbolic_zero(tangent):
        return jnp.zeros(primal.shape, dtype=out_dtype)
    return tangent.astype(out_dtype)


def _stop_optional(value: jax.Array | None) -> jax.Array | None:
    """Stop gradients through an optional residual."""
    if value is None:
        return None
    return jax.lax.stop_gradient(value)


def _is_symbolic_zero(tangent) -> bool:
    """Return whether a custom-JVP tangent is JAX's symbolic zero sentinel."""
    return (
        tangent is None
        or isinstance(tangent, jax_ad.Zero)
        or tangent.__class__.__name__ == "SymbolicZero"
    )


def _cell_tangent_system_values(
    grad_cell: jax.Array,
    tangent_cell,
) -> jax.Array:
    """Contract a cell cotangent with a cell tangent per system."""
    tcell = _tangent_or_zeros(tangent_cell, grad_cell, dtype=jnp.float64)
    values = grad_cell.astype(jnp.float64) * tcell.astype(jnp.float64)
    if values.ndim == 2:
        return jnp.array([values.sum()], dtype=jnp.float64)
    return values.sum(axis=(1, 2))


def _kvector_tangent_from_cell(
    k_vectors: jax.Array,
    cell: jax.Array,
    tangent_cell: jax.Array,
) -> jax.Array:
    """Differentiate reciprocal vectors under a cell perturbation."""
    cell_3d = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
    tcell_3d = (
        tangent_cell if tangent_cell.ndim == 3 else tangent_cell[jnp.newaxis, :, :]
    )
    k_3d = k_vectors if k_vectors.ndim == 3 else k_vectors[jnp.newaxis, :, :]
    inv_cell_t = jnp.linalg.inv(jnp.swapaxes(cell_3d, -2, -1))
    tangent = -jnp.matmul(
        jnp.matmul(k_3d, jnp.swapaxes(tcell_3d, -2, -1)),
        inv_cell_t,
    )
    if k_vectors.ndim == 2:
        return tangent[0]
    return tangent


def _per_system_atom_counts(
    batch_idx: jax.Array | None,
    num_systems: int,
    num_atoms: int,
) -> jax.Array:
    """Return per-system atom counts as float64 for tangent redistribution."""
    if batch_idx is None:
        return jnp.full((num_systems,), float(num_atoms), dtype=jnp.float64)
    bidx = batch_idx.astype(jnp.int32)
    return (
        jnp.zeros((num_systems,), dtype=jnp.float64)
        .at[bidx]
        .add(jnp.ones((num_atoms,), dtype=jnp.float64))
    )


def _system_sum_from_atoms(
    values: jax.Array,
    batch_idx: jax.Array | None,
    num_systems: int,
) -> jax.Array:
    """Sum per-atom scalar values into one scalar per system."""
    if batch_idx is None:
        return values.sum(keepdims=True)
    return (
        jnp.zeros((num_systems,), dtype=values.dtype)
        .at[batch_idx.astype(jnp.int32)]
        .add(values)
    )


def _distribute_system_values(
    system_values: jax.Array,
    batch_idx: jax.Array | None,
    num_atoms: int,
) -> jax.Array:
    """Distribute per-system values uniformly over each system's atoms."""
    if batch_idx is None:
        if num_atoms == 0:
            return jnp.zeros((0,), dtype=system_values.dtype)
        return jnp.full(
            (num_atoms,), system_values[0] / num_atoms, dtype=system_values.dtype
        )

    bidx = batch_idx.astype(jnp.int32)
    counts = _per_system_atom_counts(batch_idx, system_values.shape[0], num_atoms)
    return (system_values / jnp.maximum(counts, 1.0))[bidx]


def _real_space_energy_reference(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    batch_idx: jax.Array | None,
    neighbor_list: jax.Array | None,
    neighbor_ptr: jax.Array | None,
    neighbor_shifts: jax.Array | None,
    neighbor_matrix: jax.Array | None,
    neighbor_matrix_shifts: jax.Array | None,
    mask_value: int,
    use_matrix: bool,
) -> jax.Array:
    """Pure JAX real-space per-atom energies for transposed weighted losses."""
    dtype = _normalize_dtype(positions.dtype)
    positions = positions.astype(dtype)
    charges = charges.astype(jnp.float64)
    cell_3d = cell.astype(dtype)
    if cell_3d.ndim == 2:
        cell_3d = cell_3d[jnp.newaxis, :, :]
    alpha_arr = _prepare_alpha_array(alpha, cell_3d.shape[0], dtype=jnp.float64)
    num_atoms = positions.shape[0]
    atom_system = (
        jnp.zeros((num_atoms,), dtype=jnp.int32)
        if batch_idx is None
        else batch_idx.astype(jnp.int32)
    )

    def _pair_energy(atom_i, atom_j, shifts):
        system = atom_system[atom_i]
        shifted = jnp.einsum("...j,...jk->...k", shifts.astype(dtype), cell_3d[system])
        rij = positions[atom_j] - positions[atom_i] + shifted
        distance_sq = jnp.sum(
            rij.astype(jnp.float64) * rij.astype(jnp.float64), axis=-1
        )
        active = distance_sq > 1e-16
        safe_distance = jnp.sqrt(jnp.where(active, distance_sq, 1.0))
        qi = charges[atom_i]
        qj = charges[atom_j]
        value = 0.5 * qi * qj * erfc(alpha_arr[system] * safe_distance) / safe_distance
        return jnp.where(active, value, 0.0)

    if use_matrix:
        if neighbor_matrix is None or neighbor_matrix_shifts is None:
            raise ValueError("neighbor_matrix and neighbor_matrix_shifts are required")
        valid = neighbor_matrix != int(mask_value)
        atom_i = jnp.broadcast_to(
            jnp.arange(num_atoms, dtype=jnp.int32)[:, jnp.newaxis],
            neighbor_matrix.shape,
        )
        atom_j = jnp.where(valid, neighbor_matrix.astype(jnp.int32), 0)
        energies = _pair_energy(atom_i, atom_j, neighbor_matrix_shifts)
        return jnp.where(valid, energies, 0.0).sum(axis=1)

    if neighbor_list is None or neighbor_ptr is None or neighbor_shifts is None:
        raise ValueError(
            "neighbor_list, neighbor_ptr, and neighbor_shifts are required"
        )
    atom_j = neighbor_list[1].astype(jnp.int32)
    counts = neighbor_ptr[1:] - neighbor_ptr[:-1]
    atom_i = jnp.repeat(
        jnp.arange(num_atoms, dtype=jnp.int32),
        counts,
        total_repeat_length=atom_j.shape[0],
    )
    edge_energies = _pair_energy(atom_i, atom_j, neighbor_shifts)
    return jnp.zeros((num_atoms,), dtype=jnp.float64).at[atom_i].add(edge_energies)


def _reciprocal_space_energy_reference(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    k_vectors: jax.Array,
    alpha: jax.Array,
    batch_idx: jax.Array | None,
) -> jax.Array:
    """Pure JAX reciprocal per-atom energies for transposed weighted losses."""
    dtype = _normalize_dtype(positions.dtype)
    positions = positions.astype(dtype)
    charges = charges.astype(jnp.float64)
    cell_3d = cell.astype(dtype)
    if cell_3d.ndim == 2:
        cell_3d = cell_3d[jnp.newaxis, :, :]
    num_systems = cell_3d.shape[0]
    alpha_arr = _prepare_alpha_array(alpha, num_systems, dtype=jnp.float64)
    if k_vectors.ndim == 2:
        kv = jnp.broadcast_to(k_vectors.astype(dtype), (num_systems,) + k_vectors.shape)
    else:
        kv = k_vectors.astype(dtype)

    num_atoms = positions.shape[0]
    atom_system = (
        jnp.zeros((num_atoms,), dtype=jnp.int32)
        if batch_idx is None
        else batch_idx.astype(jnp.int32)
    )
    system_mask = atom_system[jnp.newaxis, :] == jnp.arange(num_systems)[:, jnp.newaxis]
    volumes = jnp.abs(jnp.linalg.det(cell_3d)).astype(jnp.float64)
    phase = jnp.einsum(
        "skd,nd->skn", kv.astype(jnp.float64), positions.astype(jnp.float64)
    )
    cos_phase = jnp.cos(phase)
    sin_phase = jnp.sin(phase)
    masked_charges = charges[jnp.newaxis, :] * system_mask.astype(jnp.float64)
    k_sq = jnp.sum(kv.astype(jnp.float64) * kv.astype(jnp.float64), axis=-1)
    active_k = k_sq > 1e-10
    safe_k_sq = jnp.where(active_k, k_sq, 1.0)
    green = (
        jnp.exp(-safe_k_sq / (4.0 * alpha_arr[:, jnp.newaxis] ** 2))
        * (8.0 * PI)
        / volumes[:, jnp.newaxis]
        / safe_k_sq
    )
    green = jnp.where(active_k, green, 0.0)
    real_sf = green * jnp.sum(masked_charges[:, jnp.newaxis, :] * cos_phase, axis=2)
    imag_sf = green * jnp.sum(masked_charges[:, jnp.newaxis, :] * sin_phase, axis=2)

    atom_phase = jnp.swapaxes(phase, 1, 2)[atom_system, jnp.arange(num_atoms)]
    atom_cos = jnp.cos(atom_phase)
    atom_sin = jnp.sin(atom_phase)
    raw = (
        0.5
        * charges
        * jnp.sum(
            real_sf[atom_system] * atom_cos + imag_sf[atom_system] * atom_sin,
            axis=1,
        )
    )
    total_charge_over_volume = (
        jnp.zeros((num_systems,), dtype=jnp.float64).at[atom_system].add(charges)
        / volumes
    )
    atom_alpha = alpha_arr[atom_system]
    self_energy = atom_alpha * charges * charges / jnp.sqrt(PI)
    background = (
        PI
        * charges
        * total_charge_over_volume[atom_system]
        / (2.0 * atom_alpha * atom_alpha)
    )
    return raw - self_energy - background


def _empty_i32() -> jax.Array:
    """Return a zero-size int32 sentinel for inactive Warp array slots."""
    return jnp.zeros((0,), dtype=jnp.int32)


def _empty_vec(dtype) -> jax.Array:
    """Return a zero-size vec3 sentinel for inactive Warp vector slots."""
    return jnp.zeros((0, 3), dtype=dtype)


def _empty_vec_matrix(dtype) -> jax.Array:
    """Return a zero-size vec3 matrix sentinel for inactive Warp vector2d slots."""
    return jnp.zeros((0, 0, 3), dtype=dtype)


def _empty_mat(dtype) -> jax.Array:
    """Return a zero-size mat33 sentinel for inactive Warp matrix slots."""
    return jnp.zeros((0, 3, 3), dtype=dtype)


def _empty_matrix_i32() -> jax.Array:
    """Return a zero-size int32 matrix sentinel for inactive neighbor matrices."""
    return jnp.zeros((0, 0), dtype=jnp.int32)


def _empty_shift_matrix() -> jax.Array:
    """Return a zero-size vec3i matrix sentinel for inactive shift matrices."""
    return jnp.zeros((0, 0, 3), dtype=jnp.int32)


@functools.partial(jax.custom_jvp, nondiff_argnums=(10, 11))
def _ewald_real_energy_derivatives(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    batch_idx: jax.Array | None,
    neighbor_list: jax.Array | None,
    neighbor_ptr: jax.Array | None,
    neighbor_shifts: jax.Array | None,
    neighbor_matrix: jax.Array | None,
    neighbor_matrix_shifts: jax.Array | None,
    mask_value: int,
    use_matrix: bool,
) -> tuple[jax.Array, jax.Array]:
    """Return real-space ``dE/dR`` and ``dE/dq`` with factory double-backward."""
    energy, forces, charge_grads = _ewald_real_space_impl(
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
        compute_forces=True,
        compute_charge_gradients=True,
        compute_virial=False,
    )
    del energy
    return jax.lax.stop_gradient(-forces), jax.lax.stop_gradient(charge_grads)


@_ewald_real_energy_derivatives.defjvp
def _ewald_real_energy_derivatives_jvp(
    mask_value: int,
    use_matrix: bool,
    primals: tuple[jax.Array | None, ...],
    tangents: tuple[jax.Array | None, ...],
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """JVP rule for real-space first derivatives using real-space double-backward."""
    deriv_state = (
        _DerivState.E_F if _is_symbolic_zero(tangents[1]) else _DerivState.E_F_dQ
    )
    return _ewald_real_energy_derivatives_jvp_impl(
        mask_value,
        use_matrix,
        primals,
        tangents,
        deriv_state,
    )


def _ewald_real_energy_derivatives_jvp_impl(
    mask_value: int,
    use_matrix: bool,
    primals: tuple[jax.Array | None, ...],
    tangents: tuple[jax.Array | None, ...],
    deriv_state: _DerivState,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """JVP rule body for real-space first derivatives with explicit state."""
    (
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        neighbor_list,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
    ) = primals
    (
        v_pos,
        v_charge,
        _v_cell,
        _v_alpha,
        _v_batch_idx,
        _v_neighbor_list,
        _v_neighbor_ptr,
        _v_neighbor_shifts,
        _v_neighbor_matrix,
        _v_neighbor_matrix_shifts,
    ) = tangents

    primal_out = _ewald_real_energy_derivatives(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        neighbor_list,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
        use_matrix,
    )

    dtype = _normalize_dtype(positions.dtype)
    positions_cast = positions.astype(dtype)
    charges_cast = charges.astype(dtype)
    cell_cast = cell.astype(dtype)
    if cell_cast.ndim == 2:
        cell_cast = cell_cast[jnp.newaxis, :, :]
    alpha_arr = _prepare_alpha_array(alpha, cell_cast.shape[0], dtype=dtype)

    num_atoms = positions_cast.shape[0]
    num_systems = cell_cast.shape[0]
    v_pos_arr = _tangent_or_zeros(v_pos, positions_cast, dtype=dtype)
    v_charge_arr = _tangent_or_zeros(v_charge, charges, dtype=jnp.float64)
    batch_i32 = batch_idx.astype(jnp.int32) if batch_idx is not None else _empty_i32()

    if use_matrix:
        if neighbor_matrix is None or neighbor_matrix_shifts is None:
            raise ValueError("neighbor_matrix and neighbor_matrix_shifts are required")
        idx_j = _empty_i32()
        neighbor_ptr_i32 = _empty_i32()
        shifts_i32 = _empty_vec(jnp.int32)
        matrix_i32 = neighbor_matrix.astype(jnp.int32)
        matrix_shifts_i32 = neighbor_matrix_shifts.astype(jnp.int32)
        kernel = _jax_ewald_real_double_backward(
            batch_idx is not None,
            "matrix",
            deriv_state,
            False,
        )[dtype]
    else:
        if neighbor_list is None or neighbor_ptr is None or neighbor_shifts is None:
            raise ValueError(
                "neighbor_list, neighbor_ptr, and neighbor_shifts are required"
            )
        idx_j = neighbor_list[1].astype(jnp.int32)
        neighbor_ptr_i32 = neighbor_ptr.astype(jnp.int32)
        shifts_i32 = neighbor_shifts.astype(jnp.int32)
        matrix_i32 = _empty_matrix_i32()
        matrix_shifts_i32 = _empty_shift_matrix()
        kernel = _jax_ewald_real_double_backward(
            batch_idx is not None,
            "list",
            deriv_state,
            False,
        )[dtype]

    grad_energy = jnp.ones((num_systems,), dtype=jnp.float64)
    grad_grad_energy = jnp.zeros((num_systems,), dtype=jnp.float64)
    grad_positions = jnp.zeros((num_atoms, 3), dtype=dtype)
    grad_charges = jnp.zeros((num_atoms,), dtype=jnp.float64)

    grad_grad_energy, grad_positions, grad_charges = kernel(
        v_pos_arr,
        v_charge_arr,
        _empty_mat(dtype),
        grad_energy,
        positions_cast,
        charges_cast,
        cell_cast,
        batch_i32,
        idx_j,
        neighbor_ptr_i32,
        shifts_i32,
        matrix_i32,
        matrix_shifts_i32,
        int(mask_value),
        alpha_arr,
        grad_grad_energy,
        grad_positions,
        grad_charges,
        _empty_mat(dtype),
        launch_dims=(num_atoms,),
    )
    del grad_grad_energy
    return primal_out, (grad_positions, grad_charges)


@functools.partial(jax.custom_jvp, nondiff_argnums=(6,))
def _ewald_reciprocal_energy_derivatives(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    k_vectors: jax.Array,
    alpha: jax.Array,
    batch_idx: jax.Array | None,
    max_atoms_per_system: int | None,
) -> tuple[jax.Array, jax.Array]:
    """Return reciprocal ``dE/dR`` and ``dE/dq`` with factory double-backward."""
    energy, forces, charge_grads = _ewald_reciprocal_space_impl(
        positions=positions,
        charges=charges,
        cell=cell,
        k_vectors=k_vectors,
        alpha=alpha,
        batch_idx=batch_idx,
        max_atoms_per_system=max_atoms_per_system,
        compute_forces=True,
        compute_charge_gradients=True,
        compute_virial=False,
    )
    del energy
    return jax.lax.stop_gradient(-forces), jax.lax.stop_gradient(charge_grads)


@_ewald_reciprocal_energy_derivatives.defjvp
def _ewald_reciprocal_energy_derivatives_jvp(
    max_atoms_per_system: int | None,
    primals: tuple[jax.Array | None, ...],
    tangents: tuple[jax.Array | None, ...],
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """JVP rule for reciprocal first derivatives using reciprocal double-backward."""
    positions, charges, cell, k_vectors, alpha, batch_idx = primals
    (
        v_pos,
        v_charge,
        _v_cell,
        _v_k_vectors,
        _v_alpha,
        _v_batch_idx,
    ) = tangents

    primal_out = _ewald_reciprocal_energy_derivatives(
        positions,
        charges,
        cell,
        k_vectors,
        alpha,
        batch_idx,
        max_atoms_per_system,
    )

    dtype = _normalize_dtype(positions.dtype)
    positions_cast = positions.astype(dtype)
    charges_cast = charges.astype(dtype)
    cell_cast = cell.astype(dtype)
    if cell_cast.ndim == 2:
        cell_cast = cell_cast[jnp.newaxis, :, :]
    alpha_arr = _prepare_alpha_array(alpha, cell_cast.shape[0], dtype=dtype)
    k_vectors_cast = k_vectors.astype(dtype)

    num_atoms = positions_cast.shape[0]
    num_systems = cell_cast.shape[0]
    if batch_idx is not None and k_vectors_cast.ndim == 2:
        k_vectors_2d = jnp.tile(k_vectors_cast[jnp.newaxis, :, :], (num_systems, 1, 1))
    elif batch_idx is None and k_vectors_cast.ndim == 2:
        k_vectors_2d = k_vectors_cast[jnp.newaxis, :, :]
    else:
        k_vectors_2d = k_vectors_cast
    num_k = k_vectors_2d.shape[1]

    v_pos_arr = _tangent_or_zeros(v_pos, positions_cast, dtype=dtype)
    v_charge_arr = _tangent_or_zeros(v_charge, charges, dtype=jnp.float64)
    deriv_dq = 0 if _is_symbolic_zero(v_charge) else 1
    batch_i32 = batch_idx.astype(jnp.int32) if batch_idx is not None else _empty_i32()
    if batch_idx is None:
        atom_start = _empty_i32()
        atom_end = _empty_i32()
    else:
        atom_counts = jnp.bincount(batch_i32, length=num_systems)
        atom_end = jnp.cumsum(atom_counts).astype(jnp.int32)
        atom_start = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), atom_end[:-1]])

    grad_energy = jnp.ones((num_systems,), dtype=jnp.float64)
    gA = jnp.zeros((num_systems, num_k), dtype=jnp.float64)
    gB = jnp.zeros((num_systems, num_k), dtype=jnp.float64)
    gC = jnp.zeros((num_systems, num_k), dtype=jnp.float64)
    gD = jnp.zeros((num_systems, num_k), dtype=jnp.float64)
    gP = jnp.zeros((num_systems, num_k), dtype=jnp.float64)
    gQ = jnp.zeros((num_systems, num_k), dtype=jnp.float64)
    grad_grad_energy = jnp.zeros((num_systems,), dtype=jnp.float64)

    reduce_kernel = (
        _jax_batch_ewald_reciprocal_double_backward_reduce[dtype]
        if batch_idx is not None
        else _jax_ewald_reciprocal_double_backward_reduce[dtype]
    )
    gA, gB, gC, gD, gP, gQ, grad_grad_energy = reduce_kernel(
        positions_cast,
        charges_cast,
        k_vectors_2d,
        cell_cast,
        alpha_arr,
        batch_i32,
        atom_start,
        atom_end,
        v_pos_arr,
        v_charge_arr,
        grad_energy,
        deriv_dq,
        gA,
        gB,
        gC,
        gD,
        gP,
        gQ,
        grad_grad_energy,
        0,
        jnp.zeros((0,), dtype=jnp.float64),
        _empty_vec_matrix(dtype),
        jnp.zeros((0,), dtype=jnp.float64),
        jnp.zeros((0, 0), dtype=jnp.float64),
        jnp.zeros((0, 0), dtype=jnp.float64),
        _empty_vec_matrix(dtype),
        jnp.zeros((0,), dtype=jnp.float64),
        launch_dims=(num_k, num_systems) if batch_idx is not None else (num_k,),
    )
    del grad_grad_energy

    grad_positions = jnp.zeros((num_atoms, 3), dtype=dtype)
    grad_charges = jnp.zeros((num_atoms,), dtype=jnp.float64)
    compute_kernel = (
        _jax_batch_ewald_reciprocal_double_backward_compute[dtype]
        if batch_idx is not None
        else _jax_ewald_reciprocal_double_backward_compute[dtype]
    )
    grad_positions, grad_charges = compute_kernel(
        positions_cast,
        charges_cast,
        k_vectors_2d,
        batch_i32,
        v_pos_arr,
        v_charge_arr,
        grad_energy,
        deriv_dq,
        gA,
        gB,
        gC,
        gD,
        gP,
        gQ,
        grad_positions,
        grad_charges,
        0,
        alpha_arr,
        jnp.zeros((0,), dtype=jnp.float64),
        _empty_vec_matrix(dtype),
        jnp.zeros((0,), dtype=jnp.float64),
        jnp.zeros((0, 0), dtype=jnp.float64),
        jnp.zeros((0, 0), dtype=jnp.float64),
        launch_dims=(num_atoms,),
    )

    volumes = jnp.abs(jnp.linalg.det(cell_cast)).astype(jnp.float64)
    alpha_per_atom = alpha_arr[0] if batch_idx is None else alpha_arr[batch_i32]
    self_hess = 2.0 * alpha_per_atom / jnp.sqrt(PI)
    if batch_idx is None:
        bg_coeff = PI / (alpha_arr[0] * alpha_arr[0] * volumes[0])
        bg_grad = jnp.full_like(
            charges, bg_coeff * v_charge_arr.sum(), dtype=jnp.float64
        )
    else:
        bg_coeff = PI / (alpha_arr * alpha_arr * volumes)
        vq_sum = (
            jnp.zeros((num_systems,), dtype=jnp.float64).at[batch_i32].add(v_charge_arr)
        )
        bg_grad = bg_coeff[batch_i32] * vq_sum[batch_i32]
    grad_charges = grad_charges - self_hess.astype(jnp.float64) * v_charge_arr - bg_grad

    return primal_out, (grad_positions, grad_charges)


_ewald_real_energy_derivatives_jvp_raw = _ewald_real_energy_derivatives_jvp
_ewald_reciprocal_energy_derivatives_jvp_raw = _ewald_reciprocal_energy_derivatives_jvp


def _ewald_real_hvp(
    v_pos: jax.Array,
    v_charge: jax.Array,
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    batch_idx: jax.Array | None,
    neighbor_list: jax.Array | None,
    neighbor_ptr: jax.Array | None,
    neighbor_shifts: jax.Array | None,
    neighbor_matrix: jax.Array | None,
    neighbor_matrix_shifts: jax.Array | None,
    mask_value: int,
    use_matrix: bool,
    deriv_state: _DerivState,
) -> tuple[jax.Array, jax.Array]:
    """Linear real-space HVP with an explicit transpose rule."""

    residuals = (
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        neighbor_list,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
    )

    # The HVP is a symmetric linear map in (v_pos, v_charge); custom_vjp supplies
    # its transpose (== itself) so reverse-mode over this JVP yields the Hessian.
    @jax.custom_vjp
    def _linear_hvp(lin_pos, lin_charge):
        _primal, tangent = _ewald_real_energy_derivatives_jvp_impl(
            mask_value,
            use_matrix,
            residuals,
            (lin_pos, lin_charge, None, None, None, None, None, None, None, None),
            deriv_state,
        )
        return (
            tangent[0].astype(positions.dtype),
            tangent[1].astype(charges.dtype),
        )

    def _linear_hvp_fwd(lin_pos, lin_charge):
        return _linear_hvp(lin_pos, lin_charge), None

    def _linear_hvp_bwd(_res, ct_out):
        ct_pos, ct_charge = ct_out
        return _linear_hvp(
            _tangent_or_zeros(ct_pos, positions, dtype=positions.dtype),
            _tangent_or_zeros(ct_charge, charges, dtype=charges.dtype),
        )

    _linear_hvp.defvjp(_linear_hvp_fwd, _linear_hvp_bwd)

    return _linear_hvp(v_pos, v_charge)


def _ewald_reciprocal_hvp(
    v_pos: jax.Array,
    v_charge: jax.Array,
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    k_vectors: jax.Array,
    alpha: jax.Array,
    batch_idx: jax.Array | None,
    max_atoms_per_system: int | None,
) -> tuple[jax.Array, jax.Array]:
    """Linear reciprocal HVP with an explicit transpose rule."""

    residuals = (positions, charges, cell, k_vectors, alpha, batch_idx)

    # The HVP is a symmetric linear map in (v_pos, v_charge); custom_vjp supplies
    # its transpose (== itself) so reverse-mode over this JVP yields the Hessian.
    @jax.custom_vjp
    def _linear_hvp(lin_pos, lin_charge):
        _primal, tangent = _ewald_reciprocal_energy_derivatives_jvp_raw(
            max_atoms_per_system,
            residuals,
            (lin_pos, lin_charge, None, None, None, None),
        )
        return (
            tangent[0].astype(positions.dtype),
            tangent[1].astype(charges.dtype),
        )

    def _linear_hvp_fwd(lin_pos, lin_charge):
        return _linear_hvp(lin_pos, lin_charge), None

    def _linear_hvp_bwd(_res, ct_out):
        ct_pos, ct_charge = ct_out
        return _linear_hvp(
            _tangent_or_zeros(ct_pos, positions, dtype=positions.dtype),
            _tangent_or_zeros(ct_charge, charges, dtype=charges.dtype),
        )

    _linear_hvp.defvjp(_linear_hvp_fwd, _linear_hvp_bwd)

    return _linear_hvp(v_pos, v_charge)


def _ewald_real_energy_derivatives_jvp_wrapped(
    mask_value: int,
    use_matrix: bool,
    primals: tuple[jax.Array | None, ...],
    tangents: tuple[jax.Array | None, ...],
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """JVP rule that routes real-space HVP transposes through a custom VJP."""
    (
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        neighbor_list,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
    ) = primals
    v_pos, v_charge = tangents[:2]
    deriv_state = _DerivState.E_F if _is_symbolic_zero(v_charge) else _DerivState.E_F_dQ
    primal_out = _ewald_real_energy_derivatives(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        neighbor_list,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
        use_matrix,
    )
    tangent_out = _ewald_real_hvp(
        _tangent_or_zeros(v_pos, positions, dtype=positions.dtype),
        _tangent_or_zeros(v_charge, charges, dtype=charges.dtype),
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        neighbor_list,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
        use_matrix,
        deriv_state,
    )
    return primal_out, (
        tangent_out[0].astype(primal_out[0].dtype),
        tangent_out[1].astype(primal_out[1].dtype),
    )


def _ewald_reciprocal_energy_derivatives_jvp_wrapped(
    max_atoms_per_system: int | None,
    primals: tuple[jax.Array | None, ...],
    tangents: tuple[jax.Array | None, ...],
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """JVP rule that routes reciprocal HVP transposes through a custom VJP."""
    positions, charges, cell, k_vectors, alpha, batch_idx = primals
    v_pos, v_charge = tangents[:2]
    primal_out = _ewald_reciprocal_energy_derivatives(
        positions,
        charges,
        cell,
        k_vectors,
        alpha,
        batch_idx,
        max_atoms_per_system,
    )
    tangent_out = _ewald_reciprocal_hvp(
        _tangent_or_zeros(v_pos, positions, dtype=positions.dtype),
        _tangent_or_zeros(v_charge, charges, dtype=charges.dtype),
        positions,
        charges,
        cell,
        k_vectors,
        alpha,
        batch_idx,
        max_atoms_per_system,
    )
    return primal_out, (
        tangent_out[0].astype(primal_out[0].dtype),
        tangent_out[1].astype(primal_out[1].dtype),
    )


_ewald_real_energy_derivatives.defjvp(
    _ewald_real_energy_derivatives_jvp_wrapped,
    symbolic_zeros=True,
)
_ewald_reciprocal_energy_derivatives.defjvp(
    _ewald_reciprocal_energy_derivatives_jvp_wrapped,
    symbolic_zeros=True,
)


@functools.partial(jax.custom_jvp, nondiff_argnums=(10, 11))
def _ewald_real_space_energy_jvp(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    batch_idx: jax.Array | None,
    neighbor_list: jax.Array | None,
    neighbor_ptr: jax.Array | None,
    neighbor_shifts: jax.Array | None,
    neighbor_matrix: jax.Array | None,
    neighbor_matrix_shifts: jax.Array | None,
    mask_value: int,
    use_matrix: bool,
) -> jax.Array:
    """Energy-only real-space Ewald wrapper with custom autodiff."""
    energy = _ewald_real_space_impl(
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
        compute_forces=False,
        compute_charge_gradients=False,
        compute_virial=False,
    )
    return jax.lax.stop_gradient(energy)


def _ewald_real_space_energy_jvp_rule(
    mask_value: int,
    use_matrix: bool,
    primals: tuple[jax.Array | None, ...],
    tangents: tuple[jax.Array | None, ...],
) -> tuple[jax.Array, jax.Array]:
    """JVP rule for the real-space per-atom energy vector."""
    (
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        neighbor_list,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
    ) = primals
    (
        t_positions,
        t_charges,
        t_cell,
        _t_alpha,
        _t_batch_idx,
        _t_neighbor_list,
        _t_neighbor_ptr,
        _t_neighbor_shifts,
        _t_neighbor_matrix,
        _t_neighbor_matrix_shifts,
    ) = tangents

    primal_out = _ewald_real_space_energy_jvp(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        neighbor_list,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        mask_value,
        use_matrix,
    )
    tpos = _tangent_or_zeros(t_positions, positions, dtype=positions.dtype)
    tq = _tangent_or_zeros(t_charges, charges, dtype=charges.dtype)
    if (
        not _is_symbolic_zero(t_positions)
        or not _is_symbolic_zero(t_charges)
        or not _is_symbolic_zero(t_cell)
    ):
        tcell = _tangent_or_zeros(t_cell, cell, dtype=cell.dtype)
        charges_ref = charges.astype(jnp.float64)
        tq_ref = tq.astype(jnp.float64)
        _reference_out, tangent_out = jax.jvp(
            lambda p, q, c: _real_space_energy_reference(
                p,
                q,
                c,
                alpha,
                batch_idx,
                neighbor_list,
                neighbor_ptr,
                neighbor_shifts,
                neighbor_matrix,
                neighbor_matrix_shifts,
                mask_value,
                use_matrix,
            ),
            (positions, charges_ref, cell),
            (tpos, tq_ref, tcell),
        )
        return primal_out, tangent_out.astype(primal_out.dtype)

    dpos, dq = _ewald_real_energy_derivatives(
        positions,
        charges,
        jax.lax.stop_gradient(cell),
        jax.lax.stop_gradient(alpha),
        _stop_optional(batch_idx),
        _stop_optional(neighbor_list),
        _stop_optional(neighbor_ptr),
        _stop_optional(neighbor_shifts),
        _stop_optional(neighbor_matrix),
        _stop_optional(neighbor_matrix_shifts),
        mask_value,
        use_matrix,
    )
    atom_tangent = (dpos.astype(jnp.float64) * tpos.astype(jnp.float64)).sum(axis=1)
    atom_tangent = atom_tangent + dq.astype(jnp.float64) * tq.astype(jnp.float64)

    cell_3d = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
    num_atoms = positions.shape[0]
    num_systems = cell_3d.shape[0]
    system_tangent = _system_sum_from_atoms(atom_tangent, batch_idx, num_systems)
    if not _is_symbolic_zero(t_cell):
        _energy, _forces, _charge_grads, virial = ewald_real_space(
            positions=jax.lax.stop_gradient(positions),
            charges=jax.lax.stop_gradient(charges),
            cell=jax.lax.stop_gradient(cell),
            alpha=alpha,
            batch_idx=_stop_optional(batch_idx),
            neighbor_list=_stop_optional(neighbor_list),
            neighbor_ptr=_stop_optional(neighbor_ptr),
            neighbor_shifts=_stop_optional(neighbor_shifts),
            neighbor_matrix=_stop_optional(neighbor_matrix),
            neighbor_matrix_shifts=_stop_optional(neighbor_matrix_shifts),
            mask_value=mask_value,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        grad_cell = _cell_grad_from_strain_virial(
            positions=positions,
            cell=cell,
            batch_idx=batch_idx,
            grad_positions=dpos,
            virial=jax.lax.stop_gradient(virial),
            grad_system=jnp.ones((num_systems,), dtype=jnp.float64),
        )
        system_tangent = system_tangent + _cell_tangent_system_values(
            jax.lax.stop_gradient(grad_cell),
            t_cell,
        )

    tangent_out = _distribute_system_values(system_tangent, batch_idx, num_atoms)
    return primal_out, tangent_out.astype(primal_out.dtype)


_ewald_real_space_energy_jvp.defjvp(
    _ewald_real_space_energy_jvp_rule,
    symbolic_zeros=True,
)


@functools.partial(jax.custom_jvp, nondiff_argnums=(6,))
def _ewald_reciprocal_space_energy_jvp(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    k_vectors: jax.Array,
    alpha: jax.Array,
    batch_idx: jax.Array | None,
    max_atoms_per_system: int | None,
) -> jax.Array:
    """Energy-only reciprocal Ewald wrapper with custom autodiff."""
    energy = _ewald_reciprocal_space_impl(
        positions=positions,
        charges=charges,
        cell=cell,
        k_vectors=k_vectors,
        alpha=alpha,
        batch_idx=batch_idx,
        max_atoms_per_system=max_atoms_per_system,
        compute_forces=False,
        compute_charge_gradients=False,
        compute_virial=False,
    )
    return jax.lax.stop_gradient(energy)


def _ewald_reciprocal_space_energy_jvp_rule(
    max_atoms_per_system: int | None,
    primals: tuple[jax.Array | None, ...],
    tangents: tuple[jax.Array | None, ...],
) -> tuple[jax.Array, jax.Array]:
    """JVP rule for the reciprocal per-atom energy vector."""
    positions, charges, cell, k_vectors, alpha, batch_idx = primals
    (
        t_positions,
        t_charges,
        t_cell,
        _t_k_vectors,
        _t_alpha,
        _t_batch_idx,
    ) = tangents

    primal_out = _ewald_reciprocal_space_energy_jvp(
        positions,
        charges,
        cell,
        k_vectors,
        alpha,
        batch_idx,
        max_atoms_per_system,
    )
    tpos = _tangent_or_zeros(t_positions, positions, dtype=positions.dtype)
    tq = _tangent_or_zeros(t_charges, charges, dtype=charges.dtype)
    tcell = _tangent_or_zeros(t_cell, cell, dtype=cell.dtype)
    tk = jnp.zeros_like(k_vectors)
    charges_ref = charges.astype(jnp.float64)
    tq_ref = tq.astype(jnp.float64)
    _reference_out, tangent_out = jax.jvp(
        lambda p, q, c, k: _reciprocal_space_energy_reference(
            p,
            q,
            c,
            k,
            alpha,
            batch_idx,
        ),
        (positions, charges_ref, cell, k_vectors),
        (tpos, tq_ref, tcell, tk),
    )
    return primal_out, tangent_out.astype(primal_out.dtype)


_ewald_reciprocal_space_energy_jvp.defjvp(
    _ewald_reciprocal_space_energy_jvp_rule,
    symbolic_zeros=True,
)


@functools.partial(jax.custom_jvp, nondiff_argnums=(11, 12, 13))
def _ewald_summation_energy_jvp(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    k_vectors: jax.Array,
    batch_idx: jax.Array | None,
    neighbor_list: jax.Array | None,
    neighbor_ptr: jax.Array | None,
    neighbor_shifts: jax.Array | None,
    neighbor_matrix: jax.Array | None,
    neighbor_matrix_shifts: jax.Array | None,
    k_vectors_are_internal: bool,
    max_atoms_per_system: int | None,
    mask_value: int,
) -> jax.Array:
    """Energy-only full Ewald wrapper with a second-order-capable custom JVP."""
    energy = _ewald_summation_impl(
        positions=positions,
        charges=charges,
        cell=cell,
        alpha=alpha,
        k_vectors=k_vectors,
        batch_idx=batch_idx,
        max_atoms_per_system=max_atoms_per_system,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        mask_value=mask_value,
        compute_forces=False,
        compute_charge_gradients=False,
        compute_virial=False,
    )
    return jax.lax.stop_gradient(energy)


def _ewald_summation_energy_jvp_rule(
    k_vectors_are_internal: bool,
    max_atoms_per_system: int | None,
    mask_value: int,
    primals: tuple[jax.Array | None, ...],
    tangents: tuple[jax.Array | None, ...],
) -> tuple[jax.Array, jax.Array]:
    """JVP rule for the full Ewald per-atom energy vector."""
    (
        positions,
        charges,
        cell,
        alpha,
        k_vectors,
        batch_idx,
        neighbor_list,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
    ) = primals
    (
        t_positions,
        t_charges,
        t_cell,
        _t_alpha,
        _t_k_vectors,
        _t_batch_idx,
        _t_neighbor_list,
        _t_neighbor_ptr,
        _t_neighbor_shifts,
        _t_neighbor_matrix,
        _t_neighbor_matrix_shifts,
    ) = tangents

    primal_out = _ewald_summation_energy_jvp(
        positions,
        charges,
        cell,
        alpha,
        k_vectors,
        batch_idx,
        neighbor_list,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        k_vectors_are_internal,
        max_atoms_per_system,
        mask_value,
    )

    tpos = _tangent_or_zeros(t_positions, positions, dtype=positions.dtype)
    tq = _tangent_or_zeros(t_charges, charges, dtype=charges.dtype)
    tcell = _tangent_or_zeros(t_cell, cell, dtype=cell.dtype)
    if not _is_symbolic_zero(t_cell):
        if not k_vectors_are_internal:
            tk = jnp.zeros_like(k_vectors)
        else:
            tk = _kvector_tangent_from_cell(k_vectors, cell, tcell)
    else:
        tk = jnp.zeros_like(k_vectors)
    use_matrix = neighbor_matrix is not None and neighbor_matrix_shifts is not None
    _component_out, tangent_out = jax.jvp(
        lambda p, q, c, k: (
            _ewald_real_space_energy_jvp(
                p,
                q,
                c,
                alpha,
                batch_idx,
                neighbor_list,
                neighbor_ptr,
                neighbor_shifts,
                neighbor_matrix,
                neighbor_matrix_shifts,
                mask_value,
                use_matrix,
            )
            + _reciprocal_space_energy_reference(
                p,
                q,
                c,
                k,
                alpha,
                batch_idx,
            )
        ),
        (positions, charges, cell, k_vectors),
        (tpos, tq, tcell, tk),
    )
    return primal_out, tangent_out.astype(primal_out.dtype)


_ewald_summation_energy_jvp.defjvp(
    _ewald_summation_energy_jvp_rule,
    symbolic_zeros=True,
)


def _resolve_ewald_summation_parameters(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: float | jax.Array | None,
    k_vectors: jax.Array | None,
    k_cutoff: float | jax.Array | None,
    miller_bounds: tuple[int, int, int] | None,
    batch_idx: jax.Array | None,
    accuracy: float,
) -> tuple[float | jax.Array, jax.Array]:
    """Resolve Ewald ``alpha`` and ``k_vectors`` once for forward/backward reuse."""
    if alpha is None or k_cutoff is None:
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

    if k_vectors is None:
        cell_3d = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
        if k_cutoff is None:
            raise ValueError("k_cutoff must be provided if k_vectors is None")
        k_vectors = generate_k_vectors_ewald_summation(
            cell=cell_3d,
            k_cutoff=k_cutoff,
            miller_bounds=miller_bounds,
        )

    return alpha, k_vectors


def _ewald_summation_impl(
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
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    hybrid_forces: bool = False,
    accuracy: float = 1e-6,
    pbc: jax.Array | None = None,
    slab_correction: bool = False,
) -> jax.Array | tuple[jax.Array, ...]:
    """Compute complete Ewald summation implementation.

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
        When supplied, treated as static metadata that corresponds to the
        current ``cell``.
    k_cutoff : float | None
        K-space cutoff. Used only if k_vectors is None.
    miller_bounds : tuple[int, int, int] | None, optional
        Precomputed maximum Miller indices (M_h, M_k, M_l). Forwarded to
        :func:`generate_k_vectors_ewald_summation` when ``k_vectors`` is ``None``.
        When provided, makes k-vector generation compatible with ``jax.jit``.
        Use :func:`generate_miller_indices` to precompute. Ignored when
        ``k_vectors`` is explicitly provided.
    batch_idx : jax.Array | None, shape (N,)
        System index for each atom. When provided, atoms must be grouped by
        system: ``batch_idx`` must be contiguous, nondecreasing, and use system
        IDs ``0..B-1``.
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
    compute_charge_gradients : bool, default=False
        Whether to compute charge gradients dE/dq_i.
    compute_virial : bool, default=False
        Whether to compute the virial tensor.
    hybrid_forces : bool, default=False
        Whether to detach the force path and inject analytical charge gradients.
    accuracy : float, default=1e-6
        Target accuracy for automatic parameter estimation.
    pbc : jax.Array, shape (3,) or (B, 3), dtype=bool, optional
        Per-system periodic boundary conditions. Required when
        ``slab_correction=True``. True marks periodic directions and False
        marks the non-periodic slab direction.
    slab_correction : bool, default=False
        If True, add the Yeh-Berkowitz/Ballenegger slab correction to the
        3D-periodic Ewald outputs.

    Returns
    -------
    energies : jax.Array, shape (N,)
        Per-atom total Ewald energy.
    forces : jax.Array, shape (N, 3), optional
        Forces (if compute_forces=True).
    charge_gradients : jax.Array, shape (N,), optional
        Charge gradients (if compute_charge_gradients=True).
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
    cell, num_systems = _prepare_cell(cell)
    if batch_idx is not None:
        batch_idx = batch_idx.astype(jnp.int32)
    if slab_correction:
        pbc = _prepare_pbc_for_slab(pbc, num_systems)

    explicit_k_vectors = k_vectors is not None
    alpha, k_vectors = _resolve_ewald_summation_parameters(
        positions=positions,
        charges=charges,
        cell=cell,
        alpha=alpha,
        k_vectors=k_vectors,
        k_cutoff=k_cutoff,
        miller_bounds=miller_bounds,
        batch_idx=batch_idx,
        accuracy=accuracy,
    )
    if explicit_k_vectors:
        k_vectors = jax.lax.stop_gradient(k_vectors)
    charges_orig = charges
    need_charge_gradients = compute_charge_gradients or hybrid_forces
    num_systems = cell.shape[0] if cell.ndim == 3 else 1
    if hybrid_forces:
        positions = jax.lax.stop_gradient(positions)
        charges = jax.lax.stop_gradient(charges)
        cell = jax.lax.stop_gradient(cell)
        alpha = jax.lax.stop_gradient(alpha)
        k_vectors = jax.lax.stop_gradient(k_vectors)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The component direct-output flag\(s\).*",
            category=DeprecationWarning,
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
            compute_charge_gradients=need_charge_gradients,
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
            compute_charge_gradients=need_charge_gradients,
            compute_virial=compute_virial,
        )

    slab_result = None
    if slab_correction:
        if compute_forces or need_charge_gradients or compute_virial:
            slab_result = _compute_slab_correction(
                positions,
                charges,
                cell,
                pbc,
                batch_idx=batch_idx,
                compute_forces=compute_forces,
                compute_charge_gradients=need_charge_gradients,
                compute_virial=compute_virial,
            )
        else:
            slab_result = _slab_correction_energy_autodiff(
                positions,
                charges,
                cell,
                pbc,
                batch_idx=batch_idx,
            )

    # Sum contributions
    component_tuples = [
        real_result if isinstance(real_result, tuple) else (real_result,),
        recip_result if isinstance(recip_result, tuple) else (recip_result,),
    ]
    if slab_result is not None:
        component_tuples.append(
            slab_result if isinstance(slab_result, tuple) else (slab_result,)
        )

    def _sum_component(tuple_index: int) -> jax.Array:
        total = component_tuples[0][tuple_index]
        for component in component_tuples[1:]:
            total = total + component[tuple_index]
        return total

    tuple_index = 0

    total_energies = _sum_component(tuple_index)
    tuple_index += 1
    total_charge_grads = None
    results: tuple[jax.Array, ...] = (total_energies,)

    if compute_forces:
        total_forces = _sum_component(tuple_index)
        results += (total_forces,)
        tuple_index += 1

    if need_charge_gradients:
        total_charge_grads = _sum_component(tuple_index)
        tuple_index += 1
        if compute_charge_gradients:
            results += (total_charge_grads,)

    if compute_virial:
        total_virial = _sum_component(tuple_index)
        results += (total_virial,)

    if hybrid_forces and total_charge_grads is not None:
        bidx_for_inject = (
            batch_idx
            if batch_idx is not None
            else jnp.zeros(positions.shape[0], dtype=jnp.int32)
        )
        total_energies = _inject_charge_grad(
            total_energies,
            charges_orig,
            total_charge_grads,
            batch_idx is not None,
            bidx_for_inject,
            num_systems,
        )
        results = (total_energies, *results[1:])

    if len(results) == 1:
        return results[0]
    return results


def ewald_summation(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: float | jax.Array | None = None,
    k_vectors: jax.Array | None = None,
    k_cutoff: float | jax.Array | None = None,
    batch_idx: jax.Array | None = None,
    max_atoms_per_system: int | None = None,
    neighbor_list: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    neighbor_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    mask_value: int | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    accuracy: float = 1e-6,
    hybrid_forces: bool = False,
    pbc: jax.Array | None = None,
    slab_correction: bool = False,
    *,
    miller_bounds: tuple[int, int, int] | None = None,
) -> jax.Array | tuple[jax.Array, ...]:
    """Compute complete Ewald summation.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    charges : jax.Array, shape (N,)
        Atomic partial charges.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrices.
    alpha : float, jax.Array, or None, default=None
        Ewald splitting parameter. If ``None``, estimated automatically.
    k_vectors : jax.Array or None, default=None
        Reciprocal lattice vectors. Generated from ``cell`` when omitted.
    k_cutoff : float, jax.Array, or None, default=None
        K-space cutoff used when generating ``k_vectors``.
    miller_bounds : tuple[int, int, int] or None, default=None, keyword-only
        Static Miller-index bounds for JIT-compatible k-vector generation.
    batch_idx : jax.Array or None, default=None
        System index for each atom. When provided, atoms must be grouped by
        system: ``batch_idx`` must be contiguous, nondecreasing, and use system
        IDs ``0..B-1``.
    max_atoms_per_system : int or None, default=None
        Static batch shape control for reciprocal kernels under ``jax.jit``.
    neighbor_list, neighbor_ptr, neighbor_shifts : jax.Array or None
        CSR neighbor-list inputs for the real-space component.
    neighbor_matrix, neighbor_matrix_shifts : jax.Array or None
        Dense neighbor-matrix inputs for the real-space component.
    mask_value : int or None, default=None
        Sentinel value for invalid neighbor-matrix entries.
    compute_forces : bool, default=False
        Deprecated direct-output flag. Compute energy and use JAX autodiff for
        differentiable forces.
    compute_charge_gradients : bool, default=False
        Deprecated direct-output flag. Compute energy and use JAX autodiff for
        ``dE/dq_i``.
    compute_virial : bool, default=False
        Deprecated direct-output flag for the virial tensor.
    accuracy : float, default=1e-6
        Target accuracy for automatic parameter estimation.
    hybrid_forces : bool, default=False
        Deprecated direct-output flag retained for transition compatibility.
    pbc : jax.Array, optional
        Per-system periodic boundary conditions for slab correction.
    slab_correction : bool, default=False
        If True, add the Yeh-Berkowitz/Ballenegger slab correction.

    Returns
    -------
    jax.Array or tuple[jax.Array, ...]
        Per-atom energy, plus deprecated direct outputs when requested.
    """
    if compute_forces or compute_virial or compute_charge_gradients or hybrid_forces:
        warnings.warn(
            _direct_output_deprecation_msg("ewald_summation"),
            DeprecationWarning,
            stacklevel=2,
        )

    if slab_correction and not (
        compute_forces or compute_charge_gradients or compute_virial or hybrid_forces
    ):
        generated_k_vectors = k_vectors is None
        cell_3d, num_systems = _prepare_cell(cell)
        pbc_prepared = _prepare_pbc_for_slab(pbc, num_systems)
        alpha_resolved, k_vectors_resolved = _resolve_ewald_summation_parameters(
            positions=positions,
            charges=charges,
            cell=cell_3d,
            alpha=alpha,
            k_vectors=k_vectors,
            k_cutoff=k_cutoff,
            miller_bounds=miller_bounds,
            batch_idx=batch_idx,
            accuracy=accuracy,
        )
        dtype = _normalize_dtype(positions.dtype)
        alpha_arr = _prepare_alpha_array(alpha_resolved, cell_3d.shape[0], dtype=dtype)
        if mask_value is None:
            mask_value = positions.shape[0]
        base_energy = _ewald_summation_energy_jvp(
            positions,
            charges,
            cell_3d,
            alpha_arr,
            k_vectors_resolved,
            batch_idx,
            neighbor_list,
            neighbor_ptr,
            neighbor_shifts,
            neighbor_matrix,
            neighbor_matrix_shifts,
            generated_k_vectors,
            max_atoms_per_system,
            mask_value,
        )
        slab_energy = _slab_correction_energy_autodiff(
            positions,
            charges,
            cell_3d,
            pbc_prepared,
            batch_idx=batch_idx,
        )
        return base_energy + slab_energy

    if compute_forces or compute_charge_gradients or compute_virial or hybrid_forces:
        return _ewald_summation_impl(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_vectors=k_vectors,
            k_cutoff=k_cutoff,
            miller_bounds=miller_bounds,
            batch_idx=batch_idx,
            max_atoms_per_system=max_atoms_per_system,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=mask_value,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
            accuracy=accuracy,
            hybrid_forces=hybrid_forces,
            pbc=pbc,
            slab_correction=slab_correction,
        )

    generated_k_vectors = k_vectors is None
    alpha_resolved, k_vectors_resolved = _resolve_ewald_summation_parameters(
        positions=positions,
        charges=charges,
        cell=cell,
        alpha=alpha,
        k_vectors=k_vectors,
        k_cutoff=k_cutoff,
        miller_bounds=miller_bounds,
        batch_idx=batch_idx,
        accuracy=accuracy,
    )
    dtype = _normalize_dtype(positions.dtype)
    cell_3d = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
    alpha_arr = _prepare_alpha_array(alpha_resolved, cell_3d.shape[0], dtype=dtype)
    if mask_value is None:
        mask_value = positions.shape[0]

    return _ewald_summation_energy_jvp(
        positions,
        charges,
        cell,
        alpha_arr,
        k_vectors_resolved,
        batch_idx,
        neighbor_list,
        neighbor_ptr,
        neighbor_shifts,
        neighbor_matrix,
        neighbor_matrix_shifts,
        generated_k_vectors,
        max_atoms_per_system,
        mask_value,
    )
