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

"""JAX two-dimensional slab correction bindings."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.interpreters import ad as jax_ad

from nvalchemiops.interactions.electrostatics.slab_kernels import (
    _slab_correction_backward_atoms_kernel_overload,
    _slab_correction_backward_cell_kernel_overload,
    _slab_correction_double_backward_atoms_kernel_overload,
    _slab_correction_double_backward_cell_kernel_overload,
    _slab_correction_energy_charge_grad_kernel_overload,
    _slab_correction_energy_charge_grad_virial_kernel_overload,
    _slab_correction_energy_forces_charge_grad_kernel_overload,
    _slab_correction_energy_forces_charge_grad_virial_kernel_overload,
    _slab_correction_energy_forces_kernel_overload,
    _slab_correction_energy_forces_virial_kernel_overload,
    _slab_correction_energy_kernel_overload,
    _slab_correction_energy_virial_kernel_overload,
    _slab_directional_geometry_kernel_overload,
    _slab_directional_moments_kernel_overload,
    _slab_precompute_geometry_kernel_overload,
    _slab_reduce_moments_kernel_overload,
)
from nvalchemiops.jax.interactions.electrostatics._lazy_jax_kernels import (
    _make_jax_kernels,
)
from nvalchemiops.jax.interactions.electrostatics._utils import (
    _build_electrostatic_result,
    _normalize_dtype,
    _prepare_cell,
)

__all__ = ["compute_slab_correction"]


_jax_slab_reduce_moments = _make_jax_kernels(
    _slab_reduce_moments_kernel_overload,
    3,
    ["mz", "mz2", "qtotal"],
)

_jax_slab_precompute_geometry = _make_jax_kernels(
    _slab_precompute_geometry_kernel_overload,
    4,
    ["slab_axis", "slab_normal", "slab_volume", "slab_height_sq"],
)

_jax_slab_correction_energy = _make_jax_kernels(
    _slab_correction_energy_kernel_overload,
    1,
    ["energy_out"],
)

_jax_slab_correction_energy_forces = _make_jax_kernels(
    _slab_correction_energy_forces_kernel_overload,
    2,
    ["energy_out", "forces"],
)

_jax_slab_correction_energy_forces_virial = _make_jax_kernels(
    _slab_correction_energy_forces_virial_kernel_overload,
    3,
    ["energy_out", "forces", "virial"],
)

_jax_slab_correction_energy_forces_charge_grad = _make_jax_kernels(
    _slab_correction_energy_forces_charge_grad_kernel_overload,
    3,
    ["energy_out", "forces", "charge_grads"],
)

_jax_slab_correction_energy_forces_charge_grad_virial = _make_jax_kernels(
    _slab_correction_energy_forces_charge_grad_virial_kernel_overload,
    4,
    ["energy_out", "forces", "charge_grads", "virial"],
)

_jax_slab_correction_energy_charge_grad = _make_jax_kernels(
    _slab_correction_energy_charge_grad_kernel_overload,
    2,
    ["energy_out", "charge_grads"],
)

_jax_slab_correction_energy_charge_grad_virial = _make_jax_kernels(
    _slab_correction_energy_charge_grad_virial_kernel_overload,
    3,
    ["energy_out", "charge_grads", "virial"],
)

_jax_slab_correction_energy_virial = _make_jax_kernels(
    _slab_correction_energy_virial_kernel_overload,
    2,
    ["energy_out", "virial"],
)

_jax_slab_correction_backward_atoms = _make_jax_kernels(
    _slab_correction_backward_atoms_kernel_overload,
    3,
    ["grad_positions", "grad_charges", "grad_normal"],
)

_jax_slab_correction_backward_cell = _make_jax_kernels(
    _slab_correction_backward_cell_kernel_overload,
    1,
    ["grad_cell"],
)

_jax_slab_directional_geometry = _make_jax_kernels(
    _slab_directional_geometry_kernel_overload,
    3,
    ["dnormal", "dvolume", "dheight_sq"],
)

_jax_slab_directional_moments = _make_jax_kernels(
    _slab_directional_moments_kernel_overload,
    3,
    ["dmz", "dmz2", "dqtotal"],
)

_jax_slab_correction_double_backward_atoms = _make_jax_kernels(
    _slab_correction_double_backward_atoms_kernel_overload,
    4,
    ["grad_positions", "grad_charges", "grad_normal", "h_grad_normal"],
)

_jax_slab_correction_double_backward_cell = _make_jax_kernels(
    _slab_correction_double_backward_cell_kernel_overload,
    1,
    ["grad_cell"],
)


def _prepare_pbc_for_slab(pbc: jax.Array | None, num_systems: int) -> jax.Array:
    """Normalize and validate slab pbc as ``(B, 3)``."""
    if pbc is None:
        raise ValueError(
            "slab_correction=True requires an explicit `pbc` argument. "
            "Use a boolean array with shape (3,) for a single system or "
            "(B, 3) for batched systems."
        )

    pbc = jnp.asarray(pbc)
    if pbc.dtype != jnp.bool_:
        raise ValueError(f"pbc must be a bool array, got dtype={pbc.dtype}")

    if pbc.ndim == 1:
        if pbc.shape != (3,):
            raise ValueError(f"pbc must have shape (3,) or (B, 3), got {pbc.shape}")
        if num_systems != 1:
            raise ValueError(
                "batched slab correction requires pbc with shape (B, 3); "
                "shape (3,) is only valid for single-system calls"
            )
        return pbc[jnp.newaxis, :]

    if pbc.ndim != 2 or pbc.shape[1] != 3:
        raise ValueError(f"pbc must have shape (3,) or (B, 3), got {pbc.shape}")

    if pbc.shape[0] != num_systems:
        raise ValueError(
            f"pbc has {pbc.shape[0]} rows but cell describes {num_systems} systems"
        )

    return pbc


def _slab_correction_energy_reference(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
    batch_idx: jax.Array | None = None,
) -> jax.Array:
    """Compute slab energies with JAX ops for reference checks."""
    dtype = _normalize_dtype(positions.dtype)
    positions_cast = positions.astype(dtype)
    charges_cast = charges.astype(dtype)
    cell_cast, num_systems = _prepare_cell(cell.astype(dtype))
    pbc_cast = _prepare_pbc_for_slab(pbc, num_systems)
    num_atoms = positions_cast.shape[0]
    if num_atoms == 0:
        return jnp.zeros((0,), dtype=jnp.float64)

    if batch_idx is None:
        batch_idx_i32 = jnp.zeros(num_atoms, dtype=jnp.int32)
    else:
        batch_idx_i32 = batch_idx.astype(jnp.int32)

    pos64 = positions_cast.astype(jnp.float64)
    q64 = charges_cast.astype(jnp.float64)
    cell64 = cell_cast.astype(jnp.float64)
    pbc_cast = pbc_cast.astype(jnp.bool_)

    axis_order_a = jnp.array([1, 2, 0], dtype=jnp.int32)
    axis_order_b = jnp.array([2, 0, 1], dtype=jnp.int32)
    periodic_a = cell64[:, axis_order_a, :]
    periodic_b = cell64[:, axis_order_b, :]
    normals = jnp.cross(periodic_a, periodic_b)
    normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
    volume = jnp.abs(jnp.linalg.det(cell64))
    height_sq = jnp.sum(cell64 * normals, axis=-1) ** 2
    slab_axis_mask = jnp.logical_and(
        ~pbc_cast,
        jnp.sum(~pbc_cast, axis=1, keepdims=True) == 1,
    ).astype(jnp.float64)

    normal_atoms = normals[batch_idx_i32]
    z_values = jnp.einsum("nd,nad->na", pos64, normal_atoms)
    charge_column = q64[:, jnp.newaxis]
    moments = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    projected_moment = moments.at[batch_idx_i32].add(charge_column * z_values)
    projected_second_moment = moments.at[batch_idx_i32].add(
        charge_column * z_values * z_values
    )
    total_charge = (
        jnp.zeros((num_systems,), dtype=jnp.float64).at[batch_idx_i32].add(q64)
    )

    projected_moment_atoms = projected_moment[batch_idx_i32]
    projected_second_moment_atoms = projected_second_moment[batch_idx_i32]
    total_charge_atoms = total_charge[batch_idx_i32, jnp.newaxis]
    volume_atoms = volume[batch_idx_i32, jnp.newaxis]
    height_sq_atoms = height_sq[batch_idx_i32]
    slab_axis_mask_atoms = slab_axis_mask[batch_idx_i32]
    bracket = (
        z_values * projected_moment_atoms
        - 0.5
        * (projected_second_moment_atoms + total_charge_atoms * z_values * z_values)
        - total_charge_atoms * height_sq_atoms / 12.0
    )
    axis_energies = (
        (2.0 * jnp.pi / volume_atoms) * charge_column * bracket * slab_axis_mask_atoms
    )
    return jnp.sum(axis_energies, axis=1)


def _is_symbolic_zero(tangent) -> bool:
    """Return whether a custom-JVP tangent is JAX's symbolic zero sentinel."""
    return (
        tangent is None
        or isinstance(tangent, jax_ad.Zero)
        or tangent.__class__.__name__ == "SymbolicZero"
    )


def _tangent_or_zeros(tangent, primal: jax.Array, dtype=None) -> jax.Array:
    """Materialize a custom-JVP tangent, replacing symbolic zeros with arrays."""
    out_dtype = primal.dtype if dtype is None else dtype
    if _is_symbolic_zero(tangent):
        return jnp.zeros(primal.shape, dtype=out_dtype)
    return tangent.astype(out_dtype)


def _precompute_slab_geometry(
    pbc: jax.Array,
    cell: jax.Array,
    dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return slab-axis geometry arrays consumed by atom-major kernels."""
    num_systems = cell.shape[0]
    slab_axis = jnp.zeros((num_systems,), dtype=jnp.int32)
    slab_normal = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    slab_volume = jnp.zeros((num_systems,), dtype=jnp.float64)
    slab_height_sq = jnp.zeros((num_systems,), dtype=jnp.float64)
    return _jax_slab_precompute_geometry[dtype](
        pbc,
        cell,
        slab_axis,
        slab_normal,
        slab_volume,
        slab_height_sq,
        launch_dims=(num_systems,),
    )


def _system_sum_from_atoms(
    values: jax.Array,
    batch_idx: jax.Array | None,
    num_systems: int,
) -> jax.Array:
    """Sum atom values by system."""
    if batch_idx is None:
        return values.sum(keepdims=True)
    return (
        jnp.zeros((num_systems,), dtype=values.dtype)
        .at[batch_idx.astype(jnp.int32)]
        .add(values)
    )


def _per_system_atom_counts(
    batch_idx: jax.Array | None,
    num_systems: int,
    num_atoms: int,
) -> jax.Array:
    """Return atom counts per system as float64."""
    if batch_idx is None:
        return jnp.full((num_systems,), float(num_atoms), dtype=jnp.float64)
    return (
        jnp.zeros((num_systems,), dtype=jnp.float64)
        .at[batch_idx.astype(jnp.int32)]
        .add(jnp.ones((num_atoms,), dtype=jnp.float64))
    )


def _distribute_system_values(
    system_values: jax.Array,
    batch_idx: jax.Array | None,
    num_atoms: int,
) -> jax.Array:
    """Distribute per-system values uniformly to atoms."""
    if batch_idx is None:
        if num_atoms == 0:
            return jnp.zeros((0,), dtype=system_values.dtype)
        return jnp.full(
            (num_atoms,), system_values[0] / num_atoms, dtype=system_values.dtype
        )
    bidx = batch_idx.astype(jnp.int32)
    counts = _per_system_atom_counts(batch_idx, system_values.shape[0], num_atoms)
    return (system_values / jnp.maximum(counts, 1.0))[bidx]


def _slab_correction_energy_kernel_value(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
    batch_idx: jax.Array | None = None,
) -> jax.Array:
    """Compute slab energies with the shared Warp FFI kernel."""
    dtype = _normalize_dtype(positions.dtype)
    positions_cast = positions.astype(dtype)
    charges_cast = charges.astype(dtype)
    cell_cast, num_systems = _prepare_cell(cell.astype(dtype))
    pbc_cast = _prepare_pbc_for_slab(pbc, num_systems)
    num_atoms = positions_cast.shape[0]
    if num_atoms == 0:
        return jnp.zeros((0,), dtype=jnp.float64)

    if batch_idx is None:
        batch_idx_i32 = jnp.zeros(num_atoms, dtype=jnp.int32)
    else:
        batch_idx_i32 = batch_idx.astype(jnp.int32)

    mz = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    mz2 = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    qtotal = jnp.zeros(num_systems, dtype=jnp.float64)
    mz, mz2, qtotal = _jax_slab_reduce_moments[dtype](
        positions_cast,
        charges_cast,
        batch_idx_i32,
        pbc_cast,
        cell_cast,
        mz,
        mz2,
        qtotal,
        launch_dims=(num_atoms,),
    )
    slab_axis, slab_normal, slab_volume, slab_height_sq = _precompute_slab_geometry(
        pbc_cast,
        cell_cast,
        dtype,
    )

    energy_in = jnp.zeros(num_atoms, dtype=jnp.float64)
    energy_out = jnp.zeros(num_atoms, dtype=jnp.float64)
    (energy_out,) = _jax_slab_correction_energy[dtype](
        positions_cast,
        charges_cast,
        batch_idx_i32,
        slab_axis,
        slab_normal,
        slab_volume,
        slab_height_sq,
        mz,
        mz2,
        qtotal,
        energy_in,
        energy_out,
        launch_dims=(num_atoms,),
    )
    return energy_out


def _slab_energy_derivative_values(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
    batch_idx: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute literal slab ``(dE/dR, dE/dq, dE/dcell)`` with Warp FFI kernels."""
    dtype = _normalize_dtype(positions.dtype)
    positions_cast = positions.astype(dtype)
    charges_cast = charges.astype(dtype)
    cell_cast, num_systems = _prepare_cell(cell.astype(dtype))
    pbc_cast = _prepare_pbc_for_slab(pbc, num_systems)
    num_atoms = positions_cast.shape[0]
    grad_positions = jnp.zeros((num_atoms, 3), dtype=dtype)
    grad_charges = jnp.zeros((num_atoms,), dtype=jnp.float64)
    grad_cell = jnp.zeros((num_systems, 3, 3), dtype=dtype)
    if num_atoms == 0:
        return grad_positions, grad_charges, grad_cell

    if batch_idx is None:
        batch_idx_i32 = jnp.zeros(num_atoms, dtype=jnp.int32)
    else:
        batch_idx_i32 = batch_idx.astype(jnp.int32)

    mz = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    mz2 = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    qtotal = jnp.zeros(num_systems, dtype=jnp.float64)
    mz, mz2, qtotal = _jax_slab_reduce_moments[dtype](
        positions_cast,
        charges_cast,
        batch_idx_i32,
        pbc_cast,
        cell_cast,
        mz,
        mz2,
        qtotal,
        launch_dims=(num_atoms,),
    )
    slab_axis, slab_normal, slab_volume, slab_height_sq = _precompute_slab_geometry(
        pbc_cast,
        cell_cast,
        dtype,
    )

    grad_system = jnp.ones((num_systems,), dtype=jnp.float64)
    grad_normal = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    grad_positions, grad_charges, grad_normal = _jax_slab_correction_backward_atoms[
        dtype
    ](
        positions_cast,
        charges_cast,
        batch_idx_i32,
        slab_axis,
        slab_normal,
        slab_volume,
        slab_height_sq,
        mz,
        mz2,
        qtotal,
        grad_system,
        grad_positions,
        grad_charges,
        grad_normal,
        launch_dims=(num_atoms,),
    )
    (grad_cell,) = _jax_slab_correction_backward_cell[dtype](
        pbc_cast,
        cell_cast,
        mz,
        mz2,
        qtotal,
        grad_system,
        grad_normal,
        grad_cell,
        launch_dims=(num_systems,),
    )
    return grad_positions, grad_charges, grad_cell


@jax.custom_jvp
def _slab_energy_derivatives(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
    batch_idx: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return slab first derivatives with a second-order-capable JVP."""
    dpos, dq, dcell = _slab_energy_derivative_values(
        positions, charges, cell, pbc, batch_idx
    )
    return (
        jax.lax.stop_gradient(dpos),
        jax.lax.stop_gradient(dq),
        jax.lax.stop_gradient(dcell),
    )


def _slab_energy_derivatives_jvp(
    primals: tuple[jax.Array | None, ...],
    tangents: tuple[jax.Array | None, ...],
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]
]:
    """JVP of slab first derivatives using a transposable explicit HVP."""
    positions, charges, cell, pbc, batch_idx = primals
    t_positions, t_charges, t_cell, _t_pbc, _t_batch_idx = tangents

    primal_out = _slab_energy_derivatives(positions, charges, cell, pbc, batch_idx)
    dtype = _normalize_dtype(positions.dtype)
    tpos = _tangent_or_zeros(t_positions, positions, dtype=dtype)
    tq = _tangent_or_zeros(t_charges, charges, dtype=charges.dtype)
    tcell = _tangent_or_zeros(t_cell, cell, dtype=cell.dtype)

    tangent_out = _slab_energy_hvp(
        tpos,
        tq,
        tcell,
        positions,
        charges,
        cell,
        pbc,
        batch_idx,
    )
    return primal_out, (
        tangent_out[0].astype(primal_out[0].dtype),
        tangent_out[1].astype(primal_out[1].dtype),
        tangent_out[2].astype(primal_out[2].dtype),
    )


def _slab_energy_hvp_raw(
    v_positions: jax.Array,
    v_charges: jax.Array,
    v_cell: jax.Array,
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
    batch_idx: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate slab Hessian-vector products from analytic Warp kernels."""
    dtype = _normalize_dtype(positions.dtype)
    positions_cast = positions.astype(dtype)
    charges_cast = charges.astype(dtype)
    cell_cast, num_systems = _prepare_cell(cell.astype(dtype))
    h_positions = v_positions.astype(dtype)
    h_charges = v_charges.astype(jnp.float64)
    h_cell, _ = _prepare_cell(v_cell.astype(dtype))
    pbc_cast = _prepare_pbc_for_slab(pbc, num_systems)
    num_atoms = positions_cast.shape[0]
    grad_positions = jnp.zeros((num_atoms, 3), dtype=dtype)
    grad_charges = jnp.zeros((num_atoms,), dtype=jnp.float64)
    grad_cell = jnp.zeros((num_systems, 3, 3), dtype=dtype)
    if num_atoms == 0:
        return grad_positions, grad_charges, grad_cell

    if batch_idx is None:
        batch_idx_i32 = jnp.zeros(num_atoms, dtype=jnp.int32)
    else:
        batch_idx_i32 = batch_idx.astype(jnp.int32)

    mz = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    mz2 = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    qtotal = jnp.zeros(num_systems, dtype=jnp.float64)
    mz, mz2, qtotal = _jax_slab_reduce_moments[dtype](
        positions_cast,
        charges_cast,
        batch_idx_i32,
        pbc_cast,
        cell_cast,
        mz,
        mz2,
        qtotal,
        launch_dims=(num_atoms,),
    )
    slab_axis, slab_normal, slab_volume, slab_height_sq = _precompute_slab_geometry(
        pbc_cast,
        cell_cast,
        dtype,
    )

    dmz = jnp.zeros_like(mz)
    dmz2 = jnp.zeros_like(mz2)
    dqtotal = jnp.zeros_like(qtotal)
    dnormal = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    dvolume = jnp.zeros((num_systems,), dtype=jnp.float64)
    dheight_sq = jnp.zeros_like(dvolume)
    grad_normal = jnp.zeros_like(dnormal)
    h_grad_normal = jnp.zeros_like(dnormal)
    grad_system = jnp.ones((num_systems,), dtype=jnp.float64)

    dnormal, dvolume, dheight_sq = _jax_slab_directional_geometry[dtype](
        pbc_cast,
        cell_cast,
        h_cell,
        dnormal,
        dvolume,
        dheight_sq,
        launch_dims=(num_systems,),
    )

    dmz, dmz2, dqtotal = _jax_slab_directional_moments[dtype](
        positions_cast,
        charges_cast,
        h_positions,
        h_charges,
        batch_idx_i32,
        slab_axis,
        slab_normal,
        dnormal,
        dmz,
        dmz2,
        dqtotal,
        launch_dims=(num_atoms,),
    )
    grad_positions, grad_charges, grad_normal, h_grad_normal = (
        _jax_slab_correction_double_backward_atoms[dtype](
            positions_cast,
            charges_cast,
            h_positions,
            h_charges,
            batch_idx_i32,
            slab_axis,
            slab_normal,
            slab_volume,
            slab_height_sq,
            mz,
            mz2,
            qtotal,
            dmz,
            dmz2,
            dqtotal,
            dnormal,
            dvolume,
            dheight_sq,
            grad_system,
            grad_positions,
            grad_charges,
            grad_normal,
            h_grad_normal,
            launch_dims=(num_atoms,),
        )
    )
    (grad_cell,) = _jax_slab_correction_double_backward_cell[dtype](
        pbc_cast,
        cell_cast,
        h_cell,
        mz,
        mz2,
        qtotal,
        dmz,
        dmz2,
        dqtotal,
        grad_system,
        grad_normal,
        h_grad_normal,
        grad_cell,
        launch_dims=(num_systems,),
    )
    return grad_positions, grad_charges, grad_cell


def _slab_energy_hvp(
    v_positions: jax.Array,
    v_charges: jax.Array,
    v_cell: jax.Array,
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
    batch_idx: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Linear slab HVP wrapper with an explicit transpose rule."""

    # The HVP is a symmetric linear map in (v_positions, v_charges, v_cell);
    # custom_vjp supplies its transpose (== itself) so reverse-mode over this
    # JVP yields the Hessian.
    @jax.custom_vjp
    def _linear_hvp(lin_positions, lin_charges, lin_cell):
        return _slab_energy_hvp_raw(
            lin_positions,
            lin_charges,
            lin_cell,
            positions,
            charges,
            cell,
            pbc,
            batch_idx,
        )

    def _linear_hvp_fwd(lin_positions, lin_charges, lin_cell):
        return _linear_hvp(lin_positions, lin_charges, lin_cell), None

    def _linear_hvp_bwd(_res, ct_out):
        ct_positions, ct_charges, ct_cell = ct_out
        return _linear_hvp(
            _tangent_or_zeros(ct_positions, positions, dtype=positions.dtype),
            _tangent_or_zeros(ct_charges, charges, dtype=charges.dtype),
            _tangent_or_zeros(ct_cell, cell, dtype=cell.dtype),
        )

    _linear_hvp.defvjp(_linear_hvp_fwd, _linear_hvp_bwd)

    return _linear_hvp(v_positions, v_charges, v_cell)


_slab_energy_derivatives.defjvp(_slab_energy_derivatives_jvp, symbolic_zeros=True)


@jax.custom_jvp
def _slab_correction_energy_jvp(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
    batch_idx: jax.Array | None,
) -> jax.Array:
    """Energy-only slab correction wrapper with custom derivatives."""
    return jax.lax.stop_gradient(
        _slab_correction_energy_kernel_value(positions, charges, cell, pbc, batch_idx)
    )


def _slab_correction_energy_jvp_rule(
    primals: tuple[jax.Array | None, ...],
    tangents: tuple[jax.Array | None, ...],
) -> tuple[jax.Array, jax.Array]:
    """JVP rule for per-atom slab energies."""
    positions, charges, cell, pbc, batch_idx = primals
    t_positions, t_charges, t_cell, _t_pbc, _t_batch_idx = tangents

    primal_out = _slab_correction_energy_jvp(positions, charges, cell, pbc, batch_idx)
    dtype = _normalize_dtype(positions.dtype)
    tpos = _tangent_or_zeros(t_positions, positions, dtype=dtype)
    tq = _tangent_or_zeros(t_charges, charges, dtype=charges.dtype)
    tcell = _tangent_or_zeros(t_cell, cell, dtype=cell.dtype)
    _reference_out, tangent_out = jax.jvp(
        lambda p, q, c: _slab_correction_energy_reference(p, q, c, pbc, batch_idx),
        (positions, charges.astype(jnp.float64), cell),
        (tpos, tq.astype(jnp.float64), tcell),
    )
    return primal_out, tangent_out.astype(primal_out.dtype)


_slab_correction_energy_jvp.defjvp(
    _slab_correction_energy_jvp_rule,
    symbolic_zeros=True,
)


def _slab_correction_energy_autodiff(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
    batch_idx: jax.Array | None = None,
) -> jax.Array:
    """Compute slab energies with explicit first- and second-derivative routing."""
    return _slab_correction_energy_jvp(positions, charges, cell, pbc, batch_idx)


def compute_slab_correction(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
    batch_idx: jax.Array | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
) -> jax.Array | tuple[jax.Array, ...]:
    """Yeh-Berkowitz/Ballenegger slab correction for 2D periodic systems.

    Returns the standalone slab correction contribution for JAX electrostatics
    APIs. The caller can add the returned energy, force, charge-gradient, and
    virial terms to 3D-periodic Ewald or PME component outputs. Energy-only
    calls use explicit Warp-backed derivative paths; direct-output flags remain
    forward compatibility paths.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    charges : jax.Array, shape (N,)
        Atomic charges.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrices.
    pbc : jax.Array, shape (3,) or (B, 3), dtype=bool
        Per-system periodic boundary conditions. True marks periodic directions
        and False marks the non-periodic slab direction. Systems whose pbc row
        is not slab-like contribute zero. A shape (3,) array is accepted only
        for single-system calls.
    batch_idx : jax.Array, shape (N,), dtype=int32, optional
        System index for each atom. Defaults to all zeros for a single system.
        When provided, atoms must be grouped by system: ``batch_idx`` must be
        contiguous, nondecreasing, and use system IDs ``0..B-1``.
    compute_forces : bool, default=False
        If True, return per-atom slab forces.
    compute_charge_gradients : bool, default=False
        If True, return per-atom slab charge gradients dE_slab/dq_i.
    compute_virial : bool, default=False
        If True, return per-system slab virial tensors.

    Returns
    -------
    energies : jax.Array, shape (N,)
        Per-atom slab correction energy.
    forces : jax.Array, shape (N, 3), optional
        Per-atom slab force.
    charge_gradients : jax.Array, shape (N,), optional
        Per-atom slab charge gradient.
    virial : jax.Array, shape (B, 3, 3), optional
        Per-system slab virial tensor.
    """
    dtype = _normalize_dtype(positions.dtype)
    positions_cast = positions.astype(dtype)
    charges_cast = charges.astype(dtype)
    cell_cast, num_systems = _prepare_cell(cell.astype(dtype))
    pbc_cast = _prepare_pbc_for_slab(pbc, num_systems)
    num_atoms = positions_cast.shape[0]

    if batch_idx is None:
        batch_idx_i32 = jnp.zeros(num_atoms, dtype=jnp.int32)
    else:
        batch_idx_i32 = batch_idx.astype(jnp.int32)

    if num_atoms == 0:
        return _build_electrostatic_result(
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.zeros((0, 3), dtype=dtype) if compute_forces else None,
            (jnp.zeros((0,), dtype=jnp.float64) if compute_charge_gradients else None),
            (jnp.zeros((num_systems, 3, 3), dtype=dtype) if compute_virial else None),
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    if not (compute_forces or compute_charge_gradients or compute_virial):
        return _slab_correction_energy_autodiff(
            positions_cast,
            charges_cast,
            cell_cast,
            pbc_cast,
            batch_idx_i32,
        )

    mz = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    mz2 = jnp.zeros((num_systems, 3), dtype=jnp.float64)
    qtotal = jnp.zeros(num_systems, dtype=jnp.float64)
    mz, mz2, qtotal = _jax_slab_reduce_moments[dtype](
        positions_cast,
        charges_cast,
        batch_idx_i32,
        pbc_cast,
        cell_cast,
        mz,
        mz2,
        qtotal,
        launch_dims=(num_atoms,),
    )
    slab_axis, slab_normal, slab_volume, slab_height_sq = _precompute_slab_geometry(
        pbc_cast,
        cell_cast,
        dtype,
    )

    energy_in = jnp.zeros(num_atoms, dtype=jnp.float64)
    energy_out = jnp.zeros(num_atoms, dtype=jnp.float64)

    if compute_charge_gradients and compute_forces and compute_virial:
        forces = jnp.zeros((num_atoms, 3), dtype=dtype)
        charge_grads = jnp.zeros(num_atoms, dtype=jnp.float64)
        virial = jnp.zeros((num_systems, 3, 3), dtype=dtype)
        energy_out, forces, charge_grads, virial = (
            _jax_slab_correction_energy_forces_charge_grad_virial[dtype](
                positions_cast,
                charges_cast,
                batch_idx_i32,
                slab_axis,
                slab_normal,
                slab_volume,
                slab_height_sq,
                mz,
                mz2,
                qtotal,
                energy_in,
                energy_out,
                forces,
                charge_grads,
                virial,
                launch_dims=(num_atoms,),
            )
        )
        return _build_electrostatic_result(
            energy_out,
            forces,
            charge_grads,
            virial,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    if compute_charge_gradients and compute_forces:
        forces = jnp.zeros((num_atoms, 3), dtype=dtype)
        charge_grads = jnp.zeros(num_atoms, dtype=jnp.float64)
        energy_out, forces, charge_grads = (
            _jax_slab_correction_energy_forces_charge_grad[dtype](
                positions_cast,
                charges_cast,
                batch_idx_i32,
                slab_axis,
                slab_normal,
                slab_volume,
                slab_height_sq,
                mz,
                mz2,
                qtotal,
                energy_in,
                energy_out,
                forces,
                charge_grads,
                launch_dims=(num_atoms,),
            )
        )
        return _build_electrostatic_result(
            energy_out,
            forces,
            charge_grads,
            None,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    if compute_charge_gradients and compute_virial:
        charge_grads = jnp.zeros(num_atoms, dtype=jnp.float64)
        virial = jnp.zeros((num_systems, 3, 3), dtype=dtype)
        energy_out, charge_grads, virial = (
            _jax_slab_correction_energy_charge_grad_virial[dtype](
                positions_cast,
                charges_cast,
                batch_idx_i32,
                slab_axis,
                slab_normal,
                slab_volume,
                slab_height_sq,
                mz,
                mz2,
                qtotal,
                energy_in,
                energy_out,
                charge_grads,
                virial,
                launch_dims=(num_atoms,),
            )
        )
        return _build_electrostatic_result(
            energy_out,
            None,
            charge_grads,
            virial,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    if compute_charge_gradients:
        charge_grads = jnp.zeros(num_atoms, dtype=jnp.float64)
        energy_out, charge_grads = _jax_slab_correction_energy_charge_grad[dtype](
            positions_cast,
            charges_cast,
            batch_idx_i32,
            slab_axis,
            slab_normal,
            slab_volume,
            slab_height_sq,
            mz,
            mz2,
            qtotal,
            energy_in,
            energy_out,
            charge_grads,
            launch_dims=(num_atoms,),
        )
        return _build_electrostatic_result(
            energy_out,
            None,
            charge_grads,
            None,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    if compute_forces and compute_virial:
        forces = jnp.zeros((num_atoms, 3), dtype=dtype)
        virial = jnp.zeros((num_systems, 3, 3), dtype=dtype)
        energy_out, forces, virial = _jax_slab_correction_energy_forces_virial[dtype](
            positions_cast,
            charges_cast,
            batch_idx_i32,
            slab_axis,
            slab_normal,
            slab_volume,
            slab_height_sq,
            mz,
            mz2,
            qtotal,
            energy_in,
            energy_out,
            forces,
            virial,
            launch_dims=(num_atoms,),
        )
        return _build_electrostatic_result(
            energy_out,
            forces,
            None,
            virial,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    if compute_forces:
        forces = jnp.zeros((num_atoms, 3), dtype=dtype)
        energy_out, forces = _jax_slab_correction_energy_forces[dtype](
            positions_cast,
            charges_cast,
            batch_idx_i32,
            slab_axis,
            slab_normal,
            slab_volume,
            slab_height_sq,
            mz,
            mz2,
            qtotal,
            energy_in,
            energy_out,
            forces,
            launch_dims=(num_atoms,),
        )
        return _build_electrostatic_result(
            energy_out,
            forces,
            None,
            None,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    if compute_virial:
        virial = jnp.zeros((num_systems, 3, 3), dtype=dtype)
        energy_out, virial = _jax_slab_correction_energy_virial[dtype](
            positions_cast,
            charges_cast,
            batch_idx_i32,
            slab_axis,
            slab_normal,
            slab_volume,
            slab_height_sq,
            mz,
            mz2,
            qtotal,
            energy_in,
            energy_out,
            virial,
            launch_dims=(num_atoms,),
        )
        return _build_electrostatic_result(
            energy_out,
            None,
            None,
            virial,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    return energy_out
