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

r"""JAX bindings for dispersion PME (LJ-PME).

Mirrors ``nvalchemiops.jax.interactions.electrostatics.pme``: the reciprocal
mesh pipeline (spline spread/gather, JAX FFT, B-spline deconvolution,
k-vectors) is reused; only the per-k convolution factor and the energy
corrections differ. See ``interactions/dispersion/lj_pme_kernels.py``.

``alpha`` is the dispersion splitting parameter :math:`\beta` (no ``/√2``).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from nvalchemiops.interactions.dispersion.lj_pme_kernels import (
    _batch_lj_pme_convolve_kernel_overload,
    _batch_lj_pme_energy_corrections_kernel_overload,
    _lj_pme_convolve_kernel_overload,
    _lj_pme_energy_corrections_kernel_overload,
)
from nvalchemiops.jax.interactions.dispersion.lj_dispersion import (
    lj_dispersion_energy,
    lj_dispersion_energy_forces,
    sigma_epsilon_to_dispersion_charge,
)
from nvalchemiops.jax.interactions.dispersion.parameters import (
    DispersionPMEParameters,
    estimate_dispersion_pme_parameters,
)
from nvalchemiops.jax.interactions.electrostatics._lazy_jax_kernels import (
    make_jax_kernels as _make_jax_kernels,
)
from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
    generate_k_vectors_pme,
)
from nvalchemiops.jax.interactions.electrostatics.parameters import (
    mesh_spacing_to_dimensions,
)
from nvalchemiops.jax.interactions.electrostatics.pme import compute_bspline_moduli_1d
from nvalchemiops.jax.spline import (
    spline_gather,
    spline_gather_with_force,
    spline_spread,
)

__all__ = [
    "dispersion_reciprocal_space",
    "dispersion_pme",
    "DispersionPMEParameters",
    "estimate_dispersion_pme_parameters",
]

SQRT_PI = math.sqrt(math.pi)

_jax_lj_convolve = _make_jax_kernels(
    _lj_pme_convolve_kernel_overload, num_outputs=1, in_out_argnames=["convolved_mesh"]
)
_jax_batch_lj_convolve = _make_jax_kernels(
    _batch_lj_pme_convolve_kernel_overload,
    num_outputs=1,
    in_out_argnames=["convolved_mesh"],
)
_jax_lj_corrections = _make_jax_kernels(
    _lj_pme_energy_corrections_kernel_overload,
    num_outputs=1,
    in_out_argnames=["corrected_energies"],
)
_jax_batch_lj_corrections = _make_jax_kernels(
    _batch_lj_pme_energy_corrections_kernel_overload,
    num_outputs=1,
    in_out_argnames=["corrected_energies"],
)


def _normalize_dtype(dtype):
    return jnp.float32 if jnp.dtype(dtype) == jnp.float32 else jnp.float64


def _lj_pme_fused_convolve(
    mesh_fft, k_squared, moduli_x, moduli_y, moduli_z, alpha, volume, is_batch
):
    """Dispersion convolve via the warp kernel (single launch)."""
    real_dtype = jnp.float32 if mesh_fft.dtype == jnp.complex64 else jnp.float64
    complex_dtype = mesh_fft.dtype
    input_dtype = _normalize_dtype(real_dtype)

    squeeze_output = False
    if is_batch and k_squared.ndim == 3:
        k_squared = k_squared[jnp.newaxis, ...]
    if is_batch and mesh_fft.ndim == 3:
        mesh_fft = mesh_fft[jnp.newaxis, ...]
        squeeze_output = True

    mesh_fft_real = mesh_fft.view(real_dtype).reshape(*mesh_fft.shape, 2)
    alpha = alpha.astype(real_dtype)
    volume = volume.astype(real_dtype)
    if alpha.ndim == 0:
        alpha = alpha.reshape(1)
    if volume.ndim == 0:
        volume = volume.reshape(1)
    moduli_x = moduli_x.astype(real_dtype)
    moduli_y = moduli_y.astype(real_dtype)
    moduli_z = moduli_z.astype(real_dtype)
    k_squared = k_squared.astype(real_dtype)
    convolved_real = jnp.zeros_like(mesh_fft_real)

    kernel = (_jax_batch_lj_convolve if is_batch else _jax_lj_convolve)[input_dtype]
    (convolved_real,) = kernel(
        mesh_fft_real,
        k_squared,
        moduli_x,
        moduli_y,
        moduli_z,
        alpha,
        volume,
        convolved_real,
        launch_dims=mesh_fft.shape,
    )
    convolved = convolved_real.reshape(*mesh_fft.shape[:-1], -1).view(complex_dtype)
    if squeeze_output:
        convolved = convolved.squeeze(0)
    return convolved


def _lj_pme_corrections(raw_energies, b, alpha, batch_idx):
    """Dispersion energy corrections: E_i = b_i φ_i + (α^6/12) b_i²."""
    input_dtype = _normalize_dtype(raw_energies.dtype)
    num_atoms = raw_energies.shape[0]
    if alpha.ndim == 0:
        alpha = alpha.reshape(1)
    alpha = alpha.astype(input_dtype)
    corrected = jnp.zeros(num_atoms, dtype=input_dtype)
    if batch_idx is None:
        kernel = _jax_lj_corrections[input_dtype]
        (out,) = kernel(
            raw_energies.astype(input_dtype),
            b.astype(input_dtype),
            alpha,
            corrected,
            launch_dims=(num_atoms,),
        )
    else:
        kernel = _jax_batch_lj_corrections[input_dtype]
        (out,) = kernel(
            raw_energies.astype(input_dtype),
            b.astype(input_dtype),
            batch_idx.astype(jnp.int32),
            alpha,
            corrected,
            launch_dims=(num_atoms,),
        )
    return out


def _dispersion_reciprocal_virial(
    mesh_fft_raw, convolved_mesh, k_vectors, k_squared, alpha, mesh_dimensions, is_batch
):
    """JAX dispersion reciprocal virial (mirrors the torch implementation)."""
    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
    acc_dtype = _normalize_dtype(k_squared.dtype)
    complex_dtype = jnp.complex64 if acc_dtype == jnp.float32 else jnp.complex128

    energy_density = (
        mesh_fft_raw.astype(complex_dtype)
        * jnp.conj(convolved_mesh.astype(complex_dtype))
    ).real
    weight = jnp.full_like(energy_density, 2.0)
    weight = weight.at[..., 0].set(1.0)
    if mesh_nz % 2 == 0:
        weight = weight.at[..., -1].set(1.0)
    weighted_energy = weight * energy_density

    k_sq_acc = k_squared.astype(acc_dtype)
    alpha_acc = alpha.astype(acc_dtype)
    if is_batch and k_sq_acc.ndim == 3:
        k_sq_acc = jnp.expand_dims(k_sq_acc, axis=0)
    if is_batch and alpha_acc.ndim == 1:
        alpha_view = alpha_acc.reshape(-1, 1, 1, 1)
    else:
        alpha_view = alpha_acc.reshape(-1) if alpha_acc.ndim == 0 else alpha_acc

    b2 = k_sq_acc * (0.25 / (alpha_view**2))
    b = jnp.sqrt(jnp.maximum(b2, 0.0))
    e = jnp.exp(-b2)
    erfcb = jax.scipy.special.erfc(b)
    f = (1.0 - 2.0 * b2) * e + 2.0 * SQRT_PI * b2 * b * erfcb
    g1 = 3.0 * (SQRT_PI * b * erfcb - e)
    f_safe = jnp.where(jnp.abs(f) > 1e-300, f, jnp.ones_like(f))
    k_factor = -g1 / (2.0 * (alpha_view**2) * f_safe)

    k_vecs_acc = k_vectors.astype(acc_dtype)
    if is_batch and k_vecs_acc.ndim == 4:
        k_vecs_acc = jnp.expand_dims(k_vecs_acc, axis=0)

    masked_energy_kf = weighted_energy * k_factor
    sum_dims = (1, 2, 3) if is_batch else (0, 1, 2)
    trace_term = weighted_energy.sum(axis=sum_dims)

    kx = k_vecs_acc[..., 0]
    ky = k_vecs_acc[..., 1]
    kz = k_vecs_acc[..., 2]
    xx = (kx * kx * masked_energy_kf).sum(axis=sum_dims)
    yy = (ky * ky * masked_energy_kf).sum(axis=sum_dims)
    zz = (kz * kz * masked_energy_kf).sum(axis=sum_dims)
    xy = (kx * ky * masked_energy_kf).sum(axis=sum_dims)
    xz = (kx * kz * masked_energy_kf).sum(axis=sum_dims)
    yz = (ky * kz * masked_energy_kf).sum(axis=sum_dims)

    eye = jnp.eye(3, dtype=acc_dtype)
    if is_batch:
        kk = jnp.stack(
            [
                jnp.stack([xx, xy, xz], axis=-1),
                jnp.stack([xy, yy, yz], axis=-1),
                jnp.stack([xz, yz, zz], axis=-1),
            ],
            axis=-2,
        )
        virial = eye * trace_term[:, jnp.newaxis, jnp.newaxis] - kk
    else:
        kk = jnp.stack(
            [jnp.stack([xx, xy, xz]), jnp.stack([xy, yy, yz]), jnp.stack([xz, yz, zz])]
        )
        virial = (eye * trace_term - kk)[jnp.newaxis, :, :]
    return virial.astype(acc_dtype)


def dispersion_reciprocal_space(
    positions: jax.Array,
    sigma: jax.Array,
    epsilon: jax.Array,
    cell: jax.Array,
    alpha: float | jax.Array,
    mesh_dimensions: tuple[int, int, int] | None = None,
    mesh_spacing: float | None = None,
    spline_order: int = 4,
    batch_idx: jax.Array | None = None,
    k_vectors: jax.Array | None = None,
    k_squared: jax.Array | None = None,
    volume: jax.Array | None = None,
    cell_inv_t: jax.Array | None = None,
    moduli_x: jax.Array | None = None,
    moduli_y: jax.Array | None = None,
    moduli_z: jax.Array | None = None,
    compute_forces: bool = False,
    compute_virial: bool = False,
) -> jax.Array | tuple[jax.Array, ...]:
    r"""Reciprocal-space dispersion PME (JAX). See the torch binding."""
    input_dtype = _normalize_dtype(positions.dtype)
    is_batch = batch_idx is not None
    fft_dims = (1, 2, 3) if is_batch else (0, 1, 2)

    cell3 = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
    num_systems = cell3.shape[0]

    alpha_arr = jnp.asarray(alpha, dtype=input_dtype)
    if alpha_arr.ndim == 0:
        alpha_arr = jnp.full((num_systems,), float(alpha), dtype=input_dtype)

    if mesh_dimensions is None:
        if mesh_spacing is None:
            raise ValueError("Either mesh_dimensions or mesh_spacing must be provided")
        mesh_dimensions = mesh_spacing_to_dimensions(cell3, mesh_spacing)
    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions

    b = sigma_epsilon_to_dispersion_charge(sigma, epsilon).astype(input_dtype)
    positions = positions.astype(input_dtype)
    cell_c = cell.astype(input_dtype)

    if cell_inv_t is None:
        cell_inv = jnp.linalg.inv(cell3.astype(input_dtype))
        cell_inv_t = jnp.transpose(cell_inv, (0, 2, 1)).astype(input_dtype)
    else:
        cell_inv_t = cell_inv_t.astype(input_dtype)
        if cell_inv_t.ndim == 2:
            cell_inv_t = cell_inv_t[jnp.newaxis, :, :]
        cell_inv = jnp.transpose(cell_inv_t, (0, 2, 1))

    mesh_grid = spline_spread(
        positions,
        b,
        cell_c,
        mesh_dims=mesh_dimensions,
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t,
    )
    mesh_fft = jnp.fft.rfftn(mesh_grid, axes=fft_dims, norm="backward")

    if k_vectors is None or k_squared is None:
        reciprocal_cell = (2.0 * jnp.pi) * cell_inv
        k_vectors, k_squared = generate_k_vectors_pme(
            cell_c,
            mesh_dimensions,
            reciprocal_cell=reciprocal_cell,
        )

    if moduli_x is None or moduli_y is None or moduli_z is None:
        miller_x = jnp.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx).astype(input_dtype)
        miller_y = jnp.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny).astype(input_dtype)
        miller_z = jnp.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz).astype(input_dtype)
        moduli_x = compute_bspline_moduli_1d(miller_x, mesh_nx, spline_order)
        moduli_y = compute_bspline_moduli_1d(miller_y, mesh_ny, spline_order)
        moduli_z = compute_bspline_moduli_1d(miller_z, mesh_nz, spline_order)

    if volume is None:
        volume = jnp.abs(jnp.linalg.det(cell3.astype(input_dtype))).astype(input_dtype)
    else:
        volume = volume.astype(input_dtype)
        if volume.ndim == 0:
            volume = volume.reshape(1)

    complex_dtype = jnp.complex64 if input_dtype == jnp.float32 else jnp.complex128
    mesh_fft = mesh_fft.astype(complex_dtype)
    mesh_fft_raw = mesh_fft if compute_virial else None

    convolved_mesh = _lj_pme_fused_convolve(
        mesh_fft,
        k_squared.astype(input_dtype),
        moduli_x,
        moduli_y,
        moduli_z,
        alpha_arr,
        volume,
        is_batch,
    )

    virial = None
    if compute_virial:
        virial = _dispersion_reciprocal_virial(
            mesh_fft_raw,
            convolved_mesh,
            k_vectors,
            k_squared,
            alpha_arr,
            mesh_dimensions,
            is_batch,
        )

    potential_mesh = jnp.fft.irfftn(
        convolved_mesh, s=mesh_dimensions, axes=fft_dims, norm="forward"
    )

    if compute_forces:
        raw_energies, gathered_force = spline_gather_with_force(
            positions,
            b,
            potential_mesh,
            cell_c,
            spline_order=spline_order,
            batch_idx=batch_idx,
            cell_inv_t=cell_inv_t,
        )
    else:
        raw_energies = spline_gather(
            positions,
            potential_mesh,
            cell_c,
            spline_order=spline_order,
            batch_idx=batch_idx,
            cell_inv_t=cell_inv_t,
        )
        gathered_force = None

    energies = _lj_pme_corrections(raw_energies, b, alpha_arr, batch_idx)
    forces = 2.0 * gathered_force if compute_forces else None

    if compute_forces and compute_virial:
        return energies, forces, virial
    elif compute_forces:
        return energies, forces
    elif compute_virial:
        return energies, virial
    return energies


def dispersion_pme(
    positions: jax.Array,
    sigma: jax.Array,
    epsilon: jax.Array,
    cell: jax.Array,
    alpha: float | jax.Array | None = None,
    mesh_spacing: float | None = None,
    mesh_dimensions: tuple[int, int, int] | None = None,
    spline_order: int = 4,
    batch_idx: jax.Array | None = None,
    k_vectors: jax.Array | None = None,
    k_squared: jax.Array | None = None,
    cell_inv_t: jax.Array | None = None,
    volume: jax.Array | None = None,
    moduli_x: jax.Array | None = None,
    moduli_y: jax.Array | None = None,
    moduli_z: jax.Array | None = None,
    real_space_cutoff: float | None = None,
    neighbor_list: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    neighbor_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    fill_value: int | None = None,
    compute_forces: bool = False,
    accuracy: float = 1e-6,
) -> jax.Array | tuple[jax.Array, ...]:
    r"""Complete dispersion (LJ-PME) calculation: real + reciprocal (JAX).

    Mirrors the torch binding; ``alpha`` is the splitting parameter
    :math:`\beta` (auto-estimated when None). Virial is not returned by this
    convenience wrapper (use ``dispersion_reciprocal_space`` for the reciprocal
    virial).
    """
    cell3 = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]

    if alpha is None:
        params = estimate_dispersion_pme_parameters(
            positions, cell3, batch_idx, accuracy, real_space_cutoff
        )
        alpha = params.alpha
        if real_space_cutoff is None:
            real_space_cutoff = float(params.real_space_cutoff.max())
        if mesh_dimensions is None and mesh_spacing is None:
            mesh_dimensions = tuple(params.mesh_dimensions)

    alpha_arr = jnp.asarray(alpha, dtype=_normalize_dtype(positions.dtype))
    beta = float(jnp.reshape(alpha_arr, (-1,))[0])

    if real_space_cutoff is None:
        raise ValueError(
            "real_space_cutoff must be provided (or let alpha=None auto-estimate it)"
        )

    real_kwargs = dict(
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        fill_value=fill_value,
        batch_idx=batch_idx,
    )
    if compute_forces:
        e_real, f_real = lj_dispersion_energy_forces(
            positions, sigma, epsilon, cell, real_space_cutoff, beta, **real_kwargs
        )
    else:
        e_real = lj_dispersion_energy(
            positions, sigma, epsilon, cell, real_space_cutoff, beta, **real_kwargs
        )
        f_real = None

    rec = dispersion_reciprocal_space(
        positions,
        sigma,
        epsilon,
        cell,
        alpha_arr,
        mesh_dimensions=mesh_dimensions,
        mesh_spacing=mesh_spacing,
        spline_order=spline_order,
        batch_idx=batch_idx,
        k_vectors=k_vectors,
        k_squared=k_squared,
        volume=volume,
        cell_inv_t=cell_inv_t,
        moduli_x=moduli_x,
        moduli_y=moduli_y,
        moduli_z=moduli_z,
        compute_forces=compute_forces,
    )
    if compute_forces:
        e_rec, f_rec = rec
    else:
        e_rec, f_rec = rec, None

    energies = e_real.astype(positions.dtype) + e_rec.astype(positions.dtype)
    if compute_forces:
        return energies, f_real + f_rec
    return energies
