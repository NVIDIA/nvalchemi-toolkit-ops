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
Parameter Estimation for Ewald and PME Methods (JAX)
=====================================================

This module provides functions to automatically estimate optimal parameters
for Ewald summation and Particle Mesh Ewald (PME) calculations using JAX.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from nvalchemiops.interactions.electrostatics.parameter_estimation import (
    alpha_from_cutoff,
    find_optimal_pme_cutoff,
)

__all__ = [
    "EwaldParameters",
    "PMEParameters",
    "estimate_ewald_parameters",
    "estimate_pme_parameters",
    "estimate_pme_mesh_dimensions",
    "mesh_spacing_to_dimensions",
]


@dataclass
class EwaldParameters:
    """Container for Ewald summation parameters.

    All values are arrays of shape (B,), for
    single system calculations, the shape is (1,).

    Attributes
    ----------
    alpha : jax.Array, shape (B,)
        Ewald splitting parameter (inverse length units).
    real_space_cutoff : jax.Array, shape (B,)
        Real-space cutoff distance.
    reciprocal_space_cutoff : jax.Array, shape (B,)
        Reciprocal-space cutoff (:math:`|k|` in inverse length units).
    """

    alpha: jax.Array
    real_space_cutoff: jax.Array
    reciprocal_space_cutoff: jax.Array


@dataclass
class PMEParameters:
    """Container for PME parameters.

    Attributes
    ----------
    alpha : jax.Array, shape (B,)
        Ewald splitting parameter.
    mesh_dimensions : tuple[int, int, int], shape (3,)
        Mesh dimensions (nx, ny, nz).
    mesh_spacing : jax.Array, shape (B, 3)
        Actual mesh spacing in each direction.
    real_space_cutoff : jax.Array, shape (B,)
        Real-space cutoff distance.
    """

    alpha: jax.Array
    mesh_dimensions: tuple[int, int, int]
    mesh_spacing: jax.Array
    real_space_cutoff: jax.Array


def _count_atoms_per_system(
    positions: jax.Array, num_systems: int, batch_idx: jax.Array | None = None
) -> jax.Array:
    """Count number of atoms per system."""
    if batch_idx is None:
        return jnp.array([positions.shape[0]], dtype=jnp.int32)

    counts = jnp.zeros(num_systems, dtype=jnp.int32)
    ones = jnp.ones_like(batch_idx)
    return counts.at[batch_idx].add(ones)


def estimate_ewald_parameters(
    positions: jax.Array,
    cell: jax.Array,
    batch_idx: jax.Array | None = None,
    accuracy: float = 1e-6,
) -> EwaldParameters:
    """Estimate optimal Ewald summation parameters for a given accuracy.

    Uses the Kolafa-Perram formula to balance real-space and reciprocal-space
    contributions for optimal efficiency at the target accuracy.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrix.
    batch_idx : jax.Array, shape (N,), dtype=int32, optional
        System index for each atom. If None, single-system mode.
    accuracy : float, default=1e-6
        Target accuracy (relative error tolerance).

    Returns
    -------
    EwaldParameters
        Dataclass containing alpha, real_space_cutoff, reciprocal_space_cutoff
        as ``jax.Array`` objects.
    """
    if cell.ndim == 2:
        cell = cell[None, ...]
    num_systems = cell.shape[0]

    # Compute volume per system: (B,)
    volume = jnp.abs(jnp.linalg.det(cell))

    # Get number of atoms per system: (B,)
    num_atoms = _count_atoms_per_system(positions, num_systems, batch_idx).astype(
        positions.dtype
    )

    # Intermediate parameter eta: (B,)
    eta = (volume**2 / num_atoms) ** (1.0 / 6.0) / math.sqrt(2.0 * math.pi)

    # Error factor from log(accuracy)
    error_factor = math.sqrt(-2.0 * math.log(accuracy))

    # Real-space cutoff: (B,)
    real_space_cutoff = error_factor * eta

    # Reciprocal-space cutoff: (B,)
    reciprocal_space_cutoff = error_factor / eta

    # Splitting parameter alpha: (B,)
    alpha = 1.0 / (math.sqrt(2.0) * eta)

    return EwaldParameters(
        alpha=alpha,
        real_space_cutoff=real_space_cutoff,
        reciprocal_space_cutoff=reciprocal_space_cutoff,
    )


def estimate_pme_mesh_dimensions(
    cell: jax.Array,
    alpha: jax.Array,
    accuracy: float = 1e-6,
    mesh_safety_factor: float = 1.0,
) -> tuple[int, int, int]:
    """Estimate PME mesh dimensions for a given accuracy.

    The mesh size along each axis is chosen as

        K_i = ceil(mesh_safety_factor · 2 α L_i / (3 ε^{1/5}))

    rounded up to the next power of 2. The fifth-root scaling
    ``ε^{1/5}`` is the standard heuristic used by production PME
    codes; it grows the safety margin faster than ``√(-ln ε)`` as
    ``ε`` tightens, which is empirically necessary to cover both the
    Gaussian-decay truncation and the B-spline aliasing error at the
    accuracies typically requested (1e-3 to 1e-6) across a wide
    ``(α, L, spline_order)`` envelope.

    The canonical Essmann lower bound ``2 α L √(-ln ε) / π`` is the
    Gaussian-decay term only; it can under-allocate by 2-4× at low
    α (large rc), where the B-spline aliasing term dominates.

    Parameters
    ----------
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrix.
    alpha : jax.Array, shape (B,)
        Ewald splitting parameter.
    accuracy : float, default=1e-6
        Target relative accuracy.
    mesh_safety_factor : float, default=1.0
        Multiplier on the standard heuristic. ``1.0`` is the
        well-tested default that meets accuracy across the
        configurations covered by the convergence script. Raise for
        extra paranoia at tight accuracy. **Lower at your own risk:**
        values below 1.0 can fail the accuracy guarantee on
        low-α / large-L systems (verify with the convergence script
        before using).

    Returns
    -------
    tuple[int, int, int]
        Maximum mesh dimensions (nx, ny, nz) across all systems in batch.
    """
    if cell.ndim == 2:
        cell = cell[None, ...]

    cell_lengths = jnp.linalg.norm(cell, axis=2)  # (B, 3)

    # K = 2 α L / (3 ε^0.2), with optional safety multiplier + pow-2 snap.
    accuracy_factor = 3.0 * (accuracy**0.2)
    n = (
        mesh_safety_factor
        * 2.0
        * alpha[:, None]
        * cell_lengths
        / accuracy_factor
    )  # (B, 3)

    max_n = jnp.max(n, axis=0)  # (3,)
    mesh_dims = jnp.power(2, jnp.ceil(jnp.log2(max_n))).astype(jnp.int32)
    return (
        int(mesh_dims[0].item()),
        int(mesh_dims[1].item()),
        int(mesh_dims[2].item()),
    )


def estimate_pme_parameters(
    positions: jax.Array,
    cell: jax.Array,
    batch_idx: jax.Array | None = None,
    accuracy: float = 1e-6,
    real_space_cutoff: float | None = None,
    cost_ratio_pair_to_fft: float = 1.0,
    mesh_safety_factor: float = 1.0,
) -> PMEParameters:
    """Estimate optimal PME parameters for a given accuracy.

    Unlike pure Ewald, PME's reciprocal-space cost is FFT-dominated
    (``K^3 log(K)``) rather than k-sum-dominated (``K^3``). The
    cost-optimal real-space cutoff is therefore decoupled from the
    Kolafa-Perram balance used in ``estimate_ewald_parameters`` — it is
    found by a 1D minimization of the PME cost model (see
    ``nvalchemiops.interactions.electrostatics.parameter_estimation``).
    Callers who already know their preferred cutoff (e.g. tied to neighbor-
    list update frequency in MD) should pass it via ``real_space_cutoff``.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrix.
    batch_idx : jax.Array, shape (N,), dtype=int32, optional
        System index for each atom.
    accuracy : float, default=1e-6
        Target accuracy.
    real_space_cutoff : float, optional
        If provided, used as-is; ``α`` and mesh dimensions are then
        derived from it. If ``None`` (default), the cost-optimal cutoff
        is found via golden-section minimization on the PME cost model.
    cost_ratio_pair_to_fft : float, default=1.0
        Hardware-dependent weighting of FFT vs pair-operation cost in
        the cost model. The default ``1.0`` is a generic mid-ground.
        Empirical fits on real GPUs may motivate values in
        ``[0.001, 1.0]``. Ignored when ``real_space_cutoff`` is given.
    mesh_safety_factor : float, default=1.0
        Multiplier on the standard mesh-size heuristic
        ``K = 2 α L / (3 ε^{1/5})``. ``1.0`` matches the default
        behavior of established production PME implementations.
        Raise for extra safety at tight ε. Lower with care: values
        below 1.0 can fail the accuracy guarantee on low-α systems.

    Returns
    -------
    PMEParameters
        Dataclass containing alpha, mesh dimensions, spacing, and cutoffs.
        Tensor fields are ``jax.Array`` objects.
    """
    if cell.ndim == 2:
        cell = cell[None, ...]

    num_systems = cell.shape[0]
    volume = jnp.abs(jnp.linalg.det(cell))
    num_atoms = _count_atoms_per_system(positions, num_systems, batch_idx).astype(
        positions.dtype
    )
    cell_lengths = jnp.linalg.norm(cell, axis=2)  # (B, 3)

    # Choose real-space cutoff: caller-supplied, or cost-optimal from the
    # PME cost model. For batched inputs, optimize using the median system
    # properties — the resulting cutoff is shared across the batch.
    if real_space_cutoff is None:
        if num_systems == 1:
            n_repr = float(num_atoms[0])
            v_repr = float(volume[0])
            l_repr = tuple(cell_lengths[0].tolist())
        else:
            n_repr = float(jnp.median(num_atoms))
            v_repr = float(jnp.median(volume))
            l_repr = tuple(jnp.median(cell_lengths, axis=0).tolist())
        rc_value = find_optimal_pme_cutoff(
            num_atoms=n_repr,
            volume=v_repr,
            cell_lengths=l_repr,
            accuracy=accuracy,
            cost_ratio_pair_to_fft=cost_ratio_pair_to_fft,
            mesh_safety_factor=mesh_safety_factor,
        )
    else:
        rc_value = float(real_space_cutoff)

    # Derive alpha from the real-space accuracy constraint at the chosen rc.
    alpha_value = alpha_from_cutoff(rc_value, accuracy)

    alpha = jnp.full((num_systems,), alpha_value, dtype=positions.dtype)
    rc_array = jnp.full((num_systems,), rc_value, dtype=positions.dtype)

    # Estimate mesh dimensions at the chosen alpha.
    mesh_dims = estimate_pme_mesh_dimensions(
        cell, alpha, accuracy, mesh_safety_factor=mesh_safety_factor,
    )

    # Compute actual mesh spacing
    mesh_dims_tensor = jnp.array(mesh_dims, dtype=cell_lengths.dtype)
    mesh_spacing = cell_lengths / mesh_dims_tensor  # (B, 3)

    return PMEParameters(
        alpha=alpha,
        mesh_dimensions=mesh_dims,
        mesh_spacing=mesh_spacing,
        real_space_cutoff=rc_array,
    )


def mesh_spacing_to_dimensions(
    cell: jax.Array,
    mesh_spacing: float | jax.Array,
) -> tuple[int, int, int]:
    """Convert mesh spacing to mesh dimensions.

    Parameters
    ----------
    cell : jax.Array
        Unit cell matrix.
    mesh_spacing : float | jax.Array
        Target mesh spacing.

    Returns
    -------
    tuple[int, int, int]
        Mesh dimensions, rounded up to powers of 2.
    """
    if cell.ndim == 2:
        cell = cell[None, ...]

    cell_lengths = jnp.linalg.norm(cell, axis=2)  # (B, 3)

    if isinstance(mesh_spacing, (float, int)):
        mesh_dims = jnp.ceil(cell_lengths / mesh_spacing)
    elif isinstance(mesh_spacing, jax.Array):
        if mesh_spacing.ndim == 1:
            if mesh_spacing.shape[0] != cell.shape[0]:
                raise ValueError(
                    f"mesh_spacing shape {mesh_spacing.shape} incompatible with "
                    f"cell batch size {cell.shape[0]}"
                )
            mesh_dims = jnp.ceil(cell_lengths / mesh_spacing[:, None])
        else:
            if mesh_spacing.shape != cell_lengths.shape:
                raise ValueError(
                    f"mesh_spacing shape {mesh_spacing.shape} incompatible with "
                    f"cell_lengths shape {cell_lengths.shape}"
                )
            mesh_dims = jnp.ceil(cell_lengths / mesh_spacing)
    else:
        raise TypeError(
            f"mesh_spacing must be float or jax.Array, got {type(mesh_spacing)}"
        )

    mesh_dims = jnp.power(2, jnp.ceil(jnp.log2(mesh_dims))).astype(jnp.int32)

    max_mesh_dims = jnp.max(mesh_dims, axis=0)
    return (
        int(max_mesh_dims[0].item()),
        int(max_mesh_dims[1].item()),
        int(max_mesh_dims[2].item()),
    )
