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
import numpy as np

from nvalchemiops.interactions.electrostatics._pme_mesh import (
    _DEFAULT_PME_FFT_PADDING_FRACTION,
    _DEFAULT_PME_SPLINE_ORDER,
    _round_pme_mesh_dimensions,
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
    *,
    spline_order: int = _DEFAULT_PME_SPLINE_ORDER,
    fft_padding_fraction: float = _DEFAULT_PME_FFT_PADDING_FRACTION,
) -> tuple[int, int, int]:
    r"""Return accuracy-based FFT mesh dimensions with spline support.

    The mesh size along each axis is chosen as

    .. math::

        K_i = \lceil \text{mesh\_safety\_factor} \cdot 2 \alpha L_i / (3 \varepsilon^{1/5}) \rceil

    and rounded upward to a mesh with prime factors 2, 3, 5, and 7. For the
    default order five, snap to powers of two first, then smooth multiples
    of four, within the configured budget of extra total mesh points.
    Other spline orders retain upstream power-of-two accuracy sizing. Each
    dimension covers the continuous target and assignment support.
    ``accuracy`` controls the historical fifth-root sizing rule. The estimate
    retains its relative sizing convention; independent force checks determine
    the measured error for a particular charge distribution.

    Parameters
    ----------
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrix.
    alpha : jax.Array, shape (B,)
        Ewald splitting parameter.
    accuracy : float, default=1e-6
        Dimensionless input to the legacy mesh-size heuristic.
    mesh_safety_factor : float, default=1.0
        Caller-requested multiplier on the continuous size estimate. The
        default leaves the estimate unchanged.
    spline_order : int, default=5
        B-spline assignment order. The mesh covers its assignment support.

    fft_padding_fraction : float, default=0.25
        Maximum fraction of extra total mesh points for snapping to powers
        of two, then smooth multiples of four. Zero selects the smallest
        2/3/5/7-smooth mesh. Explicit other orders keep dyadic accuracy sizing.

    Returns
    -------
    tuple[int, int, int]
        Maximum mesh dimensions (nx, ny, nz) across all systems in batch.

    Notes
    -----
    Measured example on an NVIDIA H100 NVL with Torch: for 131,072 NH3 atoms,
    snapping from 486 cubed to 512 cubed increased mesh points by 16.92% and
    reduced the energy, forces, and charge-gradient call from 13.96 ms to
    11.17 ms. Both runs used order five, float64, a 9 Angstrom cutoff, and
    accuracy 1e-6. Times are medians of five groups of ten calls after three
    warmups; parameter setup and neighbor construction were excluded.
    """
    if cell.ndim == 2:
        cell = cell[None, ...]

    cell_lengths = jnp.linalg.norm(cell, axis=2)  # (B, 3)

    # Preserve the public estimator's existing relative sizing convention.
    accuracy_factor = 3.0 * (accuracy**0.2)
    n = (
        mesh_safety_factor * 2.0 * alpha[:, None] * cell_lengths / accuracy_factor
    )  # (B, 3)

    max_n = jnp.max(n, axis=0)  # (3,)
    return _round_pme_mesh_dimensions(
        np.asarray(jax.device_get(max_n)),
        spline_order=spline_order,
        power_of_two=(spline_order != _DEFAULT_PME_SPLINE_ORDER),
        fft_padding_fraction=fft_padding_fraction,
    )


def estimate_pme_parameters(
    positions: jax.Array,
    cell: jax.Array,
    batch_idx: jax.Array | None = None,
    accuracy: float = 1e-6,
    real_space_cutoff: float | None = None,
    mesh_safety_factor: float = 1.0,
    *,
    spline_order: int = _DEFAULT_PME_SPLINE_ORDER,
    fft_padding_fraction: float = _DEFAULT_PME_FFT_PADDING_FRACTION,
) -> PMEParameters:
    r"""Estimate PME splitting parameters and FFT-friendly mesh dimensions.

    Uses the closed-form Essmann/Kolafa-Perram derivation: a single
    length scale :math:`\eta = (V^2 / N)^{1/6} / \sqrt{2\pi}` determines both ``rc``
    and :math:`\alpha`. Callers who want to pin a specific cutoff (e.g. tied to
    neighbor-list update frequency in MD) should pass ``real_space_cutoff``.

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
        Caller-supplied cutoff. When given, :math:`\alpha` is derived from it via
        :math:`\alpha = \sqrt{-\log \varepsilon} / r_c`; otherwise rc and :math:`\alpha` come from :math:`\eta`.
    mesh_safety_factor : float, default=1.0
        Caller-requested multiplier on the continuous mesh-size estimate.
    spline_order : int, default=5
        B-spline assignment order used for mesh sizing.

    fft_padding_fraction : float, default=0.25
        Maximum fraction of extra total mesh points for snapping to powers
        of two, then smooth multiples of four. Zero selects the smallest
        2/3/5/7-smooth mesh. Explicit other orders keep dyadic accuracy sizing.

    Returns
    -------
    PMEParameters
        Dataclass containing alpha, mesh dimensions, spacing, and cutoffs.

    Notes
    -----
    ``accuracy`` retains the public estimator's relative sizing convention.
    Independent force checks determine the measured error for a charge
    distribution.
    """
    if cell.ndim == 2:
        cell = cell[None, ...]

    num_systems = cell.shape[0]
    volume = jnp.abs(jnp.linalg.det(cell))
    num_atoms = _count_atoms_per_system(positions, num_systems, batch_idx).astype(
        positions.dtype
    )
    cell_lengths = jnp.linalg.norm(cell, axis=2)  # (B, 3)

    # For batched inputs we share a single rc / α across the batch (one
    # neighbor cutoff for all systems). Use median-system properties.
    if real_space_cutoff is None:
        if num_systems == 1:
            n_repr = float(num_atoms[0])
            v_repr = float(volume[0])
        else:
            n_repr = float(jnp.median(num_atoms))
            v_repr = float(jnp.median(volume))
        eta = (v_repr**2 / n_repr) ** (1.0 / 6.0) / math.sqrt(2.0 * math.pi)
        rc_value = math.sqrt(-2.0 * math.log(accuracy)) * eta
        alpha_value = 1.0 / (math.sqrt(2.0) * eta)
    else:
        rc_value = float(real_space_cutoff)
        alpha_value = math.sqrt(-math.log(accuracy)) / rc_value

    alpha = jnp.full((num_systems,), alpha_value, dtype=positions.dtype)
    rc_array = jnp.full((num_systems,), rc_value, dtype=positions.dtype)

    mesh_dims = estimate_pme_mesh_dimensions(
        cell,
        alpha,
        accuracy,
        mesh_safety_factor=mesh_safety_factor,
        spline_order=spline_order,
        fft_padding_fraction=fft_padding_fraction,
    )
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
    *,
    spline_order: int = _DEFAULT_PME_SPLINE_ORDER,
    fft_padding_fraction: float = _DEFAULT_PME_FFT_PADDING_FRACTION,
) -> tuple[int, int, int]:
    """Convert mesh spacing to mesh dimensions.

    Parameters
    ----------
    cell : jax.Array
        Unit cell matrix.
    mesh_spacing : float | jax.Array
        Target mesh spacing.
    spline_order : int, default=5
        B-spline assignment order. Each axis covers its assignment support.

    fft_padding_fraction : float, default=0.25
        Maximum fraction of extra total mesh points for snapping to powers
        of two, then smooth multiples of four. Zero selects the smallest
        2/3/5/7-smooth mesh. These spacing preferences apply to every spline order.

    Returns
    -------
    tuple[int, int, int]
        Maximum dimensions across the batch, rounded upward to 2/3/5/7-smooth
        integers that cover the requested spacing and assignment support.
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

    max_mesh_dims = jnp.max(mesh_dims, axis=0)
    return _round_pme_mesh_dimensions(
        np.asarray(jax.device_get(max_mesh_dims)),
        spline_order=spline_order,
        fft_padding_fraction=fft_padding_fraction,
    )
