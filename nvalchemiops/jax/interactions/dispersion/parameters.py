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

"""Parameter estimation for dispersion PME (LJ-PME) - JAX.

Mirrors ``nvalchemiops.torch.interactions.dispersion.parameters``. The total
dispersion-PME energy is invariant to ``alpha`` (``beta``); this estimator only
picks a workable real/reciprocal balance + mesh size, reusing the electrostatics
length-scale heuristic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp

from nvalchemiops.jax.interactions.electrostatics.parameters import (
    _count_atoms_per_system,
    estimate_pme_mesh_dimensions,
)

__all__ = [
    "DispersionPMEParameters",
    "estimate_dispersion_pme_parameters",
]


@dataclass
class DispersionPMEParameters:
    """Container for dispersion-PME parameters (JAX arrays)."""

    alpha: jnp.ndarray
    mesh_dimensions: tuple[int, int, int]
    mesh_spacing: jnp.ndarray
    real_space_cutoff: jnp.ndarray


def estimate_dispersion_pme_parameters(
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    batch_idx: jnp.ndarray | None = None,
    accuracy: float = 1e-6,
    real_space_cutoff: float | None = None,
    mesh_safety_factor: float = 1.0,
) -> DispersionPMEParameters:
    """Estimate dispersion-PME parameters for a target accuracy (JAX)."""
    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    num_systems = cell.shape[0]
    volume = jnp.abs(jnp.linalg.det(cell))
    num_atoms = _count_atoms_per_system(positions, num_systems, batch_idx).astype(
        positions.dtype
    )
    cell_lengths = jnp.linalg.norm(cell, axis=2)

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
    rc_tensor = jnp.full((num_systems,), rc_value, dtype=positions.dtype)

    mesh_dims = estimate_pme_mesh_dimensions(
        cell, alpha, accuracy, mesh_safety_factor=mesh_safety_factor
    )
    mesh_dims_arr = jnp.asarray(mesh_dims, dtype=cell_lengths.dtype)
    mesh_spacing = cell_lengths / mesh_dims_arr

    return DispersionPMEParameters(
        alpha=alpha,
        mesh_dimensions=mesh_dims,
        mesh_spacing=mesh_spacing,
        real_space_cutoff=rc_tensor,
    )
