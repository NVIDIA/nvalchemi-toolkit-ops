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
Parameter estimation for dispersion PME (LJ-PME).

Mirrors ``electrostatics/parameters.py``. The total dispersion-PME energy is
invariant to the splitting parameter :math:`\\beta` (``alpha``); this estimator
only chooses a workable real/reciprocal balance and mesh size for a target
accuracy. The :math:`r^{-6}` error model differs from Coulomb's Kolafa-Perram
formula; here we reuse the same length-scale heuristic as electrostatics as a
documented, validated-by-construction default (the total energy is correct for
any consistent ``alpha``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nvalchemiops.torch.interactions.electrostatics.parameters import (
    _count_atoms_per_system,
    estimate_pme_mesh_dimensions,
)

__all__ = [
    "DispersionPMEParameters",
    "estimate_dispersion_pme_parameters",
]


@dataclass
class DispersionPMEParameters:
    """Container for dispersion-PME parameters.

    Attributes
    ----------
    alpha : torch.Tensor, shape (B,)
        Dispersion splitting parameter ``beta`` (inverse length).
    mesh_dimensions : tuple[int, int, int]
        Mesh dimensions (nx, ny, nz).
    mesh_spacing : torch.Tensor, shape (B, 3)
        Actual mesh spacing in each direction.
    real_space_cutoff : torch.Tensor, shape (B,)
        Real-space cutoff distance.
    """

    alpha: torch.Tensor
    mesh_dimensions: tuple[int, int, int]
    mesh_spacing: torch.Tensor
    real_space_cutoff: torch.Tensor


def estimate_dispersion_pme_parameters(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    accuracy: float = 1e-6,
    real_space_cutoff: float | None = None,
    mesh_safety_factor: float = 1.0,
) -> DispersionPMEParameters:
    """Estimate dispersion-PME parameters for a target accuracy.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrix.
    batch_idx : torch.Tensor, shape (N,), optional
        System index per atom.
    accuracy : float, default=1e-6
        Target relative accuracy.
    real_space_cutoff : float, optional
        Caller-supplied real-space cutoff. When given, ``alpha`` is derived from
        it as ``sqrt(-log(accuracy)) / rc``; otherwise both come from the
        length-scale heuristic ``eta = (V²/N)^{1/6}/sqrt(2π)``.
    mesh_safety_factor : float, default=1.0
        Multiplier on the mesh-size heuristic (reused from electrostatics).

    Returns
    -------
    DispersionPMEParameters
    """
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)

    num_systems = cell.shape[0]
    volume = torch.abs(torch.linalg.det(cell))
    num_atoms = _count_atoms_per_system(positions, num_systems, batch_idx).to(
        positions.dtype
    )
    cell_lengths = torch.norm(cell, dim=2)  # (B, 3)

    if real_space_cutoff is None:
        if num_systems == 1:
            n_repr = float(num_atoms[0].item())
            v_repr = float(volume[0].item())
        else:
            n_repr = float(num_atoms.median().item())
            v_repr = float(volume.median().item())
        eta = (v_repr**2 / n_repr) ** (1.0 / 6.0) / math.sqrt(2.0 * math.pi)
        rc_value = math.sqrt(-2.0 * math.log(accuracy)) * eta
        alpha_value = 1.0 / (math.sqrt(2.0) * eta)
    else:
        rc_value = float(real_space_cutoff)
        alpha_value = math.sqrt(-math.log(accuracy)) / rc_value

    alpha = torch.full(
        (num_systems,), alpha_value, dtype=positions.dtype, device=positions.device
    )
    rc_tensor = torch.full(
        (num_systems,), rc_value, dtype=positions.dtype, device=positions.device
    )

    mesh_dims = estimate_pme_mesh_dimensions(
        cell, alpha, accuracy, mesh_safety_factor=mesh_safety_factor
    )
    mesh_dims_tensor = torch.tensor(
        mesh_dims, dtype=cell_lengths.dtype, device=cell_lengths.device
    )
    mesh_spacing = cell_lengths / mesh_dims_tensor

    return DispersionPMEParameters(
        alpha=alpha,
        mesh_dimensions=mesh_dims,
        mesh_spacing=mesh_spacing,
        real_space_cutoff=rc_tensor,
    )
