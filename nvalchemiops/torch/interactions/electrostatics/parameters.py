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
Parameter Estimation for Ewald and PME Methods (PyTorch)
========================================================

This module provides functions to automatically estimate optimal parameters
for Ewald summation and Particle Mesh Ewald (PME) calculations using PyTorch.
"""

import math
from dataclasses import dataclass

import torch


@dataclass
class EwaldParameters:
    """Container for Ewald summation parameters.

    All values are tensors of shape (B,), for
    single system calculations, the shape is (1,).

    Attributes
    ----------
    alpha : torch.Tensor, shape (B,)
        Ewald splitting parameter (inverse length units).
    real_space_cutoff : torch.Tensor, shape (B,)
        Real-space cutoff distance.
    reciprocal_space_cutoff : torch.Tensor, shape (B,)
        Reciprocal-space cutoff (:math:`|k|` in inverse length units).
    """

    alpha: torch.Tensor
    real_space_cutoff: torch.Tensor
    reciprocal_space_cutoff: torch.Tensor


@dataclass
class PMEParameters:
    """Container for PME parameters.

    Attributes
    ----------
    alpha : torch.Tensor, shape (B,)
        Ewald splitting parameter.
    mesh_dimensions : tuple[int, int, int], shape (3,)
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


def _count_atoms_per_system(
    positions: torch.Tensor, num_systems: int, batch_idx: torch.Tensor | None = None
) -> torch.Tensor:
    """Count number of atoms per system."""
    if batch_idx is None:
        return torch.tensor(
            [positions.shape[0]], dtype=torch.int32, device=positions.device
        )

    counts = torch.zeros(num_systems, dtype=torch.int32, device=batch_idx.device)
    ones = torch.ones_like(batch_idx)
    return counts.scatter_add_(0, batch_idx, ones)


def estimate_ewald_parameters(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    accuracy: float = 1e-6,
) -> EwaldParameters:
    """Estimate optimal Ewald summation parameters for a given accuracy.

    Uses the Kolafa-Perram formula to balance real-space and reciprocal-space
    contributions for optimal efficiency at the target accuracy.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrix.
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index for each atom. If None, single-system mode.
    accuracy : float, default=1e-6
        Target accuracy (relative error tolerance).

    Returns
    -------
    EwaldParameters
        Dataclass containing alpha, real_space_cutoff, reciprocal_space_cutoff
        as ``torch.Tensor`` objects.
    """
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    num_systems = cell.shape[0]

    # Compute volume per system: (B,)
    volume = torch.abs(torch.linalg.det(cell)).squeeze(-1)

    # Get number of atoms per system: (B,)
    num_atoms = _count_atoms_per_system(positions, num_systems, batch_idx).to(
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
    cell: torch.Tensor,
    alpha: torch.Tensor,
    accuracy: float = 1e-6,
) -> tuple[int, int, int]:
    """Estimate optimal PME mesh dimensions for a given accuracy.

    Parameters
    ----------
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrix.
    alpha : torch.Tensor, shape (B,)
        Ewald splitting parameter.
    accuracy : float, default=1e-6
        Target accuracy.

    Returns
    -------
    tuple[int, int, int]
        Maximum mesh dimensions (nx, ny, nz) across all systems in batch.
    """
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)

    # Cell lengths along each axis
    cell_lengths = torch.norm(cell, dim=2)  # (B, 3)

    # Accuracy factor: 3 * epsilon^(1/5)
    accuracy_factor = 3.0 * (accuracy**0.2)

    n = 2 * alpha[:, None] * cell_lengths / accuracy_factor  # (B, 3)

    # Take max across batch dimension
    max_n = torch.max(n, dim=0).values  # (3,)

    # Round up to powers of 2
    mesh_dims = torch.pow(2, torch.ceil(torch.log2(max_n))).to(torch.int32)
    return (
        int(mesh_dims[0].item()),
        int(mesh_dims[1].item()),
        int(mesh_dims[2].item()),
    )


def estimate_pme_parameters(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    accuracy: float = 1e-6,
) -> PMEParameters:
    """Estimate optimal PME parameters for a given accuracy.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrix.
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index for each atom.
    accuracy : float, default=1e-6
        Target accuracy.

    Returns
    -------
    PMEParameters
        Dataclass containing alpha, mesh dimensions, spacing, and cutoffs.
        Tensor fields are ``torch.Tensor`` objects.
    """
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)

    # We need to compute alpha locally first
    num_systems = cell.shape[0]
    volume = torch.abs(torch.linalg.det(cell)).squeeze(-1)
    num_atoms = _count_atoms_per_system(positions, num_systems, batch_idx).to(
        positions.dtype
    )
    eta = (volume**2 / num_atoms) ** (1.0 / 6.0) / math.sqrt(2.0 * math.pi)
    error_factor = math.sqrt(-2.0 * math.log(accuracy))
    real_space_cutoff = error_factor * eta
    alpha = 1.0 / (math.sqrt(2.0) * eta)

    # Estimate mesh dimensions
    mesh_dims = estimate_pme_mesh_dimensions(cell, alpha, accuracy)

    # Compute actual mesh spacing
    cell_lengths = torch.norm(cell, dim=2)  # (B, 3)
    mesh_dims_tensor = torch.tensor(
        mesh_dims, dtype=cell_lengths.dtype, device=cell_lengths.device
    )
    mesh_spacing = cell_lengths / mesh_dims_tensor  # (B, 3)

    return PMEParameters(
        alpha=alpha,
        mesh_dimensions=mesh_dims,
        mesh_spacing=mesh_spacing,
        real_space_cutoff=real_space_cutoff,
    )


def mesh_spacing_to_dimensions(
    cell: torch.Tensor,
    mesh_spacing: float | torch.Tensor,
) -> tuple[int, int, int]:
    """Convert mesh spacing to mesh dimensions.

    Parameters
    ----------
    cell : torch.Tensor
        Unit cell matrix.
    mesh_spacing : float | torch.Tensor
        Target mesh spacing.

    Returns
    -------
    tuple[int, int, int]
        Mesh dimensions, rounded up to powers of 2.
    """
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)

    cell_lengths = torch.norm(cell, dim=2)  # (B, 3)

    if isinstance(mesh_spacing, float):
        mesh_dims = torch.ceil(cell_lengths / mesh_spacing)
    elif isinstance(mesh_spacing, torch.Tensor):
        if mesh_spacing.ndim == 1:
            if mesh_spacing.shape[0] != cell.shape[0]:
                raise ValueError(
                    f"mesh_spacing shape {mesh_spacing.shape} incompatible with "
                    f"cell batch size {cell.shape[0]}"
                )
            mesh_dims = torch.ceil(cell_lengths / mesh_spacing[:, None])
        else:
            if mesh_spacing.shape != cell_lengths.shape:
                raise ValueError(
                    f"mesh_spacing shape {mesh_spacing.shape} incompatible with "
                    f"cell_lengths shape {cell_lengths.shape}"
                )
            mesh_dims = torch.ceil(cell_lengths / mesh_spacing)
    else:
        raise TypeError(
            f"mesh_spacing must be float or torch.Tensor, got {type(mesh_spacing)}"
        )

    mesh_dims = torch.pow(2, torch.ceil(torch.log2(mesh_dims))).to(torch.int32)

    max_mesh_dims = torch.max(mesh_dims, dim=0).values
    return (
        int(max_mesh_dims[0].item()),
        int(max_mesh_dims[1].item()),
        int(max_mesh_dims[2].item()),
    )
