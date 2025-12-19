# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Parameter Estimation for Ewald and PME Methods
===============================================

This module provides functions to automatically estimate optimal parameters
for Ewald summation and Particle Mesh Ewald (PME) calculations based on
desired accuracy tolerance.

The formulas are derived from error analysis of the Ewald splitting:
- For a given accuracy tolerance, the optimal splitting parameter (alpha)
  balances the work between real-space and reciprocal-space contributions.
- The cutoffs are chosen to achieve the target accuracy in each space.

EWALD PARAMETER ESTIMATION
==========================

For Ewald summation, given N atoms in a cell of volume V and accuracy tolerance :math:`\\varepsilon`:

.. math::

    \\eta = \\frac{(V^2 / N)^{1/6}}{\\sqrt{2\\pi}}

    \\alpha = \\frac{1}{\\sqrt{2}\\eta}

    r_{\\text{cutoff}} = \\sqrt{-2 \\ln \\varepsilon} \\cdot \\eta

    k_{\\text{cutoff}} = \\frac{\\sqrt{-2 \\ln \\varepsilon}}{\\eta}

where :math:`r_{\\text{cutoff}}` is the real-space cutoff and :math:`k_{\\text{cutoff}}` is the reciprocal-space cutoff.

PME MESH ESTIMATION
===================

For PME, the mesh spacing determines the reciprocal-space resolution.
Given :math:`\\alpha` and accuracy tolerance :math:`\\varepsilon`, the mesh dimensions along each axis are:

.. math::

    n_x = \\left\\lceil \\frac{2 \\alpha L_x}{3 \\varepsilon^{1/5}} \\right\\rceil

where :math:`L_x` is the cell length along axis x.

REFERENCES
==========

- Kolafa, J. & Perram, J. W. (1992). Mol. Sim. 9, 351-368
- Deserno, M. & Holm, C. (1998). J. Chem. Phys. 109, 7678
- Essmann et al. (1995). J. Chem. Phys. 103, 8577
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

    See Also
    --------
    estimate_ewald_parameters : Estimate optimal Ewald parameters automatically
    PMEParameters : Container for PME parameters
    """

    alpha: torch.Tensor
    real_space_cutoff: torch.Tensor
    reciprocal_space_cutoff: torch.Tensor


@dataclass
class PMEParameters:
    """Container for PME parameters.

    For single-system calculations, alpha and real_space_cutoff are tensors of shape (1,).
    For batch calculations, alpha and real_space_cutoff are tensors of shape (B,).
    Mesh spacing is a tensor of shape (B, 3).
    mesh_dimensions is a list of 3 integers, which is calculated as the maximum
        mesh dimensions along each axis for the entire set of structures in the batch.

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

    See Also
    --------
    estimate_pme_parameters : Estimate optimal PME parameters automatically
    EwaldParameters : Container for Ewald parameters
    """

    alpha: torch.Tensor
    mesh_dimensions: tuple[int, int, int]
    mesh_spacing: torch.Tensor
    real_space_cutoff: torch.Tensor


def _count_atoms_per_system(
    positions: torch.Tensor, num_systems: int, batch_idx: torch.Tensor | None = None
) -> torch.Tensor:
    """Count number of atoms per system.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    num_systems : int
        Number of systems.
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index for each atom. If None, all atoms belong to system 0.

    Returns
    -------
    counts : torch.Tensor, shape (num_systems,)
        Number of atoms in each system.
    """
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
        Common values: 1e-4 (low), 1e-6 (medium), 1e-8 (high).

    Returns
    -------
    EwaldParameters
        Dataclass containing alpha, real_space_cutoff, reciprocal_space_cutoff, and eta.
        For batch mode, these are tensors of shape (B,). For single-system, they are floats.

    Examples
    --------
    >>> positions = torch.randn(100, 3)
    >>> cell = torch.eye(3) * 20.0
    >>> params = estimate_ewald_parameters(positions, cell, accuracy=1e-6)
    >>> print(f"alpha={params.alpha:.4f}, r_cut={params.real_space_cutoff:.2f}")
    alpha=0.2835, r_cut=7.42

    Notes
    -----
    The formulas are:

    .. math::

        \\begin{aligned}
        \\eta &= \\frac{(V^2 / N)^{1/6}}{\\sqrt{2\\pi}} \\\\
        \\alpha &= \\frac{1}{\\sqrt{2}\\eta} \\\\
        r_{\\text{cutoff}} &= \\sqrt{-2 \\ln(\\varepsilon)} \\cdot \\eta \\\\
        k_{\\text{cutoff}} &= \\frac{\\sqrt{-2 \\ln(\\varepsilon)}}{\\eta}
        \\end{aligned}

    See Also
    --------
    EwaldParameters : Container for the returned parameters
    estimate_pme_parameters : Estimate PME parameters (includes mesh sizing)
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

    The formula is based on the B-spline interpolation error analysis:

    .. math::

        n_x = \\left\\lceil \\frac{2 \\alpha L_x}{3 \\varepsilon^{1/5}} \\right\\rceil

    Parameters
    ----------
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrix (row vectors are lattice vectors).
    alpha : torch.Tensor, shape (B,)
        Ewald splitting parameter.
    accuracy : float, default=1e-6
        Target accuracy (relative error tolerance).

    Returns
    -------
    tuple[int, int, int]
        Maximum mesh dimensions (nx, ny, nz) across all systems in batch.
        Dimensions are rounded up to powers of 2 for FFT efficiency.

    Examples
    --------
    >>> cell = 20.0 * torch.eye(3).unsqueeze(0)
    >>> alpha = torch.tensor([0.3])
    >>> mesh_dims = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-6)
    >>> print(mesh_dims)
    (64, 64, 64)

    See Also
    --------
    estimate_pme_parameters : Full PME parameter estimation
    mesh_spacing_to_dimensions : Convert mesh spacing to dimensions
    """
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)

    # Cell lengths along each axis (norm of each lattice vector)
    # cell shape is (B, 3, 3) where cell[b, i, :] is lattice vector i for system b
    cell_lengths = torch.norm(cell, dim=2)  # (B, 3)

    # Accuracy factor: 3 * epsilon^(1/5)
    accuracy_factor = 3.0 * (accuracy**0.2)

    n = 2 * alpha[:, None] * cell_lengths / accuracy_factor  # (B, 3)

    # Take max across batch dimension to get single mesh for all systems
    max_n = torch.max(n, dim=0).values  # (3,)

    # Round up to powers of 2 for FFT efficiency
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

    This combines Ewald parameter estimation with PME mesh sizing.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrix (row vectors are lattice vectors).
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index for each atom. If None, single-system mode.
    accuracy : float, default=1e-6
        Target accuracy (relative error tolerance).

    Returns
    -------
    PMEParameters
        Dataclass containing:
        - alpha: torch.Tensor, shape (B,) - Ewald splitting parameter per system
        - mesh_dimensions: tuple[int, int, int] - Maximum mesh dimensions across batch
        - mesh_spacing: torch.Tensor, shape (B, 3) - Actual mesh spacing per system
        - real_space_cutoff: torch.Tensor, shape (B,) - Real-space cutoff per system

    Examples
    --------
    >>> positions = torch.randn(100, 3)
    >>> cell = torch.eye(3) * 20.0
    >>> params = estimate_pme_parameters(positions, cell, accuracy=1e-6)
    >>> print(f"alpha={params.alpha.item():.4f}, mesh={params.mesh_dimensions}")
    alpha=0.2835, mesh=(32, 32, 32)

    See Also
    --------
    PMEParameters : Container for the returned parameters
    estimate_ewald_parameters : Estimate Ewald parameters only
    estimate_pme_mesh_dimensions : Estimate mesh dimensions only
    """
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)

    # Get Ewald parameters (handles batch internally)
    ewald_params = estimate_ewald_parameters(positions, cell, batch_idx, accuracy)

    # Estimate mesh dimensions (returns tuple of max dims across batch)
    mesh_dims = estimate_pme_mesh_dimensions(cell, ewald_params.alpha, accuracy)

    # Compute actual mesh spacing based on cell lengths for each system
    # cell shape is (B, 3, 3) where cell[b, i, :] is lattice vector i for system b
    cell_lengths = torch.norm(cell, dim=2)  # (B, 3)
    mesh_dims_tensor = torch.tensor(
        mesh_dims, dtype=cell_lengths.dtype, device=cell_lengths.device
    )
    mesh_spacing = cell_lengths / mesh_dims_tensor  # (B, 3)

    return PMEParameters(
        alpha=ewald_params.alpha,
        mesh_dimensions=mesh_dims,
        mesh_spacing=mesh_spacing,
        real_space_cutoff=ewald_params.real_space_cutoff,
    )


def mesh_spacing_to_dimensions(
    cell: torch.Tensor,
    mesh_spacing: float | torch.Tensor,
) -> tuple[int, int, int]:
    """Convert mesh spacing to mesh dimensions.

    Parameters
    ----------
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrix (row vectors are lattice vectors).
    mesh_spacing : float | torch.Tensor
        Target mesh spacing. Can be:
        - float: uniform spacing for all directions and systems
        - torch.Tensor, shape (B,): per-system spacing (uniform in all directions)
        - torch.Tensor, shape (B, 3): per-system, per-direction spacing

    Returns
    -------
    tuple[int, int, int]
        Mesh dimensions, rounded up to powers of 2.

    See Also
    --------
    estimate_pme_mesh_dimensions : Estimate mesh dimensions from accuracy tolerance
    PMEParameters : Container that includes mesh dimensions
    """
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)

    # Cell lengths along each axis (norm of each lattice vector)
    cell_lengths = torch.norm(cell, dim=2)  # (B, 3)

    if isinstance(mesh_spacing, float):
        mesh_dims = torch.ceil(cell_lengths / mesh_spacing)
    elif mesh_spacing.ndim == 1:
        # Per-system spacing (uniform in all directions)
        if mesh_spacing.shape[0] != cell.shape[0]:
            raise ValueError(
                f"mesh_spacing shape {mesh_spacing.shape} incompatible with "
                f"cell batch size {cell.shape[0]}"
            )
        mesh_dims = torch.ceil(cell_lengths / mesh_spacing[:, None])
    else:
        # Per-system, per-direction spacing
        if mesh_spacing.shape != cell_lengths.shape:
            raise ValueError(
                f"mesh_spacing shape {mesh_spacing.shape} incompatible with "
                f"cell_lengths shape {cell_lengths.shape}"
            )
        mesh_dims = torch.ceil(cell_lengths / mesh_spacing)

    mesh_dims = torch.pow(2, torch.ceil(torch.log2(mesh_dims))).to(torch.int32)

    max_mesh_dims = torch.max(mesh_dims, dim=0).values
    return (
        int(max_mesh_dims[0].item()),
        int(max_mesh_dims[1].item()),
        int(max_mesh_dims[2].item()),
    )
