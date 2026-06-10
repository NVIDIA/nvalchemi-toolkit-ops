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
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrix.
    alpha : torch.Tensor, shape (B,)
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
        cell = cell.unsqueeze(0)

    cell_lengths = torch.norm(cell, dim=2)  # (B, 3)

    # K = 2 α L / (3 ε^0.2), with optional safety multiplier + pow-2 snap.
    accuracy_factor = 3.0 * (accuracy**0.2)
    n = (
        mesh_safety_factor * 2.0 * alpha[:, None] * cell_lengths / accuracy_factor
    )  # (B, 3)

    max_n = torch.max(n, dim=0).values  # (3,)
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
    real_space_cutoff: float | None = None,
    mesh_safety_factor: float = 1.0,
) -> PMEParameters:
    """Estimate PME parameters for a given accuracy.

    Uses the closed-form Essmann/Kolafa-Perram derivation: a single
    length scale ``η = (V² / N)^{1/6} / √(2π)`` determines both ``rc``
    and ``α``. Callers who want to pin a specific cutoff (e.g. tied to
    neighbor-list update frequency in MD) should pass ``real_space_cutoff``.

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
    real_space_cutoff : float, optional
        Caller-supplied cutoff. When given, ``α`` is derived from it via
        ``α = √(-log ε) / rc``; otherwise rc and α come from η.
    mesh_safety_factor : float, default=1.0
        Multiplier on the standard mesh-size heuristic
        ``K = 2 α L / (3 ε^{1/5})``. Raise for extra safety at tight ε.

    Returns
    -------
    PMEParameters
        Dataclass containing alpha, mesh dimensions, spacing, and cutoffs.
    """
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)

    num_systems = cell.shape[0]
    volume = torch.abs(torch.linalg.det(cell))
    num_atoms = _count_atoms_per_system(positions, num_systems, batch_idx).to(
        positions.dtype
    )
    cell_lengths = torch.norm(cell, dim=2)  # (B, 3)

    # For batched inputs we share a single rc / α across the batch (one
    # neighbor cutoff for all systems). Use median-system properties.
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
        (num_systems,),
        alpha_value,
        dtype=positions.dtype,
        device=positions.device,
    )
    rc_tensor = torch.full(
        (num_systems,),
        rc_value,
        dtype=positions.dtype,
        device=positions.device,
    )

    mesh_dims = estimate_pme_mesh_dimensions(
        cell,
        alpha,
        accuracy,
        mesh_safety_factor=mesh_safety_factor,
    )
    mesh_dims_tensor = torch.tensor(
        mesh_dims, dtype=cell_lengths.dtype, device=cell_lengths.device
    )
    mesh_spacing = cell_lengths / mesh_dims_tensor  # (B, 3)

    return PMEParameters(
        alpha=alpha,
        mesh_dimensions=mesh_dims,
        mesh_spacing=mesh_spacing,
        real_space_cutoff=rc_tensor,
    )


@dataclass
class MultipoleEwaldParameters:
    """Container for GTO-Ewald multipole parameters.

    Like :class:`EwaldParameters` but with the GTO basis width ``sigma``
    propagated through. The Kolafa-Perram balance for the multipole case
    has the same ``rcut`` / ``kcut`` formulas as the monopole case, but
    ``alpha`` differs because the effective Ewald split width is
    ``sigma_c = sqrt(sigma**2 + 1/(4 alpha**2))`` rather than ``1/(alpha
    sqrt(2))``.

    Attributes
    ----------
    alpha : torch.Tensor, shape (B,)
        Ewald splitting parameter (inverse length units).
    sigma : torch.Tensor, shape (B,)
        GTO basis width (passed through; physics).
    real_space_cutoff : torch.Tensor, shape (B,)
        Real-space cutoff distance.
    reciprocal_space_cutoff : torch.Tensor, shape (B,)
        Reciprocal-space cutoff (``|k|`` in inverse length units).
    """

    alpha: torch.Tensor
    sigma: torch.Tensor
    real_space_cutoff: torch.Tensor
    reciprocal_space_cutoff: torch.Tensor


@dataclass
class MultipolePMEParameters:
    """Container for GTO-Ewald multipole PME parameters.

    Attributes
    ----------
    alpha : torch.Tensor, shape (B,)
        Ewald splitting parameter.
    sigma : torch.Tensor, shape (B,)
        GTO basis width (passed through; physics).
    mesh_dimensions : tuple[int, int, int]
        Mesh dimensions ``(nx, ny, nz)`` (max across batch).
    mesh_spacing : torch.Tensor, shape (B, 3)
        Actual mesh spacing per direction.
    real_space_cutoff : torch.Tensor, shape (B,)
        Real-space cutoff distance.
    """

    alpha: torch.Tensor
    sigma: torch.Tensor
    mesh_dimensions: tuple[int, int, int]
    mesh_spacing: torch.Tensor
    real_space_cutoff: torch.Tensor


def _kp_eta(volume: torch.Tensor, num_atoms: torch.Tensor) -> torch.Tensor:
    """Kolafa-Perram cost-balance length scale ``η = (V**2/N)**(1/6) / sqrt(2π)``.

    Common to both the monopole and multipole estimators — captures the
    geometric balance between real-space-pair count and reciprocal-space
    k-vector count at fixed accuracy.
    """
    return (volume**2 / num_atoms) ** (1.0 / 6.0) / math.sqrt(2.0 * math.pi)


def _prepare_sigma(
    sigma: float | torch.Tensor,
    num_systems: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Broadcast a scalar or per-system ``sigma`` into shape ``(B,)``."""
    if isinstance(sigma, torch.Tensor):
        sig = sigma.to(dtype=dtype, device=device)
        if sig.ndim == 0:
            sig = sig.expand(num_systems).clone()
        elif sig.shape != (num_systems,):
            raise ValueError(
                f"sigma must be scalar or shape ({num_systems},); "
                f"got {tuple(sig.shape)}"
            )
        return sig
    return torch.full((num_systems,), float(sigma), dtype=dtype, device=device)


def estimate_multipole_ewald_parameters(
    positions: torch.Tensor,
    cell: torch.Tensor,
    sigma: float | torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    accuracy: float = 1e-6,
    cost_ratio: float = 1.0,
) -> MultipoleEwaldParameters:
    """Estimate GTO-Ewald multipole parameters at a given target accuracy.

    Mirrors :func:`estimate_ewald_parameters` semantics ("relative
    energy-error" accuracy via the Kolafa-Perram envelope), adjusted for
    the multipole case where the effective Ewald-split width is
    ``sigma_c = sqrt(sigma**2 + 1/(4 alpha**2))`` rather than
    ``1/(alpha sqrt(2))``.

    Derivation
    ----------
    Both the real-space tail (``erfc(r/(2 sigma_c))``) and the
    reciprocal-space envelope (``exp(-k**2 sigma_c**2)``) decay with the
    same effective width ``sigma_c sqrt(2)``. Substituting that for the
    monopole's ``eta = 1/(alpha sqrt(2))`` in Kolafa-Perram gives
    ``rcut = error_factor * eta``, ``kcut = error_factor / eta``, and
    ``alpha = 1 / (sqrt(2) sqrt(eta**2 - 2 sigma**2))``. The
    ``sigma -> 0`` limit recovers the monopole formula.

    Cost-ratio correction
    ---------------------
    The textbook Kolafa-Perram balance assumes the per-real-space-pair
    cost equals the per-k-vector cost. On real hardware (and especially
    for the lmax=1 multipole tile kernels) those costs differ — measured
    ``C_r / C_k`` is in the 20-40x range for our cluster-pair tile
    kernels at fp64. The cost-balanced optimum scales as
    ``eta_eff = eta_KP / cost_ratio**(1/6)``: a 30x cost ratio shrinks
    rcut by ~1.76x (and grows kcut by the same factor), which can cut
    the real-space pair count by ~5x at the same target accuracy.

    The math: with cost ratio ``R = C_r / C_k``, the cost-balanced
    formula becomes ``eta_eff = (V^2 / (N * R))^(1/6) / sqrt(2 pi)``.
    Setting ``R = 1`` (default) reproduces the canonical KP estimator.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3) or (N_total, 3)
        Atomic coordinates.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrix (matches the multipole-Ewald convention).
    sigma : float or torch.Tensor
        GTO basis width — same value used by the multipole-Ewald kernel.
        Scalar or shape ``(B,)``.
    batch_idx : torch.Tensor, shape (N_total,), int32, optional
        System index per atom. ``None`` selects single-system mode.
    accuracy : float, default 1e-6
        Target relative accuracy (matches monopole convention).
    cost_ratio : float, default 1.0
        Empirical ``C_r / C_k`` ratio — per-pair real-space cost divided
        by per-k-vector reciprocal cost on the target hardware. ``1.0``
        (default) reproduces canonical Kolafa-Perram. Higher values
        shift the optimum toward smaller rcut (fewer pairs) + larger
        kcut (more k-vectors), which wins when the per-pair cluster-pair
        tile kernel dominates. For the lmax=1 multipole kernels on a
        GB10-class GPU, ``cost_ratio`` ~ 30 is a reasonable starting
        point — measure on your own hardware via the per-pair / per-k
        timing probe (see ``docs/learnings/`` if archived). Setting
        below 1 is allowed but rarely useful (the formula is
        symmetric).

    Returns
    -------
    MultipoleEwaldParameters
        ``(alpha, sigma, real_space_cutoff, reciprocal_space_cutoff)``.

    Raises
    ------
    ValueError
        If any system has ``eta_eff <= sigma * sqrt(2)`` — meaning the
        cost-balanced Ewald split is degenerate at this size + sigma
        combination. Note: the validity threshold scales as
        ``cost_ratio**(-1/6)`` — large ``cost_ratio`` makes the split
        more likely to be invalid for small/dense systems.
    """
    if cost_ratio <= 0.0:
        raise ValueError(f"cost_ratio must be positive, got {cost_ratio}")

    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    num_systems = cell.shape[0]
    dtype = positions.dtype
    device = positions.device

    volume = torch.abs(torch.linalg.det(cell)).squeeze(-1)
    num_atoms = _count_atoms_per_system(positions, num_systems, batch_idx).to(dtype)

    eta_kp = _kp_eta(volume, num_atoms)
    eta = eta_kp / (cost_ratio ** (1.0 / 6.0))
    error_factor = math.sqrt(-2.0 * math.log(accuracy))

    real_space_cutoff = error_factor * eta
    reciprocal_space_cutoff = error_factor / eta

    sigma_t = _prepare_sigma(sigma, num_systems, dtype, device)

    # alpha = 1 / (sqrt(2) * sqrt(eta**2 - 2 sigma**2)). The validity
    # check uses the cost-corrected eta — large cost_ratio shrinks eta,
    # which can push borderline systems into the invalid regime.
    discriminant = eta * eta - 2.0 * sigma_t * sigma_t
    if torch.any(discriminant <= 0.0):
        bad = (discriminant <= 0.0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            "Multipole Ewald parameter estimation: GTO sigma is too large "
            f"relative to the system size for systems {bad} at cost_ratio="
            f"{cost_ratio}. The cost-balanced eta={eta.tolist()} satisfies "
            f"eta**2 <= 2 sigma**2 (sigma={sigma_t.tolist()}). Either reduce "
            "sigma, increase the system, drop cost_ratio toward 1.0, or use "
            "direct k-space (multipole_electrostatic_energy)."
        )
    alpha = 1.0 / (math.sqrt(2.0) * torch.sqrt(discriminant))

    return MultipoleEwaldParameters(
        alpha=alpha,
        sigma=sigma_t,
        real_space_cutoff=real_space_cutoff,
        reciprocal_space_cutoff=reciprocal_space_cutoff,
    )


def estimate_multipole_pme_parameters(
    positions: torch.Tensor,
    cell: torch.Tensor,
    sigma: float | torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    accuracy: float = 1e-6,
    cost_ratio: float = 1.0,
) -> MultipolePMEParameters:
    """Estimate GTO-Ewald multipole PME parameters at a given target accuracy.

    Same Kolafa-Perram backbone as
    :func:`estimate_multipole_ewald_parameters`. Mesh dimensions follow
    the standard B-spline-error formula
    ``n_per_dim = 2 alpha_eff L / (3 accuracy**0.2)`` with
    ``alpha_eff = 1 / (sqrt(2) eta)`` — the monopole-equivalent alpha at
    the same eta.

    The ``cost_ratio`` knob has the same meaning as in
    :func:`estimate_multipole_ewald_parameters` (per-pair vs per-k cost
    asymmetry), and shifts the rcut/kcut/mesh balance the same way:
    ``eta_eff = eta_KP / cost_ratio**(1/6)``. A larger ``cost_ratio``
    grows the FFT mesh and shrinks the real-space cutoff. Note that
    PME's true reciprocal cost is FFT (``M log M``) plus
    spread/gather (``N p**3``), which is not the same shape as the
    Ewald per-k-vector cost — so the optimal ``cost_ratio`` for PME may
    differ from the Ewald optimum even on the same hardware. Default
    ``1.0`` (canonical KP) is a safe starting point.

    Returns
    -------
    MultipolePMEParameters
        ``(alpha, sigma, mesh_dimensions, mesh_spacing, real_space_cutoff)``.
    """
    if cost_ratio <= 0.0:
        raise ValueError(f"cost_ratio must be positive, got {cost_ratio}")

    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    num_systems = cell.shape[0]
    dtype = positions.dtype
    device = positions.device

    volume = torch.abs(torch.linalg.det(cell)).squeeze(-1)
    num_atoms = _count_atoms_per_system(positions, num_systems, batch_idx).to(dtype)

    eta_kp = _kp_eta(volume, num_atoms)
    eta = eta_kp / (cost_ratio ** (1.0 / 6.0))
    error_factor = math.sqrt(-2.0 * math.log(accuracy))
    real_space_cutoff = error_factor * eta

    sigma_t = _prepare_sigma(sigma, num_systems, dtype, device)
    discriminant = eta * eta - 2.0 * sigma_t * sigma_t
    if torch.any(discriminant <= 0.0):
        bad = (discriminant <= 0.0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            "Multipole PME parameter estimation: GTO sigma too large for "
            f"systems {bad} at cost_ratio={cost_ratio} (eta**2 <= 2 sigma**2). "
            "Reduce sigma, drop cost_ratio toward 1.0, or use direct k-space."
        )
    alpha = 1.0 / (math.sqrt(2.0) * torch.sqrt(discriminant))

    # Mesh resolution from the reciprocal-space B-spline error envelope.
    # Width is set by eta; alpha_eff = 1/(sqrt(2) eta) (monopole-equivalent).
    alpha_eff = torch.full_like(eta, 1.0 / math.sqrt(2.0)) / eta
    mesh_dims = estimate_pme_mesh_dimensions(cell, alpha_eff, accuracy)

    cell_lengths = torch.norm(cell, dim=2)  # (B, 3)
    mesh_dims_tensor = torch.tensor(mesh_dims, dtype=cell_lengths.dtype, device=device)
    mesh_spacing = cell_lengths / mesh_dims_tensor

    return MultipolePMEParameters(
        alpha=alpha,
        sigma=sigma_t,
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
