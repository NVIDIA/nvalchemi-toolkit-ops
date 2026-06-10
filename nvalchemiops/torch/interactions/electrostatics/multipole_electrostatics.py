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

r"""
PyTorch bindings for the direct-k-space multipole electrostatics.

This module exposes ``multipole_electrostatic_energy(...)``, the public entry
point that composes the Warp kernels into a single electrostatic energy
calculator. It is designed as a drop-in companion to the customer reference
``GTOElectrostaticEnergy`` with:

* a friendlier API (Cartesian dipoles instead of pre-permuted ``(y, z, x)``,
  cell matrix instead of pre-generated k-vectors);
* Warp-accelerated kernels on CPU and CUDA;
* bit-for-bit parity with the reference at float64 for ``l_max in {0, 1}``
  under matched inputs.

The forward path returns per-atom energies :math:`(N,)` that are
autograd-connected to ``positions`` and ``multipole_moments``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    split_packed_for_kernels,
)
from nvalchemiops.torch.math.gto import NormMode

if TYPE_CHECKING:
    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
        MultipoleSCFCache,
    )


def _resolve_norm_mode(mode: NormMode | int | str) -> NormMode:
    if isinstance(mode, str):
        return NormMode[mode.upper()]
    return NormMode(mode)


def _prepend_origin(k_vectors: torch.Tensor) -> torch.Tensor:
    """Prepend a ``(0, 0, 0)`` row so k=0 is present as index 0.

    ``generate_k_vectors_ewald_summation`` excludes k=0 (division by zero in the
    Ewald Green's function); the direct-k-space pipeline needs it explicitly so
    the ``per_k_factor`` array can be indexed uniformly. The k=0 term vanishes
    because the caller zeros ``per_k_factor[0]``.
    """
    origin = k_vectors.new_zeros((1, 3))
    return torch.cat([origin, k_vectors], dim=0)


def multipole_electrostatic_energy(
    positions: torch.Tensor,
    multipole_moments: torch.Tensor,
    cell: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None = None,
    sigma: float,
    k_cutoff: float | None = None,
    k_vectors: torch.Tensor | None = None,
    normalize: NormMode | int | str = NormMode.MULTIPOLES,
    include_self_interaction: bool = False,
) -> torch.Tensor:
    r"""Total PBC electrostatic energy via direct k-space summation.

    Computes

    .. math::

        E \;=\; \frac{1}{2} \cdot \frac{V}{(2\pi)^6}
                \sum_{\mathbf{k}} 2\,\text{Re}\!\left[\rho^{*}(\mathbf{k})\,
                V(\mathbf{k})\right]
                \;-\; \tfrac{1}{2} E_{\text{self}},

    where :math:`\rho(\mathbf{k})` and :math:`V(\mathbf{k}) = F \cdot \rho(\mathbf{k}) / k^2`
    are assembled from per-atom ``multipole_moments`` via the Warp kernels.
    Matches the customer reference ``GTOElectrostaticEnergy`` bit-for-bit at
    ``l_max in {0, 1}`` under matched inputs.

    Single-system vs batched dispatch
    ---------------------------------
    Mirrors :func:`multipole_ewald_summation`: pass ``cell`` of shape
    ``(3, 3)`` (single) or ``(B, 3, 3)`` (batched) and use ``batch_idx`` to
    select the batched path (returns per-atom :math:`(N_\text{total},)`).
    Batched mode requires ``k_cutoff`` (a pre-generated ``k_vectors`` is
    single-system only).

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions, shape ``(N, 3)`` or ``(N_total, 3)`` (flat across
        systems in the batched case), ``float32`` or ``float64``.
    multipole_moments : torch.Tensor
        Packed per-atom multipole moments, shape ``(N, (l_max+1)**2)``,
        in e3nn spherical layout: ``[q]`` (l_max=0), ``[q, μ_y, μ_z, μ_x]``
        (l_max=1), or the l_max=1 block plus the five traceless l=2 channels
        (l_max=2). The l=2 quadrupole is expanded to the Cartesian symmetric
        ``(N, 3, 3)`` form and threaded through the SCF-cache Q channel.
    cell : torch.Tensor
        Unit-cell matrix (lattice vectors as rows), shape ``(3, 3)``, or
        ``B`` per-system cells ``(B, 3, 3)`` (batched).
    batch_idx : torch.Tensor, optional, shape (N_total,), int32
        Per-atom system index (expected sorted). Required when ``cell`` is
        ``(B, 3, 3)``; must be ``None`` for a single ``(3, 3)`` cell.
    sigma : float
        Density-basis Gaussian width. Used for both the source GTO basis and
        the self-interaction overlap (matches ``GTOElectrostaticEnergy``).
    k_cutoff : float, optional
        Maximum ``|k|`` to include in the reciprocal-space sum. Required when
        ``k_vectors`` is not supplied; ignored when it is.
    k_vectors : torch.Tensor, optional
        Pre-computed k-grid, shape ``(N_k, 3)``, ``float64``. **Must include
        ``(0, 0, 0)`` as the first row** — the kernel's ``V(k=0) = 0``
        convention expects the origin explicitly and downstream indexing
        assumes row 0 is it. Pass this when amortizing k-vector generation
        across many energy evaluations for the same geometry (MD steps at
        fixed cell, SCF iterations, benchmark loops). Must live on
        ``positions.device``. When omitted, the function generates k-vectors
        internally via ``generate_k_vectors_ewald_summation(cell, k_cutoff)``
        and prepends the origin.
    normalize : NormMode | int | str
        Normalization convention for the density basis. Defaults to
        ``NormMode.MULTIPOLES`` (the only physically meaningful choice for
        source moments; the other modes exist for debugging / cross-checks).
    include_self_interaction : bool
        If False (default), subtracts ``0.5 · E_self`` where
        :math:`E_\text{self} = \sum_i \mathrm{oc}[0]\,q_i^2 +
        \mathrm{oc}[1]\,|\boldsymbol{\mu}_i|^2` and ``oc`` comes from
        :func:`nvalchemiops.torch.math.compute_overlap_constants`.

    Returns
    -------
    torch.Tensor
        Per-atom :math:`(N,)` :math:`\text{float64}` (single) or
        :math:`(N_\text{total},)` (batched, flat across systems) on
        ``positions.device``. Call ``.sum()`` for the total energy or
        ``torch.zeros(B).scatter_add(0, batch_idx, E)`` for per-system totals;
        forces/stress/charge-grads flow from ``grad(E.sum(), ...)``.
        Autograd-connected to ``positions`` and ``multipole_moments``.
    """
    is_batch = batch_idx is not None
    if is_batch:
        if cell.ndim != 3 or cell.shape[-2:] != (3, 3):
            raise ValueError(f"batched cell must be (B, 3, 3), got {tuple(cell.shape)}")
        if k_vectors is not None:
            raise ValueError(
                "k_vectors is not supported for batched energy; pass k_cutoff instead."
            )
    elif cell.shape != (3, 3):
        raise ValueError(f"cell must be (3, 3) or (B, 3, 3), got {tuple(cell.shape)}")
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"positions must be (N, 3), got {tuple(positions.shape)}")
    if multipole_moments.ndim != 2 or multipole_moments.shape[0] != positions.shape[0]:
        raise ValueError(
            "multipole_moments must be (N, (l_max+1)^2) matching positions[0]; "
            f"got {tuple(multipole_moments.shape)}"
        )
    if is_batch and batch_idx.shape[0] != positions.shape[0]:
        raise ValueError(
            f"batch_idx must match N_total={positions.shape[0]}, "
            f"got {tuple(batch_idx.shape)}"
        )
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if k_vectors is None and (k_cutoff is None or k_cutoff <= 0.0):
        raise ValueError(
            "Either k_vectors must be supplied, or k_cutoff must be a "
            f"positive float (got k_cutoff={k_cutoff})."
        )

    # Split into the l<=1 e3nn block + the Cartesian quadrupole (None for l<2).
    source_feats_l1, quadrupoles, l_max = split_packed_for_kernels(multipole_moments)
    norm_mode = _resolve_norm_mode(normalize)

    # Delayed imports avoid a circular module graph.
    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
        prepare_multipole_scf_cache,
    )
    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
        multipole_scf_step_energy,
    )

    cache = prepare_multipole_scf_cache(
        cell,
        sigma=sigma,
        receiver_sigmas=[sigma],
        k_cutoff=k_cutoff,
        k_vectors=None if is_batch else k_vectors,
        l_max=l_max,
        density_normalize=norm_mode,
        feature_normalize=norm_mode,
        device=positions.device,
    )
    return multipole_scf_step_energy(
        cache,
        positions,
        source_feats_l1,
        batch_idx=batch_idx,
        include_self_interaction=include_self_interaction,
        quadrupoles=quadrupoles,
    )


def multipole_reciprocal_space_energy(
    positions: torch.Tensor,
    multipole_moments: torch.Tensor,
    cell: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None = None,
    sigma: float,
    alpha: float,
    k_cutoff: float | None = None,
    normalize: NormMode | int | str = NormMode.MULTIPOLES,
    cache: MultipoleSCFCache | None = None,
) -> torch.Tensor:
    r"""Reciprocal-space half of an Ewald-split multipole electrostatic energy.

    Same pipeline as :func:`multipole_electrostatic_energy` but with a
    Gaussian-damped per-k kernel:

    .. math::

        V(\mathbf{k})
            = \frac{F \, e^{-|\mathbf{k}|^2 / (4\alpha^2)}}{|\mathbf{k}|^2}
              \, \rho(\mathbf{k}),

    (``k = 0`` zeroed). Intended to be paired with a real-space
    erfc-damped contribution (see :func:`multipole_real_space_energy`)
    at the same ``α`` to assemble the full Ewald-split Coulomb sum.

    Single-system vs batched dispatch
    ---------------------------------
    Mirrors :func:`multipole_ewald_summation`: pass ``cell`` of shape
    ``(3, 3)`` (single) or ``(B, 3, 3)`` (batched) and use ``batch_idx`` to
    select the batched path (returns per-atom :math:`(N_\text{total},)`).
    Both single and batched modes build their k-grid from ``k_cutoff`` (or
    reuse a pre-built ``cache``).

    Parameters
    ----------
    positions, multipole_moments, cell, sigma, k_cutoff, normalize
        Same as :func:`multipole_electrostatic_energy`.
    batch_idx : torch.Tensor, optional, shape (N_total,), int32
        Per-atom system index (expected sorted). Required when ``cell`` is
        ``(B, 3, 3)``; must be ``None`` for a single ``(3, 3)`` cell.
    alpha : float
        Ewald splitting parameter (must be positive). The caller's
        real-space kernel should use the same ``alpha``.
    cache : MultipoleSCFCache, optional
        Pre-built reciprocal cache (from :func:`prepare_multipole_scf_cache`)
        holding the position-independent k-grid / GTO-Fourier (``phi_hat``) /
        per-k-factor tables. When given, the per-call cache rebuild is skipped
        (MD / inference steady state) — the analog of passing precomputed
        ``k_squared`` to PME. ``cell``/``sigma``/``alpha``/``k_cutoff`` are then
        ignored for the reciprocal (the cache already encodes them); the caller
        owns matching the cache to the system.

        .. warning::
            A pre-built cache holds a fixed (detached) k-grid and volume, so
            ``grad(E, cell)`` (stress) does **not** flow through it — pass
            ``cache=None`` for stress / cell-gradient training. Forces
            (``grad(E, positions)``) are unaffected (positions enter per call).

    Returns
    -------
    torch.Tensor
        Per-atom :math:`(N,)` :math:`\text{float64}` (single) or
        :math:`(N_\text{total},)` (batched, flat across systems) on
        ``positions.device``. Does **not** subtract any self-interaction
        correction — the caller combines this with the real-space and
        self / background terms to get the full Ewald total.
        Call ``.sum()`` for the total or
        ``torch.zeros(B).scatter_add(0, batch_idx, E)`` for per-system totals.
    """
    is_batch = batch_idx is not None
    if is_batch:
        if cell.ndim != 3 or cell.shape[-2:] != (3, 3):
            raise ValueError(f"batched cell must be (B, 3, 3), got {tuple(cell.shape)}")
    elif cell.shape != (3, 3):
        raise ValueError(f"cell must be (3, 3) or (B, 3, 3), got {tuple(cell.shape)}")
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"positions must be (N, 3), got {tuple(positions.shape)}")
    if multipole_moments.ndim != 2 or multipole_moments.shape[0] != positions.shape[0]:
        raise ValueError(
            "multipole_moments must be (N, (l_max+1)^2) matching positions[0]; "
            f"got {tuple(multipole_moments.shape)}"
        )
    if is_batch and batch_idx.shape[0] != positions.shape[0]:
        raise ValueError(
            f"batch_idx must match N_total={positions.shape[0]}, "
            f"got {tuple(batch_idx.shape)}"
        )
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if cache is None and (k_cutoff is None or k_cutoff <= 0.0):
        raise ValueError(
            "Either a pre-built cache or a positive k_cutoff "
            f"must be supplied (got k_cutoff={k_cutoff})."
        )

    norm_mode = _resolve_norm_mode(normalize)
    # Packed e3nn moments -> l<=1 SCF-step block + Cartesian l=2 channel.
    source_feats, quadrupoles, l_max = split_packed_for_kernels(multipole_moments)

    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
        multipole_scf_step_energy,
    )

    if cache is None:
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
            prepare_multipole_scf_cache,
        )

        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=[sigma],
            k_cutoff=k_cutoff,
            l_max=l_max,
            density_normalize=norm_mode,
            feature_normalize=norm_mode,
            alpha=alpha,
            device=positions.device,
        )
    else:
        # Caller-supplied cache: cheap structural checks (the caller owns
        # matching cell/sigma/alpha; see the cache= warning in the docstring).
        if cache.is_batched != is_batch:
            raise ValueError(
                f"cache.is_batched={cache.is_batched} does not match the call "
                f"(batch_idx {'set' if is_batch else 'None'}); build the cache "
                "from a (B, 3, 3) cell for batched runs and a (3, 3) cell "
                "otherwise."
            )
        if cache.l_max < l_max:
            raise ValueError(
                f"cache.l_max={cache.l_max} is below the moment order l_max="
                f"{l_max}; rebuild the cache with l_max>={l_max}."
            )
    # Return the raw reciprocal-space sum; the caller subtracts the Ewald
    # self-term alongside their real-space erfc contribution.
    return multipole_scf_step_energy(
        cache,
        positions,
        source_feats,
        batch_idx=batch_idx,
        include_self_interaction=True,
        quadrupoles=quadrupoles,
    )
