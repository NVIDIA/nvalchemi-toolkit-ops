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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    split_packed_for_kernels,
)
from nvalchemiops.torch.math import FIELD_CONSTANT, compute_overlap_constants
from nvalchemiops.torch.math.gto import NormMode, inv_cl

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


@dataclass(frozen=True)
class _MultipoleEnergyCache:
    """Source-only geometry state consumed by the SCF energy step."""

    cell: torch.Tensor
    k_vectors: torch.Tensor
    source_phi_hat: torch.Tensor
    per_k_factor: torch.Tensor
    source_overlap_constants: torch.Tensor
    volume: torch.Tensor
    source_coeff2: torch.Tensor | None
    l_max: int
    n_systems: int
    valid_k_counts: torch.Tensor | None

    @property
    def is_batched(self) -> bool:
        return self.cell.ndim == 3

    @property
    def device(self) -> torch.device:
        return self.k_vectors.device


def _prepare_explicit_energy_cache(
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    *,
    sigma: float,
    l_max: int,
    normalize: NormMode,
    valid_k_counts: torch.Tensor | None,
    source_overlap_constants: torch.Tensor | None,
    compute_overlap: bool,
    device: torch.device,
) -> _MultipoleEnergyCache:
    """Prepare only source Fourier state for caller-supplied k-vectors.

    This private cache is deliberately limited to the fields read by the
    energy step. In particular, this path does not prepare receiver features
    or enumerate per-system grids.
    """
    is_batch = cell.ndim == 3
    k_vectors = k_vectors.to(device=device, dtype=torch.float64)
    if is_batch:
        n_systems, n_k = k_vectors.shape[:2]
        if valid_k_counts is None:
            valid_k_counts = torch.full_like(k_vectors[:, 0, 0], n_k, dtype=torch.int32)
        else:
            if valid_k_counts.shape != (n_systems,):
                raise ValueError(
                    "valid_k_counts must have shape (B,), matching batched "
                    f"k_vectors; got {tuple(valid_k_counts.shape)}"
                )
            if valid_k_counts.device != device:
                raise ValueError(
                    f"valid_k_counts must live on device={device}, "
                    f"got {valid_k_counts.device}"
                )
            valid_k_counts = valid_k_counts.to(dtype=torch.int32)
        k_index = torch.arange(n_k, device=device)
        valid_k_mask = k_index.unsqueeze(0) < valid_k_counts.unsqueeze(1)
        k_vectors = torch.where(
            valid_k_mask.unsqueeze(-1), k_vectors, torch.zeros_like(k_vectors)
        ).contiguous()
        k_norm2 = (k_vectors * k_vectors).sum(dim=-1)
        flat_k_vectors = k_vectors.reshape(-1, 3)
        flat_k_norm2 = k_norm2.reshape(-1)
    else:
        n_systems = 1
        valid_k_counts = None
        valid_k_mask = None
        k_norm2 = (k_vectors * k_vectors).sum(dim=-1)
        flat_k_vectors = k_vectors
        flat_k_norm2 = k_norm2

    # SourcePhiHatFunction is a traceable Warp op chain with analytical
    # backward. Flattening B x K lets it handle padded batches in one launch.
    from nvalchemiops.torch.interactions.electrostatics.multipole_autograd_kernels import (
        SourcePhiHatFunction,
    )

    icl0 = inv_cl(sigma, 0, normalize)
    icl1 = inv_cl(sigma, 1, normalize) if l_max >= 1 else 1.0
    source_phi_hat = SourcePhiHatFunction.apply(
        flat_k_vectors, flat_k_norm2, sigma, icl0, icl1
    )
    if is_batch:
        source_phi_hat = source_phi_hat.reshape(n_systems, n_k, 4, 2)
        source_phi_hat = torch.where(
            valid_k_mask.unsqueeze(-1).unsqueeze(-1),
            source_phi_hat,
            torch.zeros_like(source_phi_hat),
        ).contiguous()

    safe_k2 = torch.where(k_norm2 == 0.0, torch.ones_like(k_norm2), k_norm2)
    per_k_factor = torch.where(
        k_norm2 == 0.0,
        torch.zeros_like(k_norm2),
        FIELD_CONSTANT / safe_k2,
    )
    if is_batch:
        per_k_factor = torch.where(
            valid_k_mask, per_k_factor, torch.zeros_like(per_k_factor)
        ).contiguous()

    source_coeff2 = None
    if l_max >= 2:
        from nvalchemiops.math.spherical_harmonics import Y00_COEFF

        coeff2_prefac = (
            -0.5
            * float(icl0)
            * (4.0 * math.pi * math.sqrt(math.pi / 2.0))
            * float(sigma) ** 3
            * float(Y00_COEFF)
        )
        source_coeff2 = coeff2_prefac * torch.exp(-0.5 * k_norm2 * float(sigma) ** 2)
        if is_batch:
            source_coeff2 = torch.where(
                valid_k_mask, source_coeff2, torch.zeros_like(source_coeff2)
            ).contiguous()

    if source_overlap_constants is not None:
        if source_overlap_constants.shape != (3,):
            raise ValueError(
                "source_overlap_constants must have shape (3,), got "
                f"{tuple(source_overlap_constants.shape)}"
            )
        if source_overlap_constants.device != device:
            raise ValueError(
                f"source_overlap_constants must live on device={device}, "
                f"got {source_overlap_constants.device}"
            )
        source_oc = source_overlap_constants.to(dtype=torch.float64)
    elif compute_overlap:
        overlap = compute_overlap_constants(
            max_L=l_max,
            sigma_source=sigma,
            sigmas_receive=[sigma],
            normalize_source=normalize,
            normalize_receive=normalize,
        )
        source_row = overlap[0].to(device=device, dtype=torch.float64)
        source_oc = torch.zeros(3, dtype=torch.float64, device=device)
        source_oc[: l_max + 1] = source_row
        if l_max >= 2:
            # Cartesian-Frobenius quadrupole norm contributes the angular 3/2 factor.
            source_oc[2] *= 1.5
    else:
        source_oc = torch.zeros(3, dtype=torch.float64, device=device)

    volume = torch.det(cell.to(device=device, dtype=torch.float64)).abs()
    return _MultipoleEnergyCache(
        cell=cell,
        k_vectors=k_vectors,
        source_phi_hat=source_phi_hat,
        per_k_factor=per_k_factor,
        source_overlap_constants=source_oc,
        volume=volume,
        source_coeff2=source_coeff2,
        l_max=l_max,
        n_systems=n_systems,
        valid_k_counts=valid_k_counts,
    )


def multipole_electrostatic_energy(
    positions: torch.Tensor,
    multipole_moments: torch.Tensor,
    cell: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None = None,
    sigma: float,
    k_cutoff: float | None = None,
    k_vectors: torch.Tensor | None = None,
    valid_k_counts: torch.Tensor | None = None,
    source_overlap_constants: torch.Tensor | None = None,
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
    Batched mode can generate a grid from ``k_cutoff`` or consume explicit
    padded ``k_vectors`` of shape ``(B, K, 3)`` with optional
    ``valid_k_counts``.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions, shape ``(N, 3)`` or ``(N_total, 3)`` (flat across
        systems in the batched case), ``float32`` or ``float64``.
    multipole_moments : torch.Tensor
        Packed per-atom multipole moments, shape ``(N, (l_max+1)**2)``,
        in e3nn spherical layout: ``[q]`` (l_max=0), ``[q, mu_y, mu_z, mu_x]``
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
        Pre-computed k-grid, shape ``(N_k, 3)`` (single) or ``(B, K, 3)``
        (batched), ``float64``. **Must include
        ``(0, 0, 0)`` as the first row** — the kernel's ``V(k=0) = 0``
        convention expects the origin explicitly and downstream indexing
        assumes row 0 is it. Pass this when amortizing k-vector generation
        across many energy evaluations for the same geometry (MD steps at
        fixed cell, SCF iterations, benchmark loops). Must live on
        ``positions.device``. Explicit vectors select the source-only energy
        path, which avoids grid generation and receiver Fourier setup.
    valid_k_counts : torch.Tensor, optional, shape (B,), int32
        Number of valid rows in each padded batched k-grid. Rows at indices
        ``>= valid_k_counts[b]`` are masked from the reciprocal sum. Defaults
        to ``K`` for every system. Only valid with batched ``k_vectors``.
    source_overlap_constants : torch.Tensor, optional, shape (3,), float64
        Precomputed source self-overlap coefficients for l=0, l=1, and l=2.
        Supplying this tensor skips overlap quadrature; the l=2 entry includes
        the Cartesian-Frobenius angular factor. If omitted while self-interaction
        is subtracted, the coefficients are computed with tensor quadrature.
        Symbolic tracing requires static ``sigma`` and normalization metadata.
        Callers that reuse a prepared coefficient tensor can amortize this
        calculation across repeated calls.
    normalize : NormMode | int | str
        Normalization convention for the density basis. Defaults to
        ``NormMode.MULTIPOLES`` (the only physically meaningful choice for
        source moments; the other modes exist for debugging / cross-checks).
    include_self_interaction : bool
        If False (default), subtracts :math:`0.5 \cdot E_{\text{self}}` where
        :math:`E_\text{self} = \sum_i \mathrm{oc}[0]\,q_i^2 +
        \mathrm{oc}[1]\,|\boldsymbol{\mu}_i|^2` and ``oc`` comes from
        :func:`nvalchemiops.torch.math.compute_overlap_constants`.

    Notes
    -----

    **Cell gradients with explicit k-vectors**

    The cell-volume contribution is differentiated through ``cell``. The
    reciprocal-vector contribution is differentiated through ``k_vectors``;
    include its dependence on the cell by constructing ``k_vectors``
    differentiably from ``cell`` inside the traced function. A fixed grid
    constructed independently of ``cell`` omits that reciprocal-vector
    dependence.

    Returns
    -------
    torch.Tensor
        Per-atom :math:`(N,)` :math:`\text{float64}` (single) or
        :math:`(N_\text{total},)` (batched, flat across systems) on
        ``positions.device``. Call ``.sum()`` for the total energy or
        ``E.new_zeros(B).scatter_add(0, batch_idx, E)`` for per-system totals;
        forces/stress/charge-grads flow from ``grad(E.sum(), ...)``.
        Autograd-connected to ``positions`` and ``multipole_moments``.
    """
    is_batch = batch_idx is not None
    if is_batch:
        if cell.ndim != 3 or cell.shape[-2:] != (3, 3):
            raise ValueError(f"batched cell must be (B, 3, 3), got {tuple(cell.shape)}")
        if k_vectors is not None and (
            k_vectors.ndim != 3
            or k_vectors.shape[0] != cell.shape[0]
            or k_vectors.shape[-1] != 3
        ):
            raise ValueError(
                "batched k_vectors must have shape (B, K, 3), matching cell; "
                f"got {tuple(k_vectors.shape)}"
            )
    elif cell.shape != (3, 3):
        raise ValueError(f"cell must be (3, 3) or (B, 3, 3), got {tuple(cell.shape)}")
    elif k_vectors is not None and (k_vectors.ndim != 2 or k_vectors.shape[-1] != 3):
        raise ValueError(
            f"single-system k_vectors must be (N_k, 3), got {tuple(k_vectors.shape)}"
        )
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
    if valid_k_counts is not None and (not is_batch or k_vectors is None):
        raise ValueError(
            "valid_k_counts requires batched explicit k_vectors and batch_idx."
        )
    if source_overlap_constants is not None and k_vectors is None:
        raise ValueError(
            "source_overlap_constants is only used with explicit k_vectors."
        )
    if k_vectors is not None and k_vectors.device != positions.device:
        raise ValueError(
            f"k_vectors must live on positions.device={positions.device}, "
            f"got {k_vectors.device}"
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

    if k_vectors is not None:
        cache = _prepare_explicit_energy_cache(
            cell,
            k_vectors,
            sigma=sigma,
            l_max=l_max,
            normalize=norm_mode,
            valid_k_counts=valid_k_counts,
            source_overlap_constants=source_overlap_constants,
            compute_overlap=(
                not include_self_interaction and source_overlap_constants is None
            ),
            device=positions.device,
        )
    else:
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=[sigma],
            k_cutoff=k_cutoff,
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
    at the same :math:`\alpha` to assemble the full Ewald-split Coulomb sum.

    **Single-system vs batched dispatch**

    Mirrors :func:`multipole_ewald_summation`: pass ``cell`` of shape
    ``(3, 3)`` (single) or ``(B, 3, 3)`` (batched) and use ``batch_idx`` to
    select the batched path (returns per-atom :math:`(N_\text{total},)`).
    Both single and batched modes build their k-grid from ``k_cutoff`` (or
    reuse a pre-built ``cache``).

    **Cached-cell gradients**

    A pre-built cache holds a fixed (detached) k-grid and volume, so
    ``grad(E, cell)`` (stress) does **not** flow through it. Pass ``cache=None``
    for stress or cell-gradient training. Forces
    (``grad(E, positions)``) are unaffected because positions enter per call.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions, shape ``(N, 3)`` or ``(N_total, 3)`` (flat across
        systems in the batched case), ``float32`` or ``float64``.
    multipole_moments : torch.Tensor
        Packed per-atom multipole moments, shape ``(N, (l_max+1)**2)``,
        in e3nn spherical layout: ``[q]`` (l_max=0), ``[q, mu_y, mu_z, mu_x]``
        (l_max=1), or the l_max=1 block plus the five traceless l=2 channels
        (l_max=2). The l=2 quadrupole is expanded to the Cartesian symmetric
        ``(N, 3, 3)`` form and threaded through the SCF-cache Q channel.
    cell : torch.Tensor
        Unit-cell matrix (lattice vectors as rows), shape ``(3, 3)``, or
        ``B`` per-system cells ``(B, 3, 3)`` (batched).
    sigma : float
        Density-basis Gaussian width. Used for both the source GTO basis and
        the self-interaction overlap (matches ``GTOElectrostaticEnergy``).
    k_cutoff : float, optional
        Maximum ``|k|`` to include in the reciprocal-space sum. Required when
        ``cache`` is not supplied; ignored when a pre-built cache is passed.
    normalize : NormMode | int | str
        Normalization convention for the density basis. Defaults to
        ``NormMode.MULTIPOLES`` (the only physically meaningful choice for
        source moments; the other modes exist for debugging / cross-checks).
        Still resolved when ``cache`` is supplied.
    batch_idx : torch.Tensor, optional, shape (N_total,), int32
        Per-atom system index (expected sorted). Required when ``cell`` is
        ``(B, 3, 3)``; must be ``None`` for a single ``(3, 3)`` cell.
    alpha : float
        Ewald splitting parameter (must be positive). The caller's
        real-space kernel should use the same ``alpha``.
    cache : MultipoleSCFCache, optional
        Pre-built reciprocal cache (from :func:`prepare_multipole_scf_cache`)
        holding the position-independent k-grid / GTO-Fourier (``phi_hat``) /
        per-k-factor tables. When given, reciprocal-state rebuild is skipped
        (MD / inference steady state). ``cell`` shape, positive ``sigma`` and
        ``alpha``, and ``normalize`` are still validated or resolved before
        cache use; ``k_cutoff`` is ignored because the cache already encodes
        the k-grid. The caller owns matching the cache to the system.

    Returns
    -------
    torch.Tensor
        Per-atom :math:`(N,)` :math:`\text{float64}` (single) or
        :math:`(N_\text{total},)` (batched, flat across systems) on
        ``positions.device``. Does **not** subtract any self-interaction
        correction — the caller combines this with the real-space and
        self / background terms to get the full Ewald total.
        Call ``.sum()`` for the total or
        ``E.new_zeros(B).scatter_add(0, batch_idx, E)`` for per-system totals.

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
