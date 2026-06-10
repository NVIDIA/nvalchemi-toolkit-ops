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
SCF precomputation cache for direct-k-space multipole electrostatics.

Exposes :class:`MultipoleSCFCache` and the builder
:func:`prepare_multipole_scf_cache`. The cache materializes every
geometry-only tensor the energy and feature step functions consume:
structure-factor tables, source / receiver :math:`\hat\phi`, Coulomb and
projection per-k factors, overlap constants, and output LUTs. Per-step
calls in an SCF or MD loop then only need to feed their current
``(charges, dipoles)`` through a single Warp custom op.

The work is split into two phases:

* :func:`prepare_multipole_scf_cache` runs once per geometry, doing the
  geometry-only Warp kernel launches and returning an immutable dataclass
  of device tensors.
* ``multipole_scf_step_energy`` / ``multipole_scf_step_features`` each make
  a single ``@warp_custom_op`` call that reads the cache tensors and the
  current moments, runs the :math:`\rho(k)` + :math:`V(k)` + energy /
  feature pipeline, and returns the scalar / feature tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_autograd_kernels import (
    ReceiverPhiHatFunction,
    ReceiverPhiHatQuadrupoleFunction,
    SourcePhiHatFunction,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_electrostatics import (
    _prepend_origin,
    _resolve_norm_mode,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_features import (
    _build_out_col_lut,
)
from nvalchemiops.torch.math import FIELD_CONSTANT, compute_overlap_constants
from nvalchemiops.torch.math.gto import NormMode, inv_cl


@dataclass(frozen=True)
class MultipoleSCFCache:
    r"""Frozen bundle of geometry-only direct-k-space tensors for the SCF step functions.

    Represents a **single system** (``n_systems == 1``) or a **batch** of
    ``B`` systems (``n_systems == B``) under one unified dataclass. Built by
    :func:`prepare_multipole_scf_cache` from a ``(3, 3)`` (single) or
    ``(B, 3, 3)`` (batched) ``cell``; consumed by ``multipole_scf_step_energy``
    / ``multipole_scf_step_features`` (single) and their batched branches.

    For the batched case every per-k tensor carries a leading-``B`` layout and
    is uniform-shape ``(B, K_max, ...)`` with zero padding beyond each system's
    ``K_b`` valid k-vectors. Pad rows get ``k_vectors = 0`` (so ``k_α = 0`` in
    any k-weighted sum), ``per_k_factor = 0`` (zero Coulomb contribution), and
    ``k_factor_proj = 0`` (zero feature-projection weight); that is sufficient
    to make every kernel in the direct-k-space pipeline ignore pad rows without
    in-kernel branching. ``valid_k_counts`` records each system's ``K_b``.

    All tensor fields are ``float64`` on the same device.

    Attributes
    ----------
    k_vectors : torch.Tensor, float64
        Single: ``(N_k, 3)`` reciprocal-lattice k-grid with ``(0, 0, 0)`` at
        row 0. Batched: ``(B, K_max, 3)`` per-system grids zero-padded to
        ``K_max``.
    k_norm2 : torch.Tensor, float64
        ``|k|²``. Single ``(N_k,)`` / batched ``(B, K_max)``.
    source_phi_hat : torch.Tensor, float64
        Source-basis GTO Fourier coefficients
        :math:`\hat\phi_{l,m}^{\sigma}(\mathbf{k})`. Single ``(N_k, 4, 2)`` /
        batched ``(B, K_max, 4, 2)`` (pad rows zeroed).
    receiver_phi_hat : torch.Tensor, float64
        Receiver-basis GTO Fourier coefficients across all receiver σ. Single
        ``(N_k, N_σ, 4|9, 2)`` / batched ``(B, K_max, N_σ, 4|9, 2)``.
    per_k_factor : torch.Tensor, float64
        Coulomb multiplier ``FIELD_CONSTANT / k²`` (or the Ewald damped form)
        with the ``k = 0`` entry zeroed. Single ``(N_k,)`` / batched
        ``(B, K_max)`` (pad + ``k = 0`` rows zeroed).
    k_factor_proj : torch.Tensor, float64
        Feature projection weight: ``0.5`` at real ``k = 0``, ``1`` at real
        nonzero k, ``0`` at pad rows. Single ``(N_k,)`` / batched ``(B, K_max)``.
    source_overlap_constants : torch.Tensor, shape (3,), float64
        Per-l self-overlap constants for the source basis (l=0, l=1, l=2),
        shared across the batch.
    feature_overlap_constants : torch.Tensor, shape (N_σ, 2), float64
        Per-(σ, l) self-overlap constants for the receiver basis, shared
        across the batch. The ``l=1`` column is zeroed when ``l_max == 0``.
    out_col_lut_natural : torch.Tensor, shape (N_σ, 4|9), int32
        Natural row-major output LUT for the feature projection kernel.
    out_col_lut_permuted : torch.Tensor, shape (N_σ, 4|9), int32
        Permuted output LUT for the feature projection kernel.
    out_col_inv_perm : torch.Tensor, shape (N_σ · (feature_max_l+1)**2,), int64
        Precomputed inverse permutation ``argsort(out_col_lut_permuted)`` that
        maps the natural feature layout to the reference permuted flat layout.
        Cached here (it is position-independent) so the feature step does not
        re-run ``torch.argsort`` every call.
    volume : torch.Tensor, float64
        ``|det(cell)|``. Single shape ``()`` / batched ``(B,)``.
    cell : torch.Tensor, float64
        The original unit-cell matrix/matrices. Single ``(3, 3)`` / batched
        ``(B, 3, 3)``.
    n_systems : int
        Number of systems: ``1`` for single, ``B`` for batched.
    valid_k_counts : torch.Tensor or None
        Batched only: ``(B,)`` int32 of per-system valid k-counts ``K_b``
        (``K_max = valid_k_counts.max()``). ``None`` for the single-system
        cache.
    sigma : float
        Density-side Gaussian width.
    alpha : float or None
        Ewald splitting parameter the cache was built with. ``None`` selects
        the direct-k-space Coulomb factor ``per_k_factor = F / k^2``; a positive
        value selects the Ewald-damped reciprocal-space factor
        ``F exp(-k^2/(4 alpha^2)) / k^2``.
    receiver_sigmas : tuple of float
        Receiver (feature) :math:`\sigma` widths, as an immutable tuple.
    l_max : int
        Effective source multipole order this cache was built for (``0``
        for charges-only, ``1`` otherwise). Used by the step functions for
        bookkeeping; the kernels always run the l_max=1 path with zeros for
        missing components.
    density_normalize, feature_normalize : NormMode
        Normalization modes used when the cache was built, stored so
        ``multipole_scf_step_*`` can validate consistency.
    """

    k_vectors: torch.Tensor
    k_norm2: torch.Tensor
    source_phi_hat: torch.Tensor
    receiver_phi_hat: torch.Tensor
    per_k_factor: torch.Tensor
    k_factor_proj: torch.Tensor
    source_overlap_constants: torch.Tensor
    feature_overlap_constants: torch.Tensor
    out_col_lut_natural: torch.Tensor
    out_col_lut_permuted: torch.Tensor
    out_col_inv_perm: torch.Tensor
    volume: torch.Tensor
    cell: torch.Tensor
    sigma: float
    alpha: float | None
    receiver_sigmas: tuple[float, ...]
    l_max: int
    density_normalize: NormMode
    feature_normalize: NormMode
    n_systems: int = 1
    """Number of systems represented: ``1`` (single) or ``B`` (batched)."""
    valid_k_counts: torch.Tensor | None = None
    """Batched only: ``(B,)`` int32 per-system valid k-counts. ``None`` single."""
    feature_max_l: int = 1
    """Receiver feature angular cap, independent of ``l_max`` (the source cap).
    ``receiver_phi_hat`` is ``(..., 4, 2)`` for ``feature_max_l <= 1`` and
    ``(..., 9, 2)`` for ``feature_max_l == 2``; feature output width is
    ``(feature_max_l + 1)**2`` per σ."""
    source_coeff2: torch.Tensor | None = None
    """Cartesian-quadrupole per-k coefficient ``coeff2(k) = -0.5*phi0(k)``.
    Single ``(N_k,)`` / batched ``(B, K_max)`` float64. ``None`` when
    ``l_max < 2``. Geometry-only (depends on k + σ, not moments); consumed by
    the l=2 reciprocal channel."""
    feature_overlap_l2: torch.Tensor | None = None
    """l=2 receiver self-overlap constant, shape ``(N_σ,)`` float64, shared
    across the batch. ``None`` when ``feature_max_l < 2``. Kept separate from
    the ``(N_σ, 2)`` ``feature_overlap_constants`` so the l<=1 projection
    kernel's shape contract is unchanged."""

    @property
    def is_batched(self) -> bool:
        """``True`` when this cache holds a batch (``cell`` is ``(B, 3, 3)``)."""
        return self.cell.ndim == 3

    @property
    def batch_size(self) -> int:
        """Number of systems in the batch (``n_systems``; ``1`` if single)."""
        return self.n_systems

    @property
    def n_k(self) -> int:
        """Number of k-vectors in the (single-system) grid, including origin row 0."""
        return int(self.k_vectors.shape[-2 if self.is_batched else 0])

    @property
    def n_k_max(self) -> int:
        """Padded per-system k-vector count ``K_max`` (batched layout)."""
        return int(self.k_vectors.shape[1]) if self.is_batched else self.n_k

    @property
    def n_sigma(self) -> int:
        """Number of receiver σ widths."""
        return len(self.receiver_sigmas)

    @property
    def device(self) -> torch.device:
        """Device all tensors live on."""
        return self.k_vectors.device


def prepare_multipole_scf_cache(
    cell: torch.Tensor,
    *,
    sigma: float,
    receiver_sigmas: list[float] | tuple[float, ...] | torch.Tensor,
    k_cutoff: float | None = None,
    k_vectors: torch.Tensor | None = None,
    l_max: int = 1,
    feature_max_l: int = 1,
    density_normalize: NormMode | int | str = NormMode.MULTIPOLES,
    feature_normalize: NormMode | int | str = NormMode.RECEIVER,
    alpha: float | None = None,
    device: torch.device | str | None = None,
) -> MultipoleSCFCache:
    r"""Build a :class:`MultipoleSCFCache` from the position-independent inputs.

    Runs the position-independent direct-k-space geometry kernels
    (``eval_gto_fourier_dipole`` for the source basis,
    ``eval_receiver_gto_fourier_dipole`` for the receiver basis) and
    precomputes the per-k and per-:math:`\sigma` factor tables that the step
    functions consume.

    The structure-factor table (cos/sin) is not part of the cache: it
    depends on atomic positions, which flow through
    :class:`MultipoleRhoFunction` with autograd, so the table is recomputed
    from positions on every step to wire up the position gradient.

    Single-system vs batched dispatch
    ---------------------------------
    ``cell`` of shape ``(3, 3)`` builds a single-system cache. ``cell`` of
    shape ``(B, 3, 3)`` builds a batched cache (``n_systems == B``) whose
    per-k tensors carry a leading-``B`` zero-padded layout; in that case
    ``k_cutoff`` is required (a pre-generated ``k_vectors`` is not
    supported for the batched build).

    Parameters
    ----------
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Single unit cell, or ``B`` per-system unit cells (batched build).
    sigma : float
        Density-basis Gaussian width.
    receiver_sigmas : list / tuple / 1-D tensor of floats
        Multi-σ receiver widths. Must be non-empty.
    k_cutoff, k_vectors
        Same semantics as the step functions: either pass
        ``k_cutoff`` to generate the k-grid internally (with origin
        prepended), or pass a pre-generated ``k_vectors`` tensor
        (``(N_k, 3)`` float64 with origin at row 0). The batched build
        (``cell`` ``(B, 3, 3)``) requires ``k_cutoff``.
    l_max : int
        Source multipole order the cache is being built for. ``0`` or
        ``1``. Only affects the ``feature_overlap_constants`` layout
        (the ``l=1`` column is zeroed when ``l_max = 0``); all other
        tensors are ``l_max`` independent.
    feature_max_l : int, default 1
        Receiver feature angular cap (independent of the source ``l_max``).
        ``0``, ``1``, or ``2``. Selects the receiver :math:`\hat\phi` width
        (``(..., 4, 2)`` for :math:`\le 1`, ``(..., 9, 2)`` for ``2``) and the
        feature output width ``(feature_max_l + 1)**2`` per σ.
    density_normalize, feature_normalize : NormMode | int | str
        Normalization conventions.
    alpha : float, optional
        Ewald splitting parameter. ``None`` (default) selects the
        direct-k-space Coulomb factor :math:`\text{per\_k\_factor} = F / k^2`
        with the origin zeroed. A positive value selects the Ewald-damped
        reciprocal-space factor:
        :math:`\text{per\_k\_factor} = F \exp(-k^2/(4\alpha^2)) / k^2` with
        the origin zeroed, the Gaussian-smoothed reciprocal-space half of an
        Ewald split, paired with a real-space erfc contribution to assemble
        the full Coulomb sum.
    device : torch.device | str, optional
        Device for all cache tensors. Defaults to ``cell.device``.

    Returns
    -------
    MultipoleSCFCache
        Frozen dataclass. All tensors live on the resolved device.
    """
    # Batched build: dispatch to the (B, 3, 3) branch.
    if cell.ndim == 3:
        return _prepare_multipole_scf_cache_batch(
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            k_cutoff=k_cutoff,
            l_max=l_max,
            feature_max_l=feature_max_l,
            density_normalize=density_normalize,
            feature_normalize=feature_normalize,
            alpha=alpha,
            device=device,
        )

    # --- Validation ---------------------------------------------------------
    if cell.shape != (3, 3):
        raise ValueError(f"cell must be (3, 3) or (B, 3, 3), got {tuple(cell.shape)}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if l_max not in (0, 1, 2):
        raise ValueError(f"l_max must be 0, 1, or 2, got {l_max}")
    if feature_max_l not in (0, 1, 2):
        raise ValueError(f"feature_max_l must be 0, 1, or 2, got {feature_max_l}")
    if alpha is not None and alpha <= 0.0:
        raise ValueError(f"alpha, when given, must be positive, got {alpha}")
    if k_vectors is None and (k_cutoff is None or k_cutoff <= 0.0):
        raise ValueError(
            "Either k_vectors must be supplied, or k_cutoff must be a "
            f"positive float (got k_cutoff={k_cutoff})."
        )

    if isinstance(receiver_sigmas, torch.Tensor):
        sigmas_list = receiver_sigmas.detach().cpu().to(torch.float64).tolist()
    else:
        sigmas_list = [float(s) for s in receiver_sigmas]
    if len(sigmas_list) == 0:
        raise ValueError("receiver_sigmas must be non-empty")
    if any(s <= 0.0 for s in sigmas_list):
        raise ValueError(f"receiver_sigmas must all be positive, got {sigmas_list}")
    n_sigma = len(sigmas_list)

    if device is None:
        device = cell.device
    else:
        device = torch.device(device)
    density_mode = _resolve_norm_mode(density_normalize)
    feature_mode = _resolve_norm_mode(feature_normalize)

    # --- k-vectors ----------------------------------------------------------
    if k_vectors is None:
        cell_on_device = cell.to(device)
        k_vectors_half = generate_k_vectors_ewald_summation(
            cell_on_device, float(k_cutoff)
        )
        if k_vectors_half.ndim != 2:
            raise ValueError(
                "Batched cell shapes are not supported yet; pass a single (3, 3) cell."
            )
        k_vectors = _prepend_origin(k_vectors_half).to(dtype=torch.float64)
    else:
        if k_vectors.ndim != 2 or k_vectors.shape[-1] != 3:
            raise ValueError(
                f"k_vectors must be (N_k, 3), got {tuple(k_vectors.shape)}"
            )
        if k_vectors.device != device:
            raise ValueError(
                f"k_vectors must live on device={device}, got {k_vectors.device}"
            )
        k_vectors = k_vectors.to(dtype=torch.float64).contiguous()

    k_norm2 = (k_vectors * k_vectors).sum(dim=-1)
    volume = torch.det(cell.to(device)).abs().to(torch.float64)

    # Pure-torch closed form (not the once-differentiable Warp
    # SourcePhiHatFunction) so the k_vectors/k_norm2 -> source_phi_hat graph is
    # TWICE differentiable -> reciprocal stress-loss (∂²E/∂cell∂θ). Pay-once
    # O(N_k) setup, so torch here costs nothing on the per-atom hot path.
    icl0_source = inv_cl(sigma, 0, density_mode)
    icl1_source = inv_cl(sigma, 1, density_mode) if l_max >= 1 else 1.0
    source_phi_hat_t = SourcePhiHatFunction.apply(
        k_vectors, k_norm2, sigma, icl0_source, icl1_source
    )

    # Cartesian-quadrupole per-k coefficient coeff2(k) = -0.5 * phi0(k).
    # Computed in torch (not the detached Warp eval_gto_fourier_q) so it
    # carries the k_norm2 -> cell autograd graph for the l=2 reciprocal stress.
    if l_max >= 2:
        import math as _math

        from nvalchemiops.math.spherical_harmonics import Y00_COEFF as _Y00

        _coeff2_prefac = (
            -0.5
            * float(icl0_source)
            * (4.0 * _math.pi * _math.sqrt(_math.pi / 2.0))
            * float(sigma) ** 3
            * float(_Y00)
        )
        source_coeff2_t = _coeff2_prefac * torch.exp(-0.5 * k_norm2 * float(sigma) ** 2)
    else:
        source_coeff2_t = None

    inv_cl_receiver_t = torch.tensor(
        [
            [inv_cl(float(s), 0, feature_mode), inv_cl(float(s), 1, feature_mode)]
            for s in sigmas_list
        ],
        dtype=torch.float64,
        device=device,
    ).contiguous()
    sigmas_t = torch.tensor(
        sigmas_list, dtype=torch.float64, device=device
    ).contiguous()
    receiver_phi_hat_t = ReceiverPhiHatFunction.apply(
        k_vectors, k_norm2, sigmas_t, inv_cl_receiver_t
    )
    # When the receiver projects l=2, append the 5 real l=2 columns
    # (kernel-produced, cell-differentiable) -> (..., 9, 2).
    if feature_max_l >= 2:
        inv_cl_l2_recv_t = torch.tensor(
            [inv_cl(float(s), 2, feature_mode) for s in sigmas_list],
            dtype=torch.float64,
            device=device,
        ).contiguous()
        receiver_phi_hat_l2 = ReceiverPhiHatQuadrupoleFunction.apply(
            k_vectors, k_norm2, sigmas_t, inv_cl_l2_recv_t
        )  # (N_k, N_σ, 5, 2)
        receiver_phi_hat_t = torch.cat(
            [receiver_phi_hat_t, receiver_phi_hat_l2], dim=-2
        )  # (N_k, N_σ, 9, 2)

    # Direct-kspace (alpha=None): per_k_factor = FIELD_CONSTANT / k^2.
    # Ewald-reciprocal (alpha > 0):
    #   per_k_factor = FIELD_CONSTANT * exp(-k^2/(4 alpha^2)) / k^2.
    # Both zero the k=0 entry.
    safe_k2 = torch.where(k_norm2 == 0.0, torch.ones_like(k_norm2), k_norm2)
    if alpha is None:
        per_k_factor_nonzero = FIELD_CONSTANT / safe_k2
    else:
        gaussian_damp = torch.exp(-k_norm2 / (4.0 * float(alpha) ** 2))
        per_k_factor_nonzero = FIELD_CONSTANT * gaussian_damp / safe_k2
    per_k_factor = torch.where(
        k_norm2 == 0.0,
        torch.zeros_like(k_norm2),
        per_k_factor_nonzero,
    )
    # k_factor_proj: 0.5 at origin, 1 elsewhere.
    k_factor_proj = torch.where(
        k_norm2 == 0.0,
        torch.full_like(k_norm2, 0.5),
        torch.ones_like(k_norm2),
    )

    # source_overlap_constants for the energy self-interaction subtract,
    # which needs oc[l=0]*Sum q^2 + oc[l=1]*Sum |mu|^2.
    source_oc_np = compute_overlap_constants(
        max_L=l_max,
        sigma_source=sigma,
        sigmas_receive=[sigma],
        normalize_source=density_mode,
        normalize_receive=density_mode,
    )
    source_oc = torch.zeros(3, dtype=torch.float64, device=device)
    source_oc[0] = float(source_oc_np[0, 0])
    if l_max >= 1:
        source_oc[1] = float(source_oc_np[0, 1])
    if l_max >= 2:
        # x3/2: the Cartesian-Frobenius |Q|_F^2 energy-self needs the angular
        # (k.Q.k)^2 contraction factor on top of the bare overlap constant.
        # See _multipole_ewald_self_energy_per_atom.
        source_oc[2] = 1.5 * float(source_oc_np[0, 2])

    # feature_overlap_constants for the feature-step self-interaction subtract,
    # up to max(l_max, feature_max_l) so the receiver l=2 self constant exists
    # when decoupled (feature_max_l=2, l_max<2).
    feature_oc_np = compute_overlap_constants(
        max_L=max(l_max, feature_max_l),
        sigma_source=sigma,
        sigmas_receive=sigmas_list,
        normalize_source=density_mode,
        normalize_receive=feature_mode,
    )
    feature_oc = torch.zeros((n_sigma, 2), dtype=torch.float64, device=device)
    feature_oc[:, 0] = torch.as_tensor(
        feature_oc_np[:, 0], dtype=torch.float64, device=device
    )
    if l_max >= 1:
        feature_oc[:, 1] = torch.as_tensor(
            feature_oc_np[:, 1], dtype=torch.float64, device=device
        )
    # Separate l=2 receiver self-overlap column (N_σ,), kept out of the
    # (N_σ, 2) tensor so the l<=1 projection kernel's shape contract is intact.
    feature_oc_l2 = None
    if feature_max_l >= 2:
        feature_oc_l2 = torch.as_tensor(
            feature_oc_np[:, 2], dtype=torch.float64, device=device
        ).contiguous()

    out_col_lut_natural = _build_out_col_lut(
        n_sigma, permuted=False, device=device, feature_max_l=feature_max_l
    )
    out_col_lut_permuted = _build_out_col_lut(
        n_sigma, permuted=True, device=device, feature_max_l=feature_max_l
    )
    # Precompute the natural->permuted inverse permutation once (it only depends
    # on the position-independent output LUT), so the feature step reuses it
    # instead of re-running torch.argsort every call.
    out_col_inv_perm = torch.argsort(out_col_lut_permuted.flatten().long())

    return MultipoleSCFCache(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        source_phi_hat=source_phi_hat_t,
        receiver_phi_hat=receiver_phi_hat_t,
        per_k_factor=per_k_factor,
        k_factor_proj=k_factor_proj,
        source_overlap_constants=source_oc,
        feature_overlap_constants=feature_oc,
        out_col_lut_natural=out_col_lut_natural,
        out_col_lut_permuted=out_col_lut_permuted,
        out_col_inv_perm=out_col_inv_perm,
        volume=volume,
        cell=cell.to(device=device, dtype=torch.float64),
        sigma=float(sigma),
        alpha=float(alpha) if alpha is not None else None,
        receiver_sigmas=tuple(sigmas_list),
        l_max=int(l_max),
        feature_max_l=int(feature_max_l),
        density_normalize=density_mode,
        feature_normalize=feature_mode,
        source_coeff2=source_coeff2_t,
        feature_overlap_l2=feature_oc_l2,
    )


# =============================================================================
# Batched build (folded into the unified MultipoleSCFCache)
# =============================================================================


def _prepare_multipole_scf_cache_batch(
    cells: torch.Tensor,
    *,
    sigma: float,
    receiver_sigmas: list[float] | tuple[float, ...] | torch.Tensor,
    k_cutoff: float,
    l_max: int = 1,
    feature_max_l: int = 1,
    density_normalize: NormMode | int | str = NormMode.MULTIPOLES,
    feature_normalize: NormMode | int | str = NormMode.RECEIVER,
    alpha: float | None = None,
    device: torch.device | str | None = None,
) -> MultipoleSCFCache:
    r"""Build a batched :class:`MultipoleSCFCache` from ``B`` unit cells.

    Internal batched branch of :func:`prepare_multipole_scf_cache`. Every
    system in the batch gets its own k-vector grid generated via
    :func:`generate_k_vectors_ewald_summation` at the shared
    ``k_cutoff``. The per-system grids are padded to the max-K
    across the batch; padding is zero-filled in ``k_vectors``, and all
    dependent per-k tensors (``per_k_factor``, ``k_factor_proj``,
    ``source_phi_hat``, ``receiver_phi_hat``) have their pad rows
    explicitly zeroed so downstream kernels ignore them without
    masking.

    The per-batch :math:`\hat\phi` evaluation happens system-by-system
    through the existing :class:`SourcePhiHatFunction` /
    :class:`ReceiverPhiHatFunction` so cell-autograd still flows through each
    system's k-vectors to ``cells[b]``.

    Parameters
    ----------
    cells : torch.Tensor, shape (B, 3, 3)
        Per-system unit cells.
    sigma, receiver_sigmas, k_cutoff, l_max, feature_max_l,
    density_normalize, feature_normalize, alpha
        Same semantics as :func:`prepare_multipole_scf_cache`; the
        values apply uniformly to every system in the batch. ``alpha``
        (default ``None``) selects the direct-k-space vs the Ewald-damped
        reciprocal ``per_k_factor`` exactly as in the single-system builder.
    device : torch.device or str, optional
        Defaults to ``cells.device``.

    Returns
    -------
    MultipoleSCFCache
        Frozen dataclass with ``n_systems == B`` whose per-k tensors are
        shape ``(B, K_max, ...)`` with zero padding beyond per-system ``K_b``.
    """
    # --- Validation ---------------------------------------------------------
    if cells.ndim != 3 or cells.shape[-2:] != (3, 3):
        raise ValueError(f"cells must be (B, 3, 3), got {tuple(cells.shape)}")
    batch_size = int(cells.shape[0])
    if batch_size == 0:
        raise ValueError("cells must have at least one system (B >= 1)")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if l_max not in (0, 1, 2):
        raise ValueError(f"l_max must be 0, 1, or 2, got {l_max}")
    if feature_max_l not in (0, 1, 2):
        raise ValueError(f"feature_max_l must be 0, 1, or 2, got {feature_max_l}")
    if k_cutoff is None or k_cutoff <= 0.0:
        raise ValueError(f"k_cutoff must be a positive float, got {k_cutoff}")

    if isinstance(receiver_sigmas, torch.Tensor):
        sigmas_list = receiver_sigmas.detach().cpu().to(torch.float64).tolist()
    else:
        sigmas_list = [float(s) for s in receiver_sigmas]
    if len(sigmas_list) == 0:
        raise ValueError("receiver_sigmas must be non-empty")
    if any(s <= 0.0 for s in sigmas_list):
        raise ValueError(f"receiver_sigmas must all be positive, got {sigmas_list}")
    n_sigma = len(sigmas_list)

    if device is None:
        device = cells.device
    else:
        device = torch.device(device)
    density_mode = _resolve_norm_mode(density_normalize)
    feature_mode = _resolve_norm_mode(feature_normalize)

    cells_dev = cells.to(device=device, dtype=torch.float64)

    # Each system has its own valid-k count K_b; pad to K_max for uniform shapes.
    per_system_k: list[torch.Tensor] = []
    for b in range(batch_size):
        k_half_b = generate_k_vectors_ewald_summation(cells_dev[b], float(k_cutoff))
        if k_half_b.ndim != 2:
            raise ValueError(
                f"generate_k_vectors_ewald_summation returned rank-{k_half_b.ndim} "
                f"for system {b}; expected 2-D (N_k, 3)."
            )
        k_full_b = _prepend_origin(k_half_b).to(dtype=torch.float64)
        per_system_k.append(k_full_b)

    # Per-system k-counts are Python ints (materialized-tensor .shape[0]), so
    # take K_max host-side — no device round trip / .item() sync.
    k_counts = [int(k.shape[0]) for k in per_system_k]
    valid_k_counts = torch.tensor(k_counts, dtype=torch.int32, device=device)
    k_max = max(k_counts)

    # Pad rows are zero so they contribute nothing to any k-weighted sum.
    k_vectors = torch.zeros((batch_size, k_max, 3), dtype=torch.float64, device=device)
    for b in range(batch_size):
        k_b = per_system_k[b]
        k_vectors[b, : k_b.shape[0]] = k_b

    k_norm2 = (k_vectors * k_vectors).sum(dim=-1)  # (B, K_max)
    volume = torch.det(cells_dev).abs()  # (B,)

    # Pure-torch closed form per system (twice differentiable in cell -> stress-
    # loss; pay-once O(K) setup). Cell autograd flows through each system's
    # k-vectors independently. Pad rows are zeroed afterwards.
    icl0_source = inv_cl(sigma, 0, density_mode)
    icl1_source = inv_cl(sigma, 1, density_mode) if l_max >= 1 else 1.0

    source_phi_per_system: list[torch.Tensor] = []
    for b in range(batch_size):
        phi_b_padded = SourcePhiHatFunction.apply(
            k_vectors[b],
            k_norm2[b],
            sigma,
            icl0_source,
            icl1_source,
        )
        source_phi_per_system.append(phi_b_padded)
    source_phi_hat = torch.stack(source_phi_per_system, dim=0)  # (B, K_max, 4, 2)

    inv_cl_receiver_t = torch.tensor(
        [
            [inv_cl(float(s), 0, feature_mode), inv_cl(float(s), 1, feature_mode)]
            for s in sigmas_list
        ],
        dtype=torch.float64,
        device=device,
    ).contiguous()
    sigmas_t = torch.tensor(
        sigmas_list, dtype=torch.float64, device=device
    ).contiguous()

    receiver_phi_per_system: list[torch.Tensor] = []
    for b in range(batch_size):
        phi_b_padded = ReceiverPhiHatFunction.apply(
            k_vectors[b], k_norm2[b], sigmas_t, inv_cl_receiver_t
        )
        receiver_phi_per_system.append(phi_b_padded)
    receiver_phi_hat = torch.stack(
        receiver_phi_per_system, dim=0
    )  # (B, K_max, N_σ, 4, 2)

    # Append the 5 real l=2 receiver columns per system (kernel-produced,
    # cell-differentiable) -> (B, K_max, N_σ, 9, 2). Concat before the pad-mask
    # below so the full 9-col tensor is zeroed at pad rows in one shot.
    if feature_max_l >= 2:
        inv_cl_l2_recv_t = torch.tensor(
            [inv_cl(float(s), 2, feature_mode) for s in sigmas_list],
            dtype=torch.float64,
            device=device,
        ).contiguous()
        recv_l2_per_system: list[torch.Tensor] = []
        for b in range(batch_size):
            recv_l2_b = ReceiverPhiHatQuadrupoleFunction.apply(
                k_vectors[b], k_norm2[b], sigmas_t, inv_cl_l2_recv_t
            )  # (K_max, N_σ, 5, 2)
            recv_l2_per_system.append(recv_l2_b)
        receiver_phi_hat_l2 = torch.stack(recv_l2_per_system, dim=0)
        receiver_phi_hat = torch.cat(
            [receiver_phi_hat, receiver_phi_hat_l2], dim=-2
        )  # (B, K_max, N_σ, 9, 2)

    # valid_k_mask[b, k] = True iff k < valid_k_counts[b].
    k_indices = torch.arange(k_max, device=device)  # (K_max,)
    valid_k_mask = k_indices.unsqueeze(0) < valid_k_counts.unsqueeze(1)  # (B, K_max)

    # Zero phi_hat at pad rows. Real k=0 rows keep their nonzero l=0 value;
    # per_k_factor = 0 at k=0 zeros any contribution regardless.
    pad_mask_phi = valid_k_mask.unsqueeze(-1).unsqueeze(-1)  # (B, K_max, 1, 1)
    source_phi_hat = torch.where(
        pad_mask_phi, source_phi_hat, torch.zeros_like(source_phi_hat)
    )

    # Cartesian-quadrupole per-k coefficient coeff2(k) = -0.5 * phi0(k);
    # geometry-only per-k scalar. Pad rows zeroed via valid_k_mask.
    if l_max >= 2:
        import math as _math

        from nvalchemiops.math.spherical_harmonics import Y00_COEFF as _Y00

        _coeff2_prefac = (
            -0.5
            * float(icl0_source)
            * (4.0 * _math.pi * _math.sqrt(_math.pi / 2.0))
            * float(sigma) ** 3
            * float(_Y00)
        )
        source_coeff2 = _coeff2_prefac * torch.exp(
            -0.5 * k_norm2 * float(sigma) ** 2
        )  # (B, K_max)
        source_coeff2 = torch.where(
            valid_k_mask, source_coeff2, torch.zeros_like(source_coeff2)
        ).contiguous()
    else:
        source_coeff2 = None

    pad_mask_recv = valid_k_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    receiver_phi_hat = torch.where(
        pad_mask_recv, receiver_phi_hat, torch.zeros_like(receiver_phi_hat)
    )

    # per_k_factor: 0 at k=0 AND pad rows (pads carry k=0 via zero padding, so
    # the k_norm2 == 0 branch handles both). Direct k-space (alpha=None): F / k^2.
    # Ewald reciprocal (alpha>0): F * exp(-k^2/(4 alpha^2)) / k^2.
    safe_k2 = torch.where(k_norm2 == 0.0, torch.ones_like(k_norm2), k_norm2)
    if alpha is None:
        per_k_factor_nonzero = FIELD_CONSTANT / safe_k2
    else:
        if alpha <= 0.0:
            raise ValueError(f"alpha, when given, must be positive, got {alpha}")
        gaussian_damp = torch.exp(-k_norm2 / (4.0 * float(alpha) ** 2))
        per_k_factor_nonzero = FIELD_CONSTANT * gaussian_damp / safe_k2
    per_k_factor = torch.where(
        k_norm2 == 0.0,
        torch.zeros_like(k_norm2),
        per_k_factor_nonzero,
    )

    # k_factor_proj: 0.5 at REAL k=0, 1 elsewhere, 0 at pad rows. The
    # valid-k mask disambiguates real k=0 from pads (which also have k_norm2==0).
    kfp_base = torch.where(
        k_norm2 == 0.0,
        torch.full_like(k_norm2, 0.5),
        torch.ones_like(k_norm2),
    )
    k_factor_proj = torch.where(valid_k_mask, kfp_base, torch.zeros_like(kfp_base))

    # Overlap constants, shared across the batch.
    source_oc_np = compute_overlap_constants(
        max_L=l_max,
        sigma_source=sigma,
        sigmas_receive=[sigma],
        normalize_source=density_mode,
        normalize_receive=density_mode,
    )
    source_oc = torch.zeros(3, dtype=torch.float64, device=device)
    source_oc[0] = float(source_oc_np[0, 0])
    if l_max >= 1:
        source_oc[1] = float(source_oc_np[0, 1])
    if l_max >= 2:
        # x3/2: the Cartesian-Frobenius |Q|_F^2 energy-self angular factor.
        source_oc[2] = 1.5 * float(source_oc_np[0, 2])

    # Computed up to max(l_max, feature_max_l) so the receiver l=2 self
    # constant exists when decoupled (feature_max_l=2, l_max<2).
    feature_oc_np = compute_overlap_constants(
        max_L=max(l_max, feature_max_l),
        sigma_source=sigma,
        sigmas_receive=sigmas_list,
        normalize_source=density_mode,
        normalize_receive=feature_mode,
    )
    feature_oc = torch.zeros((n_sigma, 2), dtype=torch.float64, device=device)
    feature_oc[:, 0] = torch.as_tensor(
        feature_oc_np[:, 0], dtype=torch.float64, device=device
    )
    if l_max >= 1:
        feature_oc[:, 1] = torch.as_tensor(
            feature_oc_np[:, 1], dtype=torch.float64, device=device
        )
    # Separate l=2 receiver self-overlap column (N_σ,), shared across the batch.
    # Kept out of the (N_σ, 2) tensor to preserve the l<=1 contract.
    feature_oc_l2 = None
    if feature_max_l >= 2:
        feature_oc_l2 = torch.as_tensor(
            feature_oc_np[:, 2], dtype=torch.float64, device=device
        ).contiguous()

    # --- Output LUTs (shared) ----------------------------------------------
    out_col_lut_natural = _build_out_col_lut(
        n_sigma, permuted=False, device=device, feature_max_l=feature_max_l
    )
    out_col_lut_permuted = _build_out_col_lut(
        n_sigma, permuted=True, device=device, feature_max_l=feature_max_l
    )
    # Position-independent inverse permutation — cached once (see single-system
    # builder), so the batched feature step skips a per-call torch.argsort.
    out_col_inv_perm = torch.argsort(out_col_lut_permuted.flatten().long())

    return MultipoleSCFCache(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        source_phi_hat=source_phi_hat,
        receiver_phi_hat=receiver_phi_hat,
        per_k_factor=per_k_factor,
        k_factor_proj=k_factor_proj,
        source_overlap_constants=source_oc,
        feature_overlap_constants=feature_oc,
        out_col_lut_natural=out_col_lut_natural,
        out_col_lut_permuted=out_col_lut_permuted,
        out_col_inv_perm=out_col_inv_perm,
        volume=volume,
        cell=cells_dev,
        sigma=float(sigma),
        alpha=float(alpha) if alpha is not None else None,
        receiver_sigmas=tuple(sigmas_list),
        l_max=int(l_max),
        density_normalize=density_mode,
        feature_normalize=feature_mode,
        n_systems=batch_size,
        valid_k_counts=valid_k_counts,
        source_coeff2=source_coeff2,
        feature_max_l=int(feature_max_l),
        feature_overlap_l2=feature_oc_l2,
    )
