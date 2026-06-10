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
SCF step entry points for direct-k-space multipole electrostatics.

Exposes user-facing functions that consume a prebuilt
:class:`MultipoleSCFCache` and run one autograd-connected pipeline per
call:

* :func:`multipole_scf_step_energy` — per-atom periodic electrostatic
  energies :math:`(N,)` :math:`\text{float64}`.
* :func:`multipole_scf_step_features` — atom-centered features in
  the reference permuted flat :math:`(N_\text{atoms}, N_\sigma \cdot 4)`
  layout.

The wrapper validates, unpacks, and dispatches; the underlying autograd
Functions own the device work. Cache tensors and per-step moments cross
the boundary as typed torch tensors. A batched :class:`MultipoleSCFCache`
(built from a ``(B, 3, 3)`` cell) is selected with ``batch_idx``, and Ewald
variants reuse an Ewald cache for the reciprocal-space term.
"""

from __future__ import annotations

import math

import torch

from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    split_packed_for_kernels,
    split_source_feats,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
    MultipoleSCFCache,
)
from nvalchemiops.torch.math import FIELD_CONSTANT

_TWO_PI_6 = (2.0 * math.pi) ** 6


def _resolve_step_moments(
    source_feats: torch.Tensor,
    quadrupoles: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize step inputs to ``(l<=1 block, quadrupoles)``.

    Accepts either the e3nn l<=1 block (``(N, 1)`` / ``(N, 4)``) with the l=2
    moment passed via ``quadrupoles``, or a fully packed ``(N, 9)`` tensor,
    which is split here so the packed form never silently drops the l=2 block.
    """
    if source_feats.ndim == 2 and source_feats.shape[-1] == 9:
        if quadrupoles is not None:
            raise ValueError(
                "pass either a packed (N, 9) source_feats or the l<=1 block plus "
                "quadrupoles=, not both."
            )
        l1, quadrupoles, _ = split_packed_for_kernels(source_feats)
        return l1, quadrupoles
    return source_feats, quadrupoles


def _self_energy_op(
    cache: MultipoleSCFCache,
    charges: torch.Tensor,
    dipoles_cart: torch.Tensor | None,
    quadrupoles: torch.Tensor | None,
) -> torch.Tensor:
    """Per-atom self-energy ``Σ_l c_l moment_l^2`` via the shared Warp op.

    Wraps :func:`multipole_self_energy`, feeding the cache's overlap-constant
    coefficients and gating the l=1 / l=2 channels by presence. Returns the
    per-atom ``(N,)`` density; the caller reduces.
    """
    # Delayed import keeps the kernels module out of the import-time graph.
    from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
        multipole_self_energy,
    )

    oc = cache.source_overlap_constants
    return multipole_self_energy(
        charges,
        dipoles_cart,
        quadrupoles,
        oc[0].reshape(1),
        oc[1].reshape(1),
        oc[2].reshape(1),
    )


def _per_k_energy_op(
    cache: MultipoleSCFCache,
    rho: torch.Tensor,
) -> torch.Tensor:
    """Per-k energy ``2 f_k |rho_k|^2`` via the shared Warp op.

    ``rho`` is ``(K, 2)`` single / ``(B, K, 2)`` batched; returns ``(K,)`` /
    ``(B, K)``. The caller applies the ``0.5 V / (2pi)^6`` scale + reduction.
    """
    from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
        multipole_reciprocal_rho_energy,
    )

    return multipole_reciprocal_rho_energy(rho, cache.per_k_factor)


def _check_batch_dispatch(
    cache: MultipoleSCFCache, batch_idx: torch.Tensor | None
) -> None:
    """Validate the ``batch_idx`` / cache batched-ness pairing.

    Mirrors :func:`multipole_ewald_summation`: ``batch_idx`` is required iff
    the cache is batched (``(B, 3, 3)`` cell), and must be ``None`` for a
    single-system cache.
    """
    if batch_idx is not None and not cache.is_batched:
        raise ValueError(
            "batch_idx was provided but the cache is single-system; build the "
            "cache from a (B, 3, 3) cell for batched runs."
        )
    if batch_idx is None and cache.is_batched:
        raise ValueError(
            "cache is batched (built from a (B, 3, 3) cell) but batch_idx was "
            "not provided; pass batch_idx=... for batched runs."
        )


# =============================================================================
# Energy step
# =============================================================================


def multipole_scf_step_energy(
    cache: MultipoleSCFCache,
    positions: torch.Tensor,
    source_feats: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None = None,
    include_self_interaction: bool = False,
    quadrupoles: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Per-atom PBC electrostatic energies for a single or batched SCF step.

    Consumes the position-independent tensors in ``cache`` and the
    per-step ``(positions, source_feats)``. Returns per-atom
    :math:`(N,)` :math:`\text{float64}` (single) or
    :math:`(N_\text{total},)` (batched, flat across all systems).
    The result is autograd-connected to both inputs.

    The caller owns the reduction: ``.sum()`` for the total energy, or
    ``torch.zeros(B).scatter_add(0, batch_idx, E)`` for per-system totals.
    Forces, stress, and charge-grads are obtained via
    ``grad(E.sum(), ...)``. The per-atom energies sum to the same total
    bit-for-bit as the old scalar / ``(B,)`` convention.

    Parameters
    ----------
    cache : MultipoleSCFCache
        Prebuilt cache from :func:`prepare_multipole_scf_cache`. Holds
        the position-independent state (k-vectors, :math:`\hat\phi`,
        per-k factors, overlap constants, LUTs). Single or batched.
    positions : torch.Tensor, shape (N_atoms, 3) or (N_total, 3)
        Atomic positions (flat across systems in the batched case).
    source_feats : torch.Tensor, shape (N, (l_max + 1)**2)
        Packed per-atom moments in e3nn spherical layout. :math:`(N, 1)`
        for ``l_max=0``, :math:`(N, 4)` for ``l_max=1``
        (``[q, mu_y, mu_z, mu_x]``). Must match ``cache.l_max``.
    batch_idx : torch.Tensor, optional, shape (N_total,), int32
        Per-atom system index (expected sorted so atoms group by system).
        Required when ``cache`` is batched; must be ``None`` for a single
        system. Mirrors :func:`multipole_ewald_summation`'s convention.
    include_self_interaction : bool
        If ``False`` (default), subtract ``0.5 · E_self`` using
        ``cache.source_overlap_constants``. ``True`` returns the raw
        reciprocal sum (the Ewald path subtracts the self term itself).
    quadrupoles : torch.Tensor, optional, shape (N, 3, 3)
        Cartesian symmetric source quadrupole. When supplied, the additive
        :math:`\rho_Q(k)` channel and its :math:`l=2` self term are included
        (requires a cache built with source ``l_max>=2``). ``None`` (default)
        is the :math:`l_{max} \le 1` path.

    Returns
    -------
    torch.Tensor
        Per-atom :math:`(N,)` :math:`\text{float64}` (single) or
        :math:`(N_\text{total},)` (batched, flat across systems) on
        ``cache.device``. Autograd-connected to positions and source_feats.
        Call ``.sum()`` for the total energy or
        ``torch.zeros(B).scatter_add(0, batch_idx, E)`` for per-system totals;
        forces/stress/charge-grads flow from ``grad(E.sum(), ...)``.
    """
    _check_batch_dispatch(cache, batch_idx)
    if batch_idx is not None:
        return _multipole_scf_step_energy_batch(
            cache,
            positions,
            source_feats,
            batch_idx,
            include_self_interaction=include_self_interaction,
            quadrupoles=quadrupoles,
        )

    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"positions must be (N, 3), got {tuple(positions.shape)}")
    n_atoms = positions.shape[0]
    if source_feats.ndim != 2 or source_feats.shape[0] != n_atoms:
        raise ValueError(
            "source_feats must be (N_atoms, (l_max+1)^2) matching positions[0]; "
            f"got {tuple(source_feats.shape)}"
        )
    if positions.device != cache.device or source_feats.device != cache.device:
        raise ValueError(
            f"positions/source_feats must live on cache.device={cache.device}"
        )

    source_feats, quadrupoles = _resolve_step_moments(source_feats, quadrupoles)
    charges, dipoles_cart, l_max = split_source_feats(source_feats)
    dip = (
        dipoles_cart.contiguous()
        if dipoles_cart is not None
        else torch.zeros((n_atoms, 3), dtype=source_feats.dtype, device=cache.device)
    )

    # Delayed import registers the multipole_rho op chain and breaks the
    # multipole_autograd <-> multipole_scf_step import cycle. Forward and
    # (AOT-traced) backward are opaque custom ops -> 0 torch.compile breaks.
    import nvalchemiops.torch.interactions.electrostatics.multipole_autograd  # noqa: F401

    rho = torch.ops.nvalchemiops.multipole_rho(
        charges, dip, positions, cache.source_phi_hat, cache.k_vectors, cache.volume
    )
    # The Cartesian-quadrupole channel is additive in rho(k); the cross terms
    # then fall out of |rho|^2 automatically.
    if quadrupoles is not None:
        from nvalchemiops.torch.interactions.electrostatics.multipole_autograd import (
            MultipoleRhoQFunction,
        )

        if cache.source_coeff2 is None:
            raise ValueError(
                "quadrupoles requires a cache built with l_max>=2 "
                "(cache.source_coeff2 is None)."
            )
        if quadrupoles.shape != (n_atoms, 3, 3):
            raise ValueError(
                f"quadrupoles must be (N, 3, 3); got {tuple(quadrupoles.shape)}"
            )
        rho = rho + MultipoleRhoQFunction.apply(
            quadrupoles, positions, cache.source_coeff2, cache.k_vectors, cache
        )
    # The rho-assembly kernels bake the (2pi)^3/V scale with V detached.
    # Reintroduce its volume-grad via a value-preserving factor (== 1) so
    # dE/dcell captures rho's 1/V dependence.
    vol_ratio = cache.volume.detach() / cache.volume
    rho = rho * vol_ratio
    # Per-atom reciprocal energy via the spread-transpose gather: with the
    # per-k potential phi_hat = 2 f_k rho, ``E_i = scale * m_i · (Sᵀ phi_hat)_i``
    # and ``Σ_i E_i`` equals the collective ``scale · Σ_k 2 f_k |rho|²``
    # bit-for-bit. The gather restores the moment-grad's detached 1/V via the
    # value-preserving ``vol_ratio`` (== 1) so dE/dcell stays exact.
    phi_hat = (2.0 * cache.per_k_factor).unsqueeze(-1) * rho
    g = (
        torch.ops.nvalchemiops.multipole_rho_gather_t(
            phi_hat, positions, cache.source_phi_hat, cache.k_vectors, cache.volume
        )
        * vol_ratio
    )
    scale_const = 0.5 * cache.volume / _TWO_PI_6
    raw_per_atom = scale_const * (charges * g[:, 0] + (dip * g[:, [3, 1, 2]]).sum(-1))
    if quadrupoles is not None:
        g_q = (
            torch.ops.nvalchemiops.multipole_rho_q_gather_t(
                phi_hat, positions, cache.source_coeff2, cache.k_vectors, cache.volume
            )
            * vol_ratio
        )
        raw_per_atom = raw_per_atom + scale_const * (quadrupoles * g_q).sum((-1, -2))
    if include_self_interaction:
        return raw_per_atom.to(torch.float64)

    # Per-atom Σ_l c_l moment_l^2 via the shared self-energy op (per-atom, no
    # reduction — caller owns it). dipoles only when l_max>=1.
    dip_self = dipoles_cart if l_max == 1 else None
    atom_self = _self_energy_op(cache, charges, dip_self, quadrupoles)
    return (raw_per_atom - 0.5 * atom_self).to(torch.float64)


# =============================================================================
# Feature step
# =============================================================================


def multipole_scf_step_features(
    cache: MultipoleSCFCache,
    positions: torch.Tensor,
    source_feats: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None = None,
    include_self_interaction: bool = False,
    quadrupoles: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Atom-centered multipole features for a single or batched SCF step.

    Consumes the position-independent ``cache`` plus the per-step
    ``(positions, source_feats)`` and returns a
    :math:`(N, N_\sigma \cdot (\text{feature\_max\_l}+1)^2)` features tensor
    in the reference permuted flat layout (grouped by l-block). With a batched
    cache (``cache.is_batched`` and ``batch_idx`` supplied) the rows span all
    systems flat-packed.

    Parameters
    ----------
    cache : MultipoleSCFCache
        Single or batched.
    positions : torch.Tensor, shape (N_atoms, 3) or (N_total, 3)
    source_feats : torch.Tensor, shape (N, (l_max + 1)**2)
        Packed per-atom moments in e3nn spherical layout.
    batch_idx : torch.Tensor, optional, shape (N_total,), int32
        Per-atom system index (expected sorted). Required when ``cache`` is
        batched; must be ``None`` for a single system.
    include_self_interaction : bool
        If ``False`` (default), subtract the self-interaction correction
        using ``cache.feature_overlap_constants``.
    quadrupoles : torch.Tensor, optional, shape (N, 3, 3)
        Cartesian-quadrupole source moments. When supplied, the additive
        :math:`\rho_Q(k)` channel enriches the projected potential.
        Requires a cache built with source ``l_max>=2``. This is the
        source l=2 contribution, decoupled from the receiver
        ``feature_max_l`` (which lives on the cache and controls how many
        l-blocks are projected out).

    Returns
    -------
    torch.Tensor
        :math:`\text{float64}` on ``cache.device``, shape
        :math:`(N, N_\sigma \cdot (\text{feature\_max\_l}+1)^2)`
        in the reference permuted flat layout (grouped by l-block).
        Autograd-connected to ``positions`` and ``source_feats``.
    """
    _check_batch_dispatch(cache, batch_idx)
    if batch_idx is not None:
        return _multipole_scf_step_features_batch(
            cache,
            positions,
            source_feats,
            batch_idx,
            include_self_interaction=include_self_interaction,
            quadrupoles=quadrupoles,
        )

    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"positions must be (N, 3), got {tuple(positions.shape)}")
    n_atoms = positions.shape[0]
    if source_feats.ndim != 2 or source_feats.shape[0] != n_atoms:
        raise ValueError(
            "source_feats must be (N_atoms, (l_max+1)^2) matching positions[0]; "
            f"got {tuple(source_feats.shape)}"
        )
    if positions.device != cache.device or source_feats.device != cache.device:
        raise ValueError(
            f"positions/source_feats must live on cache.device={cache.device}"
        )

    source_feats, quadrupoles = _resolve_step_moments(source_feats, quadrupoles)
    charges, dipoles_cart, l_max = split_source_feats(source_feats)
    dip = (
        dipoles_cart.contiguous()
        if dipoles_cart is not None
        else torch.zeros((n_atoms, 3), dtype=source_feats.dtype, device=cache.device)
    )

    # Delayed import registers the multipole_rho + feature-projection op chains.
    import nvalchemiops.torch.interactions.electrostatics.multipole_autograd  # noqa: F401

    rho = torch.ops.nvalchemiops.multipole_rho(
        charges, dip, positions, cache.source_phi_hat, cache.k_vectors, cache.volume
    )
    # Additive Cartesian-quadrupole source channel; mirrors the energy step.
    if quadrupoles is not None:
        from nvalchemiops.torch.interactions.electrostatics.multipole_autograd import (
            MultipoleRhoQFunction,
        )

        if cache.source_coeff2 is None:
            raise ValueError(
                "quadrupoles requires a cache built with l_max>=2 "
                "(cache.source_coeff2 is None)."
            )
        if quadrupoles.shape != (n_atoms, 3, 3):
            raise ValueError(
                f"quadrupoles must be (N, 3, 3); got {tuple(quadrupoles.shape)}"
            )
        rho = rho + MultipoleRhoQFunction.apply(
            quadrupoles, positions, cache.source_coeff2, cache.k_vectors, cache
        )
    # The rho assembly bakes a detached (2pi)^3/V scale; the feature
    # projection cancels the (2pi)^3 but leaves an explicit 1/V whose
    # volume-grad would otherwise be dropped. The value-preserving ratio (== 1)
    # restores it so dfeatures/dcell is complete. cache.volume == det(cell).
    vol_ratio = cache.volume.detach() / cache.volume
    potential = cache.per_k_factor.unsqueeze(-1) * rho * vol_ratio  # (N_k, 2)

    # Raw features (N_atoms, N_sigma, 4), natural layout. Thread the
    # cell-differentiable receiver_phi_hat (l<=1 block) + k_vectors so
    # dfeatures/dcell flows.
    raw_features = torch.ops.nvalchemiops.multipole_project_raw_features(
        potential,
        positions,
        cache.receiver_phi_hat[:, :, :4, :],
        cache.k_vectors,
        cache.k_factor_proj,
    )

    # Self-interaction subtract. source_feats is already in the e3nn
    # [q, mu_y, mu_z, mu_x] layout; zero-pad when l_max=0 so the (N_sigma, 4)
    # broadcast works without a separate branch.
    if not include_self_interaction:
        if l_max == 0:
            src_lm = torch.cat(
                [
                    source_feats.to(torch.float64),
                    torch.zeros((n_atoms, 3), dtype=torch.float64, device=cache.device),
                ],
                dim=-1,
            )
        else:
            src_lm = source_feats.to(torch.float64)
        # Broadcast the overlap constant across the 3 l=1 components.
        oc = cache.feature_overlap_constants  # (N_sigma, 2)
        oc_per_lm = torch.cat(
            [oc[:, 0:1], oc[:, 1:2].expand(-1, 3)], dim=-1
        )  # (N_sigma, 4)
        self_corr = src_lm.unsqueeze(1) * oc_per_lm.unsqueeze(0)
        features_natural = raw_features - self_corr
    else:
        features_natural = raw_features

    # l=2 receiver block (decoupled from source l_max). The l=2 self-subtract
    # uses the e3nn 5-channel source l=2 derived from the Cartesian quadrupoles,
    # which is zero when the source has no l=2 moment.
    if cache.feature_max_l == 0:
        features_natural = features_natural[:, :, :1]  # l=0-only receiver
    elif cache.feature_max_l >= 2:
        raw_l2 = torch.ops.nvalchemiops.multipole_project_raw_features_quadrupole(
            potential,
            positions,
            cache.receiver_phi_hat[:, :, 4:9, :],
            cache.k_vectors,
            cache.k_factor_proj,
        )  # (N_atoms, N_σ, 5)
        if not include_self_interaction and quadrupoles is not None:
            from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
                cartesian_quadrupole_to_e3nn,
            )

            src_e3nn_l2 = cartesian_quadrupole_to_e3nn(
                quadrupoles.to(torch.float64)
            )  # (N_atoms, 5)
            self_corr_l2 = src_e3nn_l2.unsqueeze(1) * cache.feature_overlap_l2.reshape(
                1, cache.n_sigma, 1
            )
            features_l2 = raw_l2 - self_corr_l2
        else:
            features_l2 = raw_l2
        features_natural = torch.cat(
            [features_natural, features_l2], dim=-1
        )  # (N_atoms, N_sigma, 9)

    # Reference-format output: permuted flat, grouped by l-block.
    # Inverse-permute using the cached LUT.
    n_lm = (cache.feature_max_l + 1) ** 2
    flat_natural = features_natural.reshape(n_atoms, cache.n_sigma * n_lm)
    # Inverse-permute to the reference flat layout via the cache's precomputed
    # argsort (position-independent — no per-call sort).
    return flat_natural.index_select(-1, cache.out_col_inv_perm)


# =============================================================================
# Batched SCF step entry points
# =============================================================================


def _validate_batch_inputs(
    cache: MultipoleSCFCache,
    positions: torch.Tensor,
    source_feats: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """Shared input validation for batched step entries."""
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(
            f"positions must be (N_total, 3), got {tuple(positions.shape)}"
        )
    n_total = positions.shape[0]
    if source_feats.ndim != 2 or source_feats.shape[0] != n_total:
        raise ValueError(
            "source_feats must be (N_total, (l_max+1)^2) matching positions[0]; "
            f"got {tuple(source_feats.shape)}"
        )
    if batch_idx.ndim != 1 or batch_idx.shape[0] != n_total:
        raise ValueError(
            f"batch_idx must be (N_total={n_total},), got {tuple(batch_idx.shape)}"
        )
    if positions.device != cache.device or source_feats.device != cache.device:
        raise ValueError(
            f"positions/source_feats must live on cache.device={cache.device}, "
            f"got {positions.device} / {source_feats.device}"
        )


def _multipole_scf_step_energy_batch(
    cache: MultipoleSCFCache,
    positions: torch.Tensor,
    source_feats: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    include_self_interaction: bool = False,
    quadrupoles: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Batched branch of :func:`multipole_scf_step_energy`.

    Consumes a batched :class:`MultipoleSCFCache` and a flat
    :math:`(N_\text{total}, \dots)` set of per-atom inputs with a ``batch_idx``
    mapping atoms to systems. Returns per-atom
    :math:`(N_\text{total},)` :math:`\text{float64}` (flat across all systems),
    autograd-connected to ``positions`` and ``source_feats``.
    Call ``.sum()`` for the total energy or
    ``torch.zeros(B).scatter_add(0, batch_idx, E)`` for per-system totals.

    Parameters
    ----------
    cache : MultipoleSCFCache
        Batched cache (``cache.is_batched``).
    positions : torch.Tensor, shape (N_total, 3)
    source_feats : torch.Tensor, shape (N_total, (l_max+1)**2)
        Packed per-atom moments in e3nn spherical layout.
    batch_idx : torch.Tensor, shape (N_total,), int32
        System index per atom (expected sorted so atoms group by system).
    include_self_interaction : bool
        ``False`` (default): subtract per-system ``0.5 · E_self``.
    quadrupoles : torch.Tensor, optional, shape (N_total, 3, 3)
        Cartesian symmetric source quadrupole. When supplied, the additive
        batched :math:`\rho_Q(k)` channel and its per-system :math:`l=2` self
        term are included (requires a cache built with source ``l_max>=2``).
        ``None`` (default) is the :math:`l_{max} \le 1` path.

    Returns
    -------
    torch.Tensor
        Per-atom energies, shape :math:`(N_\text{total},)`,
        :math:`\text{float64}` on ``cache.device``. Flat across all systems;
        forces/stress/charge-grads flow from ``grad(E.sum(), ...)``.
    """
    _validate_batch_inputs(cache, positions, source_feats, batch_idx)
    n_total = positions.shape[0]

    source_feats, quadrupoles = _resolve_step_moments(source_feats, quadrupoles)
    charges, dipoles_cart, l_max = split_source_feats(source_feats)
    dip = (
        dipoles_cart.contiguous()
        if dipoles_cart is not None
        else torch.zeros((n_total, 3), dtype=source_feats.dtype, device=cache.device)
    )

    # Delayed import registers the batch_multipole_rho op chain and breaks the
    # multipole_autograd_batch <-> multipole_scf_step cycle. Forward and
    # (AOT-traced) backward are opaque custom ops -> 0 torch.compile breaks.
    import nvalchemiops.torch.interactions.electrostatics.multipole_autograd_batch  # noqa: F401

    rho = torch.ops.nvalchemiops.batch_multipole_rho(
        charges,
        dip,
        positions,
        cache.source_phi_hat,
        cache.k_vectors,
        cache.volume,
        batch_idx,
    )
    # Additive Cartesian-quadrupole channel (batched).
    if quadrupoles is not None:
        if cache.source_coeff2 is None:
            raise ValueError(
                "quadrupoles requires a cache built with l_max>=2 "
                "(cache.source_coeff2 is None)."
            )
        if quadrupoles.shape != (n_total, 3, 3):
            raise ValueError(
                f"quadrupoles must be (N_total, 3, 3); got {tuple(quadrupoles.shape)}"
            )
        rho = rho + torch.ops.nvalchemiops.batch_multipole_rho_q(
            quadrupoles,
            positions,
            cache.source_coeff2,
            cache.k_vectors,
            cache.volume,
            batch_idx,
        )
    # Per-system value-preserving volume ratio (== 1) so dE/dcell captures
    # rho's 1/V_b dependence. rho is (B, K, 2).
    vol_ratio = cache.volume.detach() / cache.volume  # (B,)
    rho = rho * vol_ratio.reshape(-1, 1, 1)
    # Per-atom reciprocal energy via the batched spread-transpose gathers
    # (phi_hat = 2 f_k rho, (B, K, 2)); ``Σ_i over a system`` equals the
    # collective per-system raw reciprocal bit-for-bit. The gathers map the
    # (B, K, 2) field to per-atom (N_total, ...) via batch_idx; the detached
    # 1/V_b is restored per atom via vol_ratio[batch_idx].
    bl = batch_idx.long()
    phi_hat = (2.0 * cache.per_k_factor).unsqueeze(-1) * rho  # (B, K, 2)
    vr_atom = vol_ratio.index_select(0, bl)
    g = torch.ops.nvalchemiops.batch_multipole_rho_gather_t(
        phi_hat,
        positions,
        cache.source_phi_hat,
        cache.k_vectors,
        cache.volume,
        batch_idx,
    ) * vr_atom.reshape(-1, 1)
    scale_atom = (0.5 * cache.volume / _TWO_PI_6).index_select(0, bl)
    raw_per_atom = scale_atom * (charges * g[:, 0] + (dip * g[:, [3, 1, 2]]).sum(-1))
    if quadrupoles is not None:
        g_q = torch.ops.nvalchemiops.batch_multipole_rho_q_gather_t(
            phi_hat,
            positions,
            cache.source_coeff2,
            cache.k_vectors,
            cache.volume,
            batch_idx,
        ) * vr_atom.reshape(-1, 1, 1)
        raw_per_atom = raw_per_atom + scale_atom * (quadrupoles * g_q).sum((-1, -2))

    if include_self_interaction:
        return raw_per_atom.to(torch.float64)

    # Per-atom Σ_l c_l moment_l^2 via the shared self-energy op (per-atom, no
    # reduction — caller owns it). dipoles only when l_max>=1.
    dip_self = dipoles_cart if l_max == 1 else None
    atom_self = _self_energy_op(cache, charges, dip_self, quadrupoles)
    return (raw_per_atom - 0.5 * atom_self).to(torch.float64)


def _multipole_scf_step_features_batch(
    cache: MultipoleSCFCache,
    positions: torch.Tensor,
    source_feats: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    include_self_interaction: bool = False,
    quadrupoles: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Batched branch of :func:`multipole_scf_step_features`.

    Consumes a batched :class:`MultipoleSCFCache`. Returns
    :math:`(N_\text{total}, N_\sigma \cdot (\text{feature\_max\_l}+1)^2)` in
    the reference permuted flat layout, autograd-connected to ``positions`` and
    ``source_feats``.

    ``quadrupoles`` : optional :math:`(N_\text{total}, 3, 3)` Cartesian-Q
    source moments. Adds the batched :math:`\rho_Q` channel so an l=2 source
    enriches the projected potential. Requires a cache built with source
    ``l_max>=2``. Decoupled from the receiver ``feature_max_l``.

    Parameters
    ----------
    cache : MultipoleSCFCache
        Prebuilt batched cache (``cache.is_batched``).
    positions : torch.Tensor, shape (N_total, 3)
    source_feats : torch.Tensor, shape (N_total, (l_max+1)**2)
        Packed per-atom moments in e3nn spherical layout.
    batch_idx : torch.Tensor, shape (N_total,), int32
        System index per atom (expected sorted so atoms group by system).
    include_self_interaction : bool
        If ``False`` (default), subtract the per-system self-interaction
        correction using ``cache.feature_overlap_constants``.
    quadrupoles : torch.Tensor, optional, shape (N_total, 3, 3)
        Cartesian symmetric source quadrupole; see the prose above.

    Returns
    -------
    torch.Tensor
        :math:`\text{float64}` on ``cache.device``, shape
        :math:`(N_\text{total}, N_\sigma \cdot (\text{feature\_max\_l}+1)^2)`
        in the reference permuted flat layout. Autograd-connected to
        ``positions`` and ``source_feats``.
    """
    _validate_batch_inputs(cache, positions, source_feats, batch_idx)
    n_total = positions.shape[0]

    source_feats, quadrupoles = _resolve_step_moments(source_feats, quadrupoles)
    charges, dipoles_cart, l_max = split_source_feats(source_feats)
    dip = (
        dipoles_cart.contiguous()
        if dipoles_cart is not None
        else torch.zeros((n_total, 3), dtype=source_feats.dtype, device=cache.device)
    )

    # Delayed import registers the batch_multipole_rho + feature-projection op
    # chains while breaking the module cycle.
    import nvalchemiops.torch.interactions.electrostatics.multipole_autograd_batch  # noqa: F401, E501

    rho = torch.ops.nvalchemiops.batch_multipole_rho(
        charges,
        dip,
        positions,
        cache.source_phi_hat,
        cache.k_vectors,
        cache.volume,
        batch_idx,
    )
    # Additive batched Cartesian-quadrupole source channel.
    if quadrupoles is not None:
        if cache.source_coeff2 is None:
            raise ValueError(
                "quadrupoles requires a cache built with l_max>=2 "
                "(cache.source_coeff2 is None)."
            )
        if quadrupoles.shape != (n_total, 3, 3):
            raise ValueError(
                f"quadrupoles must be (N_total, 3, 3); got {tuple(quadrupoles.shape)}"
            )
        rho = rho + torch.ops.nvalchemiops.batch_multipole_rho_q(
            quadrupoles,
            positions,
            cache.source_coeff2,
            cache.k_vectors,
            cache.volume,
            batch_idx,
        )
    # Per-system value-preserving volume ratio so the explicit 1/V in the
    # feature carries its (diagonal) volume-grad. cache.volume is per-system
    # det(cell).
    vol_ratio = (cache.volume.detach() / cache.volume).reshape(-1, 1, 1)
    potential = cache.per_k_factor.unsqueeze(-1) * rho * vol_ratio

    # Raw features (N_total, N_sigma, 4), natural layout. Thread the
    # cell-differentiable receiver_phi_hat (l<=1 block) + k_vectors so
    # dfeatures/dcell flows.
    raw_features = torch.ops.nvalchemiops.batch_multipole_project_raw_features(
        potential,
        positions,
        cache.receiver_phi_hat[:, :, :, :4],
        cache.k_vectors,
        cache.k_factor_proj,
        batch_idx,
    )

    # Self-interaction subtract. source_feats is already in the
    # [q, mu_y, mu_z, mu_x] layout; zero-pad when l_max=0 so the (N_sigma, 4)
    # broadcast works without a branch.
    if not include_self_interaction:
        if l_max == 0:
            src_lm = torch.cat(
                [
                    source_feats.to(torch.float64),
                    torch.zeros((n_total, 3), dtype=torch.float64, device=cache.device),
                ],
                dim=-1,
            )
        else:
            src_lm = source_feats.to(torch.float64)
        oc = cache.feature_overlap_constants  # (N_sigma, 2)
        oc_per_lm = torch.cat(
            [oc[:, 0:1], oc[:, 1:2].expand(-1, 3)], dim=-1
        )  # (N_sigma, 4)
        self_corr = src_lm.unsqueeze(1) * oc_per_lm.unsqueeze(0)
        features_natural = raw_features - self_corr
    else:
        features_natural = raw_features

    # Batched l=2 receiver block (decoupled from source l_max).
    if cache.feature_max_l == 0:
        features_natural = features_natural[:, :, :1]  # l=0-only receiver
    elif cache.feature_max_l >= 2:
        raw_l2 = torch.ops.nvalchemiops.batch_multipole_project_raw_features_quadrupole(
            potential,
            positions,
            cache.receiver_phi_hat[:, :, :, 4:9],
            cache.k_vectors,
            cache.k_factor_proj,
            batch_idx,
        )  # (N_total, N_σ, 5)
        if not include_self_interaction and quadrupoles is not None:
            from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
                cartesian_quadrupole_to_e3nn,
            )

            src_e3nn_l2 = cartesian_quadrupole_to_e3nn(quadrupoles.to(torch.float64))
            self_corr_l2 = src_e3nn_l2.unsqueeze(1) * cache.feature_overlap_l2.reshape(
                1, cache.n_sigma, 1
            )
            features_l2 = raw_l2 - self_corr_l2
        else:
            features_l2 = raw_l2
        features_natural = torch.cat([features_natural, features_l2], dim=-1)

    n_lm = (cache.feature_max_l + 1) ** 2
    flat_natural = features_natural.reshape(n_total, cache.n_sigma * n_lm)
    # Inverse-permute to the reference flat layout via the cache's precomputed
    # argsort (position-independent — no per-call sort).
    return flat_natural.index_select(-1, cache.out_col_inv_perm)


# =============================================================================
# Ewald SCF step — cache-aware multipole_ewald_summation
# =============================================================================


def multipole_ewald_scf_step_energy(
    cache: MultipoleSCFCache,
    positions: torch.Tensor,
    multipole_moments: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""GTO-Ewald total energy using a pre-built SCF cache.

    SCF-loop friendly analog of :func:`multipole_ewald_summation`: the
    position-independent k-space state (k-vectors, :math:`\hat\phi` tables,
    damped ``per_k_factor``, overlap constants) lives in ``cache``, so each
    step only recomputes the moment-dependent arithmetic and the
    neighbor-list pair sum, saving the per-iteration k-vector and
    :math:`\exp(-\sigma_c^2 k^2)` generation.

    Cache requirements
    ------------------
    ``cache`` must have been built with ``alpha`` (non-None):

        cache = prepare_multipole_scf_cache(
            cell, sigma=sigma, alpha=alpha, k_cutoff=kcut
        )

    A direct-k-space cache (``alpha=None``) would give the wrong reciprocal
    kernel (:math:`F/k^2` instead of
    :math:`F \exp(-k^2 / (4 \alpha^2)) / k^2`) and is rejected. Use
    :func:`multipole_scf_step_energy` for the direct-k-space path.

    Single-system vs batched dispatch
    ---------------------------------
    Same convention as :func:`multipole_ewald_summation`:

    * ``batch_idx=None`` — single system; cache must be single-system
      (built from a ``(3, 3)`` cell); returns per-atom ``(N,)`` float64.
    * ``batch_idx`` provided — B systems flat-packed; cache must be batched
      (built from a ``(B, 3, 3)`` cell); returns per-atom
      ``(N_total,)`` (flat across systems).

    Parameters
    ----------
    cache :
        Prebuilt Ewald cache. Must be built with source ``l_max>=2`` for the
        :math:`(N, 9)` (quadrupole) path.
    positions : torch.Tensor, shape (N, 3) or (N_total, 3)
    multipole_moments : torch.Tensor, shape (N, (l_max + 1)**2)
        Packed e3nn moments: :math:`(N, 1)` / :math:`(N, 4)` / :math:`(N, 9)`
        for :math:`l_{max} = 0/1/2`.
    idx_j, neighbor_ptr, unit_shifts : torch.Tensor
        Flat CSR neighbor list; same convention as
        :func:`multipole_ewald_summation`.
    batch_idx : torch.Tensor, optional
        ``(N_total,)`` int32. ``None`` selects single-system mode.

    Returns
    -------
    torch.Tensor
        Per-atom :math:`(N,)` :math:`\text{float64}` (single) or
        :math:`(N_\text{total},)` (batched, flat across systems).
        Call ``.sum()`` for the total energy or
        ``torch.zeros(B).scatter_add(0, batch_idx, E)`` for per-system totals;
        forces/stress/charge-grads flow from ``grad(E.sum(), ...)``.
        Autograd-connected to ``positions`` and ``multipole_moments``.
    """
    if cache.alpha is None:
        raise ValueError(
            "multipole_ewald_scf_step_energy requires an Ewald cache "
            "built with alpha=α (non-None). Got alpha=None (direct-k-space cache); "
            "use multipole_scf_step_energy for the direct-k-space path."
        )
    sigma = cache.sigma
    alpha = cache.alpha

    _check_batch_dispatch(cache, batch_idx)
    is_batch = batch_idx is not None

    # Delayed imports to avoid circular module graph.
    from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
        _multipole_ewald_self_energy_per_atom,
        multipole_real_space_energy,
    )

    # Split the packed moments: the l<=1 block (+ separate Cartesian quadrupole)
    # is what the reciprocal step and self-energy consume; the real-space term
    # takes the full packed tensor.
    source_feats_l1, quadrupoles, l_max = split_packed_for_kernels(multipole_moments)
    if quadrupoles is not None and cache.source_coeff2 is None:
        raise ValueError(
            "an (N, 9) multipole_moments requires a cache built with l_max>=2 "
            "(cache.source_coeff2 is None)."
        )

    device = positions.device
    input_dtype = positions.dtype
    coulomb_scale = FIELD_CONSTANT / (4.0 * math.pi)

    if is_batch:
        B = cache.batch_size
        sigmas = torch.full((B,), sigma, dtype=input_dtype, device=device)
        alphas = torch.full((B,), alpha, dtype=input_dtype, device=device)
        real = multipole_real_space_energy(
            positions,
            multipole_moments,
            cache.cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx=batch_idx,
        )
    else:
        sigma_t = torch.tensor([sigma], dtype=input_dtype, device=device)
        alpha_t = torch.tensor([alpha], dtype=input_dtype, device=device)
        real = multipole_real_space_energy(
            positions,
            multipole_moments,
            cache.cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigma_t,
            alpha_t,
        )
    # multipole_real_space_energy returns per-atom (N,)/(N_total,) for all l_max.
    e_real_per_atom = coulomb_scale * real

    # Reciprocal via cache (no per-step k-space rebuild).
    # include_self_interaction=True returns the raw reciprocal; the Ewald self
    # term is handled analytically below.
    # multipole_scf_step_energy returns per-atom (N,)/(N_total,).
    e_recip_per_atom = multipole_scf_step_energy(
        cache,
        positions,
        source_feats_l1,
        batch_idx=batch_idx,
        include_self_interaction=True,
        quadrupoles=quadrupoles,
    )

    atom_self = _multipole_ewald_self_energy_per_atom(
        source_feats_l1, sigma, alpha, quadrupoles=quadrupoles
    )
    # Return per-atom (N,)/(N_total,) — caller owns the reduction.
    return (e_real_per_atom + e_recip_per_atom - atom_self).to(torch.float64)
