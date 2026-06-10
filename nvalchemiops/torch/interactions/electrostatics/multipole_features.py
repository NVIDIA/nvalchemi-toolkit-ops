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
PyTorch bindings for direct k-space atom-centered electrostatic features.

Exposes ``multipole_electrostatic_features(...)``, the feature-extraction
counterpart to ``multipole_electrostatic_energy``. Produces LODE/ACE-style
per-atom descriptors by projecting :math:`V(\mathbf{k})` back onto a multi-σ
receiver basis. Bit-for-bit parity with the customer reference
``GTOElectrostaticFeatures`` at ``density_max_l`` in
:math:`\{0, 1\}` and ``feature_max_l = 1``.
"""

from __future__ import annotations

import torch

from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    infer_l_max,
    split_packed_for_kernels,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_electrostatics import (
    _resolve_norm_mode,
)
from nvalchemiops.torch.math.gto import NormMode


def _build_out_col_lut(
    n_sigma: int, permuted: bool, device: torch.device, feature_max_l: int = 1
) -> torch.Tensor:
    r"""Per-:math:`(\sigma, lm)` flat output-column LUT for the feature projection kernel.

    The kernel writes each per-:math:`(\sigma, lm)` output to
    ``features[i, lut[sigma, lm]]``. ``feature_max_l`` selects
    ``n_lm = (feature_max_l + 1)**2`` (4 for :math:`l \le 1`, 9 for
    :math:`l = 2`). Two layouts are supported:

    * ``permuted=False`` -> ``lut[s, lm] = s * n_lm + lm`` (row-major natural
      layout; a ``.reshape(N, N_sigma, n_lm)`` on the output recovers 3-D).
    * ``permuted=True`` -> the customer-drop-in layout:
      l-blocks grouped, each in :math:`(\sigma, m)` order:
      ``[l0_s0..l0_s{K-1}, l1_m-1_s0, l1_m0_s0, l1_m+1_s0, l1_m-1_s1, ...,
      (l2 block: 5 m per sigma)]``.

    The LUT is a small ``(N_sigma, n_lm) int32`` tensor.
    """
    n_lm = (feature_max_l + 1) ** 2
    lut = torch.empty((n_sigma, n_lm), dtype=torch.int32, device=device)
    if permuted:
        s_ar = torch.arange(n_sigma, dtype=torch.int32, device=device)
        offset = 0
        col = 0
        for ell in range(feature_max_l + 1):
            width = 2 * ell + 1
            block_starts = offset + s_ar * width
            lut[:, col : col + width] = block_starts.unsqueeze(1) + torch.arange(
                width, dtype=torch.int32, device=device
            ).unsqueeze(0)
            offset += width * n_sigma
            col += width
    else:
        s_idx = torch.arange(n_sigma, dtype=torch.int32, device=device)
        lm_idx = torch.arange(n_lm, dtype=torch.int32, device=device)
        lut[:] = s_idx.unsqueeze(1) * n_lm + lm_idx.unsqueeze(0)
    return lut


def multipole_electrostatic_features(
    positions: torch.Tensor,
    multipole_moments: torch.Tensor,
    cell: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None = None,
    sigma: float,
    receiver_sigmas: list[float] | tuple[float, ...] | torch.Tensor,
    k_cutoff: float | None = None,
    k_vectors: torch.Tensor | None = None,
    feature_max_l: int = 1,
    density_normalize: NormMode | int | str = NormMode.MULTIPOLES,
    feature_normalize: NormMode | int | str = NormMode.RECEIVER,
    include_self_interaction: bool = False,
) -> torch.Tensor:
    r"""Atom-centered electrostatic features via direct k-space projection.

    Computes

    .. math::

        f_{i, \sigma_r, l, m} \;=\; \frac{2}{(2\pi)^3}
            \sum_{\mathbf{k}} w(\mathbf{k}) \cdot
            \text{Re}\!\left[V^{*}(\mathbf{k})\,
                \hat\phi_{l,m}^{\sigma_r}(\mathbf{k})\,
                e^{i\mathbf{k}\cdot\mathbf{r}_i}\right],

    where :math:`V(\mathbf{k})` is the periodic electrostatic potential
    assembled from ``multipole_moments`` as in the companion energy binding,
    :math:`\hat\phi_{l,m}^{\sigma_r}` is the receiver-basis GTO Fourier
    coefficient, and :math:`w(\mathbf{k})` is :math:`0.5` at :math:`k = 0`
    and :math:`1` elsewhere (the half-space-with-origin convention matching
    the reference ``k_factor_proj``).

    Bit-for-bit parity with the customer reference ``GTOElectrostaticFeatures``
    at ``density_max_l`` in :math:`\{0, 1\}`, ``feature_max_l = 1``, under
    matched inputs.

    Single-system vs batched dispatch
    ---------------------------------
    Mirrors :func:`multipole_ewald_summation`: pass ``cell`` of shape
    ``(3, 3)`` (single) or ``(B, 3, 3)`` (batched) and use ``batch_idx`` to
    select the batched path. Batched mode requires ``k_cutoff`` (a
    pre-generated ``k_vectors`` is single-system only).

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3) or (N_total, 3)
        Atomic positions; flat across systems in the batched case.
    multipole_moments : torch.Tensor, shape (N, (l_max+1)**2)
        Packed per-atom source moments in e3nn spherical layout. Source
        ``l_max`` is inferred; an :math:`l_{max}=2` ``(N, 9)`` tensor enriches
        the projected potential with its Cartesian-quadrupole channel.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Single unit cell or ``B`` per-system cells (batched).
    batch_idx : torch.Tensor, optional, shape (N_total,), int32
        Per-atom system index (expected sorted). Required when ``cell`` is
        ``(B, 3, 3)``; must be ``None`` for a single ``(3, 3)`` cell.
    sigma : float
        Density-side Gaussian width.
    receiver_sigmas : list of floats, tuple, or 1-D tensor
        Multi-σ receiver basis widths. Must be non-empty.
    k_cutoff, k_vectors
        Same semantics as :func:`multipole_electrostatic_energy`. Pass
        ``k_vectors`` to amortize setup across calls for fixed geometry
        (single-system only). Batched mode requires ``k_cutoff``.
    feature_max_l : int, default 1
        Receiver angular cap: how many l-blocks are projected out per σ,
        independent of the source ``l_max``. ``1`` -> ``(N_σ * 4)`` features
        (:math:`l \le 1`); ``2`` -> ``(N_σ * 9)`` (adds the 5 :math:`l = 2`
        receiver channels). The :math:`l = 2` self-interaction subtract uses
        the source's e3nn :math:`l = 2` moment (zero when the source has no
        quadrupole).
    density_normalize : NormMode | int | str
        Source-basis normalization. Defaults to ``MULTIPOLES``.
    feature_normalize : NormMode | int | str
        Receiver-basis normalization. Defaults to ``RECEIVER``, matching
        ``GTOElectrostaticFeatures``'s ``integral_normalization`` default.
    include_self_interaction : bool
        If ``False`` (default), subtract the self-interaction term using
        :func:`compute_overlap_constants`.

    Returns
    -------
    torch.Tensor
        ``float64`` on ``positions.device``, shape
        ``(N, N_σ * (feature_max_l + 1)**2)`` in the reference permuted-flat
        layout (grouped by l-block). Autograd-connected to ``positions`` and
        ``multipole_moments``.
    """
    is_batch = batch_idx is not None
    if is_batch:
        if cell.ndim != 3 or cell.shape[-2:] != (3, 3):
            raise ValueError(f"batched cell must be (B, 3, 3), got {tuple(cell.shape)}")
        if k_vectors is not None:
            raise ValueError(
                "k_vectors is not supported for batched features; pass "
                "k_cutoff instead."
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

    if isinstance(receiver_sigmas, torch.Tensor):
        sigmas_list = receiver_sigmas.detach().cpu().to(torch.float64).tolist()
    else:
        sigmas_list = [float(s) for s in receiver_sigmas]
    if len(sigmas_list) == 0:
        raise ValueError("receiver_sigmas must be non-empty")
    if any(s <= 0.0 for s in sigmas_list):
        raise ValueError(f"receiver_sigmas must all be positive, got {sigmas_list}")

    if feature_max_l not in (0, 1, 2):
        raise ValueError(f"feature_max_l must be 0, 1, or 2, got {feature_max_l}")
    l_max = infer_l_max(multipole_moments)
    density_mode = _resolve_norm_mode(density_normalize)
    feature_mode = _resolve_norm_mode(feature_normalize)

    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
        prepare_multipole_scf_cache,
    )
    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
        multipole_scf_step_features,
    )

    cache = prepare_multipole_scf_cache(
        cell,
        sigma=sigma,
        receiver_sigmas=sigmas_list,
        k_cutoff=k_cutoff,
        k_vectors=None if is_batch else k_vectors,
        l_max=l_max,
        feature_max_l=feature_max_l,
        density_normalize=density_mode,
        feature_normalize=feature_mode,
        device=positions.device,
    )
    source_feats_l1, quadrupoles_cart, _ = split_packed_for_kernels(multipole_moments)
    return multipole_scf_step_features(
        cache,
        positions,
        source_feats_l1,
        batch_idx=batch_idx,
        quadrupoles=quadrupoles_cart,
        include_self_interaction=include_self_interaction,
    )
