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
GTO Basis Normalization — Host-Side Scaffolding
===============================================

Binding-layer helpers for building the :math:`(l, \sigma)`-dependent
normalization scaling tables that the multipole Ewald/PME and atom-centered
feature kernels consume as Warp constant arrays.

The Warp ``@wp.func`` / ``@wp.kernel`` primitives for GTO basis evaluation
live in :mod:`nvalchemiops.math.gto`. This module is strictly CPU scaffolding
— it runs once at module-construction time to produce small constant tables
that are then shipped to the GPU.
"""

from __future__ import annotations

import math
from enum import IntEnum

import warp as wp

from nvalchemiops.math.gto import (
    _eval_gto_density_kernel,
    _eval_gto_fourier_kernel,
)


class NormMode(IntEnum):
    r"""GTO basis normalization conventions used by the multipole pipeline.

    The Fourier transform of a GTO basis function carries a per-:math:`(l, \sigma)`
    scaling factor :math:`1 / C_l(\sigma, \text{mode})`. Three mode choices are
    supported, corresponding to the three normalization denominators defined below:

    ``MULTIPOLES``
        .. math::

            C_l^{-1}(\sigma) = \sqrt{\frac{4\pi}{2l+1}} \cdot 2^{(2l+1)/2}
                               \cdot \Gamma\!\left(\frac{2l+3}{2}\right) \cdot \sigma^{2l+3}

        A GTO with coefficient :math:`q` integrates to a charge distribution whose
        :math:`(2l+1)`-component multipole moment is exactly :math:`q`. Use on the
        density/source side of the Ewald/PME pipeline.

    ``RECEIVER``
        .. math::

            C_l^{-1}(\sigma) = 2^{(l+1)/2} \cdot \Gamma\!\left(\frac{l+3}{2}\right) \cdot \sigma^{l+3}

        A GTO tested against a uniform potential gives a unit projection. Use on the
        feature/projection side of LODE-style atom-centered electrostatic descriptors.

    ``NONE``
        :math:`C_l^{-1}(\sigma) = 1`. Un-normalized basis; primarily for debugging
        and cross-checks.

    Notes
    -----
    The enum values are stable integers (0, 1, 2). Downstream Warp kernels consume
    the scaling factors via ``wp.constant`` arrays built by :func:`inv_cl_table`;
    kernels that branch on mode do so on the ``int`` underlying the enum.
    """

    MULTIPOLES = 0
    RECEIVER = 1
    NONE = 2


def inv_cl(sigma: float, L: int, mode: NormMode) -> float:
    r"""Scaling factor :math:`1 / C_l(\sigma, \text{mode})` for GTO basis Fourier coefficients.

    Parameters
    ----------
    sigma
        Gaussian width parameter. Must be positive.
    L
        Angular momentum quantum number (``l`` in the math). Must be non-negative.
    mode
        Normalization convention. See :class:`NormMode`.

    Returns
    -------
    float
        The multiplier to apply when normalizing a GTO Fourier coefficient.

    Raises
    ------
    ValueError
        If ``sigma <= 0`` or ``L < 0``.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if L < 0:
        raise ValueError(f"L must be non-negative, got {L}")
    mode = NormMode(mode)
    if mode == NormMode.NONE:
        return 1.0
    if mode == NormMode.MULTIPOLES:
        l_part = (
            math.sqrt(4.0 * math.pi / (2 * L + 1))
            * 2.0 ** ((2 * L + 1) / 2.0)
            * math.gamma((2 * L + 3) / 2.0)
        )
        denom = l_part * sigma ** (2 * L + 3)
    else:  # RECEIVER
        l_part = 2.0 ** ((L + 1) / 2.0) * math.gamma((L + 3) / 2.0)
        denom = l_part * sigma ** (L + 3)
    return 1.0 / denom


def inv_cl_table(sigma: float, max_L: int, mode: NormMode) -> list[float]:
    r"""Precomputed table of :func:`inv_cl` for :math:`L = 0, 1, \ldots, \text{max\_L}`.

    Parameters
    ----------
    sigma
        Gaussian width parameter. Must be positive.
    max_L
        Maximum angular momentum (inclusive). Must be non-negative.
    mode
        Normalization convention. See :class:`NormMode`.

    Returns
    -------
    list[float]
        Length-``max_L + 1`` list of scaling factors. Entry ``L`` holds
        ``inv_cl(sigma, L, mode)``.
    """
    if max_L < 0:
        raise ValueError(f"max_L must be non-negative, got {max_L}")
    return [inv_cl(sigma, L, mode) for L in range(max_L + 1)]


# =============================================================================
# PyTorch launch wrappers
# =============================================================================


def eval_gto_density_pytorch(
    positions,
    sigma: float,
    L_max: int = 2,
    device=None,
):
    """Evaluate GTO densities from PyTorch tensors.

    Parameters
    ----------
    positions : torch.Tensor
        Input positions [N, 3] as float64.
    sigma : float
        Gaussian width parameter.
    L_max : int
        Maximum angular momentum (0, 1, or 2). Default: 2.
    device : torch.device, optional
        Device for computation.

    Returns
    -------
    torch.Tensor
        GTO density values [N, num_components].
    """
    import torch

    if device is None:
        device = positions.device

    N = positions.shape[0]
    num_components = {0: 1, 1: 4, 2: 9}[L_max]

    output = torch.zeros((N, num_components), dtype=torch.float64, device=device)

    wp_device = wp.device_from_torch(device)
    wp_positions = wp.from_torch(positions.contiguous(), dtype=wp.vec3d)
    wp_output = wp.from_torch(output, dtype=wp.float64)

    wp.launch(
        kernel=_eval_gto_density_kernel,
        dim=N,
        inputs=[wp_positions, wp.float64(sigma), L_max],
        outputs=[wp_output],
        device=wp_device,
    )

    return output


def eval_gto_fourier_pytorch(
    k_vectors,
    sigma: float,
    L_max: int = 2,
    device=None,
):
    """Evaluate GTO Fourier transforms from PyTorch tensors.

    Parameters
    ----------
    k_vectors : torch.Tensor
        Input wave vectors [K, 3] as float64.
    sigma : float
        Gaussian width parameter.
    L_max : int
        Maximum angular momentum (0, 1, or 2). Default: 2.
    device : torch.device, optional
        Device for computation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (real_part, imag_part) each of shape [K, num_components].
    """
    import torch

    if device is None:
        device = k_vectors.device

    K = k_vectors.shape[0]
    num_components = {0: 1, 1: 4, 2: 9}[L_max]

    output_real = torch.zeros((K, num_components), dtype=torch.float64, device=device)
    output_imag = torch.zeros((K, num_components), dtype=torch.float64, device=device)

    wp_device = wp.device_from_torch(device)
    wp_k = wp.from_torch(k_vectors.contiguous(), dtype=wp.vec3d)
    wp_real = wp.from_torch(output_real, dtype=wp.float64)
    wp_imag = wp.from_torch(output_imag, dtype=wp.float64)

    wp.launch(
        kernel=_eval_gto_fourier_kernel,
        dim=K,
        inputs=[wp_k, wp.float64(sigma), L_max],
        outputs=[wp_real, wp_imag],
        device=wp_device,
    )

    return output_real, output_imag
