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
GTO-GTO Self-Overlap Constants (Host-Side Scaffolding)
======================================================

Binding-layer helper that precomputes the per-:math:`(L, \sigma_{\text{receive}})`
constants appearing in self-interaction subtraction for both the multipole
Ewald/PME energy and the atom-centered electrostatic feature pipelines.

This module is CPU-only host-side scaffolding: it runs once at module-construction
time to produce a small ``(N_sigma, L+1)`` table that is subsequently shipped to
the GPU as a Warp constant array. The downstream hot kernels that consume the
table live in :mod:`nvalchemiops.math`.

For a density-basis GTO centered on atom :math:`i` with width :math:`\sigma_s` and
multipole coefficient :math:`Q_{L,m}^i`, and a receiver-basis GTO with width
:math:`\sigma_r`, the self-field contribution at the same atom is

.. math::

    \text{overlap}(L, \sigma_s, \sigma_r, \text{mode}_s, \text{mode}_r)
    \cdot Q_{L,m}^i

where the overlap constant is the same for every :math:`m \in \{-L, \ldots, L\}` at
a given :math:`(L, \sigma_r)`. The constant is

.. math::

    \text{overlap}(L, \sigma_s, \sigma_r) = \frac{1}{C_L(\sigma_s, \text{mode}_s)}
    \cdot \frac{1}{C_L(\sigma_r, \text{mode}_r)} \cdot \frac{F_{\text{field}}}{2L+1}
    \cdot I_L(\sigma_s, \sigma_r)

with :math:`1/C_L(\cdot)` from :func:`nvalchemiops.torch.math.gto.inv_cl` and the
radial integral

.. math::

    I_L(\sigma_s, \sigma_r) = \int_0^\infty r^{L+2} \cdot
    \exp\!\left(-\tfrac{r^2}{2\sigma_s^2}\right) \cdot
    \left[F_1(r, L, \sigma_r) + F_2(r, L, \sigma_r)\right] \, dr,

where :math:`F_1` and :math:`F_2` are the radial pieces of the Coulomb-kernel
response of a GTO charge distribution. The integral is evaluated by trapezoidal
quadrature on a dense linear grid; default grid parameters match the customer
reference ``GTOSelfInteractionBlock``.

The output is a ``torch.Tensor`` shaped ``(N_sigma_receive, max_L + 1)`` that is
ready to be shipped into Warp kernels as a constant table. A companion helper
:func:`flatten_to_reference_layout` converts to the flat customer layout for
parity testing.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from nvalchemiops.torch.math.gto import NormMode, inv_cl

# Electrostatic prefactor 1 / ε₀ in (electron charge, Volt, Ångström) units.
# Value matches the customer reference's ``FIELD_CONSTANT``.
FIELD_CONSTANT: float = 1.0 / 5.526349406e-3


def _radial_F_total(r: torch.Tensor, L: int, sigma: float) -> torch.Tensor:
    r"""Radial response ``F_1 + F_2`` from the reference implementation.

    .. math::

        F_1(r, L, \sigma) &= 2^{L+1/2} \, \sigma^{2L+3} \,
            \gamma\!\left(\tfrac{2L+3}{2}, \tfrac{r^2}{2\sigma^2}\right) \, r^{-(L+1)} \\
        F_2(r, L, \sigma) &= \sigma^2 \, r^L \, \exp\!\left(-\tfrac{r^2}{2\sigma^2}\right)

    where :math:`\gamma(a, x)` is the lower incomplete gamma function.
    """
    a = (2 * L + 3) / 2.0
    # ``torch.special.gammainc`` is the *regularized* lower incomplete gamma
    # γ(a, x) / Γ(a); multiply back by Γ(a) = exp(lgamma(a)) to recover γ(a, x).
    a_t = torch.full_like(r, a)
    gamma_a = torch.exp(torch.lgamma(torch.tensor(a, dtype=r.dtype)))
    gamma_inc = torch.special.gammainc(a_t, 0.5 * r * r / (sigma * sigma)) * gamma_a
    F1 = 2.0 ** (L + 0.5) * sigma ** (2 * L + 3) * gamma_inc * r.pow(-(L + 1))
    F2 = sigma * sigma * r.pow(L) * torch.exp(-0.5 * r * r / (sigma * sigma))
    return F1 + F2


def _overlap_radial_integral(
    L: int,
    sigma_source: float,
    sigma_receive: float,
    grid_size: int,
    r_max_factor: float,
) -> float:
    r"""Trapezoidal evaluation of the radial overlap integral :math:`I_L`.

    Grid: linear from ``1e-4`` to ``r_max_factor * max(σ_s, σ_r)`` with
    ``grid_size`` samples — same discretization as the customer reference.
    """
    r_max = r_max_factor * max(sigma_receive, sigma_source)
    grid = torch.linspace(1e-4, r_max, grid_size, dtype=torch.float64)
    F_total = _radial_F_total(grid, L, sigma_receive)
    integrand = (
        grid.pow(L + 2)
        * torch.exp(-0.5 * grid * grid / (sigma_source * sigma_source))
        * F_total
    )
    return float(torch.trapezoid(integrand, x=grid))


def compute_overlap_constants(
    *,
    max_L: int,
    sigma_source: float,
    sigmas_receive: Sequence[float] | torch.Tensor,
    normalize_source: NormMode | int = NormMode.MULTIPOLES,
    normalize_receive: NormMode | int = NormMode.RECEIVER,
    field_constant: float = FIELD_CONSTANT,
    grid_size: int = 10000,
    r_max_factor: float = 10.0,
) -> torch.Tensor:
    r"""Precompute GTO-GTO self-overlap constants for the multipole pipeline.

    Parameters
    ----------
    max_L
        Maximum angular momentum (inclusive). Must be non-negative.
    sigma_source
        Single-:math:`\sigma` density basis width. Must be positive.
    sigmas_receive
        Multi-:math:`\sigma` receiver basis widths. Each must be positive; at
        least one value is required.
    normalize_source, normalize_receive
        Normalization conventions. See :class:`nvalchemiops.torch.math.gto.NormMode`.
    field_constant
        Electrostatic prefactor (``1/ε₀`` in the chosen unit system). Defaults
        to :data:`FIELD_CONSTANT` (``e``, ``V``, ``Å`` units).
    grid_size, r_max_factor
        Numerical quadrature parameters. The integrand is evaluated on a linear
        grid from ``1e-4`` to ``r_max_factor * max(σ_s, σ_r)`` with
        ``grid_size`` points, then integrated via the trapezoidal rule.
        Defaults match the reference implementation.

    Returns
    -------
    torch.Tensor of shape ``(N_sigma_receive, max_L + 1)``, float64 (CPU)
        Entry ``[i, L]`` is the overlap constant for receiver width
        ``sigmas_receive[i]`` and angular momentum ``L``. The same constant
        applies to every :math:`m \in \{-L, \ldots, L\}` for that
        ``(L, σ_r)``.

    Raises
    ------
    ValueError
        On non-positive ``sigma_source``, non-positive entries in
        ``sigmas_receive``, empty ``sigmas_receive``, or negative ``max_L``.
    """
    if max_L < 0:
        raise ValueError(f"max_L must be non-negative, got {max_L}")
    if sigma_source <= 0.0:
        raise ValueError(f"sigma_source must be positive, got {sigma_source}")
    sigmas_list = [float(s) for s in sigmas_receive]
    if len(sigmas_list) == 0:
        raise ValueError("sigmas_receive must contain at least one value")
    if any(s <= 0.0 for s in sigmas_list):
        raise ValueError("all sigmas_receive entries must be positive")

    mode_s = NormMode(normalize_source)
    mode_r = NormMode(normalize_receive)

    out = torch.zeros((len(sigmas_list), max_L + 1), dtype=torch.float64)
    for L in range(max_L + 1):
        cl_s = inv_cl(sigma_source, L, mode_s)
        l_weight = field_constant / (2 * L + 1)
        for i, sigma_r in enumerate(sigmas_list):
            radial = _overlap_radial_integral(
                L, sigma_source, sigma_r, grid_size, r_max_factor
            )
            cl_r = inv_cl(sigma_r, L, mode_r)
            out[i, L] = cl_s * cl_r * l_weight * radial
    return out


def flatten_to_reference_layout(
    constants_by_sigma_L: torch.Tensor, max_L: int
) -> torch.Tensor:
    r"""Convert ``(N_sigma, L+1)`` layout to the customer reference flat layout.

    The customer's index scheme is

    .. math::

        \text{idx}(L, m, i_\sigma) = N_\sigma \cdot L^2 + m + i_\sigma \cdot (2L + 1)

    Each overlap constant is broadcast to ``2L + 1`` ``m``-slots (the constant
    does not depend on ``m``). Useful only for parity testing against the
    reference implementation — production code should index the clean
    ``(N_sigma, L+1)`` tensor directly.
    """
    if max_L < 0:
        raise ValueError(f"max_L must be non-negative, got {max_L}")
    constants = torch.as_tensor(constants_by_sigma_L, dtype=torch.float64)
    if tuple(constants.shape) != (constants.shape[0], max_L + 1):
        raise ValueError(
            "constants_by_sigma_L must have shape (N_sigma, max_L + 1); "
            f"got {tuple(constants.shape)} for max_L={max_L}"
        )
    n_sigma = constants.shape[0]
    total_len = n_sigma * (max_L + 1) ** 2
    flat = torch.zeros(total_len, dtype=torch.float64)
    for L in range(max_L + 1):
        base = n_sigma * L * L
        width = 2 * L + 1
        for i_sigma in range(n_sigma):
            val = float(constants[i_sigma, L])
            for m in range(width):
                flat[base + m + i_sigma * width] = val
    return flat
