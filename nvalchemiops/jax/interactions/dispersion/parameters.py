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

"""Parameter estimation for LJ-PME (dispersion PME).

Implements the GROMACS-style two-tail accuracy criterion of Wennberg
et al. (JCTC 2013): pick a real-space cutoff, then choose β so the
damped real-space tail at the cutoff is below the target accuracy, and
choose mesh dimensions so the reciprocal-space truncation tail matches
the same accuracy.

The relative real-space tail at cutoff is

.. math::

    \\frac{|V_{\\text{real}}(r_c) - V(r_c)|}{|V(r_c)|}
    \\approx g(\\beta r_c) = e^{-(\\beta r_c)^2}
        \\bigl(1 + (\\beta r_c)^2 + (\\beta r_c)^4/2\\bigr).

Setting :math:`g(x_\\varepsilon) = \\varepsilon` and writing
:math:`\\beta = x_\\varepsilon / r_c` gives a one-parameter family of
matched (β, r_c). The reciprocal-space tail uses the same threshold
:math:`x_\\varepsilon` to bound the high-k truncation:

.. math::

    \\frac{\\pi N}{2 L \\beta} \\geq x_\\varepsilon
    \\;\\Longleftrightarrow\\;
    N \\geq \\frac{2 L \\beta x_\\varepsilon}{\\pi}.

GROMACS's default ``ewald-rtol-lj = 1e-3`` is a reasonable starting
accuracy (looser than Coulomb).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = [
    "PMEDispersionParameters",
    "estimate_pme_dispersion_parameters",
    "estimate_pme_dispersion_mesh_dimensions",
    "solve_dispersion_beta",
    "DISPERSION_DEFAULT_ACCURACY",
    "DISPERSION_DEFAULT_CUTOFF",
]


# GROMACS default for the dispersion-PME accuracy. Looser than Coulomb
# (1e-5/1e-6) because the r^-6 kernel is short-range relative to 1/r.
DISPERSION_DEFAULT_ACCURACY = 1e-3
DISPERSION_DEFAULT_CUTOFF = 9.0  # Å — typical LJ cutoff


@dataclass
class PMEDispersionParameters:
    """Container for LJ-PME parameters.

    Attributes
    ----------
    beta : jax.Array, shape (B,)
        Dispersion Ewald splitting parameter.
    cutoff : float
        Real-space cutoff (shared across systems).
    mesh_dimensions : tuple[int, int, int]
        FFT mesh dimensions (nx, ny, nz), the largest needed across the
        batch.
    mesh_spacing : jax.Array, shape (B, 3)
        Actual mesh spacing along each lattice direction.
    accuracy : float
        Target relative accuracy (e.g. 1e-3).
    """

    beta: jax.Array
    cutoff: float
    mesh_dimensions: tuple[int, int, int]
    mesh_spacing: jax.Array
    accuracy: float


def solve_dispersion_beta(cutoff: float, accuracy: float) -> float:
    """Solve :math:`g(\\beta \\cdot r_c) = \\varepsilon` for β·r_c, return β.

    Uses bisection over :math:`x = \\beta r_c \\in [0.1, 20]`. :math:`g(x)` is
    monotonically decreasing on this interval, so bisection is unconditionally
    convergent.

    Parameters
    ----------
    cutoff : float
        Real-space cutoff :math:`r_c`.
    accuracy : float
        Target relative accuracy :math:`\\varepsilon` (e.g. 1e-3).

    Returns
    -------
    float
        Dispersion splitting parameter β.
    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if not (0 < accuracy < 1):
        raise ValueError(f"accuracy must be in (0, 1), got {accuracy}")

    def g(x: float) -> float:
        return math.exp(-x * x) * (1.0 + x * x + 0.5 * x * x * x * x)

    lo, hi = 0.1, 20.0
    if g(lo) < accuracy:
        # Threshold already met at small x: β can be very small.
        return lo / cutoff
    if g(hi) > accuracy:
        # Threshold unreachable in our window: clamp to maximum β.
        return hi / cutoff

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if g(mid) > accuracy:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    x_eps = 0.5 * (lo + hi)
    return x_eps / cutoff


def _next_fft_friendly(n: int) -> int:
    """Round ``n`` up to the next integer of the form :math:`2^a 3^b 5^c`.

    FFT libraries are typically fastest on such mesh dimensions. Falls back
    to the next power of 2 if no small composite is close enough.
    """
    if n <= 1:
        return 1
    candidate = n
    while True:
        m = candidate
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return candidate
        candidate += 1
        if candidate > 4 * n:
            # safeguard: fall back to next power of 2
            return 1 << (n - 1).bit_length()


def estimate_pme_dispersion_mesh_dimensions(
    cell: jax.Array,
    beta: jax.Array,
    accuracy: float = DISPERSION_DEFAULT_ACCURACY,
) -> tuple[int, int, int]:
    """Pick mesh dimensions matching the dispersion-PME accuracy target.

    The high-k truncation tail of the dispersion Green's function is

    .. math::

        \\frac{|V_{\\text{recip,trunc}}|}{|V_{\\text{recip}}|}
        \\approx g\\!\\left(\\tfrac{\\pi N}{2 L \\beta}\\right),

    so requiring this to be below :math:`\\varepsilon` gives
    :math:`N \\geq 2 L \\beta x_\\varepsilon / \\pi` with the same
    :math:`x_\\varepsilon` that fixes β.

    Parameters
    ----------
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrix.
    beta : jax.Array, scalar or shape (B,)
        Dispersion Ewald splitting parameter.
    accuracy : float
        Target relative accuracy.

    Returns
    -------
    tuple[int, int, int]
        Mesh dimensions, shared across the batch (max over systems and
        axes), rounded up to a 2-3-5-smooth integer.
    """
    if cell.ndim == 2:
        cell = cell[None, ...]

    # Cell lengths along each axis: (B, 3)
    cell_lengths = jnp.linalg.norm(cell, axis=2)

    # Solve g(x_eps) = accuracy once (host-side); reuse for all axes.
    def g(x: float) -> float:
        return math.exp(-x * x) * (1.0 + x * x + 0.5 * x * x * x * x)

    lo, hi = 0.1, 20.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if g(mid) > accuracy:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    x_eps = 0.5 * (lo + hi)

    beta_arr = jnp.asarray(beta, dtype=cell_lengths.dtype)
    if beta_arr.ndim == 0:
        beta_arr = beta_arr[None]
    # N >= 2 L beta x_eps / pi
    n_per_axis = 2.0 * cell_lengths * beta_arr[:, None] * x_eps / math.pi
    max_n = jnp.max(n_per_axis, axis=0)  # (3,)
    raw = [int(math.ceil(float(max_n[i]))) for i in range(3)]
    return (
        _next_fft_friendly(raw[0]),
        _next_fft_friendly(raw[1]),
        _next_fft_friendly(raw[2]),
    )


def estimate_pme_dispersion_parameters(
    cell: jax.Array,
    cutoff: float = DISPERSION_DEFAULT_CUTOFF,
    accuracy: float = DISPERSION_DEFAULT_ACCURACY,
) -> PMEDispersionParameters:
    """Estimate (β, mesh) for LJ-PME at a target accuracy.

    Picks β so that :math:`g(\\beta r_c) \\le \\varepsilon` (real-space tail at
    the cutoff) and then mesh dimensions so that the reciprocal-space
    truncation tail matches the same threshold.

    Parameters
    ----------
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrix.
    cutoff : float, default=9.0
        Real-space cutoff in cell-length units (Å for typical inputs).
    accuracy : float, default=1e-3
        Target relative accuracy. GROMACS's default for LJ-PME.

    Returns
    -------
    PMEDispersionParameters
        β (per system), shared cutoff, mesh dimensions, mesh spacing,
        and the accuracy threshold used.
    """
    if cell.ndim == 2:
        cell = cell[None, ...]
    num_systems = cell.shape[0]

    beta_scalar = solve_dispersion_beta(cutoff, accuracy)
    beta = jnp.full(num_systems, beta_scalar, dtype=cell.dtype)

    mesh_dims = estimate_pme_dispersion_mesh_dimensions(cell, beta, accuracy)

    cell_lengths = jnp.linalg.norm(cell, axis=2)  # (B, 3)
    mesh_dims_arr = jnp.asarray(mesh_dims, dtype=cell_lengths.dtype)
    mesh_spacing = cell_lengths / mesh_dims_arr

    return PMEDispersionParameters(
        beta=beta,
        cutoff=cutoff,
        mesh_dimensions=mesh_dims,
        mesh_spacing=mesh_spacing,
        accuracy=accuracy,
    )
