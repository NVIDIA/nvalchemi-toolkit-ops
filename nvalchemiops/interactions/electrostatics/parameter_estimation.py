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

"""Framework-agnostic helpers for PME parameter estimation.

The cost-optimal real-space cutoff for PME differs from pure Ewald
because the reciprocal-space cost scales as ``K^3 log(K)`` (FFT) rather
than ``K^3`` (k-space sum). This module exposes a pure-Python (stdlib-
only) cost model + 1D golden-section minimizer that both the torch and
JAX parameter estimators dispatch to when the caller doesn't supply an
explicit ``real_space_cutoff``.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence


def _next_pow2(x: float) -> int:
    """Smallest power-of-2 ``≥ max(x, 1)``."""
    return 1 << max(0, math.ceil(math.log2(max(x, 1.0))))


def pme_cost_model(
    rc: float,
    *,
    num_atoms: float,
    volume: float,
    cell_lengths: Sequence[float],
    accuracy: float,
    cost_ratio_pair_to_fft: float,
    mesh_safety_factor: float = 1.0,
) -> float:
    """Approximate total PME cost (arbitrary units) at a given real-space cutoff.

    The cost decomposes as

    .. math::
        C(r_c) = N \\rho \\frac{4\\pi}{3} r_c^3
                 + \\tau \\, K_{tot} \\log K_{tot}

    where the second term is the FFT cost (proportional to mesh size and
    its log) and ``τ`` (``cost_ratio_pair_to_fft``) re-weights the FFT
    contribution to match a given hardware. Both ``α`` and the per-axis
    mesh sizes ``K_i`` are derived from ``r_c`` via the standard accuracy
    constraints:

      - ``α = √(-log(2 ε)) / r_c``  (real-space erfc-truncation error ≤ ε)
      - ``K_i = next_pow2(mesh_safety_factor · 2 α L_i / (3 ε^{1/5}))``
        (matches the formula in ``estimate_pme_mesh_dimensions``)

    The cost model MUST use the same K formula as the runtime mesh
    estimator; otherwise the rc-minimizer balances against the wrong
    FFT cost and biases the chosen rc.

    Parameters
    ----------
    rc : float
        Trial real-space cutoff (Å, in whatever length unit the caller uses).
    num_atoms : float
        Atom count of a representative system.
    volume : float
        Cell volume.
    cell_lengths : sequence of 3 floats
        Per-axis cell lengths.
    accuracy : float
        Target accuracy (relative truncation error).
    cost_ratio_pair_to_fft : float
        Hardware-dependent scaling factor. ``1.0`` weights pair operations
        and FFT butterflies equally; smaller values favor smaller-mesh
        / larger-cutoff solutions.
    mesh_safety_factor : float, default=1.0
        Mirror the value passed to ``estimate_pme_mesh_dimensions`` so
        the predicted K matches the runtime K.

    Returns
    -------
    float
        Total cost in arbitrary units (only the ratio between calls matters).
    """
    rho = num_atoms / volume
    real_cost = num_atoms * rho * (4.0 / 3.0) * math.pi * rc**3

    c_acc_real = math.sqrt(-math.log(2.0 * accuracy))
    alpha = c_acc_real / rc

    accuracy_factor = 3.0 * (accuracy**0.2)
    k_total = 1
    for L in cell_lengths:
        k_min = mesh_safety_factor * 2.0 * alpha * L / accuracy_factor
        k_total *= _next_pow2(k_min)

    fft_cost = k_total * math.log(max(k_total, 2.0))
    return real_cost + cost_ratio_pair_to_fft * fft_cost


def golden_section_minimize(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 0.01,
    max_iter: int = 100,
) -> float:
    """Find ``x*`` minimizing a unimodal ``f`` on ``[a, b]`` to tolerance ``tol``.

    Pure-stdlib 1D minimizer (no scipy). Converges linearly with ratio
    ``φ^-1 ≈ 0.618`` per iteration — ~25 iterations reach 0.01-Å precision
    on a typical 4–25 Å bracket.

    ``f`` should be approximately unimodal; the PME cost surface is
    piecewise-monotone with discrete jumps at mesh-tier boundaries
    (power-of-2 ``next_pow2``), which golden section handles correctly by
    evaluating ``f`` at the bracket points without needing derivatives.
    """
    if b <= a:
        return a
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.618
    x1 = b - inv_phi * (b - a)
    x2 = a + inv_phi * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(max_iter):
        if b - a <= tol:
            break
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - inv_phi * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + inv_phi * (b - a)
            f2 = f(x2)
    return 0.5 * (a + b)


def find_optimal_pme_cutoff(
    num_atoms: float,
    volume: float,
    cell_lengths: Sequence[float],
    accuracy: float = 1e-6,
    cost_ratio_pair_to_fft: float = 1.0,
    mesh_safety_factor: float = 1.0,
    rc_min: float = 2.0,
    rc_max: float | None = None,
) -> float:
    """Cost-optimal PME real-space cutoff via 1D golden-section search.

    Wraps ``pme_cost_model`` + ``golden_section_minimize``. ``rc_max``
    defaults to ``min(cell_lengths) / 2`` (PBC sanity — no atom in the
    minimum image can be further than half the smallest cell length).

    ``mesh_safety_factor`` is forwarded to the cost model so the K it
    predicts at each trial rc matches the K the runtime mesh estimator
    will actually use. A mismatch here biases the chosen rc upward.
    """
    if rc_max is None:
        rc_max = 0.5 * min(cell_lengths)
    if rc_max <= rc_min:
        return rc_min

    def _cost(rc: float) -> float:
        return pme_cost_model(
            rc,
            num_atoms=num_atoms,
            volume=volume,
            cell_lengths=cell_lengths,
            accuracy=accuracy,
            cost_ratio_pair_to_fft=cost_ratio_pair_to_fft,
            mesh_safety_factor=mesh_safety_factor,
        )

    return golden_section_minimize(_cost, rc_min, rc_max, tol=0.01)


def alpha_from_cutoff(real_space_cutoff: float, accuracy: float) -> float:
    """Solve ``α`` from the real-space accuracy constraint.

    ``erfc(α·r_c) / r_c ≤ ε``  ⇒  ``α·r_c ≈ √(-log(2 ε))``

    so ``α = √(-log(2 ε)) / r_c``. Used by the framework-specific PME
    parameter estimators once a real-space cutoff has been chosen (or
    supplied by the caller).
    """
    return math.sqrt(-math.log(2.0 * accuracy)) / real_space_cutoff
