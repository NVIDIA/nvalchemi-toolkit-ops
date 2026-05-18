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

"""Parameter estimation for LJ-PME (PyTorch).

Mirror of ``nvalchemiops/jax/interactions/dispersion/parameters.py`` —
see that module for the underlying math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "PMEDispersionParameters",
    "estimate_pme_dispersion_parameters",
    "estimate_pme_dispersion_mesh_dimensions",
    "solve_dispersion_beta",
    "DISPERSION_DEFAULT_ACCURACY",
    "DISPERSION_DEFAULT_CUTOFF",
]


DISPERSION_DEFAULT_ACCURACY = 1e-3
DISPERSION_DEFAULT_CUTOFF = 9.0


@dataclass
class PMEDispersionParameters:
    """Container for LJ-PME parameters (PyTorch).

    Attributes
    ----------
    beta : torch.Tensor, shape (B,)
        Dispersion Ewald splitting parameter.
    cutoff : float
        Real-space cutoff (shared across systems).
    mesh_dimensions : tuple[int, int, int]
        FFT mesh dimensions.
    mesh_spacing : torch.Tensor, shape (B, 3)
        Actual mesh spacing along each lattice direction.
    accuracy : float
        Target relative accuracy.
    """

    beta: torch.Tensor
    cutoff: float
    mesh_dimensions: tuple[int, int, int]
    mesh_spacing: torch.Tensor
    accuracy: float


def solve_dispersion_beta(cutoff: float, accuracy: float) -> float:
    """Solve :math:`g(\\beta \\cdot r_c) = \\varepsilon` for :math:`\\beta r_c`, return β."""
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if not (0 < accuracy < 1):
        raise ValueError(f"accuracy must be in (0, 1), got {accuracy}")

    def g(x: float) -> float:
        return math.exp(-x * x) * (1.0 + x * x + 0.5 * x * x * x * x)

    lo, hi = 0.1, 20.0
    if g(lo) < accuracy:
        return lo / cutoff
    if g(hi) > accuracy:
        return hi / cutoff

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if g(mid) > accuracy:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi) / cutoff


def _next_fft_friendly(n: int) -> int:
    """Round ``n`` up to the next integer of the form :math:`2^a 3^b 5^c`."""
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
            return 1 << (n - 1).bit_length()


def estimate_pme_dispersion_mesh_dimensions(
    cell: torch.Tensor,
    beta: float | torch.Tensor,
    accuracy: float = DISPERSION_DEFAULT_ACCURACY,
) -> tuple[int, int, int]:
    """Pick mesh dimensions for LJ-PME at a target accuracy."""
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)

    cell_lengths = torch.linalg.norm(cell, dim=2)  # (B, 3)

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

    if isinstance(beta, (int, float)):
        beta_t = torch.tensor(
            [float(beta)], dtype=cell_lengths.dtype, device=cell.device
        )
    else:
        beta_t = beta.to(dtype=cell_lengths.dtype, device=cell.device)
        if beta_t.dim() == 0:
            beta_t = beta_t.reshape(1)

    n_per_axis = 2.0 * cell_lengths * beta_t.view(-1, 1) * x_eps / math.pi
    max_n = n_per_axis.max(dim=0).values
    raw = [int(math.ceil(float(max_n[i]))) for i in range(3)]
    return (
        _next_fft_friendly(raw[0]),
        _next_fft_friendly(raw[1]),
        _next_fft_friendly(raw[2]),
    )


def estimate_pme_dispersion_parameters(
    cell: torch.Tensor,
    cutoff: float = DISPERSION_DEFAULT_CUTOFF,
    accuracy: float = DISPERSION_DEFAULT_ACCURACY,
) -> PMEDispersionParameters:
    """Estimate (β, mesh) for LJ-PME at a target accuracy."""
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    num_systems = cell.shape[0]

    beta_scalar = solve_dispersion_beta(cutoff, accuracy)
    beta = torch.full((num_systems,), beta_scalar, dtype=cell.dtype, device=cell.device)

    mesh_dims = estimate_pme_dispersion_mesh_dimensions(cell, beta, accuracy)

    cell_lengths = torch.linalg.norm(cell, dim=2)
    mesh_dims_t = torch.tensor(
        mesh_dims, dtype=cell_lengths.dtype, device=cell_lengths.device
    )
    mesh_spacing = cell_lengths / mesh_dims_t

    return PMEDispersionParameters(
        beta=beta,
        cutoff=cutoff,
        mesh_dimensions=mesh_dims,
        mesh_spacing=mesh_spacing,
        accuracy=accuracy,
    )
