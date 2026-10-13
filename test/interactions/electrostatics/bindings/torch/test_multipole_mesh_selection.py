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

"""Automatic multipole PME mesh sizing retains the execution spline support."""

import inspect
import math

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics.parameters import (
    estimate_multipole_pme_parameters,
)
from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
    multipole_particle_mesh_ewald,
)
from nvalchemiops.torch.neighbors import neighbor_list


def test_multipole_estimator_exposes_the_shared_padding_budget() -> None:
    """Multipole setup exposes the same keyword-only mesh padding default."""
    parameter = inspect.signature(estimate_multipole_pme_parameters).parameters[
        "fft_padding_fraction"
    ]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 0.25


def test_multipole_estimator_can_disable_mesh_snapping() -> None:
    """Multipole padding changes its mesh while preserving physical parameters."""
    positions = torch.zeros((128, 3), dtype=torch.float64)
    cell = 20.0 * torch.eye(3, dtype=torch.float64)
    smallest = estimate_multipole_pme_parameters(
        positions, cell, sigma=0.1, accuracy=1e-6, fft_padding_fraction=0
    )
    snapped = estimate_multipole_pme_parameters(
        positions, cell, sigma=0.1, accuracy=1e-6
    )
    assert smallest.mesh_dimensions == (45, 45, 45)
    assert snapped.mesh_dimensions == (48, 48, 48)
    assert math.prod(snapped.mesh_dimensions) <= 1.25 * math.prod(
        smallest.mesh_dimensions
    )
    torch.testing.assert_close(snapped.alpha, smallest.alpha, rtol=0, atol=0)
    torch.testing.assert_close(
        snapped.real_space_cutoff, smallest.real_space_cutoff, rtol=0, atol=0
    )
    torch.testing.assert_close(snapped.sigma, smallest.sigma, rtol=0, atol=0)


@pytest.mark.parametrize("padding", [-1, math.nan, math.inf, True])
def test_multipole_estimator_rejects_invalid_padding_budget(padding: float) -> None:
    """Multipole sizing validates its extra-point budget consistently."""
    with pytest.raises(ValueError, match="fft_padding_fraction"):
        estimate_multipole_pme_parameters(
            torch.zeros((128, 3), dtype=torch.float64),
            20.0 * torch.eye(3, dtype=torch.float64),
            sigma=0.1,
            fft_padding_fraction=padding,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_multipole_estimate_covers_requested_assignment_order(device: str) -> None:
    """Order six covers its support when the upstream continuous target is below six."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, device=device)
    cell = 20.0 * torch.eye(3, dtype=torch.float64, device=device)
    default = estimate_multipole_pme_parameters(
        positions, cell, sigma=0.1, accuracy=1e-2
    )
    explicit_four = estimate_multipole_pme_parameters(
        positions, cell, sigma=0.1, accuracy=1e-2, spline_order=4
    )
    order_six = estimate_multipole_pme_parameters(
        positions, cell, sigma=0.1, accuracy=1e-2, spline_order=6
    )
    assert default.mesh_dimensions == (5, 5, 5)
    assert explicit_four.mesh_dimensions == (4, 4, 4)
    assert order_six.mesh_dimensions == (8, 8, 8)
    torch.testing.assert_close(order_six.alpha, default.alpha, rtol=0, atol=0)
    torch.testing.assert_close(
        order_six.real_space_cutoff, default.real_space_cutoff, rtol=0, atol=0
    )
    torch.testing.assert_close(order_six.sigma, default.sigma, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("explicit_alpha", [False, True])
def test_multipole_automatic_order_six_matches_explicit_mesh(
    explicit_alpha: bool,
) -> None:
    """Full multipole PME resolves order-six support for both automatic-alpha paths."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    positions = torch.tensor(
        [[1.0, 2.0, 3.0]], dtype=torch.float64, device="cuda", requires_grad=True
    )
    moments = torch.tensor(
        [[1.0]], dtype=torch.float64, device="cuda", requires_grad=True
    )
    cell = 20.0 * torch.eye(3, dtype=torch.float64, device="cuda")
    parameters = estimate_multipole_pme_parameters(
        positions, cell, sigma=0.1, accuracy=1e-2, spline_order=6
    )
    pairs, pointers, shifts = neighbor_list(
        positions,
        float(parameters.real_space_cutoff.max()),
        cell=cell,
        pbc=torch.ones(3, dtype=torch.bool, device="cuda"),
        method="naive",
        return_neighbor_list=True,
    )
    common = {
        "sigma": 0.1,
        "spline_order": 6,
        "accuracy": 1e-2,
    }
    alpha = float(parameters.alpha.item())
    arguments = (positions, moments, cell, pairs[1].contiguous(), pointers, shifts)
    automatic = multipole_particle_mesh_ewald(
        *arguments, alpha=alpha if explicit_alpha else None, **common
    )
    explicit = multipole_particle_mesh_ewald(
        *arguments, alpha=alpha, mesh_dimensions=parameters.mesh_dimensions, **common
    )
    assert torch.isfinite(automatic).all()
    torch.testing.assert_close(automatic, explicit, rtol=1e-12, atol=1e-12)
    automatic_gradients = torch.autograd.grad(automatic.sum(), (positions, moments))
    explicit_gradients = torch.autograd.grad(explicit.sum(), (positions, moments))
    for actual, expected in zip(automatic_gradients, explicit_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
