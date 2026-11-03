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

"""Force-accuracy regression for automatic PME sizing on a displaced crystal."""

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    estimate_pme_parameters,
    ewald_summation,
    generate_k_vectors_pme,
    particle_mesh_ewald,
)
from nvalchemiops.torch.neighbors import neighbor_list
from test.interactions.electrostatics.conftest import create_cscl_supercell


@pytest.mark.parametrize("spline_order", [None, 4, 6])
@pytest.mark.gpu
def test_automatic_mesh_preserves_upstream_force_accuracy(
    spline_order: int | None,
) -> None:
    """Automatic sizing preserves the independently checked upstream force error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    crystal = create_cscl_supercell(4)
    positions = torch.as_tensor(crystal.positions, dtype=torch.float64, device="cuda")
    phase = torch.arange(positions.numel(), device="cuda").reshape_as(positions)
    positions = (positions + 0.07 * torch.sin(phase)).requires_grad_(True)
    charges = torch.as_tensor(crystal.charges, dtype=torch.float64, device="cuda")
    cell = torch.as_tensor(crystal.cell, dtype=torch.float64, device="cuda")
    order_kwargs = {} if spline_order is None else {"spline_order": spline_order}
    parameters = estimate_pme_parameters(
        positions, cell, accuracy=1e-6, real_space_cutoff=9.0, **order_kwargs
    )
    if spline_order is None:
        assert parameters.mesh_dimensions == (80, 80, 80)
    pairs, pointers, shifts = neighbor_list(
        positions.detach(),
        9.0,
        cell=cell,
        pbc=torch.ones(3, dtype=torch.bool, device="cuda"),
        method="naive",
        return_neighbor_list=True,
    )
    common = dict(
        alpha=parameters.alpha,
        neighbor_list=pairs,
        neighbor_ptr=pointers,
        neighbor_shifts=shifts,
    )
    # Use the same real-space truncation to isolate reciprocal mesh error.
    reference = ewald_summation(positions, charges, cell, k_cutoff=5.0, **common)
    expected_forces = -torch.autograd.grad(reference.sum(), positions)[0]
    upstream_mesh = (128, 128, 128)
    vectors, squared = generate_k_vectors_pme(cell, upstream_mesh)
    upstream = particle_mesh_ewald(
        positions,
        charges,
        cell,
        mesh_dimensions=upstream_mesh,
        spline_order=4 if spline_order is None else spline_order,
        k_vectors=vectors,
        k_squared=squared,
        **common,
    )
    upstream_forces = -torch.autograd.grad(upstream.sum(), positions)[0]
    automatic = particle_mesh_ewald(
        positions, charges, cell, accuracy=1e-6, **order_kwargs, **common
    )
    actual_forces = -torch.autograd.grad(automatic.sum(), positions)[0]
    upstream_error = torch.linalg.vector_norm(upstream_forces - expected_forces)
    actual_error = torch.linalg.vector_norm(actual_forces - expected_forces)
    assert actual_error <= upstream_error + torch.finfo(torch.float64).eps ** 0.5
    if spline_order is not None:
        torch.testing.assert_close(
            actual_forces, upstream_forces, rtol=1e-10, atol=1e-12
        )
