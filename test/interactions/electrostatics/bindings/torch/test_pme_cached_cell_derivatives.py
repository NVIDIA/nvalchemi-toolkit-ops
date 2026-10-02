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

"""Cell derivatives through public PME calls with supplied setup caches."""

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    generate_k_squared_pme,
    generate_k_vectors_pme,
    particle_mesh_ewald,
    pme_reciprocal_space,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("cache_mode", ["squared_only", "paired", "all_cell_metadata"])
@pytest.mark.parametrize("full_api", [False, True])
@pytest.mark.parametrize("spline_order", [4, 5])
def test_cached_cell_first_and_second_derivatives_match_uncached(
    device: str,
    batched: bool,
    weighted: bool,
    cache_mode: str,
    full_api: bool,
    spline_order: int,
) -> None:
    """Cached cell metadata preserves weighted gradients and directional Hessians."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    dtype = torch.float64
    positions = torch.tensor(
        [
            [1.13, 1.07, 1.19],
            [2.13, 1.57, 1.19],
            [1.63, 2.57, 2.19],
            [3.13, 2.07, 1.19],
        ],
        dtype=dtype,
        device=device,
    )
    charges = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=dtype, device=device)
    cell = torch.diag(torch.tensor([8.0, 9.0, 10.0], dtype=dtype, device=device))
    batch_idx = None
    if batched:
        positions = torch.cat([positions, positions + 0.31])
        charges = torch.cat([charges, charges * 0.7])
        cell = torch.stack([cell, cell * 1.1])
        batch_idx = torch.repeat_interleave(
            torch.arange(2, dtype=torch.int32, device=device), 4
        )
    mesh_dimensions = (9, 10, 12)
    if cache_mode == "squared_only":
        cache = {"k_squared": generate_k_squared_pme(cell, mesh_dimensions)}
    else:
        k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dimensions)
        cache = {"k_vectors": k_vectors, "k_squared": k_squared}
        if cache_mode == "all_cell_metadata":
            cache.update(
                volume=torch.abs(torch.linalg.det(cell)),
                cell_inv_t=torch.linalg.inv(cell).transpose(-1, -2).contiguous(),
            )
    common = {
        "alpha": 0.3,
        "mesh_dimensions": mesh_dimensions,
        "spline_order": spline_order,
        "batch_idx": batch_idx,
    }
    if full_api:
        pairs = [
            (offset + source, offset + destination)
            for offset in range(0, len(positions), 4)
            for source in range(4)
            for destination in range(4)
            if source != destination
        ]
        common.update(
            neighbor_list=torch.tensor(pairs, dtype=torch.int32, device=device)
            .transpose(0, 1)
            .contiguous(),
            neighbor_ptr=torch.arange(
                0, 3 * len(positions) + 1, 3, dtype=torch.int32, device=device
            ),
            neighbor_shifts=torch.zeros(
                len(pairs), 3, dtype=torch.int32, device=device
            ),
        )
    selected_api = particle_mesh_ewald if full_api else pme_reciprocal_space
    weights = (
        torch.linspace(0.2, 1.1, len(positions), dtype=dtype, device=device)
        if weighted
        else torch.ones(len(positions), dtype=dtype, device=device)
    )
    direction = (
        torch.arange(cell.numel(), dtype=dtype, device=device).reshape_as(cell) / 100
    )

    def derivatives(setup_cache: dict, create_graph: bool):
        current_cell = cell.detach().clone().requires_grad_(True)
        energy = selected_api(positions, charges, current_cell, **common, **setup_cache)
        gradient = torch.autograd.grad(
            torch.sum(energy * weights), current_cell, create_graph=create_graph
        )[0]
        hessian_vector = (
            torch.autograd.grad(torch.sum(gradient * direction), current_cell)[0]
            if create_graph
            else None
        )
        return energy.detach(), gradient.detach(), hessian_vector

    reference_energy, reference_gradient, reference_hessian = derivatives({}, True)
    cached_energy, cached_gradient, _ = derivatives(cache, False)
    cached_graph_energy, cached_graph_gradient, cached_hessian = derivatives(
        cache, True
    )
    for actual in (cached_energy, cached_graph_energy):
        torch.testing.assert_close(actual, reference_energy, rtol=1e-10, atol=1e-10)
    for actual in (cached_gradient, cached_graph_gradient):
        torch.testing.assert_close(actual, reference_gradient, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(cached_hessian, reference_hessian, rtol=1e-8, atol=1e-10)
