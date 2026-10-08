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

"""Cell derivatives through public JAX PME calls with supplied setup caches."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.interactions.electrostatics import (
    generate_k_squared_pme,
    generate_k_vectors_pme,
    particle_mesh_ewald,
    pme_reciprocal_space,
)

pytestmark = pytest.mark.usefixtures("device")


def test_squared_cache_preserves_reciprocal_cell_gradient() -> None:
    """A prepared squared-grid cache preserves physical first cell derivatives."""
    positions = jnp.array([[1.13, 1.07, 1.19], [2.13, 1.57, 1.19]], dtype=jnp.float64)
    charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
    cell = 8.0 * jnp.eye(3, dtype=jnp.float64)
    mesh_dimensions = (8, 8, 8)
    k_squared = generate_k_squared_pme(cell, mesh_dimensions)

    def energy(current_cell, cache):
        return pme_reciprocal_space(
            positions,
            charges,
            current_cell,
            alpha=jnp.array([0.3], dtype=jnp.float64),
            mesh_dimensions=mesh_dimensions,
            **cache,
        ).sum()

    reference = jax.grad(lambda current_cell: energy(current_cell, {}))(cell)
    actual = jax.grad(
        lambda current_cell: energy(current_cell, {"k_squared": k_squared})
    )(cell)
    np.testing.assert_allclose(actual, reference, rtol=1e-8, atol=1e-10)


def _cached_fixture(
    batched: bool, cache_mode: str, full_api: bool, spline_order: int = 4
) -> tuple:
    """Prepare a single or heterogeneous pair fixture through public setup APIs."""
    positions = jnp.array([[1.13, 1.07, 1.19], [2.13, 1.57, 1.19]], dtype=jnp.float64)
    charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
    cell = jnp.diag(jnp.array([8.0, 9.0, 10.0], dtype=jnp.float64))
    batch_idx = None
    if batched:
        positions = jnp.concatenate([positions, positions + 0.31])
        charges = jnp.concatenate([charges, charges * 0.7])
        cell = jnp.stack([cell, cell * 1.1])
        batch_idx = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 2)
    mesh_dimensions = (9, 10, 12)
    if cache_mode == "squared_only":
        cache = {"k_squared": generate_k_squared_pme(cell, mesh_dimensions)}
    else:
        k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dimensions)
        cache = {"k_vectors": k_vectors, "k_squared": k_squared}
        if cache_mode == "all_cell_metadata":
            cache.update(
                volume=jnp.abs(jnp.linalg.det(cell)),
                cell_inv_t=jnp.swapaxes(jnp.linalg.inv(cell), -1, -2),
            )
    common = {
        "alpha": jnp.full((2 if batched else 1,), 0.3, dtype=jnp.float64),
        "mesh_dimensions": mesh_dimensions,
        "spline_order": spline_order,
        "batch_idx": batch_idx,
    }
    if full_api:
        sources = jnp.arange(len(positions), dtype=jnp.int32)
        destinations = sources ^ 1
        common.update(
            neighbor_list=jnp.stack([sources, destinations]),
            neighbor_ptr=jnp.arange(len(positions) + 1, dtype=jnp.int32),
            neighbor_shifts=jnp.zeros((len(positions), 3), dtype=jnp.int32),
        )
    selected_api = particle_mesh_ewald if full_api else pme_reciprocal_space
    return positions, charges, cell, selected_api, common, cache


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("cache_mode", ["squared_only", "paired", "all_cell_metadata"])
@pytest.mark.parametrize("full_api", [False, True])
@pytest.mark.parametrize("spline_order", [4, 5])
@pytest.mark.parametrize("jit", [False, True])
def test_cached_cell_gradient_matches_uncached(
    batched: bool,
    weighted: bool,
    cache_mode: str,
    full_api: bool,
    jit: bool,
    spline_order: int,
) -> None:
    """Prepared metadata preserves eager and JIT cell gradients for per-atom losses."""
    positions, charges, cell, selected_api, common, cache = _cached_fixture(
        batched, cache_mode, full_api, spline_order
    )
    weights = (
        jnp.linspace(0.2, 1.1, len(positions), dtype=jnp.float64)
        if weighted
        else jnp.ones(len(positions), dtype=jnp.float64)
    )

    def evaluate(setup_cache):
        def loss(current_cell):
            energies = selected_api(
                positions, charges, current_cell, **common, **setup_cache
            )
            return jnp.sum(energies * weights), energies

        function = jax.value_and_grad(loss, has_aux=True)
        if jit:
            function = jax.jit(function)
        return function(cell)

    (reference_loss, reference_energy), reference_gradient = evaluate({})
    (actual_loss, actual_energy), actual_gradient = evaluate(cache)
    np.testing.assert_allclose(actual_loss, reference_loss, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual_energy, reference_energy, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        actual_gradient, reference_gradient, rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("cache_mode", ["squared_only", "paired", "all_cell_metadata"])
@pytest.mark.parametrize("full_api", [False, True])
def test_cached_cell_hvp_preserves_supported_derivative_boundary(
    batched: bool, cache_mode: str, full_api: bool
) -> None:
    """Supplied caches preserve the public first-order-only cell derivative boundary."""
    positions, charges, cell, selected_api, common, cache = _cached_fixture(
        batched, cache_mode, full_api
    )

    def loss(current_cell):
        return selected_api(positions, charges, current_cell, **common, **cache).sum()

    with pytest.raises(
        (NotImplementedError, TypeError),
        match="cell/strain HVPs|forward-mode autodiff",
    ):
        jax.jvp(jax.grad(loss), (cell,), (jnp.ones_like(cell),))


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("cache_mode", ["squared_only", "paired", "all_cell_metadata"])
@pytest.mark.parametrize("spline_order", [4, 5])
@pytest.mark.parametrize("full_api", [False, True])
def test_cached_position_charge_hvp_matches_uncached(
    batched: bool, cache_mode: str, full_api: bool, spline_order: int
) -> None:
    """Fixed-cell setup caches preserve joint position and charge Hessian products."""
    positions, charges, cell, selected_api, common, cache = _cached_fixture(
        batched, cache_mode, full_api, spline_order
    )
    direction_positions = (
        jnp.arange(positions.size, dtype=jnp.float64).reshape(positions.shape) / 100
    )
    direction_charges = jnp.linspace(0.2, 0.7, len(charges), dtype=jnp.float64)

    def evaluate(setup_cache):
        def loss(current_positions, current_charges):
            return selected_api(
                current_positions, current_charges, cell, **common, **setup_cache
            ).sum()

        return jax.jvp(
            jax.grad(loss, argnums=(0, 1)),
            (positions, charges),
            (direction_positions, direction_charges),
        )

    reference_gradients, reference_hvps = evaluate({})
    actual_gradients, actual_hvps = evaluate(cache)
    for actual, expected in zip(
        (*actual_gradients, *actual_hvps),
        (*reference_gradients, *reference_hvps),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("spline_order", [4, 5, 6])
@pytest.mark.parametrize("compiled", [False, True])
def test_shared_cell_squared_cache_matches_batched_reciprocal_derivatives(
    spline_order: int, compiled: bool
) -> None:
    """A common reciprocal grid matches dense metadata for repeated cells."""
    positions = jnp.array(
        [
            [1.13, 1.07, 1.19],
            [2.13, 1.57, 1.19],
            [1.44, 1.38, 1.50],
            [2.44, 1.88, 1.50],
        ],
        dtype=jnp.float64,
    )
    charges = jnp.array([1.0, -1.0, 0.7, -0.7], dtype=jnp.float64)
    cell = jnp.broadcast_to(8.0 * jnp.eye(3, dtype=jnp.float64), (2, 3, 3))
    batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    mesh_dimensions = (9, 10, 12)
    shared = generate_k_squared_pme(cell[0], mesh_dimensions)
    dense = generate_k_squared_pme(cell, mesh_dimensions)

    def evaluate(cache, pos, charge, current_cell):
        def energy(p, q, c):
            return pme_reciprocal_space(
                p,
                q,
                c,
                alpha=jnp.array([0.3, 0.35], dtype=jnp.float64),
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                batch_idx=batch_idx,
                k_squared=cache,
            ).sum()

        return jax.value_and_grad(energy, argnums=(0, 1, 2))(pos, charge, current_cell)

    run = jax.jit(evaluate) if compiled else evaluate
    actual = run(shared, positions, charges, cell)
    expected = run(dense, positions, charges, cell)
    for result, reference in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(result, reference, rtol=1e-8, atol=1e-10)
