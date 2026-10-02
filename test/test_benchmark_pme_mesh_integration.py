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

"""CPU checks for upstream PME setup and the shared mesh selector in benchmarks."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from benchmarks.interactions.electrostatics.benchmark_electrostatics_suite import (
    _canonical_pme_input_sha256,
    _el_estimate_ewald_params,
    _el_estimate_pme_params,
    _el_parameter_metadata,
    _el_runtime_pme_input_metadata,
)
from nvalchemiops.jax.interactions.electrostatics import parameters as jax_parameters
from nvalchemiops.torch.interactions.electrostatics import (
    parameters as torch_parameters,
)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("accuracy", [1e-4, 1e-6])
@pytest.mark.parametrize("spline_order", [4, 5, 6])
def test_benchmark_uses_upstream_cutoff_alpha_and_public_mesh_selector(
    backend: str, accuracy: float, spline_order: int
) -> None:
    """Benchmark setup preserves the configured cutoff and ordinary API values."""
    if backend == "torch":
        positions = torch.zeros((128, 3), dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64)[None] * 20.0
        batch_idx = torch.zeros(128, dtype=torch.int32)
        parameters = torch_parameters
        jax_api = None
    else:
        positions = jnp.zeros((128, 3), dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64)[None] * 20.0
        batch_idx = jnp.zeros(128, dtype=jnp.int32)
        parameters = jax_parameters
        jax_api = {
            "estimate_pme_parameters": parameters.estimate_pme_parameters,
            "estimate_ewald_parameters": parameters.estimate_ewald_parameters,
        }

    cutoff_limit = 9.0
    pme = _el_estimate_pme_params(
        positions,
        cell,
        batch_idx,
        backend,
        accuracy,
        jax_api,
        cutoff_limit,
        spline_order,
    )
    ewald_alpha, ewald_real_cutoff, ewald_reciprocal_cutoff = _el_estimate_ewald_params(
        positions, cell, batch_idx, backend, accuracy, jax_api
    )
    expected_pme = parameters.estimate_pme_parameters(
        positions,
        cell,
        batch_idx=batch_idx,
        accuracy=accuracy,
        real_space_cutoff=cutoff_limit,
        spline_order=spline_order,
    )
    expected_ewald = parameters.estimate_ewald_parameters(
        positions,
        cell,
        batch_idx=batch_idx,
        accuracy=accuracy,
    )
    np.testing.assert_allclose(np.asarray(pme.alpha), np.asarray(expected_pme.alpha))
    np.testing.assert_allclose(np.asarray(pme.real_space_cutoff), cutoff_limit)
    np.testing.assert_allclose(
        np.asarray(pme.alpha), math.sqrt(-math.log(accuracy)) / cutoff_limit
    )
    assert pme.mesh_dimensions == expected_pme.mesh_dimensions
    assert pme.mesh_dimensions == parameters.estimate_pme_mesh_dimensions(
        cell,
        pme.alpha,
        accuracy=accuracy,
        spline_order=spline_order,
    )
    np.testing.assert_allclose(
        np.asarray(ewald_alpha), np.asarray(expected_ewald.alpha)
    )
    np.testing.assert_allclose(
        ewald_real_cutoff,
        np.asarray(expected_ewald.real_space_cutoff),
    )
    np.testing.assert_allclose(
        ewald_reciprocal_cutoff,
        np.asarray(expected_ewald.reciprocal_space_cutoff),
    )


def test_benchmark_preserves_automatic_cutoff_below_configured_limit() -> None:
    """A generous configured limit leaves upstream cutoff and alpha unchanged."""
    positions = torch.zeros((128, 3), dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64)[None] * 20.0
    batch_idx = torch.zeros(128, dtype=torch.int32)
    expected = torch_parameters.estimate_pme_parameters(
        positions, cell, batch_idx, 1e-6
    )
    limit = float(expected.real_space_cutoff.max()) * 2.0
    actual = _el_estimate_pme_params(
        positions, cell, batch_idx, "torch", 1e-6, None, limit, 5
    )
    torch.testing.assert_close(actual.alpha, expected.alpha, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.real_space_cutoff, expected.real_space_cutoff, rtol=0, atol=0
    )
    assert actual.mesh_dimensions == expected.mesh_dimensions


def test_runtime_input_identity_hashes_scientific_array_content() -> None:
    """Equal arrays have equal hashes; changing a position changes the identity."""
    positions = np.array([[1.0, 1.0, 1.0], [2.0, 1.5, 1.0]], dtype=np.float64)
    charges = np.array([1.0, -1.0], dtype=np.float64)
    cell = np.eye(3, dtype=np.float64)[None] * 8.0
    batch_idx = np.zeros(2, dtype=np.int32)
    pbc = np.ones((1, 3), dtype=np.bool_)
    arguments = (positions, charges, cell, batch_idx, pbc)
    expected = _canonical_pme_input_sha256(*arguments)
    assert expected == _canonical_pme_input_sha256(
        *(value.copy() for value in arguments)
    )
    metadata = _el_runtime_pme_input_metadata(
        *(torch.from_numpy(value) for value in arguments),
        backend="torch",
        jax_api=None,
    )
    assert metadata["pme_runtime_input_sha256"] == expected
    moved = positions.copy()
    moved[0, 0] += 0.25
    assert expected != _canonical_pme_input_sha256(moved, *arguments[1:])


@pytest.mark.parametrize("method", ["pme", "ewald"])
def test_row_metadata_records_actual_execution_parameters(method: str) -> None:
    """Result metadata records executed meshes or Ewald reciprocal cutoffs."""
    metadata = _el_parameter_metadata(
        method,
        torch.tensor([0.3, 0.3]),
        9.0,
        (35, 40, 48),
        reciprocal_space_cutoff=2.0,
    )
    assert metadata["real_space_cutoff"] == 9.0
    assert metadata["alpha"] == pytest.approx(0.3)
    if method == "pme":
        assert (metadata["mesh_nx"], metadata["mesh_ny"], metadata["mesh_nz"]) == (
            35,
            40,
            48,
        )
        assert metadata["reciprocal_grid_points"] == 35 * 40 * 25
    else:
        assert metadata["reciprocal_space_cutoff"] == 2.0
