# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Shared fixtures for JAX electrostatics tests.

This module provides common fixtures and utilities used across JAX electrostatics
test modules (Coulomb, Ewald, PME, etc.).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


def place_on_device(arr: jax.Array, device_type: str) -> jax.Array:
    """
    Place a JAX array on the specified device type.

    Parameters
    ----------
    arr : jax.Array
        The array to place on the device.
    device_type : str
        Device type, either "cpu" or "gpu".

    Returns
    -------
    jax.Array
        Array placed on the specified device.
    """
    if device_type == "cpu":
        device = jax.devices("cpu")[0]
    else:
        device = jax.devices("gpu")[0]
    return jax.device_put(arr, device)


@pytest.fixture()
def device():
    """
    GPU device fixture. Skips when no CUDA device is available.

    Returns
    -------
    str
        Device type string "gpu".
    """
    if len(jax.devices("gpu")) == 0:
        pytest.skip("No CUDA device available.")
    return "gpu"


@pytest.fixture()
def simple_pair_system(device):
    """
    Two-atom system for basic tests.

    Parameters
    ----------
    device : str
        Device type from the device fixture.

    Returns
    -------
    tuple
        (positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts)
        - positions: [2, 3] float64 array
        - charges: [2] float64 array
        - cell: [1, 3, 3] float64 array
        - neighbor_list: [2, 1] int32 array
        - neighbor_ptr: [2] int32 array
        - neighbor_shifts: [1, 3] int32 array
    """
    positions = place_on_device(
        jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64), device
    )
    charges = place_on_device(jnp.array([1.0, -1.0], dtype=jnp.float64), device)
    cell = place_on_device(
        jnp.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=jnp.float64,
        ),
        device,
    )
    neighbor_list = place_on_device(jnp.array([[0], [1]], dtype=jnp.int32), device)
    neighbor_ptr = place_on_device(jnp.array([0, 1], dtype=jnp.int32), device)
    neighbor_shifts = place_on_device(jnp.zeros((1, 3), dtype=jnp.int32), device)
    return positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts
