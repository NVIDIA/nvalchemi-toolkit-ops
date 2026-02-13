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
    try:
        if len(jax.devices("gpu")) == 0:
            pytest.skip("No CUDA device available.")
    except RuntimeError:
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


# ==============================================================================
# Virial Test Utilities
# ==============================================================================


def make_virial_cscl_system_jax(size: int = 2, device: str = "gpu"):
    """Create a CsCl test system for virial tests (JAX version).

    Parameters
    ----------
    size : int, default=2
        Supercell size.
    device : str, default="gpu"
        Device type to place arrays on.

    Returns
    -------
    tuple
        (positions, charges, cell) as JAX arrays on the specified device.
    """
    # Import here to avoid circular imports at module level
    from test.interactions.electrostatics.conftest import create_cscl_supercell

    crystal = create_cscl_supercell(size)
    positions = place_on_device(jnp.array(crystal.positions, dtype=jnp.float64), device)
    charges = place_on_device(jnp.array(crystal.charges, dtype=jnp.float64), device)
    cell = place_on_device(
        jnp.array(crystal.cell, dtype=jnp.float64)[jnp.newaxis, :, :], device
    )
    return positions, charges, cell


def apply_strain_jax(
    positions: jax.Array, cell: jax.Array, epsilon: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Apply infinitesimal strain: x' = (I + eps) @ x, cell' = (I + eps) @ cell.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic positions.
    cell : jax.Array, shape (B, 3, 3) or (1, 3, 3)
        Unit cell matrices.
    epsilon : jax.Array, shape (3, 3)
        Infinitesimal strain tensor.

    Returns
    -------
    tuple
        (new_positions, new_cell) with strain applied.
    """
    I_plus_eps = jnp.eye(3, dtype=jnp.float64) + epsilon
    new_positions = positions @ I_plus_eps.T
    new_cell = cell @ I_plus_eps.T
    return new_positions, new_cell


def fd_virial_full_jax(
    energy_fn, positions: jax.Array, cell: jax.Array, h: float = 1e-5
):
    """Compute full 3x3 virial tensor by finite differences (JAX version).

    The virial is defined as:
        virial_ab = -dE/d(epsilon_ab) ≈ -[E(+h) - E(-h)] / (2h)

    Parameters
    ----------
    energy_fn : callable
        Function that takes (positions, cell) and returns total energy (scalar).
    positions : jax.Array, shape (N, 3)
        Atomic positions.
    cell : jax.Array, shape (B, 3, 3) or (1, 3, 3)
        Unit cell matrices.
    h : float, default=1e-5
        Finite difference step size.

    Returns
    -------
    jax.Array, shape (3, 3)
        Virial tensor computed via finite differences.
    """
    import numpy as np

    virial = np.zeros((3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            eps_plus = jnp.zeros((3, 3), dtype=jnp.float64)
            eps_plus = eps_plus.at[a, b].set(h)
            pos_p, cell_p = apply_strain_jax(positions, cell, eps_plus)
            E_plus = float(energy_fn(pos_p, cell_p))

            eps_minus = jnp.zeros((3, 3), dtype=jnp.float64)
            eps_minus = eps_minus.at[a, b].set(-h)
            pos_m, cell_m = apply_strain_jax(positions, cell, eps_minus)
            E_minus = float(energy_fn(pos_m, cell_m))

            virial[a, b] = -(E_plus - E_minus) / (2.0 * h)
    return jnp.array(virial, dtype=jnp.float64)
