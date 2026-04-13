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

"""
Shared fixtures for JAX electrostatics tests.

This module provides common fixtures and utilities used across JAX electrostatics
test modules (Coulomb, Ewald, PME, etc.).
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="No JAX installed.")

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None
import numpy as np


@pytest.fixture(scope="session")
def device():
    """
    GPU device fixture. Skips tests when no CUDA device is available.

    This fixture serves as a test skip gate. Tests that depend on this fixture
    will be skipped if no GPU is available. The return value is kept for
    backward compatibility but should not be used for array placement.

    Returns
    -------
    str
        Device type string "gpu" (for backward compatibility).
    """
    try:
        if len(jax.devices("gpu")) == 0:
            pytest.skip("No CUDA device available.")
    except RuntimeError:
        pytest.skip("No CUDA device available.")
    return "gpu"


# ==============================================================================
# Shared System Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def simple_pair_system(device):  # noqa: ARG001
    """
    Two-atom system for basic tests using neighbor_matrix (dense) format.

    Parameters
    ----------
    device : fixture
        Device fixture dependency for GPU availability gating.
        The parameter is not used directly but ensures tests skip on CPU-only systems.

    Returns
    -------
    tuple
        (positions, charges, cell, neighbor_matrix, neighbor_matrix_shifts)
        - positions: [2, 3] float64 array
        - charges: [2] float64 array
        - cell: [1, 3, 3] float64 array
        - neighbor_matrix: [2, 1] int32 array (dense neighbor indices)
        - neighbor_matrix_shifts: [2, 1, 3] int32 array (periodic shift vectors)
    """
    positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
    charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
    cell = jnp.array(
        [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
        dtype=jnp.float64,
    )
    # Dense neighbor_matrix: atom 0 -> neighbor [1], atom 1 -> neighbor [0]
    neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
    neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)
    return positions, charges, cell, neighbor_matrix, neighbor_matrix_shifts


@pytest.fixture(scope="session")
def batched_dipole_system(device):  # noqa: ARG001
    """Two independent dipole systems in a single batch (4 atoms total).

    Creates a batched system with 2 systems of 2 atoms each, all in cubic
    cells with side length 10.0. Includes precomputed dense neighbor data
    from ``batch_cell_list``.

    Parameters
    ----------
    device : fixture
        Device fixture dependency for GPU availability gating.

    Returns
    -------
    dict
        Keys: ``positions`` [4, 3], ``charges`` [4], ``cell`` [1, 3, 3],
        ``batch_idx`` [4], ``neighbor_matrix``, ``num_neighbors``,
        ``neighbor_matrix_shifts``.
    """
    from nvalchemiops.jax.neighbors import batch_cell_list

    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    charges = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64)
    cell = jnp.array(
        [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
        dtype=jnp.float64,
    )
    batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

    cutoff = 5.0
    pbc = jnp.array([[True, True, True]])
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = batch_cell_list(
        positions, cutoff, cell, pbc, batch_idx=batch_idx, max_neighbors=32
    )

    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "batch_idx": batch_idx,
        "neighbor_matrix": neighbor_matrix,
        "num_neighbors": num_neighbors,
        "neighbor_matrix_shifts": neighbor_matrix_shifts,
    }


# ==============================================================================
# Shared Helper Functions
# ==============================================================================


def make_crystal_system_jax(
    crystal_type: str = "cscl", size: int = 2
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Create a crystal test system as JAX arrays.

    Wraps the numpy crystal generators from
    ``test.interactions.electrostatics.conftest`` and converts the result
    to JAX float64 arrays with the cell in ``(1, 3, 3)`` shape.

    Parameters
    ----------
    crystal_type : str, default="cscl"
        One of ``"cscl"``, ``"wurtzite"``, ``"zincblende"``.
    size : int, default=2
        Supercell size.

    Returns
    -------
    tuple
        ``(positions, charges, cell)`` as JAX float64 arrays.
        ``positions`` has shape ``(N, 3)``, ``charges`` ``(N,)``,
        ``cell`` ``(1, 3, 3)``.
    """
    from test.interactions.electrostatics.conftest import (
        create_cscl_supercell,
        create_wurtzite_system,
        create_zincblende_system,
    )

    generators = {
        "cscl": create_cscl_supercell,
        "wurtzite": create_wurtzite_system,
        "zincblende": create_zincblende_system,
    }
    if crystal_type not in generators:
        raise ValueError(
            f"Unknown crystal_type '{crystal_type}'. Choose from {list(generators)}."
        )
    crystal = generators[crystal_type](size=size)

    positions = jnp.array(crystal.positions, dtype=jnp.float64)
    charges = jnp.array(crystal.charges, dtype=jnp.float64)
    cell = jnp.array(crystal.cell, dtype=jnp.float64)[jnp.newaxis, :, :]
    return positions, charges, cell


def make_virial_cscl_system_jax(size: int = 2):
    """Create a CsCl test system for virial tests (JAX version).

    Parameters
    ----------
    size : int, default=2
        Supercell size.

    Returns
    -------
    tuple
        (positions, charges, cell) as JAX arrays.
    """
    return make_crystal_system_jax("cscl", size)


def cubic_cell_jax(cell_size: float = 10.0, dtype=jnp.float64) -> jax.Array:
    """Create a cubic unit cell as a JAX array.

    Parameters
    ----------
    cell_size : float, default=10.0
        Side length of the cubic cell.
    dtype : jnp.dtype, default=jnp.float64
        Data type.

    Returns
    -------
    jax.Array, shape (1, 3, 3)
        Cubic cell matrix.
    """
    return jnp.array(
        [[[cell_size, 0.0, 0.0], [0.0, cell_size, 0.0], [0.0, 0.0, cell_size]]],
        dtype=dtype,
    )


# ==============================================================================
# Virial Test Utilities
# ==============================================================================


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
