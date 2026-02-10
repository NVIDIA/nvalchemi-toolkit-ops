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
import math

import jax
import jax.numpy as jnp

# Mathematical constants
PI = math.pi
TWOPI = 2.0 * PI

__all__ = ["generate_k_vectors_ewald_summation", "generate_k_vectors_pme"]


def _generate_miller_indices(
    cell: jax.Array,
    k_cutoff: float | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate Miller indices for Ewald summation.

    Parameters
    ----------
    cell : jax.Array, shape (N, 3, 3)
        Unit cell matrices with lattice vectors as rows.
    k_cutoff : float | jax.Array
        Maximum magnitude of k-vectors to include in reciprocal summation.

    Notes
    -----
    if cell represents a single system, return max_h, max_k, max_l
    computed by taking the maximum reciprocal cell_lengths over the entire batch of systems.
    """
    cell_lengths = (jnp.linalg.norm(cell, axis=-1).max(axis=0)) / (
        2 * jnp.pi
    )  # Length of each reciprocal vector
    return jnp.ceil(k_cutoff * cell_lengths).astype(jnp.int32)


def generate_k_vectors_ewald_summation(
    cell: jax.Array,
    k_cutoff: float | jax.Array,
) -> jax.Array:
    """Generate reciprocal lattice vectors for Ewald summation (half-space).

    Creates k-vectors within the specified cutoff for the reciprocal space
    summation in the Ewald method. Uses half-space optimization to reduce
    computational cost by approximately 2x.

    Half-Space Optimization
    -----------------------
    This function generates k-vectors in the positive half-space only, exploiting
    the symmetry S(-k) = S*(k) where S(k) is the structure factor. For each pair
    of k-vectors (k, -k), only one is included.

    The half-space condition selects k-vectors where:
        - h > 0, OR
        - (h == 0 AND k > 0), OR
        - (h == 0 AND k == 0 AND l > 0)

    The kernels in ewald_kernels.py compensate by doubling the Green's function
    (using :math:`8\\pi` instead of :math:`4\\pi`), so energies, forces, and charge gradients are
    computed correctly.

    Mathematical Background
    -----------------------
    For a direct lattice defined by basis vectors {a, b, c} (rows of cell matrix),
    the reciprocal lattice vectors are:

    .. math::

        \\mathbf{a}^* &= \\frac{2\\pi (\\mathbf{b} \\times \\mathbf{c})}{V}

        \\mathbf{b}^* &= \\frac{2\\pi (\\mathbf{c} \\times \\mathbf{a})}{V}

        \\mathbf{c}^* &= \\frac{2\\pi (\\mathbf{a} \\times \\mathbf{b})}{V}

    where :math:`V = \\mathbf{a} \\cdot (\\mathbf{b} \\times \\mathbf{c})` is the cell volume.

    In matrix form: :math:`\\text{reciprocal_matrix} = 2\\pi \\cdot (\\text{cell}^T)^{-1}`

    Each k-vector is: :math:`\\mathbf{k} = h \\mathbf{a}^* + k \\mathbf{b}^* + l \\mathbf{c}^*`
    where (h, k, l) are Miller indices (integers).

    Parameters
    ----------
    cell : jax.Array
        Unit cell matrix with lattice vectors as rows.
        Shape (3, 3) for single system or (B, 3, 3) for batch.
    k_cutoff : float or jax.Array
        Maximum magnitude of k-vectors to include (:math:`|\\mathbf{k}| \\leq k_{\\text{cutoff}}`).
        Typical values: 8-12 :math:`\\text{\\AA}^{-1}` for molecular systems.
        Higher values increase accuracy but also computational cost.

    Returns
    -------
    jax.Array
        Reciprocal lattice vectors within the cutoff.
        Shape (K, 3) for single system or (B, K, 3) for batch.
        Excludes k=0 and includes only half-space vectors.

    Examples
    --------
    Single system with explicit k_cutoff::

        >>> cell = jnp.eye(3, dtype=jnp.float64) * 10.0
        >>> k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)
        >>> k_vectors.shape
        (...)  # Number depends on cell size and cutoff

    With automatic parameter estimation::

        >>> from nvalchemiops.jax.interactions.electrostatics import estimate_ewald_parameters
        >>> params = estimate_ewald_parameters(positions, cell)
        >>> k_vectors = generate_k_vectors_ewald_summation(cell, params.reciprocal_space_cutoff)

    Notes
    -----
    - The k=0 vector is always excluded (causes division by zero in Green's function).
    - For batch mode, the same set of Miller indices is used for all systems but
      transformed using each system's reciprocal cell.
    - The number of k-vectors K scales as O(k_cutoff³ · V) where V is the cell volume.

    See Also
    --------
    ewald_reciprocal_space : Uses these k-vectors for reciprocal space energy.
    estimate_ewald_parameters : Automatic parameter estimation including k_cutoff.
    """
    if cell.ndim == 2:
        cell = cell[None, ...]
    dtype = cell.dtype

    max_h, max_k, max_l = 2 * _generate_miller_indices(cell, k_cutoff) + 1

    # Generate all combinations of Miller indices
    h_range = jnp.fft.fftfreq(max_h, dtype=dtype) * max_h
    k_range = jnp.fft.fftfreq(max_k, dtype=dtype) * max_k
    l_range = jnp.fft.fftfreq(max_l, dtype=dtype) * max_l

    h_grid, k_grid, l_grid = jnp.meshgrid(h_range, k_range, l_range, indexing="ij")
    miller_indices = jnp.stack(
        [h_grid.flatten(), k_grid.flatten(), l_grid.flatten()], axis=1
    )

    # Apply half-space filter: keep only one of each +-k pair
    # Condition: h > 0 OR (h == 0 AND k > 0) OR (h == 0 AND k == 0 AND l > 0)
    h = miller_indices[:, 0]
    k = miller_indices[:, 1]
    m = miller_indices[:, 2]  # Using 'm' instead of 'l' to avoid E741 ambiguity

    halfspace_mask = (h > 0) | ((h == 0) & (k > 0)) | ((h == 0) & (k == 0) & (m > 0))

    miller_indices = miller_indices[halfspace_mask]

    # Compute reciprocal lattice vectors (2π times reciprocal of direct lattice)
    reciprocal_cell = TWOPI * jnp.linalg.inv(
        jnp.swapaxes(cell, -2, -1)
    )  # Transpose for column vectors
    k_vectors = miller_indices.astype(reciprocal_cell.dtype) @ reciprocal_cell
    if k_vectors.shape[0] == 1:
        return jnp.squeeze(k_vectors, axis=0)
    return k_vectors


def generate_k_vectors_pme(
    cell: jax.Array,
    mesh_dimensions: tuple[int, int, int],
    reciprocal_cell: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Generate reciprocal lattice vectors for Particle Mesh Ewald (PME).

    Creates k-vectors on a regular grid compatible with FFT-based reciprocal
    space calculations in PME. Uses rfft conventions (half-size in z-dimension)
    to exploit Hermitian symmetry of real-valued charge densities.

    Notes
    -----
    For a direct lattice defined by basis vectors {a, b, c} (rows of cell matrix),
    the reciprocal lattice vectors are:

    .. math::

        \\begin{aligned}
        \\mathbf{a}^* &= \\frac{2\\pi (\\mathbf{b} \\times \\mathbf{c})}{V} \\\\
        \\mathbf{b}^* &= \\frac{2\\pi (\\mathbf{c} \\times \\mathbf{a})}{V} \\\\
        \\mathbf{c}^* &= \\frac{2\\pi (\\mathbf{a} \\times \\mathbf{b})}{V}
        \\end{aligned}

    where :math:`V = \\mathbf{a} \\cdot (\\mathbf{b} \\times \\mathbf{c})` is the cell volume.

    In matrix form:

    .. math::

        \\text{reciprocal_matrix} = 2\\pi \\cdot (\\text{cell}^T)^{-1}

    Each k-vector is then:

    .. math::

        \\mathbf{k} = h \\mathbf{a}^* + k \\mathbf{b}^* + l \\mathbf{c}^*

    where (h, k, l) are Miller indices (integers).

    Parameters
    ----------
    cell : jax.Array
        Unit cell matrix with lattice vectors as rows.
        Shape (3, 3) for single system or (B, 3, 3) for batch.
    mesh_dimensions : tuple[int, int, int]
        PME mesh grid dimensions (nx, ny, nz). Should typically be chosen
        such that mesh spacing is :math:`\\sim 1 \\text{\\AA}` or finer. Power-of-2 dimensions
        are optimal for FFT performance.
    reciprocal_cell : jax.Array, optional
        Precomputed reciprocal cell matrix (:math:`2\\pi \\cdot \\text{cell}^{-1}`). If provided,
        skips the inverse computation. Shape (3, 3) or (B, 3, 3).

    Returns
    -------
    k_vectors : jax.Array, shape (nx, ny, nz//2+1, 3)
        Cartesian k-vectors at each grid point. Uses rfft convention
        where z-dimension is halved due to Hermitian symmetry.
    k_squared_safe : jax.Array, shape (nx, ny, nz//2+1)
        Squared magnitude :math:`|\\mathbf{k}|^2` for each k-vector, with k=0 set to a
        small positive value (1e-12) to avoid division by zero.

    Examples
    --------
    Basic usage::

        >>> cell = jnp.eye(3, dtype=jnp.float64) * 10.0
        >>> mesh_dims = (32, 32, 32)
        >>> k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dims)
        >>> k_vectors.shape
        (32, 32, 17, 3)

    With precomputed reciprocal cell::

        >>> reciprocal_cell = 2 * jnp.pi * jnp.linalg.inv(cell)
        >>> k_vectors, k_squared = generate_k_vectors_pme(
        ...     cell, mesh_dims, reciprocal_cell=reciprocal_cell
        ... )

    Notes
    -----
    - The z-dimension output size is nz//2+1 due to rfft symmetry.
    - Miller indices follow jnp.fft.fftfreq convention (0, 1, 2, ..., -2, -1).
    - k_squared_safe has k=0 replaced with 1e-12 to prevent division by zero
      in Green's function calculations.

    See Also
    --------
    pme_reciprocal_space : Uses these k-vectors for PME reciprocal space energy.
    pme_green_structure_factor : Computes Green's function using k_squared.
    """
    dtype = cell.dtype

    # Ensure cell has batch dimension
    cell_3d = cell if cell.ndim == 3 else jnp.expand_dims(cell, 0)

    # Compute reciprocal lattice vectors (2*pi times reciprocal of direct lattice)
    if reciprocal_cell is None:
        reciprocal_cell = TWOPI * jnp.linalg.inv(cell_3d)

    # Generate all combinations of Miller indices
    mesh_grid_x, mesh_grid_y, mesh_grid_z = mesh_dimensions

    # Generate Miller indices (h, k, l) for each FFT grid point
    # fftfreq gives frequencies normalized to sampling rate
    # Multiplying by n gives actual Miller indices
    kx = jnp.fft.fftfreq(mesh_grid_x, d=1.0, dtype=dtype) * mesh_grid_x
    ky = jnp.fft.fftfreq(mesh_grid_y, d=1.0, dtype=dtype) * mesh_grid_y
    kz = jnp.fft.rfftfreq(mesh_grid_z, d=1.0, dtype=dtype) * mesh_grid_z

    kx_grid, ky_grid, kz_grid = jnp.meshgrid(kx, ky, kz, indexing="ij")

    # Stack into Miller indices array (nx, ny, nz/2+1, 3)
    k_grid = jnp.stack([kx_grid, ky_grid, kz_grid], axis=-1)

    # Transform Miller indices to Cartesian k-vectors
    # k_cart = [h, k, l] @ reciprocal_matrix^T
    # where reciprocal_matrix has reciprocal lattice vectors as rows
    k_vectors = jnp.einsum("ijkd,bcd->bijkc", k_grid, reciprocal_cell).squeeze(0)

    # Compute k^2 for Green's function
    k_squared = jnp.sum(k_vectors**2, axis=-1)

    # Avoid division by zero at k=0
    k_squared_safe = jnp.where(k_squared > 1e-12, k_squared, jnp.array(1e-12))

    return k_vectors, k_squared_safe
