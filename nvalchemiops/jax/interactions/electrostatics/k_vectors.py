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
import math

import jax
import jax.numpy as jnp

# Mathematical constants
PI = math.pi
TWOPI = 2.0 * PI

__all__ = [
    "generate_ewald_miller_indices",
    "generate_k_vectors_ewald_summation",
    "generate_k_vectors_pme",
    "generate_miller_indices",
    "k_vectors_from_miller_indices",
]


def _prepare_k_cutoff(
    k_cutoff: float | jax.Array,
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
    """Normalize k_cutoff for shared batch Miller bounds."""
    k_cutoff_array = jnp.asarray(k_cutoff, dtype=dtype)
    if k_cutoff_array.ndim == 0 or k_cutoff_array.size == 1:
        return jnp.reshape(k_cutoff_array, ())
    return jnp.max(k_cutoff_array)


def generate_miller_indices(
    cell: jax.Array,
    k_cutoff: float | jax.Array,
) -> jax.Array:
    """Generate Miller index bounds for Ewald summation.

    Parameters
    ----------
    cell : jax.Array, shape (N, 3, 3)
        Unit cell matrices with lattice vectors as rows.
    k_cutoff : float | jax.Array
        Maximum magnitude of k-vectors to include in reciprocal summation.

    Returns
    -------
    jax.Array
        Array of shape (3,) containing the maximum Miller indices (M_h, M_k, M_l)
        for each lattice direction.

    Notes
    -----
    For batch mode, one shared set of Miller bounds is used for all systems.
    If ``k_cutoff`` is provided per system, the maximum cutoff across the batch
    is used to build those shared bounds.
    """
    shared_k_cutoff = _prepare_k_cutoff(k_cutoff, cell.dtype)
    cell_lengths = (jnp.linalg.norm(cell, axis=-1).max(axis=0)) / (
        2 * jnp.pi
    )  # Length of each reciprocal vector
    return jnp.ceil(shared_k_cutoff * cell_lengths).astype(jnp.int32)


# Backwards-compatible alias
_generate_miller_indices = generate_miller_indices


def generate_ewald_miller_indices(
    cell: jax.Array,
    k_cutoff: float | jax.Array,
    miller_bounds: tuple[int, int, int] | None = None,
) -> jax.Array:
    r"""Generate the positive-half-space Miller topology for Ewald summation.

    Derives conservative rectangular Miller bounds from ``k_cutoff`` for the
    reciprocal-space Ewald sum. Supplying ``miller_bounds`` instead enumerates
    that rectangle directly; neither mode applies a per-vector magnitude
    filter. Uses half-space optimization to reduce computational cost by
    approximately 2x.

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
    (using :math:`8\pi` instead of :math:`4\pi`), so energies, forces, and charge gradients are
    computed correctly.

    Mathematical Background
    -----------------------
    For a direct lattice defined by basis vectors {a, b, c} (rows of cell matrix),
    the reciprocal lattice vectors are:

    .. math::

        \mathbf{a}^* &= \frac{2\pi (\mathbf{b} \times \mathbf{c})}{V}

        \mathbf{b}^* &= \frac{2\pi (\mathbf{c} \times \mathbf{a})}{V}

        \mathbf{c}^* &= \frac{2\pi (\mathbf{a} \times \mathbf{b})}{V}

    where :math:`V = \mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})` is the cell volume.

    In matrix form: :math:`\text{reciprocal_matrix} = 2\pi \cdot (\text{cell}^T)^{-1}`

    Each k-vector is: :math:`\mathbf{k} = h \mathbf{a}^* + k \mathbf{b}^* + l \mathbf{c}^*`
    where (h, k, l) are Miller indices (integers).

    Parameters
    ----------
    cell : jax.Array
        Unit cell matrix with lattice vectors as rows.
        Shape (3, 3) for single system or (B, 3, 3) for batch.
    k_cutoff : float or jax.Array
        Reciprocal cutoff used to derive conservative per-axis Miller bounds.
        The resulting rectangular topology can contain vectors whose magnitude
        exceeds this value.
    miller_bounds : tuple[int, int, int] | None, optional
        Explicit Miller half-bounds ``(M_h, M_k, M_l)``. When supplied, their
        rectangle is enumerated directly and ``k_cutoff`` does not select
        individual rows. Use the legacy :func:`generate_miller_indices` bounds
        helper before ``jax.jit``; it returns bounds, not full index rows.

    Returns
    -------
    jax.Array
        Signed integer Miller indices of shape ``(K, 3)``. The rows are
        nonzero, unique, and in the positive half-space.

    Examples
    --------
    Single system with explicit k_cutoff::

        >>> cell = jnp.eye(3, dtype=jnp.float64) * 10.0
        >>> indices = generate_ewald_miller_indices(cell, k_cutoff=8.0)
        >>> k_vectors = k_vectors_from_miller_indices(cell, indices)

    With automatic parameter estimation::

        >>> from nvalchemiops.jax.interactions.electrostatics import estimate_ewald_parameters
        >>> params = estimate_ewald_parameters(positions, cell)
        >>> k_vectors = generate_k_vectors_ewald_summation(cell, params.reciprocal_space_cutoff)

    JIT-compatible usage with precomputed bounds::

        >>> from nvalchemiops.jax.interactions.electrostatics import generate_miller_indices
        >>> cell = jnp.eye(3, dtype=jnp.float64)[None, ...] * 10.0
        >>> bounds = generate_miller_indices(cell, k_cutoff=8.0)
        >>> miller_bounds = (int(bounds[0]), int(bounds[1]), int(bounds[2]))
        >>> # This can now be called inside @jax.jit
        >>> k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0, miller_bounds=miller_bounds)

    Notes
    -----
    - The k=0 vector is always excluded (causes division by zero in Green's function).
    - For batch mode, the same set of Miller indices is used for all systems but
      transformed using each system's reciprocal cell. If ``k_cutoff`` is given
      per system, the maximum cutoff across the batch determines the shared
      Miller bounds.
    - The number of k-vectors K scales as :math:`O(k_{\text{cutoff}}^3 \cdot V)` where V is the cell volume.
    - When using inside ``jax.jit``, you **must** provide ``miller_bounds``
      as a concrete ``tuple[int, int, int]``. The bounds determine array shapes
      (via ``jnp.arange``), which must be statically known at trace time.

    A retained topology must conservatively cover every cell state in which it
    will be used. Enlarging it changes ``K`` and can recompile ``jax.jit``.
    """
    if cell.ndim == 2:
        cell = cell[None, ...]
    # Get max Miller indices per direction: M_h, M_k, M_l
    if miller_bounds is not None:
        M_h, M_k, M_l = miller_bounds
    else:
        _bounds = generate_miller_indices(cell, k_cutoff)
        M_h = int(_bounds[0])
        M_k = int(_bounds[1])
        M_l = int(_bounds[2])

    # Build half-space Miller indices directly (no boolean masking)
    # Block 1: h in [1, M_h], k in [-M_k, M_k], l in [-M_l, M_l]
    h1 = jnp.arange(1, M_h + 1, dtype=jnp.int32)
    k1 = jnp.arange(-M_k, M_k + 1, dtype=jnp.int32)
    l1 = jnp.arange(-M_l, M_l + 1, dtype=jnp.int32)
    h1_grid, k1_grid, l1_grid = jnp.meshgrid(h1, k1, l1, indexing="ij")
    block1 = jnp.stack(
        [h1_grid.reshape(-1), k1_grid.reshape(-1), l1_grid.reshape(-1)], axis=1
    )

    # Block 2: h = 0, k in [1, M_k], l in [-M_l, M_l]
    k2 = jnp.arange(1, M_k + 1, dtype=jnp.int32)
    l2 = jnp.arange(-M_l, M_l + 1, dtype=jnp.int32)
    k2_grid, l2_grid = jnp.meshgrid(k2, l2, indexing="ij")
    block2 = jnp.stack(
        [
            jnp.zeros(k2_grid.size, dtype=jnp.int32),
            k2_grid.reshape(-1),
            l2_grid.reshape(-1),
        ],
        axis=1,
    )

    # Block 3: h = 0, k = 0, l in [1, M_l]
    l3 = jnp.arange(1, M_l + 1, dtype=jnp.int32)
    block3 = jnp.stack(
        [jnp.zeros(l3.size, dtype=jnp.int32), jnp.zeros(l3.size, dtype=jnp.int32), l3],
        axis=1,
    )

    return jnp.concatenate([block1, block2, block3], axis=0)


def k_vectors_from_miller_indices(
    cell: jax.Array,
    miller_indices: jax.Array,
) -> jax.Array:
    """Materialize reciprocal vectors from caller-retained Miller indices.

    The supplied ``cell`` defines the live reciprocal transform. ``miller_indices``
    must be a signed integer array of shape ``(K, 3)``; ``(0, 3)`` is valid.
    Duplicate rows, zero rows, and half-space membership are caller
    preconditions and are not checked. :func:`generate_ewald_miller_indices`
    produces valid topology. The result has shape ``(K, 3)`` for one cell or
    ``(B, K, 3)`` for a batch.
    """
    if cell.ndim not in (2, 3) or cell.shape[-2:] != (3, 3):
        raise ValueError("cell must have shape (3, 3) or (B, 3, 3)")
    if miller_indices.ndim != 2 or miller_indices.shape[-1] != 3:
        raise ValueError("miller_indices must have shape (K, 3)")
    if not jnp.issubdtype(miller_indices.dtype, jnp.signedinteger):
        raise ValueError("miller_indices must have a signed integer dtype")
    if cell.ndim == 2:
        cell = cell[None, ...]
    reciprocal_cell = TWOPI * jnp.linalg.inv(jnp.swapaxes(cell, -2, -1))
    k_vectors = miller_indices.astype(reciprocal_cell.dtype) @ reciprocal_cell
    if k_vectors.shape[0] == 1:
        return jnp.squeeze(k_vectors, axis=0)
    return k_vectors


def generate_k_vectors_ewald_summation(
    cell: jax.Array,
    k_cutoff: float | jax.Array,
    miller_bounds: tuple[int, int, int] | None = None,
) -> jax.Array:
    """Generate reciprocal lattice vectors for Ewald summation (half-space)."""
    return k_vectors_from_miller_indices(
        cell,
        generate_ewald_miller_indices(cell, k_cutoff, miller_bounds),
    )


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
    k_vectors = jnp.einsum("ijkd,bcd->bijkc", k_grid, reciprocal_cell)
    if k_vectors.shape[0] == 1:
        k_vectors = jnp.squeeze(k_vectors, axis=0)

    # Compute k^2 for Green's function
    k_squared = jnp.sum(k_vectors**2, axis=-1)

    # Avoid division by zero at k=0
    k_squared_safe = jnp.where(k_squared > 1e-12, k_squared, jnp.array(1e-12))

    return k_vectors, k_squared_safe
