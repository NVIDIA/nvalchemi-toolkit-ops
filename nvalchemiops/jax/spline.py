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

"""JAX B-Spline Interpolation Bindings.

This module provides JAX bindings for B-spline interpolation functions
used in mesh-based calculations (e.g., Particle Mesh Ewald).

This module wraps the framework-agnostic Warp kernels from
``nvalchemiops.math.spline`` with JAX jax_kernel wrappers.

SUPPORTED ORDERS
================

- Order 1: Constant (Nearest Grid Point)
- Order 2: Linear
- Order 3: Quadratic
- Order 4: Cubic (recommended for PME)

OPERATIONS
==========

1. SPREAD: Scatter atom values to mesh grid
   mesh[g] += value[atom] * weight(atom, g)

2. GATHER: Collect mesh values at atom positions
   value[atom] = Σ_g mesh[g] * weight(atom, g)

3. GATHER_VEC3: Collect 3D vector field values at atom positions
   vector[atom] = Σ_g mesh[g] * weight(atom, g)

4. GATHER_GRADIENT: Collect mesh values with weight gradients (forces)
   grad[atom] = sum_g mesh[g] * grad_weight(atom, g)

USAGE
=====

Single-system:
    from nvalchemiops.jax.spline import spline_spread, spline_gather, spline_gather_gradient

    # Spread charges to mesh
    mesh = spline_spread(positions, charges, cell, mesh_dims, spline_order=4)

    # Gather potential from mesh
    potentials = spline_gather(positions, potential_mesh, cell, spline_order=4)

    # Gather forces
    forces = spline_gather_gradient(positions, charges, potential_mesh, cell, spline_order=4)

Batched (multiple systems):
    # Spread charges to batched mesh
    mesh = spline_spread(positions, charges, cell, mesh_dims, spline_order=4, batch_idx=batch_idx)

    # Gather potential from batched mesh
    potentials = spline_gather(positions, potential_mesh, cell, spline_order=4, batch_idx=batch_idx)

REFERENCES
==========

- Essmann et al. (1995). J. Chem. Phys. 103, 8577 (PME B-splines)
"""

import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel

from nvalchemiops.math.spline import (
    _batch_bspline_gather_channels_kernel_overload as wp_batch_gather_channels,
)
from nvalchemiops.math.spline import (
    _batch_bspline_gather_gradient_kernel_overload as wp_batch_gather_gradient,
)
from nvalchemiops.math.spline import (
    _batch_bspline_gather_kernel_overload as wp_batch_gather,
)
from nvalchemiops.math.spline import (
    _batch_bspline_gather_vec3_kernel_overload as wp_batch_gather_vec3,
)
from nvalchemiops.math.spline import (
    _batch_bspline_spread_channels_kernel_overload as wp_batch_spread_channels,
)
from nvalchemiops.math.spline import (
    _batch_bspline_spread_kernel_overload as wp_batch_spread,
)
from nvalchemiops.math.spline import (
    _bspline_gather_channels_kernel_overload as wp_gather_channels,
)
from nvalchemiops.math.spline import (
    _bspline_gather_gradient_kernel_overload as wp_gather_gradient,
)
from nvalchemiops.math.spline import _bspline_gather_kernel_overload as wp_gather
from nvalchemiops.math.spline import (
    _bspline_gather_vec3_kernel_overload as wp_gather_vec3,
)
from nvalchemiops.math.spline import (
    _bspline_spread_channels_kernel_overload as wp_spread_channels,
)
from nvalchemiops.math.spline import _bspline_spread_kernel_overload as wp_spread
from nvalchemiops.math.spline import _bspline_weight_kernel_overload as wp_weight

# ==============================================================================
# Helper Functions for Dtype-Dispatched Kernel Creation
# ==============================================================================


def _make_spline_jax_kernels(
    wp_overload_dict: dict,
    num_outputs: int,
    in_out_argnames: list[str],
) -> dict:
    """Create dtype-dispatched JAX kernel wrappers from Warp overloads.

    Parameters
    ----------
    wp_overload_dict : dict
        Warp kernel overload dictionary keyed by wp.float32/wp.float64.
    num_outputs : int
        Number of output arrays returned by the kernel.
    in_out_argnames : list of str
        Names of in-place output arguments.

    Returns
    -------
    dict
        Dictionary mapping jnp.float32/jnp.float64 to jax_kernel instances.
    """
    _JAX_TO_WP = {jnp.float32: wp.float32, jnp.float64: wp.float64}
    return {
        jax_dtype: jax_kernel(
            wp_overload_dict[wp_dtype],
            num_outputs=num_outputs,
            in_out_argnames=in_out_argnames,
            enable_backward=False,
        )
        for jax_dtype, wp_dtype in _JAX_TO_WP.items()
    }


def _normalize_dtype(dtype):
    """Normalize JAX dtype for kernel dictionary lookup.

    Parameters
    ----------
    dtype : dtype-like
        Input dtype from a JAX array.

    Returns
    -------
    jnp.float32 or jnp.float64
        Normalized JAX dtype for kernel lookup.
    """
    if dtype == jnp.float32 or str(dtype) == "float32":
        return jnp.float32
    elif dtype == jnp.float64 or str(dtype) == "float64":
        return jnp.float64
    else:
        raise ValueError(f"Unsupported dtype for spline operations: {dtype}")


# ==============================================================================
# JAX Kernel Wrappers (dtype-dispatched jax_kernel around Warp overloads)
# ==============================================================================

# --- Unbatched Kernels ---

_weight_kernels = _make_spline_jax_kernels(wp_weight, 1, ["weights"])
_spread_kernels = _make_spline_jax_kernels(wp_spread, 1, ["mesh"])
_gather_kernels = _make_spline_jax_kernels(wp_gather, 1, ["output"])
_gather_vec3_kernels = _make_spline_jax_kernels(wp_gather_vec3, 1, ["output"])
_gather_gradient_kernels = _make_spline_jax_kernels(wp_gather_gradient, 1, ["forces"])
_spread_channels_kernels = _make_spline_jax_kernels(wp_spread_channels, 1, ["mesh"])
_gather_channels_kernels = _make_spline_jax_kernels(wp_gather_channels, 1, ["output"])

# --- Batched Kernels ---

_batch_spread_kernels = _make_spline_jax_kernels(wp_batch_spread, 1, ["mesh"])
_batch_gather_kernels = _make_spline_jax_kernels(wp_batch_gather, 1, ["output"])
_batch_gather_vec3_kernels = _make_spline_jax_kernels(
    wp_batch_gather_vec3, 1, ["output"]
)
_batch_gather_gradient_kernels = _make_spline_jax_kernels(
    wp_batch_gather_gradient, 1, ["forces"]
)
_batch_spread_channels_kernels = _make_spline_jax_kernels(
    wp_batch_spread_channels, 1, ["mesh"]
)
_batch_gather_channels_kernels = _make_spline_jax_kernels(
    wp_batch_gather_channels, 1, ["output"]
)

__all__ = [
    "bspline_weight",
    "spline_spread",
    "spline_gather",
    "spline_gather_vec3",
    "spline_gather_gradient",
    "spline_spread_channels",
    "spline_gather_channels",
    "compute_bspline_deconvolution",
    "compute_bspline_deconvolution_1d",
]


# ==============================================================================
# High-level Launcher Functions
# ==============================================================================


def bspline_weight(u: jnp.ndarray, order: int) -> jnp.ndarray:
    """Compute B-spline basis function M_n(u).

    Parameters
    ----------
    u : jax.Array
        Input values. dtype=float32 or float64.
    order : int
        Spline order (1-4).

    Returns
    -------
    weights : jax.Array
        B-spline weights M_n(u). dtype matches input u dtype.

    Notes
    -----
    This function computes the B-spline basis function recursively.
    For PME, typically use order=4 (cubic B-splines).

    Examples
    --------
    >>> u = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)
    >>> weights = bspline_weight(u, order=4)
    """
    num_points = u.shape[0]
    working_dtype = _normalize_dtype(u.dtype)

    # Allocate output
    weights = jnp.zeros_like(u)

    (weights_out,) = _weight_kernels[working_dtype](
        u,
        int(order),
        weights,
        launch_dims=num_points,
    )
    return weights_out


def spline_spread(
    positions: jnp.ndarray,
    values: jnp.ndarray,
    cell: jnp.ndarray,
    mesh_dims: tuple[int, int, int],
    spline_order: int = 4,
    batch_idx: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Spread values from atoms to mesh grid using B-spline interpolation.

    For each atom, distributes its value to nearby grid points weighted by the
    B-spline basis function. This is the adjoint operation to gathering.

    Formula: mesh[g] += value[atom] * w(atom, g)

    where w(atom, g) is the product of 1D B-spline weights in each dimension.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic positions in Cartesian coordinates. dtype=float32 or float64.
    values : jax.Array, shape (N,)
        Values to spread (e.g., charges). dtype=float32 or float64.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrix. For single-system: (3, 3) or (1, 3, 3).
        For batched: (B, 3, 3). dtype=float32 or float64.
        Convention: cell[i, :] is the i-th lattice vector.
    mesh_dims : tuple[int, int, int]
        Mesh dimensions (nx, ny, nz).
    spline_order : int, optional
        B-spline order (1-4, where 4=cubic). Default: 4
    batch_idx : jax.Array | None, shape (N,), dtype=int32, optional
        System index for each atom. If None, uses single-system kernel.
        Default: None

    Returns
    -------
    mesh : jax.Array
        For single-system: shape (nx, ny, nz), dtype matches positions dtype
        For batch: shape (B, nx, ny, nz), dtype matches positions dtype

    Notes
    -----
    - Uses atomic adds for thread-safe accumulation to shared grid points.
    - Grid indices are wrapped using periodic boundary conditions.

    Examples
    --------
    Single system:

    >>> positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)
    >>> charges = jnp.array([1.0, -1.0], dtype=jnp.float32)
    >>> cell = jnp.eye(3, dtype=jnp.float32) * 10.0  # 10 Bohr cubic box
    >>> mesh = spline_spread(positions, charges, cell, (32, 32, 32), spline_order=4)
    >>> mesh.shape
    (32, 32, 32)

    Batched systems:

    >>> batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    >>> cell_batch = jnp.stack([cell, cell])  # Shape (2, 3, 3)
    >>> mesh = spline_spread(positions, charges, cell_batch, (32, 32, 32),
    ...                       spline_order=4, batch_idx=batch_idx)
    >>> mesh.shape
    (2, 32, 32, 32)
    """
    num_atoms = positions.shape[0]
    num_points = spline_order**3
    mesh_nx, mesh_ny, mesh_nz = mesh_dims
    working_dtype = _normalize_dtype(positions.dtype)

    # Cast inputs to working dtype
    values_work = values.astype(working_dtype)
    cell_work = cell.astype(working_dtype)

    # Compute cell_inv_t
    if cell_work.ndim == 2:
        cell_work = cell_work[jnp.newaxis, :, :]  # Shape (1, 3, 3)

    cell_inv = jnp.linalg.inv(cell_work)
    cell_inv_t = jnp.transpose(cell_inv, (0, 2, 1))

    if batch_idx is None:
        # Single-system kernel
        mesh = jnp.zeros((mesh_nx, mesh_ny, mesh_nz), dtype=working_dtype)
        (mesh_out,) = _spread_kernels[working_dtype](
            positions,
            values_work,
            cell_inv_t,
            int(spline_order),
            mesh,
            launch_dims=(num_atoms, num_points),
        )
        return mesh_out
    else:
        # Batched kernel
        num_systems = cell_work.shape[0]
        batch_idx_i32 = batch_idx.astype(jnp.int32)

        mesh = jnp.zeros((num_systems, mesh_nx, mesh_ny, mesh_nz), dtype=working_dtype)
        (mesh_out,) = _batch_spread_kernels[working_dtype](
            positions,
            values_work,
            batch_idx_i32,
            cell_inv_t,
            int(spline_order),
            mesh,
            launch_dims=(num_atoms, num_points),
        )
        return mesh_out


def spline_gather(
    positions: jnp.ndarray,
    mesh: jnp.ndarray,
    cell: jnp.ndarray,
    spline_order: int = 4,
    batch_idx: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Gather values from mesh to atoms using B-spline interpolation.

    For each atom, interpolates the mesh value at its position by summing nearby
    grid points weighted by the B-spline basis function.

    Formula: output[atom] = Σ_g mesh[g] * w(atom, g)

    where the sum is over the order^3 grid points in the atom's stencil.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic positions in Cartesian coordinates. dtype=float32 or float64.
    mesh : jax.Array
        For single-system: shape (nx, ny, nz)
        For batch: shape (B, nx, ny, nz)
        dtype=float32 or float64.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrix. dtype=float32 or float64.
    spline_order : int, optional
        B-spline order (1-4). Default: 4
    batch_idx : jax.Array | None, shape (N,), dtype=int32, optional
        System index for each atom. If None, uses single-system kernel.
        Default: None

    Returns
    -------
    values : jax.Array, shape (N,), dtype matches positions dtype
        Interpolated values at atomic positions.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Grid indices are wrapped using periodic boundary conditions.

    Examples
    --------
    >>> potentials = spline_gather(positions, potential_mesh, cell, spline_order=4)
    """
    num_atoms = positions.shape[0]
    num_points = spline_order**3
    working_dtype = _normalize_dtype(positions.dtype)

    # Cast inputs to working dtype
    mesh_work = mesh.astype(working_dtype)
    cell_work = cell.astype(working_dtype)

    # Compute cell_inv_t
    if cell_work.ndim == 2:
        cell_work = cell_work[jnp.newaxis, :, :]  # Shape (1, 3, 3)

    cell_inv = jnp.linalg.inv(cell_work)
    cell_inv_t = jnp.transpose(cell_inv, (0, 2, 1))

    # Allocate output
    output = jnp.zeros(num_atoms, dtype=working_dtype)

    if batch_idx is None:
        # Single-system kernel
        (output_out,) = _gather_kernels[working_dtype](
            positions,
            cell_inv_t,
            int(spline_order),
            mesh_work,
            output,
            launch_dims=(num_atoms, num_points),
        )
        return output_out
    else:
        # Batched kernel
        batch_idx_i32 = batch_idx.astype(jnp.int32)

        (output_out,) = _batch_gather_kernels[working_dtype](
            positions,
            batch_idx_i32,
            cell_inv_t,
            int(spline_order),
            mesh_work,
            output,
            launch_dims=(num_atoms, num_points),
        )
        return output_out


def spline_gather_vec3(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    mesh: jnp.ndarray,
    cell: jnp.ndarray,
    spline_order: int = 4,
    batch_idx: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Gather charge-weighted 3D vector values from mesh using B-splines.

    Similar to spline_gather but multiplies by the atom's charge and
    outputs to a 3D vector array (for use with vector-valued mesh fields).

    Formula: output[atom] = q[atom] * Σ_g mesh[g] * w(atom, g)

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic positions in Cartesian coordinates. dtype=float32 or float64.
    charges : jax.Array, shape (N,)
        Atomic charges (or other scalar weights). dtype=float32 or float64.
    mesh : jax.Array
        For single-system: shape (nx, ny, nz, 3) [vec3-valued mesh]
        For batch: shape (B, nx, ny, nz, 3) [vec3-valued mesh]
        dtype=vec3f or vec3d (Warp vector type).
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrix. dtype=float32 or float64.
    spline_order : int, optional
        B-spline order (1-4). Default: 4
    batch_idx : jax.Array | None, shape (N,), dtype=int32, optional
        System index for each atom. If None, uses single-system kernel.
        Default: None

    Returns
    -------
    vectors : jax.Array, shape (N, 3), dtype matches positions dtype
        Charge-weighted interpolated vectors at atomic positions.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Grid indices are wrapped using periodic boundary conditions.
    - The mesh must be a vec3-valued mesh (Warp vector type).

    Examples
    --------
    >>> electric_field = spline_gather_vec3(positions, charges, E_mesh, cell, spline_order=4)
    """
    num_atoms = positions.shape[0]
    num_points = spline_order**3
    working_dtype = _normalize_dtype(positions.dtype)

    # Cast inputs to working dtype
    charges_work = charges.astype(working_dtype)
    mesh_work = mesh.astype(working_dtype)
    cell_work = cell.astype(working_dtype)

    # Compute cell_inv_t
    if cell_work.ndim == 2:
        cell_work = cell_work[jnp.newaxis, :, :]  # Shape (1, 3, 3)

    cell_inv = jnp.linalg.inv(cell_work)
    cell_inv_t = jnp.transpose(cell_inv, (0, 2, 1))

    # Allocate output (vec3)
    output = jnp.zeros((num_atoms, 3), dtype=working_dtype)

    if batch_idx is None:
        # Single-system kernel
        (output_out,) = _gather_vec3_kernels[working_dtype](
            positions,
            charges_work,
            cell_inv_t,
            int(spline_order),
            mesh_work,
            output,
            launch_dims=(num_atoms, num_points),
        )
        return output_out
    else:
        # Batched kernel
        batch_idx_i32 = batch_idx.astype(jnp.int32)

        (output_out,) = _batch_gather_vec3_kernels[working_dtype](
            positions,
            charges_work,
            batch_idx_i32,
            cell_inv_t,
            int(spline_order),
            mesh_work,
            output,
            launch_dims=(num_atoms, num_points),
        )
        return output_out


def spline_gather_gradient(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    mesh: jnp.ndarray,
    cell: jnp.ndarray,
    spline_order: int = 4,
    batch_idx: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute forces by gathering mesh gradients using B-spline derivatives.

    Computes:

    .. math::

        F_i = -q_i \\sum_g \\phi(g) \\nabla w(r_i, g)

    The gradient ∇w is computed in fractional coordinates and then transformed
    to Cartesian coordinates via the cell matrix.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic positions in Cartesian coordinates. dtype=float32 or float64.
    charges : jax.Array, shape (N,)
        Atomic charges. dtype=float32 or float64.
    mesh : jax.Array
        For single-system: shape (nx, ny, nz)
        For batch: shape (B, nx, ny, nz)
        Scalar-valued mesh containing potential values (e.g., electrostatic potential φ).
        dtype=float32 or float64.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrix. dtype=float32 or float64.
    spline_order : int, optional
        B-spline order (1-4). Default: 4
    batch_idx : jax.Array | None, shape (N,), dtype=int32, optional
        System index for each atom. If None, uses single-system kernel.
        Default: None

    Returns
    -------
    forces : jax.Array, shape (N, 3), dtype matches positions dtype
        Forces on atoms in Cartesian coordinates.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's force.
    - The gradient is computed in fractional coordinates, then transformed:
      F_cart = cell_inv_t^T * F_frac
    - Grid indices are wrapped using periodic boundary conditions.

    Examples
    --------
    >>> forces = spline_gather_gradient(positions, charges, potential_mesh, cell, spline_order=4)
    """
    num_atoms = positions.shape[0]
    num_points = spline_order**3
    working_dtype = _normalize_dtype(positions.dtype)

    # Cast inputs to working dtype
    charges_work = charges.astype(working_dtype)
    mesh_work = mesh.astype(working_dtype)
    cell_work = cell.astype(working_dtype)

    # Compute cell_inv_t
    if cell_work.ndim == 2:
        cell_work = cell_work[jnp.newaxis, :, :]  # Shape (1, 3, 3)

    cell_inv = jnp.linalg.inv(cell_work)
    cell_inv_t = jnp.transpose(cell_inv, (0, 2, 1))

    # Allocate forces output (vec3)
    forces = jnp.zeros((num_atoms, 3), dtype=working_dtype)

    if batch_idx is None:
        # Single-system kernel
        (forces_out,) = _gather_gradient_kernels[working_dtype](
            positions,
            charges_work,
            cell_inv_t,
            int(spline_order),
            mesh_work,
            forces,
            launch_dims=(num_atoms, num_points),
        )
        return forces_out
    else:
        # Batched kernel
        batch_idx_i32 = batch_idx.astype(jnp.int32)

        (forces_out,) = _batch_gather_gradient_kernels[working_dtype](
            positions,
            charges_work,
            batch_idx_i32,
            cell_inv_t,
            int(spline_order),
            mesh_work,
            forces,
            launch_dims=(num_atoms, num_points),
        )
        return forces_out


def spline_spread_channels(
    positions: jnp.ndarray,
    values: jnp.ndarray,
    cell: jnp.ndarray,
    mesh_dims: tuple[int, int, int],
    spline_order: int = 4,
    batch_idx: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Spread multi-channel values from atoms to mesh grid using B-spline interpolation.

    This is useful for spreading multipole coefficients (e.g., 9 channels for L_max=2:
    1 monopole + 3 dipoles + 5 quadrupoles).

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic positions in Cartesian coordinates. dtype=float32 or float64.
    values : jax.Array, shape (N, C)
        Multi-channel values to spread. C is the number of channels.
        dtype=float32 or float64.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrix. dtype=float32 or float64.
    mesh_dims : tuple[int, int, int]
        Mesh dimensions (nx, ny, nz).
    spline_order : int, optional
        B-spline order (1-4). Default: 4
    batch_idx : jax.Array | None, shape (N,), dtype=int32, optional
        System index for each atom. If None, uses single-system kernel.
        Default: None

    Returns
    -------
    mesh : jax.Array
        For single-system: shape (C, nx, ny, nz)
        For batch: shape (B, C, nx, ny, nz)
        dtype matches positions dtype

    Notes
    -----
    - Uses atomic adds for thread-safe accumulation.
    - Grid indices are wrapped using periodic boundary conditions.

    Examples
    --------
    >>> # Spread 9-channel multipole coefficients
    >>> multipoles = jnp.array(jnp.random.randn(100, 9), dtype=jnp.float32)
    >>> mesh = spline_spread_channels(positions, multipoles, cell, (16, 16, 16))
    >>> print(mesh.shape)  # (9, 16, 16, 16)
    """
    num_atoms = positions.shape[0]
    num_channels = values.shape[1]
    num_points = spline_order**3
    mesh_nx, mesh_ny, mesh_nz = mesh_dims
    working_dtype = _normalize_dtype(positions.dtype)

    # Cast inputs to working dtype
    values_work = values.astype(working_dtype)
    cell_work = cell.astype(working_dtype)

    # Compute cell_inv_t
    if cell_work.ndim == 2:
        cell_work = cell_work[jnp.newaxis, :, :]  # Shape (1, 3, 3)

    cell_inv = jnp.linalg.inv(cell_work)
    cell_inv_t = jnp.transpose(cell_inv, (0, 2, 1))

    if batch_idx is None:
        # Single-system kernel
        mesh = jnp.zeros((num_channels, mesh_nx, mesh_ny, mesh_nz), dtype=working_dtype)
        (mesh_out,) = _spread_channels_kernels[working_dtype](
            positions,
            values_work,
            cell_inv_t,
            int(spline_order),
            mesh,
            launch_dims=(num_atoms, num_points),
        )
        return mesh_out
    else:
        # Batched kernel
        num_systems = cell_work.shape[0]
        batch_idx_i32 = batch_idx.astype(jnp.int32)

        # Flatten mesh from (B, C, nx, ny, nz) to (B*C, nx, ny, nz) for Warp 4D limit
        mesh = jnp.zeros(
            (num_systems * num_channels, mesh_nx, mesh_ny, mesh_nz), dtype=working_dtype
        )
        (mesh_flat,) = _batch_spread_channels_kernels[working_dtype](
            positions,
            values_work,
            batch_idx_i32,
            cell_inv_t,
            int(spline_order),
            int(num_channels),
            mesh,
            launch_dims=(num_atoms, num_points),
        )
        # Reshape back to (B, C, nx, ny, nz)
        mesh_out = mesh_flat.reshape(
            num_systems, num_channels, mesh_nx, mesh_ny, mesh_nz
        )
        return mesh_out


def spline_gather_channels(
    positions: jnp.ndarray,
    mesh: jnp.ndarray,
    cell: jnp.ndarray,
    spline_order: int = 4,
    batch_idx: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Gather multi-channel values from mesh to atoms using B-spline interpolation.

    This is the inverse of spline_spread_channels.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic positions in Cartesian coordinates. dtype=float32 or float64.
    mesh : jax.Array
        For single-system: shape (C, nx, ny, nz)
        For batch: shape (B, C, nx, ny, nz)
        dtype=float32 or float64.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrix. dtype=float32 or float64.
    spline_order : int, optional
        B-spline order (1-4). Default: 4
    batch_idx : jax.Array | None, shape (N,), dtype=int32, optional
        System index for each atom. If None, uses single-system kernel.
        Default: None

    Returns
    -------
    values : jax.Array, shape (N, C)
        Interpolated multi-channel values at atomic positions.
        dtype matches positions dtype

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Grid indices are wrapped using periodic boundary conditions.

    Examples
    --------
    >>> # Gather 9-channel potential from mesh
    >>> potential_mesh = jnp.random.randn(9, 16, 16, 16).astype(jnp.float32)
    >>> potentials = spline_gather_channels(positions, potential_mesh, cell)
    >>> print(potentials.shape)  # (100, 9)
    """
    num_atoms = positions.shape[0]
    num_points = spline_order**3
    working_dtype = _normalize_dtype(positions.dtype)

    # Cast inputs to working dtype
    mesh_work = mesh.astype(working_dtype)
    cell_work = cell.astype(working_dtype)

    # Compute cell_inv_t
    if cell_work.ndim == 2:
        cell_work = cell_work[jnp.newaxis, :, :]  # Shape (1, 3, 3)

    cell_inv = jnp.linalg.inv(cell_work)
    cell_inv_t = jnp.transpose(cell_inv, (0, 2, 1))

    if batch_idx is None:
        # Single-system kernel
        num_channels = mesh_work.shape[0]
        output = jnp.zeros((num_atoms, num_channels), dtype=working_dtype)

        (output_out,) = _gather_channels_kernels[working_dtype](
            positions,
            cell_inv_t,
            int(spline_order),
            mesh_work,
            output,
            launch_dims=(num_atoms, num_points),
        )
        return output_out
    else:
        # Batched kernel
        batch_idx_i32 = batch_idx.astype(jnp.int32)
        num_systems = mesh_work.shape[0]
        num_channels = mesh_work.shape[1]

        # Flatten mesh from (B, C, nx, ny, nz) to (B*C, nx, ny, nz)
        mesh_nx, mesh_ny, mesh_nz = (
            mesh_work.shape[2],
            mesh_work.shape[3],
            mesh_work.shape[4],
        )
        mesh_flat = mesh_work.reshape(
            num_systems * num_channels, mesh_nx, mesh_ny, mesh_nz
        )

        output = jnp.zeros((num_atoms, num_channels), dtype=working_dtype)

        (output_out,) = _batch_gather_channels_kernels[working_dtype](
            positions,
            batch_idx_i32,
            cell_inv_t,
            int(spline_order),
            int(num_channels),
            mesh_flat,
            output,
            launch_dims=(num_atoms, num_points),
        )
        return output_out


# ==============================================================================
# Deconvolution Functions
# ==============================================================================


def _bspline_modulus(k: jnp.ndarray, n: int, order: int) -> jnp.ndarray:
    """Compute the modulus of B-spline Fourier transform.

    The B-spline function M_n(u) has Fourier transform.
    For PME, we need the modulus of this for the cardinal B-spline interpolation.

    Parameters
    ----------
    k : jax.Array
        Frequency indices (integers).
    n : int
        Grid dimension.
    order : int
        B-spline order.

    Returns
    -------
    jax.Array
        |b(k)|^2 where b(k) is the B-spline Fourier coefficient.
    """
    # Compute the exponential B-spline factors
    # Following Essmann et al. (1995) Eq. 4.7
    pi = jnp.pi

    # Handle k=0 case specially (limit is 1)
    result = jnp.ones_like(k, dtype=jnp.float64)

    # For non-zero k, compute the product
    nonzero_mask = k != 0

    # w = 2*pi * k / n
    w = 2.0 * pi * k.astype(jnp.float64) / n

    # The B-spline Fourier coefficient is:
    # b(k) = sum_{j=0}^{order-1} M_order(j+1) * exp(2*pi*i j k / n)
    # where M_order is the B-spline basis function

    # Compute M_order values at integer points 1, 2, ..., order
    m_values = _compute_bspline_coefficients(order)

    # Sum: b(k) = sum_j M_order(j+1) * exp(i w j)
    b_real = jnp.zeros_like(k, dtype=jnp.float64)
    b_imag = jnp.zeros_like(k, dtype=jnp.float64)

    for j in range(order):
        phase = w * j
        b_real = b_real + m_values[j] * jnp.cos(phase)
        b_imag = b_imag + m_values[j] * jnp.sin(phase)

    # |b(k)|^2
    b_sq = b_real**2 + b_imag**2

    # Handle k=0 case
    result = jnp.where(nonzero_mask, b_sq, result)

    return result


def _compute_bspline_coefficients(order: int) -> jnp.ndarray:
    """Compute B-spline basis function values at integer points.

    For a B-spline of order n, we need M_n(1), M_n(2), ..., M_n(n).
    These are used in the Fourier transform computation.

    Parameters
    ----------
    order : int
        B-spline order.

    Returns
    -------
    jax.Array
        B-spline values [M_n(1), M_n(2), ..., M_n(n)].
    """
    if order == 1:
        return jnp.array([1.0], dtype=jnp.float64)
    elif order == 2:
        return jnp.array([0.5, 0.5], dtype=jnp.float64)
    elif order == 3:
        return jnp.array([1 / 6, 4 / 6, 1 / 6], dtype=jnp.float64)
    elif order == 4:
        return jnp.array([1 / 24, 11 / 24, 11 / 24, 1 / 24], dtype=jnp.float64)
    elif order == 5:
        return jnp.array(
            [1 / 120, 26 / 120, 66 / 120, 26 / 120, 1 / 120],
            dtype=jnp.float64,
        )
    elif order == 6:
        return jnp.array(
            [1 / 720, 57 / 720, 302 / 720, 302 / 720, 57 / 720, 1 / 720],
            dtype=jnp.float64,
        )
    else:
        # Use recursive definition for higher orders
        # M_n(u) = u/(n-1) * M_{n-1}(u) + (n-u)/(n-1) * M_{n-1}(u-1)
        coeffs = _compute_bspline_coefficients(order - 1)
        new_coeffs = jnp.zeros(order, dtype=jnp.float64)
        for j in range(order):
            u = float(j + 1)
            if j < order - 1:
                new_coeffs = new_coeffs.at[j].add(u / (order - 1) * coeffs[j])
            if j > 0:
                new_coeffs = new_coeffs.at[j].add(
                    (order - u) / (order - 1) * coeffs[j - 1]
                )
        return new_coeffs


def compute_bspline_deconvolution(
    mesh_dims: tuple[int, int, int],
    spline_order: int = 4,
) -> jnp.ndarray:
    """Compute B-spline deconvolution factors for Fourier space correction.

    In FFT-based methods (like PME), the B-spline interpolation introduces
    smoothing in the charge distribution. This function computes the
    deconvolution factors to correct for this smoothing in Fourier space.

    The correction is: mesh_corrected_k = mesh_k * deconv

    Parameters
    ----------
    mesh_dims : tuple[int, int, int]
        Mesh dimensions (nx, ny, nz).
    spline_order : int, optional
        B-spline order. Default: 4

    Returns
    -------
    deconv : jax.Array, shape (nx, ny, nz)
        Deconvolution factors. Multiply with FFT of mesh to correct.
        dtype=float64

    Notes
    -----
    The deconvolution factor for a given k-vector is:

    D(k_x, k_y, k_z) = 1 / (|b(k_x)|^2 * |b(k_y)|^2 * |b(k_z)|^2)

    where b(k) is the Fourier transform of the 1D B-spline.

    For efficiency, this uses the separable property of the 3D B-spline.

    Examples
    --------
    >>> deconv = compute_bspline_deconvolution((16, 16, 16), spline_order=4)
    >>> mesh_fft = jnp.fft.fftn(charge_mesh)
    >>> mesh_corrected_fft = mesh_fft * deconv
    >>> charge_mesh_corrected = jnp.fft.ifftn(mesh_corrected_fft).real
    """
    nx, ny, nz = mesh_dims

    # Create frequency indices for each dimension
    # For FFT, frequencies are arranged as [0, 1, ..., n//2, -(n//2-1), ..., -1]
    kx = jnp.fft.fftfreq(nx) * nx  # Integer frequencies
    ky = jnp.fft.fftfreq(ny) * ny
    kz = jnp.fft.fftfreq(nz) * nz

    # Compute |b(k)|^2 for each dimension
    bx_sq = _bspline_modulus(kx, nx, spline_order)
    by_sq = _bspline_modulus(ky, ny, spline_order)
    bz_sq = _bspline_modulus(kz, nz, spline_order)

    # The 3D deconvolution is the product of 1D factors
    # deconv = 1 / (bx^2 * by^2 * bz^2)
    # Use outer product for efficiency
    bx_sq = bx_sq.reshape(nx, 1, 1)
    by_sq = by_sq.reshape(1, ny, 1)
    bz_sq = bz_sq.reshape(1, 1, nz)

    b_sq_3d = bx_sq * by_sq * bz_sq

    # Avoid division by zero (should not happen for reasonable orders)
    b_sq_3d = jnp.maximum(b_sq_3d, 1e-15)

    deconv = 1.0 / b_sq_3d

    return deconv


def compute_bspline_deconvolution_1d(
    n: int,
    spline_order: int = 4,
) -> jnp.ndarray:
    """Compute 1D B-spline deconvolution factors.

    Useful for separable operations or debugging.

    Parameters
    ----------
    n : int
        Grid dimension.
    spline_order : int, optional
        B-spline order. Default: 4

    Returns
    -------
    deconv_1d : jax.Array, shape (n,)
        1D deconvolution factors.
        dtype=float64

    Examples
    --------
    >>> deconv_1d = compute_bspline_deconvolution_1d(16, spline_order=4)
    """
    k = jnp.fft.fftfreq(n) * n
    b_sq = _bspline_modulus(k, n, spline_order)
    b_sq = jnp.maximum(b_sq, 1e-15)

    return 1.0 / b_sq
