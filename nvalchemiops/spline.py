# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
B-Spline Interpolation Module
=============================

This module provides B-spline interpolation functions for mesh-based calculations,
commonly used in Particle Mesh Ewald (PME) and similar methods.

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

5. SPREAD_CHANNELS: Scatter multi-channel values (e.g., multipoles) to mesh
   mesh[c, g] += values[atom, c] * weight(atom, g)

6. GATHER_CHANNELS: Collect multi-channel values from mesh
   values[atom, c] = Σ_g mesh[c, g] * weight(atom, g)

7. DECONVOLUTION: Correct B-spline approximation in Fourier space
   Used in FFT-based methods to remove B-spline smoothing artifacts.

USAGE
=====

Single-system:
    from nvalchemiops.spline import spline_spread, spline_gather, spline_gather_gradient

    # Spread charges to mesh
    mesh = spline_spread(positions, charges, cell, mesh_dims, spline_order=4)

    # Gather potential from mesh
    potentials = spline_gather(positions, potential_mesh, cell, spline_order=4)

    # Gather forces
    forces = spline_gather_gradient(positions, charges, potential_mesh, cell, spline_order=4)

Multi-channel (multipoles):
    from nvalchemiops.spline import spline_spread_channels, spline_gather_channels

    # multipoles has shape (N, num_channels) e.g. (N, 9) for L_max=2
    mesh = spline_spread_channels(positions, multipoles, cell, mesh_dims, spline_order=4)

    # Gather multi-channel potential from mesh
    potentials = spline_gather_channels(positions, potential_mesh, cell, spline_order=4)

Batched (multiple systems):
    # Spread charges to batched mesh
    mesh = spline_spread(positions, charges, cell, mesh_dims, spline_order=4, batch_idx=batch_idx)

    # Gather potential from batched mesh
    potentials = spline_gather(positions, potential_mesh, cell, spline_order=4, batch_idx=batch_idx)

Deconvolution:
    from nvalchemiops.spline import compute_bspline_deconvolution

    # Get deconvolution factors for mesh
    deconv = compute_bspline_deconvolution(mesh_dims, spline_order=4, device=device)

    # Apply in Fourier space: mesh_corrected_k = mesh_k * deconv
    mesh_fft = torch.fft.fftn(mesh)
    mesh_corrected_fft = mesh_fft * deconv
    mesh_corrected = torch.fft.ifftn(mesh_corrected_fft).real

REFERENCES
==========

- Essmann et al. (1995). J. Chem. Phys. 103, 8577 (PME B-splines)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from nvalchemiops.autograd import (
    OutputSpec,
    WarpAutogradContextManager,
    attach_for_backward,
    needs_grad,
    warp_custom_op,
    warp_from_torch,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

# Mathematical constants
PI = math.pi
TWOPI = 2.0 * PI


###########################################################################################
########################### B-Spline Weight Functions #####################################
###########################################################################################


@wp.func
def bspline_weight(u: Any, order: wp.int32) -> Any:
    """Compute B-spline basis function M_n(u).

    Parameters
    ----------
    u : float (Any)
        Parameter in [0, order). Type-generic (float32 or float64).
    order : wp.int32
        Spline order (1=constant, 2=linear, 3=quadratic, 4=cubic).

    Returns
    -------
    float (Any)
        Weight value M_n(u). Same type as input.
    """
    # Type-generic constants
    zero = type(u)(0.0)
    one = type(u)(1.0)
    two = type(u)(2.0)
    three = type(u)(3.0)
    four = type(u)(4.0)
    six = type(u)(6.0)

    if order == 4:
        if u >= zero and u < one:
            return u * u * u / six
        elif u >= one and u < two:
            u2 = u * u
            u3 = u2 * u
            return (
                type(u)(-3.0) * u3 + type(u)(12.0) * u2 - type(u)(12.0) * u + four
            ) / six
        elif u >= two and u < three:
            u2 = u * u
            u3 = u2 * u
            return (
                three * u3 - type(u)(24.0) * u2 + type(u)(60.0) * u - type(u)(44.0)
            ) / six
        elif u >= three and u < four:
            v = four - u
            return v * v * v / six
        else:
            return zero
    elif order == 3:
        if u >= zero and u < one:
            return u * u / two
        elif u >= one and u < two:
            return type(u)(0.75) - (u - type(u)(1.5)) * (u - type(u)(1.5))
        elif u >= two and u < three:
            v = three - u
            return v * v / two
        else:
            return zero
    elif order == 2:
        if u >= zero and u < one:
            return u
        elif u >= one and u < two:
            return two - u
        else:
            return zero
    elif order == 1:
        if u >= zero and u < one:
            return one
        else:
            return zero
    else:
        return zero


@wp.func
def bspline_derivative(u: Any, order: wp.int32) -> Any:
    """Compute B-spline derivative dM_n(u)/du.

    Parameters
    ----------
    u : float (Any)
        Parameter in [0, order). Type-generic (float32 or float64).
    order : wp.int32
        Spline order.

    Returns
    -------
    float (Any)
        Derivative value. Same type as input.
    """
    # Type-generic constants
    zero = type(u)(0.0)
    one = type(u)(1.0)
    two = type(u)(2.0)
    three = type(u)(3.0)
    four = type(u)(4.0)
    six = type(u)(6.0)

    if order == 4:
        if u >= zero and u < one:
            return u * u / two
        elif u >= one and u < two:
            return (type(u)(-9.0) * u * u + type(u)(24.0) * u - type(u)(12.0)) / six
        elif u >= two and u < three:
            return (type(u)(9.0) * u * u - type(u)(48.0) * u + type(u)(60.0)) / six
        elif u >= three and u < four:
            v = four - u
            return -three * v * v / six
        else:
            return zero
    elif order == 3:
        if u >= zero and u < one:
            return u
        elif u >= one and u < two:
            return -two * (u - type(u)(1.5))
        elif u >= two and u < three:
            return -(three - u)
        else:
            return zero
    elif order == 2:
        if u >= zero and u < one:
            return one
        elif u >= one and u < two:
            return -one
        else:
            return zero
    else:
        return zero


###########################################################################################
########################### Grid Utility Functions ########################################
###########################################################################################


@wp.func
def compute_fractional_coords(
    position: Any,
    cell_inv_t: Any,
    mesh_dims: wp.vec3i,
) -> Any:
    """Convert Cartesian position to mesh coordinates.

    Parameters
    ----------
    position : vec3 (Any)
        Atomic position. Type-generic (vec3f or vec3d).
    cell_inv_t : mat33 (Any)
        Transpose of inverse cell. Type-generic (mat33f or mat33d).
    mesh_dims : wp.vec3i
        Mesh dimensions.

    Returns
    -------
    base_grid : wp.vec3i
        Base grid point (floor of mesh coords).
    theta : vec3 (Any)
        Fractional part [0, 1) in each dimension. Same type as position.

    Note: Returns (base_grid, theta) as a tuple via multiple return values.
    """
    # Convert to fractional coordinates
    frac = cell_inv_t * position
    p0 = position[0]
    # Scale to mesh coordinates
    mesh_x = frac[0] * type(p0)(mesh_dims[0])
    mesh_y = frac[1] * type(p0)(mesh_dims[1])
    mesh_z = frac[2] * type(p0)(mesh_dims[2])

    # Base grid point
    mx = wp.int32(wp.floor(mesh_x))
    my = wp.int32(wp.floor(mesh_y))
    mz = wp.int32(wp.floor(mesh_z))

    # Fractional part
    theta_x = mesh_x - type(p0)(mx)
    theta_y = mesh_y - type(p0)(my)
    theta_z = mesh_z - type(p0)(mz)

    return wp.vec3i(mx, my, mz), type(position)(theta_x, theta_y, theta_z)


@wp.func
def bspline_grid_offset(
    point_idx: wp.int32,
    order: wp.int32,
    theta: Any,
) -> wp.vec3i:
    """Compute grid offset for B-spline point index.

    For B-splines, points are indexed 0 to order^3-1 and arranged in a cube.
    The offset is computed such that the B-spline parameter u is always in [0, n).

    The offset_start for each dimension is floor(theta - (n-2)/2), which ensures
    that for any theta in [0, 1), all n grid points have valid u values.

    Parameters
    ----------
    point_idx : wp.int32
        Linear point index (0 to order^3-1).
    order : wp.int32
        Spline order.
    theta : vec3 (Any)
        Fractional position within the base grid cell [0, 1) in each dimension.
        Type-generic (vec3f or vec3d).

    Returns
    -------
    wp.vec3i
        Grid offset (relative to base grid point).
    """
    order2 = order * order
    i = point_idx // order2
    j = (point_idx % order2) // order
    k = point_idx % order

    t0 = theta[0]

    # Compute offset_start = floor(theta - (n-2)/2) for each dimension
    # This ensures u = n/2 + theta - offset is always in [0, n)
    half_n_minus_1 = type(t0)(order - 2) * type(t0)(0.5)
    offset_start_x = wp.int32(wp.floor(t0 - half_n_minus_1))
    offset_start_y = wp.int32(wp.floor(theta[1] - half_n_minus_1))
    offset_start_z = wp.int32(wp.floor(theta[2] - half_n_minus_1))

    return wp.vec3i(i + offset_start_x, j + offset_start_y, k + offset_start_z)


@wp.func
def bspline_weight_3d(
    theta: Any,
    offset: wp.vec3i,
    order: wp.int32,
) -> Any:
    """Compute 3D B-spline weight (separable product).

    The B-spline parameter u is computed as:

    .. math::

        u = \\text{order}/2 + \\theta - \\text{offset}

    When offset = i + offset_start (from bspline_grid_offset), this gives
    u values in [0, n) that sum to 1 and are centered at the atom position.

    Parameters
    ----------
    theta : vec3 (Any)
        Fractional position within the base grid cell [0, 1).
        Type-generic (vec3f or vec3d).
    offset : wp.vec3i
        Grid offset from base grid point (includes offset_start adjustment).
    order : wp.int32
        Spline order.

    Returns
    -------
    float (Any)
        Weight = M(u_x) * M(u_y) * M(u_z). Same scalar type as theta.
    """
    # Get scalar type from theta vector
    t0 = theta[0]
    half_order = type(t0)(order) * type(t0)(0.5)
    zero = type(t0)(0.0)
    order_f = type(t0)(order)

    # u = n/2 + theta - offset
    u_x = half_order + t0 - type(t0)(offset[0])
    u_y = half_order + theta[1] - type(t0)(offset[1])
    u_z = half_order + theta[2] - type(t0)(offset[2])

    if (
        u_x < zero
        or u_x >= order_f
        or u_y < zero
        or u_y >= order_f
        or u_z < zero
        or u_z >= order_f
    ):
        return zero

    return (
        bspline_weight(u_x, order)
        * bspline_weight(u_y, order)
        * bspline_weight(u_z, order)
    )


@wp.func
def bspline_weight_gradient_3d(
    theta: Any,
    offset: wp.vec3i,
    order: wp.int32,
    mesh_dims: wp.vec3i,
) -> Any:
    """Compute gradient of 3D B-spline weight.

    The B-spline parameter u is computed as:

    .. math::

        u = \\text{order}/2 + \\theta - \\text{offset}

    The gradient with respect to theta is:

    .. math::

        \\begin{aligned}
        \\frac{\\partial u}{\\partial \\theta} &= +1 \\\\
        \\frac{\\partial \\text{weight}}{\\partial \\theta} &= \\frac{\\partial M}{\\partial u} \\cdot \\frac{\\partial u}{\\partial \\theta} = \\frac{\\partial M}{\\partial u}
        \\end{aligned}

    Parameters
    ----------
    theta : vec3 (Any)
        Fractional position within the base grid cell [0, 1).
        Type-generic (vec3f or vec3d).
    offset : wp.vec3i
        Grid offset from base grid point (includes offset_start adjustment).
    order : wp.int32
        Spline order.
    mesh_dims : wp.vec3i
        Mesh dimensions (for scaling to Cartesian coordinates).

    Returns
    -------
    vec3 (Any)
        Gradient :math:`\\nabla` weight in fractional coordinates (scaled by mesh_dims).
        Same type as theta.
    """
    # Get scalar type from theta vector
    t0 = theta[0]
    half_order = type(t0)(order) * type(t0)(0.5)
    zero = type(t0)(0.0)
    order_f = type(t0)(order)

    # u = n/2 + theta - offset
    u_x = half_order + t0 - type(t0)(offset[0])
    u_y = half_order + theta[1] - type(t0)(offset[1])
    u_z = half_order + theta[2] - type(t0)(offset[2])

    if (
        u_x < zero
        or u_x >= order_f
        or u_y < zero
        or u_y >= order_f
        or u_z < zero
        or u_z >= order_f
    ):
        return type(theta)(zero, zero, zero)

    w_x = bspline_weight(u_x, order)
    w_y = bspline_weight(u_y, order)
    w_z = bspline_weight(u_z, order)

    # Positive sign because u = half_order + theta - offset, so ∂u/∂theta = +1
    dw_x = bspline_derivative(u_x, order) * type(t0)(mesh_dims[0])
    dw_y = bspline_derivative(u_y, order) * type(t0)(mesh_dims[1])
    dw_z = bspline_derivative(u_z, order) * type(t0)(mesh_dims[2])

    return type(theta)(dw_x * w_y * w_z, w_x * dw_y * w_z, w_x * w_y * dw_z)


@wp.func
def wrap_grid_index(idx: wp.int32, dim: wp.int32) -> wp.int32:
    """Wrap grid index for periodic boundaries."""
    return ((idx % dim) + dim) % dim


###########################################################################################
########################### Single-System Warp Kernels ####################################
###########################################################################################


@wp.kernel
def _bspline_spread_kernel(
    positions: wp.array(dtype=Any),
    values: wp.array(dtype=Any),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array3d(dtype=Any),
):
    """Spread (scatter) values from atoms to a 3D mesh using B-spline interpolation.

    For each atom, distributes its value to nearby grid points weighted by the
    B-spline basis function. This is the adjoint operation to gathering.

    Formula: mesh[g] += value[atom] * w(atom, g)

    where w(atom, g) is the product of 1D B-spline weights in each dimension.

    Launch Grid
    -----------
    dim = [num_atoms, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates in Cartesian space.
    values : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Values to spread (e.g., charges).
    cell_inv_t : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix for fractional coordinate conversion.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array3d, shape (nx, ny, nz), dtype=wp.float32 or wp.float64
        OUTPUT: 3D mesh to accumulate values into. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds for thread-safe accumulation to shared grid points.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight skip the atomic add for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
    position = positions[atom_idx]
    value = values[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    if weight > type(value)(0.0):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        wp.atomic_add(mesh, gx, gy, gz, value * weight)


@wp.kernel
def _bspline_gather_kernel(
    positions: wp.array(dtype=Any),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array3d(dtype=Any),
    output: wp.array(dtype=Any),
):
    """Gather (interpolate) values from a 3D mesh to atom positions using B-splines.

    For each atom, interpolates the mesh value at its position by summing nearby
    grid points weighted by the B-spline basis function.

    Formula: output[atom] = Σ_g mesh[g] * w(atom, g)

    where the sum is over the order^3 grid points in the atom's stencil.

    Launch Grid
    -----------
    dim = [num_atoms, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates in Cartesian space.
    cell_inv_t : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix for fractional coordinate conversion.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array3d, shape (nx, ny, nz), dtype=wp.float32 or wp.float64
        3D mesh containing values to interpolate (e.g., electrostatic potential).
    output : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Interpolated values per atom. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight skip the atomic add for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
    position = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    mesh_val = mesh[0, 0, 0]  # Get type reference
    if weight > type(mesh_val)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[gx, gy, gz]
        wp.atomic_add(output, atom_idx, mesh_val * weight)


@wp.kernel
def _bspline_gather_vec3_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array3d(dtype=Any),
    output: wp.array(dtype=Any),
):
    """Gather charge-weighted 3D vector values from mesh to atoms using B-splines.

    Similar to _bspline_gather_kernel but multiplies by the atom's charge and
    outputs to a 3D vector array (for use with vector-valued mesh fields).

    Formula: output[atom] = q[atom] * Σ_g mesh[g] * w(atom, g)

    Launch Grid
    -----------
    dim = [num_atoms, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates in Cartesian space.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges (or other scalar weights).
    cell_inv_t : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix for fractional coordinate conversion.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array3d, shape (nx, ny, nz), dtype=wp.vec3f or wp.vec3d
        3D mesh containing vector values to interpolate.
    output : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Charge-weighted interpolated vectors per atom. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic add for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
    position = positions[atom_idx]
    charge = charges[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    if weight > type(charge)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[gx, gy, gz]
        wp.atomic_add(output, atom_idx, charge * mesh_val * weight)


@wp.kernel
def _bspline_gather_gradient_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array3d(dtype=Any),
    forces: wp.array(dtype=Any),
):
    """Compute forces by gathering mesh gradients using B-spline derivatives.

    Computes:

    .. math::

        F_i = -q_i \\sum_g \\phi(g) \\nabla w(r_i, g)

    The gradient ∇w is computed in fractional coordinates and then transformed
    to Cartesian coordinates via the cell matrix.

    Launch Grid
    -----------
    dim = [num_atoms, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates in Cartesian space.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell_inv_t : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix for fractional coordinate conversion.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array3d, shape (nx, ny, nz), dtype=wp.float32 or wp.float64
        3D mesh containing potential values (e.g., electrostatic potential φ).
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Forces per atom in Cartesian coordinates. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's force.
    - The gradient is computed in fractional coordinates, then transformed:
      F_cart = cell_inv_t^T * F_frac
    - Threads with zero gradient magnitude skip the atomic add for efficiency.
    - Grid indices are wrapped using periodic boundary conditions.
    """
    atom_idx, point_idx = wp.tid()

    mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
    position = positions[atom_idx]
    charge = charges[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    grad_frac = bspline_weight_gradient_3d(theta, offset, order, mesh_dims)

    grad_mag = wp.abs(grad_frac[0]) + wp.abs(grad_frac[1]) + wp.abs(grad_frac[2])

    if grad_mag > type(charge)(0.0):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[gx, gy, gz]

        force_frac = type(position)(
            -charge * mesh_val * grad_frac[0],
            -charge * mesh_val * grad_frac[1],
            -charge * mesh_val * grad_frac[2],
        )
        force = wp.transpose(cell_inv_t[0]) * force_frac

        wp.atomic_add(forces, atom_idx, force)


###########################################################################################
########################### Batch Warp Kernels #############################################
###########################################################################################


@wp.kernel
def _batch_bspline_spread_kernel(
    positions: wp.array(dtype=Any),
    values: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),  # (B, 3, 3)
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),  # (B, nx, ny, nz)
):
    """Spread values from atoms to a batched 4D mesh using B-splines.

    Batched version of _bspline_spread_kernel for multiple systems. Each atom
    is assigned to a system via batch_idx, and values are spread to that
    system's mesh slice.

    Formula: mesh[sys, g] += value[atom] * w(atom, g)

    Launch Grid
    -----------
    dim = [num_atoms_total, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    values : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Values to spread (e.g., charges) for all systems.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell_inv_t : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array4d, shape (B, nx, ny, nz), dtype=wp.float32 or wp.float64
        OUTPUT: 4D mesh (batch × spatial) to accumulate values. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds for thread-safe accumulation to shared grid points.
    - Each system uses its own cell matrix for fractional coordinate conversion.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic add for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    sys_idx = batch_idx[atom_idx]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]
    value = values[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[sys_idx], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    if weight > type(value)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        wp.atomic_add(mesh, sys_idx, gx, gy, gz, value * weight)


@wp.kernel
def _batch_bspline_gather_kernel(
    positions: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),  # (B, 3, 3)
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),  # (B, nx, ny, nz)
    output: wp.array(dtype=Any),
):
    """Gather values from a batched 4D mesh to atom positions using B-splines.

    Batched version of _bspline_gather_kernel for multiple systems. Each atom
    reads from its assigned system's mesh slice via batch_idx.

    Formula: output[atom] = Σ_g mesh[sys, g] * w(atom, g)

    Launch Grid
    -----------
    dim = [num_atoms_total, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell_inv_t : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array4d, shape (B, nx, ny, nz), dtype=wp.float32 or wp.float64
        4D mesh (batch × spatial) containing values to interpolate.
    output : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Interpolated values per atom. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Each system uses its own cell matrix for fractional coordinate conversion.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic add for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    sys_idx = batch_idx[atom_idx]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[sys_idx], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    mesh_val = mesh[0, 0, 0, 0]  # Get type reference
    if weight > type(mesh_val)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[sys_idx, gx, gy, gz]
        wp.atomic_add(output, atom_idx, mesh_val * weight)


@wp.kernel
def _batch_bspline_gather_vec3_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),  # (B, 3, 3)
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),  # (B, nx, ny, nz)
    output: wp.array(dtype=Any),
):
    """Gather charge-weighted 3D vector values from batched mesh using B-splines.

    Batched version of _bspline_gather_vec3_kernel for multiple systems.

    Formula: output[atom] = q[atom] * Σ_g mesh[sys, g] * w(atom, g)

    Launch Grid
    -----------
    dim = [num_atoms_total, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges (or other scalar weights).
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell_inv_t : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array4d, shape (B, nx, ny, nz), dtype=wp.vec3f or wp.vec3d
        4D mesh (batch × spatial) containing vector values.
    output : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Charge-weighted interpolated vectors per atom. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Each system uses its own cell matrix for fractional coordinate conversion.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic add for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    sys_idx = batch_idx[atom_idx]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]
    charge = charges[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[sys_idx], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    if weight > type(charge)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[sys_idx, gx, gy, gz]
        wp.atomic_add(output, atom_idx, charge * mesh_val * weight)


@wp.kernel
def _batch_bspline_gather_gradient_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),  # (B, 3, 3)
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),  # (B, nx, ny, nz)
    forces: wp.array(dtype=Any),
):
    """Compute forces by gathering mesh gradients from batched mesh using B-spline derivatives.

    Computes:

    .. math::

        F_i = -q_i \\sum_g \\phi(g) \\nabla w(r_i, g)

    The gradient ∇w is computed in fractional coordinates and then transformed
    to Cartesian coordinates via each system's cell matrix.

    Launch Grid
    -----------
    dim = [num_atoms_total, order^3]

    Each thread handles one (atom, grid_point) pair within the atom's stencil.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell_inv_t : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array4d, shape (B, nx, ny, nz), dtype=wp.float32 or wp.float64
        4D mesh (batch × spatial) containing potential values.
    forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Forces per atom in Cartesian coordinates. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's force.
    - The gradient is computed in fractional coordinates, then transformed:
      F_cart = cell_inv_t[sys]^T * F_frac
    - Each system uses its own cell matrix for the transformation.
    - Threads with zero gradient magnitude skip the atomic add for efficiency.
    - Grid indices are wrapped using periodic boundary conditions.
    """
    atom_idx, point_idx = wp.tid()

    sys_idx = batch_idx[atom_idx]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]
    charge = charges[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[sys_idx], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    grad_frac = bspline_weight_gradient_3d(theta, offset, order, mesh_dims)

    grad_mag = wp.abs(grad_frac[0]) + wp.abs(grad_frac[1]) + wp.abs(grad_frac[2])

    if grad_mag > type(charge)(0.0):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[sys_idx, gx, gy, gz]

        force_frac = type(position)(
            -charge * mesh_val * grad_frac[0],
            -charge * mesh_val * grad_frac[1],
            -charge * mesh_val * grad_frac[2],
        )
        force = wp.transpose(cell_inv_t[sys_idx]) * force_frac

        wp.atomic_add(forces, atom_idx, force)


###########################################################################################
########################### Multi-Channel Warp Kernels ####################################
###########################################################################################


@wp.kernel
def _bspline_spread_channels_kernel(
    positions: wp.array(dtype=Any),
    values: wp.array2d(dtype=Any),  # (N, C)
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),  # (C, nx, ny, nz)
):
    """Spread multi-channel values from atoms to mesh using B-splines.

    Similar to _bspline_spread_kernel but handles multiple channels per atom,
    useful for multipole moments (e.g., monopole + dipole + quadrupole).

    Formula: mesh[c, g] += values[atom, c] * w(atom, g)

    for each channel c = 0, 1, ..., C-1.

    Launch Grid
    -----------
    dim = [num_atoms, order^3]

    Each thread handles one (atom, grid_point) pair and iterates over all channels.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates in Cartesian space.
    values : wp.array2d, shape (N, C), dtype=wp.float32 or wp.float64
        Multi-channel values to spread (e.g., multipole moments).
    cell_inv_t : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix for fractional coordinate conversion.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array4d, shape (C, nx, ny, nz), dtype=wp.float32 or wp.float64
        OUTPUT: 4D mesh (channels × spatial) to accumulate values. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds for thread-safe accumulation to shared grid points.
    - Each channel is spread independently to its own mesh slice.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic adds for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    num_channels = values.shape[1]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    val = values[0, 0]  # Get type reference
    if weight > type(val)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        # Spread each channel
        for c in range(num_channels):
            val = values[atom_idx, c]
            wp.atomic_add(mesh, c, gx, gy, gz, val * weight)


@wp.kernel
def _bspline_gather_channels_kernel(
    positions: wp.array(dtype=Any),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),  # (C, nx, ny, nz)
    output: wp.array2d(dtype=Any),  # (N, C)
):
    """Gather multi-channel values from mesh to atoms using B-splines.

    Similar to _bspline_gather_kernel but handles multiple channels,
    useful for multipole-based methods.

    Formula: output[atom, c] = Σ_g mesh[c, g] * w(atom, g)

    for each channel c = 0, 1, ..., C-1.

    Launch Grid
    -----------
    dim = [num_atoms, order^3]

    Each thread handles one (atom, grid_point) pair and iterates over all channels.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates in Cartesian space.
    cell_inv_t : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix for fractional coordinate conversion.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    mesh : wp.array4d, shape (C, nx, ny, nz), dtype=wp.float32 or wp.float64
        4D mesh (channels × spatial) containing values to interpolate.
    output : wp.array2d, shape (N, C), dtype=wp.float32 or wp.float64
        OUTPUT: Interpolated multi-channel values per atom. Must be zero-initialized.

    Notes
    -----
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Each channel is gathered independently from its own mesh slice.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic adds for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    num_channels = mesh.shape[0]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    mesh_val = mesh[0, 0, 0, 0]  # Get type reference
    if weight > type(mesh_val)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        # Gather each channel
        for c in range(num_channels):
            mesh_val = mesh[c, gx, gy, gz]
            wp.atomic_add(output, atom_idx, c, mesh_val * weight)


@wp.kernel
def _batch_bspline_spread_channels_kernel(
    positions: wp.array(dtype=Any),
    values: wp.array2d(dtype=Any),  # (N, C)
    batch_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),  # (B, 3, 3)
    order: wp.int32,
    num_channels: wp.int32,
    mesh: wp.array4d(dtype=Any),  # (B*C, nx, ny, nz) - flattened batch*channel
):
    """Spread multi-channel values from atoms to batched mesh using B-splines.

    Batched version of _bspline_spread_channels_kernel. Due to Warp's 4D array
    limit, the batch and channel dimensions are flattened into a single dimension.

    Formula: mesh[sys*C + c, g] += values[atom, c] * w(atom, g)

    Launch Grid
    -----------
    dim = [num_atoms_total, order^3]

    Each thread handles one (atom, grid_point) pair and iterates over all channels.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    values : wp.array2d, shape (N_total, C), dtype=wp.float32 or wp.float64
        Multi-channel values to spread (e.g., multipole moments).
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell_inv_t : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    num_channels : wp.int32
        Number of channels (C).
    mesh : wp.array4d, shape (B*C, nx, ny, nz), dtype=wp.float32 or wp.float64
        OUTPUT: Flattened 4D mesh to accumulate values. Must be zero-initialized.

    Notes
    -----
    - Mesh storage: (B*C, nx, ny, nz) with flat_idx = sys_idx * C + channel_idx.
    - Uses atomic adds for thread-safe accumulation to shared grid points.
    - Each system uses its own cell matrix for fractional coordinate conversion.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic adds for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    sys_idx = batch_idx[atom_idx]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[sys_idx], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    val = values[0, 0]  # Get type reference
    if weight > type(val)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        # Spread each channel using flattened batch*channel indexing
        for c in range(num_channels):
            flat_idx = sys_idx * num_channels + c
            val = values[atom_idx, c]
            wp.atomic_add(mesh, flat_idx, gx, gy, gz, val * weight)


@wp.kernel
def _batch_bspline_gather_channels_kernel(
    positions: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),  # (B, 3, 3)
    order: wp.int32,
    num_channels: wp.int32,
    mesh: wp.array4d(dtype=Any),  # (B*C, nx, ny, nz) - flattened batch*channel
    output: wp.array2d(dtype=Any),  # (N, C)
):
    """Gather multi-channel values from batched mesh to atoms using B-splines.

    Batched version of _bspline_gather_channels_kernel. Due to Warp's 4D array
    limit, the batch and channel dimensions are flattened into a single dimension.

    Formula: output[atom, c] = Σ_g mesh[sys*C + c, g] * w(atom, g)

    Launch Grid
    -----------
    dim = [num_atoms_total, order^3]

    Each thread handles one (atom, grid_point) pair and iterates over all channels.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell_inv_t : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended for PME.
    num_channels : wp.int32
        Number of channels (C).
    mesh : wp.array4d, shape (B*C, nx, ny, nz), dtype=wp.float32 or wp.float64
        Flattened 4D mesh (batch*channels × spatial) containing values.
    output : wp.array2d, shape (N_total, C), dtype=wp.float32 or wp.float64
        OUTPUT: Interpolated multi-channel values per atom. Must be zero-initialized.

    Notes
    -----
    - Mesh storage: (B*C, nx, ny, nz) with flat_idx = sys_idx * C + channel_idx.
    - Uses atomic adds since multiple threads contribute to each atom's output.
    - Each system uses its own cell matrix for fractional coordinate conversion.
    - Grid indices are wrapped using periodic boundary conditions.
    - Threads with 1e-8 weight or less skip the atomic adds for efficiency.
    """
    atom_idx, point_idx = wp.tid()

    sys_idx = batch_idx[atom_idx]
    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[sys_idx], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    weight = bspline_weight_3d(theta, offset, order)

    mesh_val = mesh[0, 0, 0, 0]  # Get type reference
    if weight > type(mesh_val)(1e-8):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        # Gather each channel using flattened batch*channel indexing
        for c in range(num_channels):
            flat_idx = sys_idx * num_channels + c
            mesh_val = mesh[flat_idx, gx, gy, gz]
            wp.atomic_add(output, atom_idx, c, mesh_val * weight)


###########################################################################################
########################### Kernel Overloads for Dtype Flexibility #########################
###########################################################################################

# Type lists for creating overloads
_T = [wp.float32, wp.float64]
_V = [wp.vec3f, wp.vec3d]
_M = [wp.mat33f, wp.mat33d]

# Single-system kernel overloads
_bspline_spread_kernel_overload = {}
_bspline_gather_kernel_overload = {}
_bspline_gather_vec3_kernel_overload = {}
_bspline_gather_gradient_kernel_overload = {}

# Batch kernel overloads
_batch_bspline_spread_kernel_overload = {}
_batch_bspline_gather_kernel_overload = {}
_batch_bspline_gather_vec3_kernel_overload = {}
_batch_bspline_gather_gradient_kernel_overload = {}

# Multi-channel kernel overloads
_bspline_spread_channels_kernel_overload = {}
_bspline_gather_channels_kernel_overload = {}
_batch_bspline_spread_channels_kernel_overload = {}
_batch_bspline_gather_channels_kernel_overload = {}

for t, v, m in zip(_T, _V, _M):
    # Single-system kernels
    _bspline_spread_kernel_overload[t] = wp.overload(
        _bspline_spread_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # values
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array3d(dtype=t),  # mesh
        ],
    )
    _bspline_gather_kernel_overload[t] = wp.overload(
        _bspline_gather_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array3d(dtype=t),  # mesh
            wp.array(dtype=t),  # output
        ],
    )
    _bspline_gather_vec3_kernel_overload[t] = wp.overload(
        _bspline_gather_vec3_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array3d(dtype=v),  # mesh
            wp.array(dtype=v),  # output
        ],
    )
    _bspline_gather_gradient_kernel_overload[t] = wp.overload(
        _bspline_gather_gradient_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array3d(dtype=t),  # mesh
            wp.array(dtype=v),  # forces
        ],
    )

    # Batch kernels
    _batch_bspline_spread_kernel_overload[t] = wp.overload(
        _batch_bspline_spread_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # values
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array(dtype=t, ndim=4),  # mesh
        ],
    )
    _batch_bspline_gather_kernel_overload[t] = wp.overload(
        _batch_bspline_gather_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array(dtype=t, ndim=4),  # mesh
            wp.array(dtype=t),  # output
        ],
    )
    _batch_bspline_gather_vec3_kernel_overload[t] = wp.overload(
        _batch_bspline_gather_vec3_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array(dtype=v, ndim=4),  # mesh
            wp.array(dtype=v),  # output
        ],
    )
    _batch_bspline_gather_gradient_kernel_overload[t] = wp.overload(
        _batch_bspline_gather_gradient_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array(dtype=t, ndim=4),  # mesh
            wp.array(dtype=v),  # forces
        ],
    )

    # Multi-channel kernels
    _bspline_spread_channels_kernel_overload[t] = wp.overload(
        _bspline_spread_channels_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array2d(dtype=t),  # values
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array(dtype=t, ndim=4),  # mesh
        ],
    )
    _bspline_gather_channels_kernel_overload[t] = wp.overload(
        _bspline_gather_channels_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array(dtype=t, ndim=4),  # mesh
            wp.array2d(dtype=t),  # output
        ],
    )
    _batch_bspline_spread_channels_kernel_overload[t] = wp.overload(
        _batch_bspline_spread_channels_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array2d(dtype=t),  # values
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.int32,  # num_channels
            wp.array4d(dtype=t),  # mesh
        ],
    )
    _batch_bspline_gather_channels_kernel_overload[t] = wp.overload(
        _batch_bspline_gather_channels_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.int32,  # num_channels
            wp.array4d(dtype=t),  # mesh
            wp.array2d(dtype=t),  # output
        ],
    )


###########################################################################################
########################### Internal Custom Ops: _spline_* (Single-System) #################
###########################################################################################


@warp_custom_op(
    name="alchemiops::_spline_spread",
    outputs=[
        OutputSpec(
            "mesh",
            wp.array(dtype=Any, ndim=3),
            lambda pos, values, cell, mesh_nx, mesh_ny, mesh_nz, spline_order, *_: (
                mesh_nx,
                mesh_ny,
                mesh_nz,
            ),
        ),
    ],
    grad_arrays=[
        "mesh",
        "positions",
        "values",
        "cell_inv_t",
    ],
)
def _spline_spread(
    positions: torch.Tensor,
    values: torch.Tensor,
    cell: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal: Single-system spline spread with dtype flexibility."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    num_points = spline_order**3
    needs_grad_flag = needs_grad(positions, values, cell)

    if cell.dim() == 2:
        cell = cell.unsqueeze(0)

    if cell_inv_t is None:
        cell_inv = torch.linalg.inv_ex(cell)[0]
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()

    wp_positions = warp_from_torch(
        positions, wp_vec_dtype, requires_grad=needs_grad_flag
    )
    wp_values = warp_from_torch(
        values.to(input_dtype), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_cell_inv_t = warp_from_torch(
        cell_inv_t, wp_mat_dtype, requires_grad=needs_grad_flag
    )

    mesh = torch.zeros(
        (mesh_nx, mesh_ny, mesh_nz), device=positions.device, dtype=input_dtype
    )
    wp_mesh = warp_from_torch(mesh, wp_dtype, requires_grad=needs_grad_flag)

    kernel = _bspline_spread_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_atoms, num_points),
            inputs=[wp_positions, wp_values, wp_cell_inv_t, wp.int32(spline_order)],
            outputs=[wp_mesh],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            mesh,
            tape=tape,
            mesh=wp_mesh,
            positions=wp_positions,
            values=wp_values,
            cell_inv_t=wp_cell_inv_t,
        )
    return mesh


@warp_custom_op(
    name="alchemiops::_spline_gather",
    outputs=[
        OutputSpec(
            "values",
            wp.array(dtype=Any),
            lambda pos, *_: (pos.shape[0],),
        ),
    ],
    grad_arrays=[
        "values",
        "positions",
        "mesh",
        "cell_inv_t",
    ],
)
def _spline_gather(
    positions: torch.Tensor,
    mesh: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal: Single-system spline gather with dtype flexibility."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    num_points = spline_order**3
    needs_grad_flag = needs_grad(positions, mesh, cell)

    if cell.dim() == 2:
        cell = cell.unsqueeze(0)

    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()

    wp_positions = warp_from_torch(
        positions, wp_vec_dtype, requires_grad=needs_grad_flag
    )
    wp_cell_inv_t = warp_from_torch(
        cell_inv_t, wp_mat_dtype, requires_grad=needs_grad_flag
    )
    wp_mesh = warp_from_torch(
        mesh.to(input_dtype), wp_dtype, requires_grad=needs_grad_flag
    )

    values = torch.zeros(num_atoms, device=positions.device, dtype=input_dtype)
    wp_values = warp_from_torch(values, wp_dtype, requires_grad=needs_grad_flag)

    kernel = _bspline_gather_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_atoms, num_points),
            inputs=[wp_positions, wp_cell_inv_t, wp.int32(spline_order), wp_mesh],
            outputs=[wp_values],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            values,
            tape=tape,
            values=wp_values,
            positions=wp_positions,
            cell_inv_t=wp_cell_inv_t,
            mesh=wp_mesh,
        )
    return values


@warp_custom_op(
    name="alchemiops::_spline_gather_vec3",
    outputs=[
        OutputSpec(
            "values", wp.array(dtype=Any, ndim=2), lambda pos, *_: (pos.shape[0], 3)
        ),
    ],
    grad_arrays=[
        "values",
        "positions",
        "charges",
        "mesh",
        "cell_inv_t",
    ],
)
def _spline_gather_vec3(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal: Single-system vec3 spline gather with dtype flexibility."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    num_points = spline_order**3
    needs_grad_flag = needs_grad(positions, mesh, cell)

    if cell.dim() == 2:
        cell = cell.unsqueeze(0)

    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()

    wp_positions = warp_from_torch(
        positions, wp_vec_dtype, requires_grad=needs_grad_flag
    )
    wp_charges = warp_from_torch(
        charges.to(input_dtype), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_cell_inv_t = warp_from_torch(
        cell_inv_t, wp_mat_dtype, requires_grad=needs_grad_flag
    )
    wp_mesh = warp_from_torch(
        mesh.to(input_dtype), wp_vec_dtype, requires_grad=needs_grad_flag
    )

    values = torch.zeros((num_atoms, 3), device=positions.device, dtype=input_dtype)
    wp_values = warp_from_torch(values, wp_vec_dtype, requires_grad=needs_grad_flag)

    kernel = _bspline_gather_vec3_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_atoms, num_points),
            inputs=[
                wp_positions,
                wp_charges,
                wp_cell_inv_t,
                wp.int32(spline_order),
                wp_mesh,
            ],
            outputs=[wp_values],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            values,
            tape=tape,
            values=wp_values,
            positions=wp_positions,
            charges=wp_charges,
            cell_inv_t=wp_cell_inv_t,
            mesh=wp_mesh,
        )
    return values


@warp_custom_op(
    name="alchemiops::_spline_gather_gradient",
    outputs=[
        OutputSpec(
            "forces", wp.array(dtype=Any, ndim=2), lambda pos, *_: (pos.shape[0], 3)
        ),
    ],
    grad_arrays=[
        "forces",
        "positions",
        "charges",
        "mesh",
        "cell_inv_t",
    ],
)
def _spline_gather_gradient(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal: Single-system spline gather gradient with dtype flexibility."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    num_points = spline_order**3
    needs_grad_flag = needs_grad(positions, charges, mesh, cell)

    if cell.dim() == 2:
        cell = cell.unsqueeze(0)

    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()

    wp_positions = warp_from_torch(
        positions, wp_vec_dtype, requires_grad=needs_grad_flag
    )
    wp_charges = warp_from_torch(
        charges.to(input_dtype), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_cell_inv_t = warp_from_torch(
        cell_inv_t, wp_mat_dtype, requires_grad=needs_grad_flag
    )
    wp_mesh = warp_from_torch(
        mesh.to(input_dtype), wp_dtype, requires_grad=needs_grad_flag
    )

    forces = torch.zeros((num_atoms, 3), device=positions.device, dtype=input_dtype)
    wp_forces = warp_from_torch(forces, wp_vec_dtype, requires_grad=needs_grad_flag)

    kernel = _bspline_gather_gradient_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_atoms, num_points),
            inputs=[
                wp_positions,
                wp_charges,
                wp_cell_inv_t,
                wp.int32(spline_order),
                wp_mesh,
            ],
            outputs=[wp_forces],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            forces,
            tape=tape,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell_inv_t=wp_cell_inv_t,
            mesh=wp_mesh,
        )
    return forces


###########################################################################################
########################### Internal Custom Ops: _batch_spline_* (Batch) ###################
###########################################################################################


@warp_custom_op(
    name="alchemiops::_batch_spline_spread",
    outputs=[
        OutputSpec(
            "mesh",
            wp.array(dtype=Any, ndim=4),
            lambda pos,
            values,
            batch_idx,
            cell,
            num_systems,
            mesh_nx,
            mesh_ny,
            mesh_nz,
            spline_order,
            *_: (num_systems, mesh_nx, mesh_ny, mesh_nz),
        ),
    ],
    grad_arrays=[
        "mesh",
        "positions",
        "values",
        "cell_inv_t",
    ],
)
def _batch_spline_spread(
    positions: torch.Tensor,
    values: torch.Tensor,
    batch_idx: torch.Tensor,
    cell: torch.Tensor,
    num_systems: int,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal: Batch spline spread with dtype flexibility."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    num_points = spline_order**3
    needs_grad_flag = needs_grad(positions, values, cell)

    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()

    wp_positions = warp_from_torch(
        positions, wp_vec_dtype, requires_grad=needs_grad_flag
    )
    wp_values = warp_from_torch(
        values.to(input_dtype), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_cell_inv_t = warp_from_torch(
        cell_inv_t, wp_mat_dtype, requires_grad=needs_grad_flag
    )

    mesh = torch.zeros(
        (num_systems, mesh_nx, mesh_ny, mesh_nz),
        device=positions.device,
        dtype=input_dtype,
    )
    wp_mesh = warp_from_torch(mesh, wp_dtype, requires_grad=needs_grad_flag)

    kernel = _batch_bspline_spread_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_atoms, num_points),
            inputs=[
                wp_positions,
                wp_values,
                wp_batch_idx,
                wp_cell_inv_t,
                wp.int32(spline_order),
            ],
            outputs=[wp_mesh],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            mesh,
            tape=tape,
            mesh=wp_mesh,
            positions=wp_positions,
            values=wp_values,
            cell_inv_t=wp_cell_inv_t,
        )
    return mesh


@warp_custom_op(
    name="alchemiops::_batch_spline_gather",
    outputs=[
        OutputSpec(
            "values",
            wp.array(dtype=Any),
            lambda pos, *_: (pos.shape[0],),
        ),
    ],
    grad_arrays=[
        "values",
        "positions",
        "mesh",
        "cell_inv_t",
    ],
)
def _batch_spline_gather(
    positions: torch.Tensor,
    mesh: torch.Tensor,
    batch_idx: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal: Batch spline gather with dtype flexibility."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    num_points = spline_order**3
    needs_grad_flag = needs_grad(positions, mesh, cell)

    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()

    wp_positions = warp_from_torch(
        positions, wp_vec_dtype, requires_grad=needs_grad_flag
    )
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_cell_inv_t = warp_from_torch(
        cell_inv_t, wp_mat_dtype, requires_grad=needs_grad_flag
    )
    wp_mesh = warp_from_torch(
        mesh.to(input_dtype), wp_dtype, requires_grad=needs_grad_flag
    )

    values = torch.zeros(num_atoms, device=positions.device, dtype=input_dtype)
    wp_values = warp_from_torch(values, wp_dtype, requires_grad=needs_grad_flag)

    kernel = _batch_bspline_gather_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_atoms, num_points),
            inputs=[
                wp_positions,
                wp_batch_idx,
                wp_cell_inv_t,
                wp.int32(spline_order),
                wp_mesh,
            ],
            outputs=[wp_values],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            values,
            tape=tape,
            values=wp_values,
            positions=wp_positions,
            cell_inv_t=wp_cell_inv_t,
            mesh=wp_mesh,
        )
    return values


@warp_custom_op(
    name="alchemiops::_batch_spline_gather_vec3",
    outputs=[
        OutputSpec(
            "values", wp.array(dtype=Any, ndim=2), lambda pos, *_: (pos.shape[0], 3)
        ),
    ],
    grad_arrays=[
        "values",
        "positions",
        "charges",
        "mesh",
        "cell_inv_t",
    ],
)
def _batch_spline_gather_vec3(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    batch_idx: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal: Batch vec3 spline gather with dtype flexibility."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    num_points = spline_order**3
    needs_grad_flag = needs_grad(positions, mesh, cell)

    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()

    wp_positions = warp_from_torch(
        positions, wp_vec_dtype, requires_grad=needs_grad_flag
    )
    wp_charges = warp_from_torch(
        charges.to(input_dtype), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_cell_inv_t = warp_from_torch(
        cell_inv_t, wp_mat_dtype, requires_grad=needs_grad_flag
    )
    wp_mesh = warp_from_torch(
        mesh.to(input_dtype), wp_vec_dtype, requires_grad=needs_grad_flag
    )

    values = torch.zeros((num_atoms, 3), device=positions.device, dtype=input_dtype)
    wp_values = warp_from_torch(values, wp_vec_dtype, requires_grad=needs_grad_flag)

    kernel = _batch_bspline_gather_vec3_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_atoms, num_points),
            inputs=[
                wp_positions,
                wp_charges,
                wp_batch_idx,
                wp_cell_inv_t,
                wp.int32(spline_order),
                wp_mesh,
            ],
            outputs=[wp_values],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            values,
            tape=tape,
            values=wp_values,
            positions=wp_positions,
            charges=wp_charges,
            cell_inv_t=wp_cell_inv_t,
            mesh=wp_mesh,
        )
    return values


@warp_custom_op(
    name="alchemiops::_batch_spline_gather_gradient",
    outputs=[
        OutputSpec(
            "forces", wp.array(dtype=Any, ndim=2), lambda pos, *_: (pos.shape[0], 3)
        ),
    ],
    grad_arrays=[
        "forces",
        "positions",
        "charges",
        "mesh",
        "cell_inv_t",
    ],
)
def _batch_spline_gather_gradient(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    batch_idx: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal: Batch spline gather gradient with dtype flexibility."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    num_points = spline_order**3
    needs_grad_flag = needs_grad(positions, charges, mesh, cell)

    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()

    wp_positions = warp_from_torch(
        positions, wp_vec_dtype, requires_grad=needs_grad_flag
    )
    wp_charges = warp_from_torch(
        charges.to(input_dtype), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_cell_inv_t = warp_from_torch(
        cell_inv_t, wp_mat_dtype, requires_grad=needs_grad_flag
    )
    wp_mesh = warp_from_torch(
        mesh.to(input_dtype), wp_dtype, requires_grad=needs_grad_flag
    )

    forces = torch.zeros((num_atoms, 3), device=positions.device, dtype=input_dtype)
    wp_forces = warp_from_torch(forces, wp_vec_dtype, requires_grad=needs_grad_flag)

    kernel = _batch_bspline_gather_gradient_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_atoms, num_points),
            inputs=[
                wp_positions,
                wp_charges,
                wp_batch_idx,
                wp_cell_inv_t,
                wp.int32(spline_order),
                wp_mesh,
            ],
            outputs=[wp_forces],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            forces,
            tape=tape,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell_inv_t=wp_cell_inv_t,
            mesh=wp_mesh,
        )
    return forces


###########################################################################################
########################### Internal Custom Ops: Multi-Channel (Single-System) #############
###########################################################################################


@warp_custom_op(
    name="alchemiops::_spline_spread_channels",
    outputs=[
        OutputSpec(
            "mesh",
            wp.array(dtype=Any, ndim=4),
            lambda pos,
            values,
            cell,
            num_channels,
            mesh_nx,
            mesh_ny,
            mesh_nz,
            spline_order,
            *_: (num_channels, mesh_nx, mesh_ny, mesh_nz),
        ),
    ],
    grad_arrays=[
        "mesh",
        "positions",
        "values",
        "cell_inv_t",
    ],
)
def _spline_spread_channels(
    positions: torch.Tensor,
    values: torch.Tensor,
    cell: torch.Tensor,
    num_channels: int,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
) -> torch.Tensor:
    """Internal: Single-system multi-channel spline spread with dtype flexibility."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    num_points = spline_order**3
    needs_grad_flag = needs_grad(positions, values, cell)

    if cell.dim() == 2:
        cell = cell.unsqueeze(0)

    cell_inv = torch.linalg.inv_ex(cell)[0]
    cell_inv_t = cell_inv.transpose(-1, -2).contiguous()

    wp_positions = warp_from_torch(
        positions, wp_vec_dtype, requires_grad=needs_grad_flag
    )
    wp_values = warp_from_torch(
        values.to(input_dtype), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_cell_inv_t = warp_from_torch(
        cell_inv_t, wp_mat_dtype, requires_grad=needs_grad_flag
    )

    mesh = torch.zeros(
        (num_channels, mesh_nx, mesh_ny, mesh_nz),
        device=positions.device,
        dtype=input_dtype,
    )
    wp_mesh = warp_from_torch(mesh, wp_dtype, requires_grad=needs_grad_flag)

    kernel = _bspline_spread_channels_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_atoms, num_points),
            inputs=[wp_positions, wp_values, wp_cell_inv_t, wp.int32(spline_order)],
            outputs=[wp_mesh],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            mesh,
            tape=tape,
            mesh=wp_mesh,
            positions=wp_positions,
            values=wp_values,
            cell_inv_t=wp_cell_inv_t,
        )
    return mesh


@warp_custom_op(
    name="alchemiops::_spline_gather_channels",
    outputs=[
        OutputSpec(
            "values",
            wp.array(dtype=Any, ndim=2),
            lambda pos, mesh, *_: (pos.shape[0], mesh.shape[1]),
        ),
    ],
    grad_arrays=[
        "values",
        "positions",
        "mesh",
        "cell_inv_t",
    ],
)
def _spline_gather_channels(
    positions: torch.Tensor,
    mesh: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Internal: Single-system multi-channel spline gather with dtype flexibility."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    num_channels = mesh.shape[0]
    num_points = spline_order**3
    needs_grad_flag = needs_grad(positions, mesh, cell)

    if cell.dim() == 2:
        cell = cell.unsqueeze(0)

    cell_inv = torch.linalg.inv(cell)
    cell_inv_t = cell_inv.transpose(-1, -2).contiguous()

    wp_positions = warp_from_torch(
        positions, wp_vec_dtype, requires_grad=needs_grad_flag
    )
    wp_cell_inv_t = warp_from_torch(
        cell_inv_t, wp_mat_dtype, requires_grad=needs_grad_flag
    )
    wp_mesh = warp_from_torch(
        mesh.to(input_dtype), wp_dtype, requires_grad=needs_grad_flag
    )

    values = torch.zeros(
        (num_atoms, num_channels), device=positions.device, dtype=input_dtype
    )
    wp_values = warp_from_torch(values, wp_dtype, requires_grad=needs_grad_flag)

    kernel = _bspline_gather_channels_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_atoms, num_points),
            inputs=[wp_positions, wp_cell_inv_t, wp.int32(spline_order), wp_mesh],
            outputs=[wp_values],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            values,
            tape=tape,
            values=wp_values,
            positions=wp_positions,
            cell_inv_t=wp_cell_inv_t,
            mesh=wp_mesh,
        )
    return values


###########################################################################################
########################### Internal Custom Ops: Multi-Channel (Batch) #####################
###########################################################################################


def _batch_spline_spread_channels_output_shape(
    position,
    values,
    batch_idx,
    cell,
    num_systems,
    num_channels,
    mesh_nx,
    mesh_ny,
    mesh_nz,
    spline_order,
):
    return (num_systems, num_channels, mesh_nx, mesh_ny, mesh_nz)


@warp_custom_op(
    name="alchemiops::_batch_spline_spread_channels",
    outputs=[
        OutputSpec(
            "mesh",
            wp.array(dtype=Any, ndim=4),
            _batch_spline_spread_channels_output_shape,
        ),
    ],
    grad_arrays=[
        "mesh",
        "positions",
        "values",
        "cell_inv_t",
    ],
)
def _batch_spline_spread_channels(
    positions: torch.Tensor,
    values: torch.Tensor,
    batch_idx: torch.Tensor,
    cell: torch.Tensor,
    num_systems: int,
    num_channels: int,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
) -> torch.Tensor:
    """Internal: Batch multi-channel spline spread with dtype flexibility."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    num_points = spline_order**3
    needs_grad_flag = needs_grad(positions, values, cell)

    cell_inv = torch.linalg.inv(cell)
    cell_inv_t = cell_inv.transpose(-1, -2).contiguous()

    wp_positions = warp_from_torch(
        positions, wp_vec_dtype, requires_grad=needs_grad_flag
    )
    wp_values = warp_from_torch(
        values.to(input_dtype), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_cell_inv_t = warp_from_torch(
        cell_inv_t, wp_mat_dtype, requires_grad=needs_grad_flag
    )

    # Create mesh with flattened (B*C, nx, ny, nz) format for Warp 4D limit
    mesh_flat = torch.zeros(
        (num_systems * num_channels, mesh_nx, mesh_ny, mesh_nz),
        device=positions.device,
        dtype=input_dtype,
    )
    wp_mesh = warp_from_torch(mesh_flat, wp_dtype, requires_grad=needs_grad_flag)

    kernel = _batch_bspline_spread_channels_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_atoms, num_points),
            inputs=[
                wp_positions,
                wp_values,
                wp_batch_idx,
                wp_cell_inv_t,
                wp.int32(spline_order),
                wp.int32(num_channels),
            ],
            outputs=[wp_mesh],
            device=device,
        )

    # Reshape back to (B, C, nx, ny, nz) for output
    mesh = mesh_flat.view(num_systems, num_channels, mesh_nx, mesh_ny, mesh_nz)

    if needs_grad_flag:
        attach_for_backward(
            mesh,
            tape=tape,
            mesh=wp_mesh,
            positions=wp_positions,
            values=wp_values,
            cell_inv_t=wp_cell_inv_t,
        )
    return mesh


@warp_custom_op(
    name="alchemiops::_batch_spline_gather_channels",
    outputs=[
        OutputSpec(
            "values",
            wp.array(dtype=Any, ndim=2),
            lambda pos, mesh, *_: (pos.shape[0], mesh.shape[1]),
        ),
    ],
    grad_arrays=[
        "values",
        "positions",
        "mesh",
        "cell_inv_t",
    ],
)
def _batch_spline_gather_channels(
    positions: torch.Tensor,
    mesh: torch.Tensor,
    batch_idx: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Internal: Batch multi-channel spline gather with dtype flexibility."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    num_systems = mesh.shape[0]  # (B, C, nx, ny, nz)
    num_channels = mesh.shape[1]
    mesh_nx, mesh_ny, mesh_nz = mesh.shape[2], mesh.shape[3], mesh.shape[4]
    num_points = spline_order**3
    needs_grad_flag = needs_grad(positions, mesh, cell)

    cell_inv = torch.linalg.inv(cell)
    cell_inv_t = cell_inv.transpose(-1, -2).contiguous()

    wp_positions = warp_from_torch(
        positions, wp_vec_dtype, requires_grad=needs_grad_flag
    )
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_cell_inv_t = warp_from_torch(
        cell_inv_t, wp_mat_dtype, requires_grad=needs_grad_flag
    )

    # Flatten mesh from (B, C, nx, ny, nz) to (B*C, nx, ny, nz) for Warp 4D limit
    mesh_flat = (
        mesh.to(input_dtype)
        .view(num_systems * num_channels, mesh_nx, mesh_ny, mesh_nz)
        .contiguous()
    )
    wp_mesh = warp_from_torch(mesh_flat, wp_dtype, requires_grad=needs_grad_flag)

    values = torch.zeros(
        (num_atoms, num_channels), device=positions.device, dtype=input_dtype
    )
    wp_values = warp_from_torch(values, wp_dtype, requires_grad=needs_grad_flag)

    kernel = _batch_bspline_gather_channels_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_atoms, num_points),
            inputs=[
                wp_positions,
                wp_batch_idx,
                wp_cell_inv_t,
                wp.int32(spline_order),
                wp.int32(num_channels),
                wp_mesh,
            ],
            outputs=[wp_values],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            values,
            tape=tape,
            values=wp_values,
            positions=wp_positions,
            cell_inv_t=wp_cell_inv_t,
            mesh=wp_mesh,
        )
    return values


###########################################################################################
########################### Unified Public API #############################################
###########################################################################################


def spline_spread(
    positions: torch.Tensor,
    values: torch.Tensor,
    cell: torch.Tensor,
    mesh_dims: tuple[int, int, int],
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Spread values from atoms to mesh grid using B-spline interpolation.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic positions.
    values : torch.Tensor, shape (N,)
        Values to spread (e.g., charges).
    cell : torch.Tensor, shape (3, 3), (1, 3, 3), or (B, 3, 3)
        Unit cell matrix. For batched, shape should be (B, 3, 3).
    mesh_dims : tuple[int, int, int]
        Mesh dimensions (nx, ny, nz).
    spline_order : int, default=4
        B-spline order (1-4, where 4=cubic).
    batch_idx : torch.Tensor | None, shape (N,), dtype=int32, default=None
        System index for each atom. If None, uses single-system kernel.
    cell_inv_t : torch.Tensor | None, default=None
        Precomputed transpose of cell inverse. If provided, skips inverse computation.
        Shape (1, 3, 3) for single-system or (B, 3, 3) for batch.

    Returns
    -------
    mesh : torch.Tensor
        For single-system: shape (nx, ny, nz)
        For batch: shape (B, nx, ny, nz)
    """
    mesh_nx, mesh_ny, mesh_nz = mesh_dims

    if batch_idx is None:
        return _spline_spread(
            positions, values, cell, mesh_nx, mesh_ny, mesh_nz, spline_order, cell_inv_t
        )
    else:
        num_systems = cell.shape[0]
        if cell.dim() == 2:
            cell = cell.unsqueeze(0).expand(num_systems, -1, -1).contiguous()
        return _batch_spline_spread(
            positions,
            values,
            batch_idx,
            cell,
            num_systems,
            mesh_nx,
            mesh_ny,
            mesh_nz,
            spline_order,
            cell_inv_t,
        )


def spline_gather(
    positions: torch.Tensor,
    mesh: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather values from mesh to atoms using B-spline interpolation.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic positions.
    mesh : torch.Tensor
        For single-system: shape (nx, ny, nz)
        For batch: shape (B, nx, ny, nz)
    cell : torch.Tensor, shape (3, 3), (1, 3, 3), or (B, 3, 3)
        Unit cell matrix.
    spline_order : int, default=4
        B-spline order.
    batch_idx : torch.Tensor | None, shape (N,), dtype=int32, default=None
        System index for each atom. If None, uses single-system kernel.
    cell_inv_t : torch.Tensor | None, default=None
        Precomputed transpose of cell inverse. If provided, skips inverse computation.
        Shape (1, 3, 3) for single-system or (B, 3, 3) for batch.

    Returns
    -------
    values : torch.Tensor, shape (N,)
        Interpolated values at atomic positions.
    """
    if batch_idx is None:
        return _spline_gather(positions, mesh, cell, spline_order, cell_inv_t)
    else:
        # Ensure cell is 3D for batch operations
        if cell.dim() == 2:
            num_systems = int(batch_idx.max().item()) + 1
            cell = cell.unsqueeze(0).expand(num_systems, -1, -1).contiguous()
        return _batch_spline_gather(
            positions, mesh, batch_idx, cell, spline_order, cell_inv_t
        )


def spline_gather_vec3(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather 3D vector values from mesh to atoms using B-spline interpolation.

    This is useful for interpolating vector fields like electric fields.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic positions.
    mesh : torch.Tensor
        For single-system: shape (nx, ny, nz, 3)
        For batch: shape (B, nx, ny, nz, 3)
    cell : torch.Tensor, shape (3, 3), (1, 3, 3), or (B, 3, 3)
        Unit cell matrix.
    spline_order : int, default=4
        B-spline order.
    batch_idx : torch.Tensor | None, shape (N,), dtype=int32, default=None
        System index for each atom. If None, uses single-system kernel.
    cell_inv_t : torch.Tensor | None, default=None
        Precomputed transpose of cell inverse. If provided, skips inverse computation.
        Shape (1, 3, 3) for single-system or (B, 3, 3) for batch.

    Returns
    -------
    vectors : torch.Tensor, shape (N, 3)
        Interpolated 3D vectors at atomic positions.
    """
    if batch_idx is None:
        return _spline_gather_vec3(
            positions, charges, mesh, cell, spline_order, cell_inv_t
        )
    else:
        # Ensure cell is 3D for batch operations
        if cell.dim() == 2:
            num_systems = int(batch_idx.max().item()) + 1
            cell = cell.unsqueeze(0).expand(num_systems, -1, -1).contiguous()
        return _batch_spline_gather_vec3(
            positions, charges, mesh, batch_idx, cell, spline_order, cell_inv_t
        )


def spline_gather_gradient(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather gradient from mesh to atoms using B-spline derivatives.

    Computes forces:

    .. math::

        F_i = -q_i \\sum_g \\phi(g) \\nabla w(r_i, g)

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic positions.
    charges : torch.Tensor, shape (N,)
        Atomic charges.
    mesh : torch.Tensor
        For single-system: shape (nx, ny, nz)
        For batch: shape (B, nx, ny, nz)
    cell : torch.Tensor, shape (3, 3), (1, 3, 3), or (B, 3, 3)
        Unit cell matrix.
    spline_order : int, default=4
        B-spline order.
    batch_idx : torch.Tensor | None, shape (N,), dtype=int32, default=None
        System index for each atom. If None, uses single-system kernel.
    cell_inv_t : torch.Tensor | None, default=None
        Precomputed transpose of cell inverse. If provided, skips inverse computation.
        Shape (1, 3, 3) for single-system or (B, 3, 3) for batch.

    Returns
    -------
    forces : torch.Tensor, shape (N, 3)
        Forces on atoms.
    """
    if batch_idx is None:
        return _spline_gather_gradient(
            positions, charges, mesh, cell, spline_order, cell_inv_t
        )
    else:
        # Ensure cell is 3D for batch operations
        if cell.dim() == 2:
            num_systems = int(batch_idx.max().item()) + 1
            cell = cell.unsqueeze(0).expand(num_systems, -1, -1).contiguous()
        return _batch_spline_gather_gradient(
            positions, charges, mesh, batch_idx, cell, spline_order, cell_inv_t
        )


def spline_spread_channels(
    positions: torch.Tensor,
    values: torch.Tensor,
    cell: torch.Tensor,
    mesh_dims: tuple[int, int, int],
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Spread multi-channel values from atoms to mesh grid using B-spline interpolation.

    This is useful for spreading multipole coefficients (e.g., 9 channels for L_max=2:
    1 monopole + 3 dipoles + 5 quadrupoles).

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic positions.
    values : torch.Tensor, shape (N, C)
        Multi-channel values to spread. C is the number of channels.
    cell : torch.Tensor, shape (3, 3), (1, 3, 3), or (B, 3, 3)
        Unit cell matrix. For batched, shape should be (B, 3, 3).
    mesh_dims : tuple[int, int, int]
        Mesh dimensions (nx, ny, nz).
    spline_order : int, default=4
        B-spline order (1-4, where 4=cubic).
    batch_idx : torch.Tensor | None, shape (N,), dtype=int32, default=None
        System index for each atom. If None, uses single-system kernel.

    Returns
    -------
    mesh : torch.Tensor
        For single-system: shape (C, nx, ny, nz)
        For batch: shape (B, C, nx, ny, nz)

    Example
    -------
    >>> # Spread 9-channel multipole coefficients
    >>> multipoles = torch.randn(100, 9, dtype=torch.float64, device="cuda")
    >>> mesh = spline_spread_channels(positions, multipoles, cell, (16, 16, 16))
    >>> print(mesh.shape)  # (9, 16, 16, 16)
    """
    mesh_nx, mesh_ny, mesh_nz = mesh_dims
    num_channels = values.shape[1]

    if batch_idx is None:
        return _spline_spread_channels(
            positions,
            values,
            cell,
            num_channels,
            mesh_nx,
            mesh_ny,
            mesh_nz,
            spline_order,
        )
    else:
        if cell.dim() == 2:
            num_systems = int(batch_idx.max().item()) + 1
            cell = cell.unsqueeze(0).expand(num_systems, -1, -1).contiguous()
        else:
            num_systems = cell.shape[0]
        return _batch_spline_spread_channels(
            positions,
            values,
            batch_idx,
            cell,
            num_systems,
            num_channels,
            mesh_nx,
            mesh_ny,
            mesh_nz,
            spline_order,
        )


def spline_gather_channels(
    positions: torch.Tensor,
    mesh: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather multi-channel values from mesh to atoms using B-spline interpolation.

    This is the inverse of spline_spread_channels.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic positions.
    mesh : torch.Tensor
        For single-system: shape (C, nx, ny, nz)
        For batch: shape (B, C, nx, ny, nz)
    cell : torch.Tensor, shape (3, 3), (1, 3, 3), or (B, 3, 3)
        Unit cell matrix.
    spline_order : int, default=4
        B-spline order.
    batch_idx : torch.Tensor | None, shape (N,), dtype=int32, default=None
        System index for each atom. If None, uses single-system kernel.

    Returns
    -------
    values : torch.Tensor, shape (N, C)
        Interpolated multi-channel values at atomic positions.

    Example
    -------
    >>> # Gather 9-channel potential from mesh
    >>> potential_mesh = torch.randn(9, 16, 16, 16, dtype=torch.float64, device="cuda")
    >>> potentials = spline_gather_channels(positions, potential_mesh, cell)
    >>> print(potentials.shape)  # (100, 9)
    """
    if batch_idx is None:
        return _spline_gather_channels(positions, mesh, cell, spline_order)
    else:
        # Ensure cell is 3D for batch operations
        if cell.dim() == 2:
            num_systems = int(batch_idx.max().item()) + 1
            cell = cell.unsqueeze(0).expand(num_systems, -1, -1).contiguous()
        return _batch_spline_gather_channels(
            positions, mesh, batch_idx, cell, spline_order
        )


###########################################################################################
########################### Deconvolution Functions #######################################
###########################################################################################


def _bspline_modulus(k: torch.Tensor, n: int, order: int) -> torch.Tensor:
    """Compute the modulus of B-spline Fourier transform.

    The B-spline function :math:`M_n(u)` has Fourier transform:

    .. math::

        \\hat{M}_n(k) = \\left[\\frac{\\sin(\\pi k/n)}{\\pi k/n}\\right]^n

    For PME, we need the modulus of this for the cardinal B-spline interpolation.

    Parameters
    ----------
    k : torch.Tensor
        Frequency indices (integers).
    n : int
        Grid dimension.
    order : int
        B-spline order.

    Returns
    -------
    torch.Tensor
        :math:`|b(k)|^2` where :math:`b(k)` is the B-spline Fourier coefficient.
    """
    # Compute the exponential B-spline factors
    # Following Essmann et al. (1995) Eq. 4.7
    pi = torch.tensor(math.pi, dtype=torch.float64, device=k.device)

    # For order n B-splines, the Fourier transform involves
    # the exponential factors exp(2*pi*i m k / n) for m = 0, ..., order-1
    # summed and then raised to order power

    # Handle k=0 case specially (limit is 1)
    result = torch.ones_like(k, dtype=torch.float64)

    # For non-zero k, compute the product
    nonzero_mask = k != 0

    # w = 2*pi * k / n
    w = 2.0 * pi * k.float() / n

    # The B-spline Fourier coefficient is:
    # b(k) = sum_{j=0}^{order-1} M_order(j+1) * exp(2*pi*i j k / n)
    # where M_order is the B-spline basis function

    # Compute M_order values at integer points 1, 2, ..., order
    m_values = _compute_bspline_coefficients(order, k.device)

    # Sum: b(k) = Σ_j M_order(j+1) * exp(i w j)
    b_real = torch.zeros_like(k, dtype=torch.float64)
    b_imag = torch.zeros_like(k, dtype=torch.float64)

    for j in range(order):
        phase = w * j
        b_real = b_real + m_values[j] * torch.cos(phase)
        b_imag = b_imag + m_values[j] * torch.sin(phase)

    # |b(k)|^2
    b_sq = b_real**2 + b_imag**2

    # Handle k=0 case
    result = torch.where(nonzero_mask, b_sq, result)

    return result


def _compute_bspline_coefficients(order: int, device) -> torch.Tensor:
    """Compute B-spline basis function values at integer points.

    For a B-spline of order n, we need M_n(1), M_n(2), ..., M_n(n).
    These are used in the Fourier transform computation.

    Parameters
    ----------
    order : int
        B-spline order.
    device
        PyTorch device.

    Returns
    -------
    torch.Tensor
        B-spline values [M_n(1), M_n(2), ..., M_n(n)].
    """
    if order == 1:
        return torch.tensor([1.0], dtype=torch.float64, device=device)
    elif order == 2:
        return torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    elif order == 3:
        return torch.tensor([1 / 6, 4 / 6, 1 / 6], dtype=torch.float64, device=device)
    elif order == 4:
        return torch.tensor(
            [1 / 24, 11 / 24, 11 / 24, 1 / 24], dtype=torch.float64, device=device
        )
    elif order == 5:
        return torch.tensor(
            [1 / 120, 26 / 120, 66 / 120, 26 / 120, 1 / 120],
            dtype=torch.float64,
            device=device,
        )
    elif order == 6:
        return torch.tensor(
            [1 / 720, 57 / 720, 302 / 720, 302 / 720, 57 / 720, 1 / 720],
            dtype=torch.float64,
            device=device,
        )
    else:
        # Use recursive definition for higher orders
        # M_n(u) = u/(n-1) * M_{n-1}(u) + (n-u)/(n-1) * M_{n-1}(u-1)
        coeffs = _compute_bspline_coefficients(order - 1, device)
        new_coeffs = torch.zeros(order, dtype=torch.float64, device=device)
        for j in range(order):
            u = float(j + 1)
            if j < order - 1:
                new_coeffs[j] += u / (order - 1) * coeffs[j]
            if j > 0:
                new_coeffs[j] += (order - u) / (order - 1) * coeffs[j - 1]
        return new_coeffs


def compute_bspline_deconvolution(
    mesh_dims: tuple[int, int, int],
    spline_order: int = 4,
    device=None,
) -> torch.Tensor:
    """Compute B-spline deconvolution factors for Fourier space correction.

    In FFT-based methods (like PME), the B-spline interpolation introduces
    smoothing in the charge distribution. This function computes the
    deconvolution factors to correct for this smoothing in Fourier space.

    The correction is: mesh_corrected_k = mesh_k * deconv

    Parameters
    ----------
    mesh_dims : tuple[int, int, int]
        Mesh dimensions (nx, ny, nz).
    spline_order : int, default=4
        B-spline order.
    device : torch.device, optional
        Device for the output tensor. Default: CPU.

    Returns
    -------
    deconv : torch.Tensor, shape (nx, ny, nz)
        Deconvolution factors. Multiply with FFT of mesh to correct.

    Example
    -------
    >>> deconv = compute_bspline_deconvolution((16, 16, 16), spline_order=4)
    >>> mesh_fft = torch.fft.fftn(charge_mesh)
    >>> mesh_corrected_fft = mesh_fft * deconv
    >>> charge_mesh_corrected = torch.fft.ifftn(mesh_corrected_fft).real

    Notes
    -----
    The deconvolution factor for a given k-vector is:

    .. math::

        D(k_x, k_y, k_z) = \\frac{1}{|b(k_x)|^2 \\cdot |b(k_y)|^2 \\cdot |b(k_z)|^2}

    where :math:`b(k)` is the Fourier transform of the 1D B-spline.

    For efficiency, this uses the separable property of the 3D B-spline.
    """
    if device is None:
        device = torch.device("cpu")

    nx, ny, nz = mesh_dims

    # Create frequency indices for each dimension
    # For FFT, frequencies are arranged as [0, 1, ..., n//2, -(n//2-1), ..., -1]
    kx = torch.fft.fftfreq(nx, device=device) * nx  # Integer frequencies
    ky = torch.fft.fftfreq(ny, device=device) * ny
    kz = torch.fft.fftfreq(nz, device=device) * nz

    # Compute |b(k)|^2 for each dimension
    bx_sq = _bspline_modulus(kx, nx, spline_order)
    by_sq = _bspline_modulus(ky, ny, spline_order)
    bz_sq = _bspline_modulus(kz, nz, spline_order)

    # The 3D deconvolution is the product of 1D factors
    # deconv = 1 / (bx^2 * by^2 * bz^2)
    # Use outer product for efficiency
    bx_sq = bx_sq.view(nx, 1, 1)
    by_sq = by_sq.view(1, ny, 1)
    bz_sq = bz_sq.view(1, 1, nz)

    b_sq_3d = bx_sq * by_sq * bz_sq

    # Avoid division by zero (should not happen for reasonable orders)
    b_sq_3d = torch.clamp(b_sq_3d, min=1e-15)

    deconv = 1.0 / b_sq_3d

    return deconv


def compute_bspline_deconvolution_1d(
    n: int,
    spline_order: int = 4,
    device=None,
) -> torch.Tensor:
    """Compute 1D B-spline deconvolution factors.

    Useful for separable operations or debugging.

    Parameters
    ----------
    n : int
        Grid dimension.
    spline_order : int, default=4
        B-spline order.
    device : torch.device, optional
        Device for the output tensor.

    Returns
    -------
    deconv_1d : torch.Tensor, shape (n,)
        1D deconvolution factors.
    """
    if device is None:
        device = torch.device("cpu")

    k = torch.fft.fftfreq(n, device=device) * n
    b_sq = _bspline_modulus(k, n, spline_order)
    b_sq = torch.clamp(b_sq, min=1e-15)

    return 1.0 / b_sq


###########################################################################################
########################### Convenience Exports ###########################################
###########################################################################################


__all__ = [
    # Unified PyTorch API (scalar)
    "spline_spread",
    "spline_gather",
    "spline_gather_vec3",
    "spline_gather_gradient",
    # Unified PyTorch API (multi-channel)
    "spline_spread_channels",
    "spline_gather_channels",
    # Deconvolution
    "compute_bspline_deconvolution",
    "compute_bspline_deconvolution_1d",
    # Warp functions (for custom kernels)
    "bspline_weight",
    "bspline_derivative",
    "bspline_weight_3d",
    "bspline_weight_gradient_3d",
    "compute_fractional_coords",
    "bspline_grid_offset",
    "wrap_grid_index",
    # Warp kernels (single-system, scalar)
    "_bspline_spread_kernel",
    "_bspline_gather_kernel",
    "_bspline_gather_vec3_kernel",
    "_bspline_gather_gradient_kernel",
    # Warp kernels (batch, scalar)
    "_batch_bspline_spread_kernel",
    "_batch_bspline_gather_kernel",
    "_batch_bspline_gather_vec3_kernel",
    "_batch_bspline_gather_gradient_kernel",
    # Warp kernels (single-system, multi-channel)
    "_bspline_spread_channels_kernel",
    "_bspline_gather_channels_kernel",
    # Warp kernels (batch, multi-channel)
    "_batch_bspline_spread_channels_kernel",
    "_batch_bspline_gather_channels_kernel",
]
