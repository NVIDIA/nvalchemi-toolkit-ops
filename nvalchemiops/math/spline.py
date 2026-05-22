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
B-Spline Interpolation Kernels (Pure Warp)
==========================================

This module provides pure Warp kernels and launchers for B-spline interpolation
functions used in mesh-based calculations (e.g., Particle Mesh Ewald).

This module is framework-agnostic - it contains only Warp kernels and launchers.
For PyTorch bindings, use ``nvalchemiops.torch.spline`` instead.

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

REFERENCES
==========

- Essmann et al. (1995). J. Chem. Phys. 103, 8577 (PME B-splines)
"""

from __future__ import annotations

from typing import Any

import warp as wp

# Disable warp's automatic adjoint (backward) codegen for every kernel in
# this module. All callers route through hand-written backward chains:
# torch via register_warp_op_chain + register_autograd, JAX via
# warp.jax_experimental.jax_kernel(..., enable_backward=False). 
wp.set_module_options({"enable_backward": False})


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

    if order == 6:
        # Quintic cardinal B-spline (degree 5), 6 pieces on [0, 6).
        # Coefficients derived from M_6(u) = (1/120) Σ_{j=0..k} (-1)^j C(6,j) (u-j)^5.
        twenty_four = type(u)(24.0)
        one_twenty = type(u)(120.0)
        five = type(u)(5.0)
        if u >= zero and u < one:
            u2 = u * u
            return (u2 * u2 * u) / one_twenty
        elif u >= one and u < two:
            u2 = u * u
            u3 = u2 * u
            u4 = u2 * u2
            u5 = u4 * u
            return (
                six
                - type(u)(30.0) * u
                + type(u)(60.0) * u2
                - type(u)(60.0) * u3
                + type(u)(30.0) * u4
                - five * u5
            ) / one_twenty
        elif u >= two and u < three:
            u2 = u * u
            u3 = u2 * u
            u4 = u2 * u2
            u5 = u4 * u
            return (
                type(u)(-474.0)
                + type(u)(1170.0) * u
                - type(u)(1140.0) * u2
                + type(u)(540.0) * u3
                - type(u)(120.0) * u4
                + type(u)(10.0) * u5
            ) / one_twenty
        elif u >= three and u < four:
            u2 = u * u
            u3 = u2 * u
            u4 = u2 * u2
            u5 = u4 * u
            return (
                type(u)(4386.0)
                - type(u)(6930.0) * u
                + type(u)(4260.0) * u2
                - type(u)(1260.0) * u3
                + type(u)(180.0) * u4
                - type(u)(10.0) * u5
            ) / one_twenty
        elif u >= four and u < five:
            u2 = u * u
            u3 = u2 * u
            u4 = u2 * u2
            u5 = u4 * u
            return (
                type(u)(-10974.0)
                + type(u)(12270.0) * u
                - type(u)(5340.0) * u2
                + type(u)(1140.0) * u3
                - type(u)(120.0) * u4
                + five * u5
            ) / one_twenty
        elif u >= five and u < type(u)(6.0):
            v = type(u)(6.0) - u
            v2 = v * v
            return (v2 * v2 * v) / one_twenty
        else:
            _ = twenty_four  # unused; declared for symmetry
            return zero
    elif order == 5:
        # Quartic cardinal B-spline (degree 4), 5 pieces on [0, 5).
        # Coefficients from M_5(u) = (1/24) Σ_{j=0..k} (-1)^j C(5,j) (u-j)^4.
        twenty_four = type(u)(24.0)
        five = type(u)(5.0)
        if u >= zero and u < one:
            u2 = u * u
            return (u2 * u2) / twenty_four
        elif u >= one and u < two:
            u2 = u * u
            u3 = u2 * u
            u4 = u2 * u2
            return (
                type(u)(-5.0)
                + type(u)(20.0) * u
                - type(u)(30.0) * u2
                + type(u)(20.0) * u3
                - type(u)(4.0) * u4
            ) / twenty_four
        elif u >= two and u < three:
            u2 = u * u
            u3 = u2 * u
            u4 = u2 * u2
            return (
                type(u)(155.0)
                - type(u)(300.0) * u
                + type(u)(210.0) * u2
                - type(u)(60.0) * u3
                + type(u)(6.0) * u4
            ) / twenty_four
        elif u >= three and u < four:
            u2 = u * u
            u3 = u2 * u
            u4 = u2 * u2
            return (
                type(u)(-655.0)
                + type(u)(780.0) * u
                - type(u)(330.0) * u2
                + type(u)(60.0) * u3
                - type(u)(4.0) * u4
            ) / twenty_four
        elif u >= four and u < five:
            v = five - u
            v2 = v * v
            return (v2 * v2) / twenty_four
        else:
            return zero
    elif order == 4:
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

    if order == 6:
        # Derivatives of the quintic pieces from bspline_weight (order 6).
        one_twenty = type(u)(120.0)
        five = type(u)(5.0)
        if u >= zero and u < one:
            u2 = u * u
            return (five * u2 * u2) / one_twenty
        elif u >= one and u < two:
            u2 = u * u
            u3 = u2 * u
            u4 = u2 * u2
            return (
                type(u)(-30.0)
                + type(u)(120.0) * u
                - type(u)(180.0) * u2
                + type(u)(120.0) * u3
                - type(u)(25.0) * u4
            ) / one_twenty
        elif u >= two and u < three:
            u2 = u * u
            u3 = u2 * u
            u4 = u2 * u2
            return (
                type(u)(1170.0)
                - type(u)(2280.0) * u
                + type(u)(1620.0) * u2
                - type(u)(480.0) * u3
                + type(u)(50.0) * u4
            ) / one_twenty
        elif u >= three and u < four:
            u2 = u * u
            u3 = u2 * u
            u4 = u2 * u2
            return (
                type(u)(-6930.0)
                + type(u)(8520.0) * u
                - type(u)(3780.0) * u2
                + type(u)(720.0) * u3
                - type(u)(50.0) * u4
            ) / one_twenty
        elif u >= four and u < five:
            u2 = u * u
            u3 = u2 * u
            u4 = u2 * u2
            return (
                type(u)(12270.0)
                - type(u)(10680.0) * u
                + type(u)(3420.0) * u2
                - type(u)(480.0) * u3
                + type(u)(25.0) * u4
            ) / one_twenty
        elif u >= five and u < type(u)(6.0):
            # M_6(u) = (6-u)^5 / 120 → M_6'(u) = -5(6-u)^4 / 120 = -(6-u)^4 / 24
            v = type(u)(6.0) - u
            v2 = v * v
            return -(v2 * v2) / type(u)(24.0)
        else:
            return zero
    elif order == 5:
        # Derivatives of the quartic pieces from bspline_weight (order 5).
        twenty_four = type(u)(24.0)
        five = type(u)(5.0)
        if u >= zero and u < one:
            return (four * u * u * u) / twenty_four
        elif u >= one and u < two:
            u2 = u * u
            u3 = u2 * u
            return (
                type(u)(20.0)
                - type(u)(60.0) * u
                + type(u)(60.0) * u2
                - type(u)(16.0) * u3
            ) / twenty_four
        elif u >= two and u < three:
            u2 = u * u
            u3 = u2 * u
            return (
                type(u)(-300.0)
                + type(u)(420.0) * u
                - type(u)(180.0) * u2
                + type(u)(24.0) * u3
            ) / twenty_four
        elif u >= three and u < four:
            u2 = u * u
            u3 = u2 * u
            return (
                type(u)(780.0)
                - type(u)(660.0) * u
                + type(u)(180.0) * u2
                - type(u)(16.0) * u3
            ) / twenty_four
        elif u >= four and u < five:
            # M_5(u) = (5-u)^4 / 24 → M_5'(u) = -4(5-u)^3 / 24 = -(5-u)^3 / 6
            v = five - u
            return -(v * v * v) / six
        else:
            return zero
    elif order == 4:
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


@wp.func
def bspline_second_derivative(u: Any, order: wp.int32) -> Any:
    """Compute B-spline second derivative ``d²M_n(u)/du²``.

    Mirrors the order-2/3/4 coverage of ``bspline_derivative`` (orders 1 and
    5+ return zero — same convention as the first-derivative function).
    Used by the position-Hessian backward of ``_bspline_gather_gradient_kernel``.
    """
    zero = type(u)(0.0)
    one = type(u)(1.0)
    two = type(u)(2.0)
    three = type(u)(3.0)
    four = type(u)(4.0)
    six = type(u)(6.0)

    if order == 6:
        # Second derivatives of the quintic pieces from bspline_weight (order 6).
        one_twenty = type(u)(120.0)
        five = type(u)(5.0)
        if u >= zero and u < one:
            return (type(u)(20.0) * u * u * u) / one_twenty
        elif u >= one and u < two:
            u2 = u * u
            u3 = u2 * u
            return (
                type(u)(120.0)
                - type(u)(360.0) * u
                + type(u)(360.0) * u2
                - type(u)(100.0) * u3
            ) / one_twenty
        elif u >= two and u < three:
            u2 = u * u
            u3 = u2 * u
            return (
                type(u)(-2280.0)
                + type(u)(3240.0) * u
                - type(u)(1440.0) * u2
                + type(u)(200.0) * u3
            ) / one_twenty
        elif u >= three and u < four:
            u2 = u * u
            u3 = u2 * u
            return (
                type(u)(8520.0)
                - type(u)(7560.0) * u
                + type(u)(2160.0) * u2
                - type(u)(200.0) * u3
            ) / one_twenty
        elif u >= four and u < five:
            u2 = u * u
            u3 = u2 * u
            return (
                type(u)(-10680.0)
                + type(u)(6840.0) * u
                - type(u)(1440.0) * u2
                + type(u)(100.0) * u3
            ) / one_twenty
        elif u >= five and u < type(u)(6.0):
            # M_6 = (6-u)^5 / 120 → M_6'' = 20(6-u)^3 / 120 = (6-u)^3 / 6
            v = type(u)(6.0) - u
            return (v * v * v) / six
        else:
            return zero
    elif order == 5:
        # Second derivatives of the quartic pieces from bspline_weight (order 5).
        twenty_four = type(u)(24.0)
        five = type(u)(5.0)
        if u >= zero and u < one:
            return (type(u)(12.0) * u * u) / twenty_four
        elif u >= one and u < two:
            u2 = u * u
            return (
                type(u)(-60.0)
                + type(u)(120.0) * u
                - type(u)(48.0) * u2
            ) / twenty_four
        elif u >= two and u < three:
            u2 = u * u
            return (
                type(u)(420.0)
                - type(u)(360.0) * u
                + type(u)(72.0) * u2
            ) / twenty_four
        elif u >= three and u < four:
            u2 = u * u
            return (
                type(u)(-660.0)
                + type(u)(360.0) * u
                - type(u)(48.0) * u2
            ) / twenty_four
        elif u >= four and u < five:
            # M_5 = (5-u)^4 / 24 → M_5'' = 12(5-u)^2 / 24 = (5-u)^2 / 2
            v = five - u
            return (v * v) / two
        else:
            return zero
    elif order == 4:
        # W(u) over [k, k+1] for k in {0..3} (see bspline_weight for forms).
        if u >= zero and u < one:
            # W''(u) = u
            return u
        elif u >= one and u < two:
            # W' = (-9u² + 24u - 12)/6  →  W'' = -3u + 4
            return -three * u + four
        elif u >= two and u < three:
            # W' = (9u² - 48u + 60)/6  →  W'' = 3u - 8
            return three * u - type(u)(8.0)
        elif u >= three and u < four:
            # W = (4-u)³/6 → W'' = 4 - u
            return four - u
        else:
            return zero
    elif order == 3:
        if u >= zero and u < one:
            return one
        elif u >= one and u < two:
            return -two
        elif u >= two and u < three:
            return one
        else:
            return zero
    elif order == 2:
        # First derivative is piecewise constant, so second derivative is 0.
        return zero
    else:
        return zero


@wp.func
def bspline_weight_hessian_dot_vec3(
    theta: Any,
    offset: wp.vec3i,
    order: wp.int32,
    mesh_dims: wp.vec3i,
    v: Any,
) -> Any:
    """Compute ``H @ v`` where ``H`` is the scaled 3x3 Hessian of the 3D
    B-spline weight at the given stencil point.

    The Hessian is symmetric and has entries
    ``H[c, d] = mesh_dims[c] · ∂²W/∂θ_c∂θ_d · mesh_dims[d]`` (matching the
    ``mesh_dims``-scaling convention used by ``bspline_weight_gradient_3d``).
    Off-diagonal entries are products of two 1D first-derivatives; diagonal
    entries multiply the 1D second-derivative by the other two 1D weights.

    Returning ``H @ v`` directly avoids constructing a generic-dtype
    ``mat33`` in Warp (which is awkward across float32/float64) and saves
    the calling kernel from doing the matrix-vector product separately.
    """
    t0 = theta[0]
    half_order = type(t0)(order) * type(t0)(0.5)
    zero = type(t0)(0.0)
    order_f = type(t0)(order)

    u_x = half_order + t0 - type(t0)(offset[0])
    u_y = half_order + theta[1] - type(t0)(offset[1])
    u_z = half_order + theta[2] - type(t0)(offset[2])

    if (
        u_x < zero or u_x >= order_f
        or u_y < zero or u_y >= order_f
        or u_z < zero or u_z >= order_f
    ):
        return type(v)(zero, zero, zero)

    w_x = bspline_weight(u_x, order)
    w_y = bspline_weight(u_y, order)
    w_z = bspline_weight(u_z, order)

    dw_x = bspline_derivative(u_x, order)
    dw_y = bspline_derivative(u_y, order)
    dw_z = bspline_derivative(u_z, order)

    d2w_x = bspline_second_derivative(u_x, order)
    d2w_y = bspline_second_derivative(u_y, order)
    d2w_z = bspline_second_derivative(u_z, order)

    mx = type(t0)(mesh_dims[0])
    my = type(t0)(mesh_dims[1])
    mz = type(t0)(mesh_dims[2])

    Hxx = d2w_x * w_y * w_z * mx * mx
    Hyy = w_x * d2w_y * w_z * my * my
    Hzz = w_x * w_y * d2w_z * mz * mz
    Hxy = dw_x * dw_y * w_z * mx * my
    Hxz = dw_x * w_y * dw_z * mx * mz
    Hyz = w_x * dw_y * dw_z * my * mz

    return type(v)(
        Hxx * v[0] + Hxy * v[1] + Hxz * v[2],
        Hxy * v[0] + Hyy * v[1] + Hyz * v[2],
        Hxz * v[0] + Hyz * v[1] + Hzz * v[2],
    )


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
    # This ensures u = n/2 + theta - offset is always in [0, n).
    # Warp 1.13.0 adjoint-codegen mis-types `type(t0)(order - 2)` so use
    # an int variable for the subtraction and cast once at the end.
    n_minus_2 = order - 2  # int32, no float involvement
    half_n_minus_1 = type(t0)(n_minus_2) * type(t0)(0.5)
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
def _bspline_weight_kernel(
    u: wp.array(dtype=Any),
    order: wp.int32,
    weights: wp.array(dtype=Any),
):
    """Compute B-spline weights for an array of inputs.

    Parameters
    ----------
    u : wp.array, shape (N,)
        Input values.
    order : wp.int32
        Spline order.
    weights : wp.array, shape (N,)
        Output weights.
    """
    i = wp.tid()
    weights[i] = bspline_weight(u[i], order)


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
    - Layout is per-(atom, stencil-point) rather than per-atom. A
      per-atom + register accumulation variant regressed by 9.4x at
      single_128k: serializing the 64 atomic_adds within a thread loses more
      from reduced occupancy than it gains from eliminated launch overhead.
      The per-order specialized kernels (see ``_PER_ORDER_*`` below) take a
      different approach: full unroll of the order^3 stencil per atom, used
      for orders 2-6.
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
    - A per-atom register-accumulation variant regressed by 9.2x at
      single_128k: the atomic-elimination win on ``output[atom_idx]`` was
      overwhelmed by uncoalesced mesh reads (threads in a warp now hit
      different atoms' scattered stencils instead of the same atom's
      contiguous patch). Atom sorting + tighter codegen could close that
      gap but is out of scope here.
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
def _bspline_gather_with_force_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array3d(dtype=Any),
    output: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
):
    """Single-pass interpolation: gather potential AND spline-derivative force.

    Reads each mesh stencil cell ONCE, accumulating both:
      - ``output[atom] += Σ_g mesh[g] * w(atom, g)``           (raw potential)
      - ``forces[atom] += -q_atom * Σ_g mesh[g] * (Cell^{-T} ∇w)``  (Cartesian force)

    This replaces calling ``_bspline_gather_kernel`` followed by
    ``_bspline_gather_gradient_kernel`` on the same mesh — they would each
    re-read every stencil cell and recompute the per-thread weight
    derivatives. The fused kernel halves the mesh DRAM traffic for the
    PME-with-forces path and reuses one set of 1D weight evaluations across
    both output channels.

    Launch Grid
    -----------
    dim = [num_atoms, order^3]

    Parameters
    ----------
    positions, charges, cell_inv_t, order, mesh :
        Same as ``_bspline_gather_kernel`` / ``_bspline_gather_gradient_kernel``.
    output : wp.array, shape (N,), dtype=float32/float64
        OUTPUT: raw potential per atom. Must be zero-initialized.
    forces : wp.array, shape (N,), dtype=vec3f/vec3d
        OUTPUT: Cartesian force per atom (already including −q). Must be
        zero-initialized.
    """
    atom_idx, point_idx = wp.tid()

    mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
    position = positions[atom_idx]
    charge = charges[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)

    t0 = theta[0]
    half_order = type(t0)(order) * type(t0)(0.5)
    zero = type(t0)(0.0)
    order_f = type(t0)(order)

    u_x = half_order + theta[0] - type(t0)(offset[0])
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
        return

    # One 1D-weight + derivative evaluation per axis per thread, reused for
    # both the scalar potential weight and the three gradient components.
    w_x = bspline_weight(u_x, order)
    w_y = bspline_weight(u_y, order)
    w_z = bspline_weight(u_z, order)
    dw_x = bspline_derivative(u_x, order) * type(t0)(mesh_dims[0])
    dw_y = bspline_derivative(u_y, order) * type(t0)(mesh_dims[1])
    dw_z = bspline_derivative(u_z, order) * type(t0)(mesh_dims[2])

    gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
    gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
    gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

    mesh_val = mesh[gx, gy, gz]

    # Scalar potential contribution. The `weight > 1e-8` cutoff matches the
    # original ``_bspline_gather_kernel`` so the fused output is byte-identical
    # to the (un-fused) two-kernel path; without it, near-zero stencil-edge
    # contributions get included and the accumulated sum order changes enough
    # to violate the tight momentum-conservation tolerance.
    weight = w_x * w_y * w_z
    if weight > type(mesh_val)(1e-8):
        wp.atomic_add(output, atom_idx, mesh_val * weight)

    # Fractional-coordinate gradient → Cartesian force. ``grad_mag > 0`` mirrors
    # the original ``_bspline_gather_gradient_kernel``.
    grad_x = dw_x * w_y * w_z
    grad_y = w_x * dw_y * w_z
    grad_z = w_x * w_y * dw_z
    grad_mag = wp.abs(grad_x) + wp.abs(grad_y) + wp.abs(grad_z)
    if grad_mag > type(charge)(0.0):
        force_frac = type(position)(
            -charge * mesh_val * grad_x,
            -charge * mesh_val * grad_y,
            -charge * mesh_val * grad_z,
        )
        force = wp.transpose(cell_inv_t[0]) * force_frac
        wp.atomic_add(forces, atom_idx, force)


###########################################################################################
####### Per-order specialized fused gather — per-atom + register accumulation #############
###########################################################################################
#
# For each supported spline order, a Python factory builds a kernel that:
#   * captures ORDER as a Python int (so Warp's codegen treats it as
#     compile-time and fully unrolls the order^3 stencil loop),
#   * uses dim=[num_atoms] (one thread per atom),
#   * pre-computes the 1D weights + derivatives into a length-ORDER vec,
#   * accumulates the potential and Cartesian-frame force gradient into
#     registers, then writes ONCE per channel — no atomics on the output.
#
# This pattern requires ORDER to be a Python int literal so Warp can unroll
# the per-thread loop at codegen time. Microbenched at single_8k mesh=64^3
# order=6: 9.26x speedup over the runtime-order kernel
# (0.627 ms → 0.068 ms) with bit-identical output.
#
# Each (order, dtype) pair is assigned to its OWN named warp module via
# ``@wp.kernel(module=wp.get_module(...))`` so that warp compiles only the
# orders the user actually launches. Otherwise all 5 orders × both dtypes
# would compile eagerly on the first launch, ballooning cold-cache compile
# time. Module options (``enable_backward=False``) are set per-module
# inside the factory.

_PER_ORDER_VEC = {
    (2, wp.float32): wp.types.vector(length=2, dtype=wp.float32),
    (2, wp.float64): wp.types.vector(length=2, dtype=wp.float64),
    (3, wp.float32): wp.types.vector(length=3, dtype=wp.float32),
    (3, wp.float64): wp.types.vector(length=3, dtype=wp.float64),
    (4, wp.float32): wp.types.vector(length=4, dtype=wp.float32),
    (4, wp.float64): wp.types.vector(length=4, dtype=wp.float64),
    (5, wp.float32): wp.types.vector(length=5, dtype=wp.float32),
    (5, wp.float64): wp.types.vector(length=5, dtype=wp.float64),
    (6, wp.float32): wp.types.vector(length=6, dtype=wp.float32),
    (6, wp.float64): wp.types.vector(length=6, dtype=wp.float64),
}


def _per_order_module(kind: str, order: int, scalar_dtype) -> "wp.Module":
    """Return a named warp module for a (kind, order, dtype) per-order kernel.

    Using a distinct warp module per tuple means warp compiles ONLY the
    orders/dtypes the user actually launches at runtime. Module options
    (``enable_backward=False``) are applied here, isolated from the
    parent ``nvalchemiops.math.spline`` module's settings.
    """
    dtype_tag = "fp32" if scalar_dtype is wp.float32 else "fp64"
    mod = wp.get_module(
        f"nvalchemiops.math.spline_per_order.{kind}_order{order}_{dtype_tag}"
    )
    # The adjoint is never used (see top-of-file note); skipping it saves
    # ~70% of generated code for these heavily-unrolled kernels.
    mod.options["enable_backward"] = False
    return mod


def _make_bspline_gather_with_force_kernel(
    ORDER: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    """Factory: per-order specialized fused gather kernel.

    Returns a Warp kernel parameterized for the given spline ``ORDER``. The
    kernel walks the order^3 stencil entirely in registers — fully unrolled
    by Warp's codegen because ORDER is a Python int literal in scope — and
    writes potential and force without atomics.
    """
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]
    _vec_pos = vec_pos_dtype
    # Pre-compute Python-side float constants so the kernel never has to
    # do an int→float cast that Warp 1.13.0's adjoint codegen mishandles.
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5

    @wp.kernel(module=_per_order_module("gather_with_force", ORDER, scalar_dtype))
    def kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        cell_inv_t: wp.array(dtype=Any),
        mesh: wp.array3d(dtype=Any),
        output: wp.array(dtype=Any),
        forces: wp.array(dtype=Any),
    ):
        atom_idx = wp.tid()
        mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
        position = positions[atom_idx]
        charge = charges[atom_idx]

        base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)

        t0 = theta[0]
        half_order = type(t0)(HALF_ORDER_PY)
        half_n_minus_2 = type(t0)(HALF_N_MINUS_2_PY)
        zero = type(t0)(0.0)

        offset_start_x = wp.int32(wp.floor(theta[0] - half_n_minus_2))
        offset_start_y = wp.int32(wp.floor(theta[1] - half_n_minus_2))
        offset_start_z = wp.int32(wp.floor(theta[2] - half_n_minus_2))

        # 1D weights + derivatives per axis (3 * ORDER evaluations total).
        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        dwx = vec_ord()
        dwy = vec_ord()
        dwz = vec_ord()
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            dwx[k] = bspline_derivative(u_x, ORDER) * type(t0)(mesh_dims[0])
            dwy[k] = bspline_derivative(u_y, ORDER) * type(t0)(mesh_dims[1])
            dwz[k] = bspline_derivative(u_z, ORDER) * type(t0)(mesh_dims[2])

        # Register accumulators (no atomics).
        phi_acc = zero
        gx_acc = zero
        gy_acc = zero
        gz_acc = zero

        # Triple loop — fully unrolled at compile time.
        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wij = wxi * wy[j]
                dwxij_x = dwxi * wy[j]
                dwxij_y = wxi * dwy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(base_grid[2] + offset_start_z + k, mesh_dims[2])
                    val = mesh[gx, gy, gz]
                    wzk = wz[k]
                    phi_acc = phi_acc + val * (wij * wzk)
                    gx_acc = gx_acc + val * (dwxij_x * wzk)
                    gy_acc = gy_acc + val * (dwxij_y * wzk)
                    gz_acc = gz_acc + val * (wij * dwz[k])

        # Single non-atomic write per output channel.
        output[atom_idx] = phi_acc
        grad_frac = _vec_pos(gx_acc, gy_acc, gz_acc)
        force_frac = _vec_pos(
            -charge * grad_frac[0],
            -charge * grad_frac[1],
            -charge * grad_frac[2],
        )
        forces[atom_idx] = wp.transpose(cell_inv_t[0]) * force_frac

    return kernel


# Pre-compile per-order specializations for production orders.
# {scalar_dtype: {order: overload}}
_PER_ORDER_GATHER_WITH_FORCE_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
_SUPPORTED_PER_ORDER = (2, 3, 4, 5, 6)
for _order in _SUPPORTED_PER_ORDER:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_bspline_gather_with_force_kernel(_order, _scalar, _vec, _mat)
        # Register a concrete-type overload so launch can resolve the kernel
        # without inspecting the Any-typed annotations (which are strings under
        # the file's `from __future__ import annotations`).
        _PER_ORDER_GATHER_WITH_FORCE_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),       # positions
                wp.array(dtype=_scalar),    # charges
                wp.array(dtype=_mat),       # cell_inv_t
                wp.array3d(dtype=_scalar),  # mesh
                wp.array(dtype=_scalar),    # output
                wp.array(dtype=_vec),       # forces
            ],
        )


def _make_batch_bspline_gather_with_force_kernel(
    ORDER: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    """Batched variant of ``_make_bspline_gather_with_force_kernel``.

    Each thread handles one atom; the system index is looked up via
    ``batch_idx[atom_idx]`` and used to index the 4D mesh and per-system
    inverse-cell. Same per-atom + register-accumulation pattern as the
    single-system kernel; ORDER is a Python int literal so the inner
    order^3 stencil loop unrolls fully at codegen time.
    """
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]
    _vec_pos = vec_pos_dtype
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5

    @wp.kernel(module=_per_order_module(
        "batch_gather_with_force", ORDER, scalar_dtype,
    ))
    def kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        batch_idx: wp.array(dtype=wp.int32),
        cell_inv_t: wp.array(dtype=Any),  # (B, 3, 3)
        mesh: wp.array(dtype=Any, ndim=4),  # (B, nx, ny, nz)
        output: wp.array(dtype=Any),
        forces: wp.array(dtype=Any),
    ):
        atom_idx = wp.tid()
        sys_idx = batch_idx[atom_idx]
        mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
        position = positions[atom_idx]
        charge = charges[atom_idx]

        base_grid, theta = compute_fractional_coords(
            position, cell_inv_t[sys_idx], mesh_dims
        )

        t0 = theta[0]
        half_order = type(t0)(HALF_ORDER_PY)
        half_n_minus_2 = type(t0)(HALF_N_MINUS_2_PY)
        zero = type(t0)(0.0)

        offset_start_x = wp.int32(wp.floor(theta[0] - half_n_minus_2))
        offset_start_y = wp.int32(wp.floor(theta[1] - half_n_minus_2))
        offset_start_z = wp.int32(wp.floor(theta[2] - half_n_minus_2))

        wx = vec_ord(); wy = vec_ord(); wz = vec_ord()
        dwx = vec_ord(); dwy = vec_ord(); dwz = vec_ord()
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            dwx[k] = bspline_derivative(u_x, ORDER) * type(t0)(mesh_dims[0])
            dwy[k] = bspline_derivative(u_y, ORDER) * type(t0)(mesh_dims[1])
            dwz[k] = bspline_derivative(u_z, ORDER) * type(t0)(mesh_dims[2])

        phi_acc = zero
        gx_acc = zero
        gy_acc = zero
        gz_acc = zero

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]; dwxi = dwx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wij = wxi * wy[j]
                dwxij_x = dwxi * wy[j]
                dwxij_y = wxi * dwy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(base_grid[2] + offset_start_z + k, mesh_dims[2])
                    val = mesh[sys_idx, gx, gy, gz]
                    wzk = wz[k]
                    phi_acc = phi_acc + val * (wij * wzk)
                    gx_acc = gx_acc + val * (dwxij_x * wzk)
                    gy_acc = gy_acc + val * (dwxij_y * wzk)
                    gz_acc = gz_acc + val * (wij * dwz[k])

        output[atom_idx] = phi_acc
        grad_frac = _vec_pos(gx_acc, gy_acc, gz_acc)
        force_frac = _vec_pos(
            -charge * grad_frac[0],
            -charge * grad_frac[1],
            -charge * grad_frac[2],
        )
        forces[atom_idx] = wp.transpose(cell_inv_t[sys_idx]) * force_frac

    return kernel


# Pre-compile batch per-order specializations alongside the single-system ones.
_PER_ORDER_BATCH_GATHER_WITH_FORCE_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _SUPPORTED_PER_ORDER:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_batch_bspline_gather_with_force_kernel(_order, _scalar, _vec, _mat)
        _PER_ORDER_BATCH_GATHER_WITH_FORCE_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),         # positions
                wp.array(dtype=_scalar),      # charges
                wp.array(dtype=wp.int32),     # batch_idx
                wp.array(dtype=_mat),         # cell_inv_t (B, 3, 3)
                wp.array(dtype=_scalar, ndim=4),  # mesh (B, nx, ny, nz)
                wp.array(dtype=_scalar),      # output
                wp.array(dtype=_vec),         # forces
            ],
        )


###########################################################################################
########################### Per-order spread kernels #######################################
###########################################################################################
#
# Switches the order=2..6 spread path from (atom × order^3)-thread atomic
# scatter to one-thread-per-atom + fully unrolled order^3 stencil. Mirrors
# ``_make_bspline_gather_with_force_kernel`` (same 1D weight precompute +
# register reuse), but writes the (value × weight) products into ``mesh``
# via ``atomic_add`` (still needed because multiple atoms can share a
# mesh cell). Up to ~34× faster than the generic kernel at order=6.


def _make_bspline_spread_kernel(
    ORDER: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    """Factory: per-order specialized single-system spread kernel.

    One thread per atom, 1D weights in registers (``ORDER`` scalars per
    axis), fully-unrolled order^3 inner loop. Eliminates the
    per-(atom, stencil_pt) thread
    explosion of the generic ``_bspline_spread_kernel`` (which spawns
    ``num_atoms × order^3`` threads each computing one atomic_add).
    """
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5

    @wp.kernel(module=_per_order_module("spread", ORDER, scalar_dtype))
    def kernel(
        positions: wp.array(dtype=Any),
        values: wp.array(dtype=Any),
        cell_inv_t: wp.array(dtype=Any),  # (1, 3, 3)
        mesh: wp.array3d(dtype=Any),       # (nx, ny, nz) — atomic-add target
    ):
        atom_idx = wp.tid()
        mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
        position = positions[atom_idx]
        value = values[atom_idx]

        base_grid, theta = compute_fractional_coords(
            position, cell_inv_t[0], mesh_dims
        )

        t0 = theta[0]
        half_order = type(t0)(HALF_ORDER_PY)
        half_n_minus_2 = type(t0)(HALF_N_MINUS_2_PY)

        offset_start_x = wp.int32(wp.floor(theta[0] - half_n_minus_2))
        offset_start_y = wp.int32(wp.floor(theta[1] - half_n_minus_2))
        offset_start_z = wp.int32(wp.floor(theta[2] - half_n_minus_2))

        # 1D B-spline weights in registers (3 × ORDER scalars; no derivatives
        # are needed for forward spread).
        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)

        # Fully-unrolled stencil walk; ORDER is a Python int literal so
        # Warp's codegen unrolls all three loops at compile time.
        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wij = wxi * wy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    contrib = value * (wij * wz[k])
                    wp.atomic_add(mesh, gx, gy, gz, contrib)

    return kernel


def _make_batch_bspline_spread_kernel(
    ORDER: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    """Batched variant of ``_make_bspline_spread_kernel``."""
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5

    @wp.kernel(module=_per_order_module("batch_spread", ORDER, scalar_dtype))
    def kernel(
        positions: wp.array(dtype=Any),
        values: wp.array(dtype=Any),
        batch_idx: wp.array(dtype=wp.int32),
        cell_inv_t: wp.array(dtype=Any),     # (B, 3, 3)
        mesh: wp.array(dtype=Any, ndim=4),   # (B, nx, ny, nz)
    ):
        atom_idx = wp.tid()
        sys_idx = batch_idx[atom_idx]
        mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
        position = positions[atom_idx]
        value = values[atom_idx]

        base_grid, theta = compute_fractional_coords(
            position, cell_inv_t[sys_idx], mesh_dims
        )

        t0 = theta[0]
        half_order = type(t0)(HALF_ORDER_PY)
        half_n_minus_2 = type(t0)(HALF_N_MINUS_2_PY)

        offset_start_x = wp.int32(wp.floor(theta[0] - half_n_minus_2))
        offset_start_y = wp.int32(wp.floor(theta[1] - half_n_minus_2))
        offset_start_z = wp.int32(wp.floor(theta[2] - half_n_minus_2))

        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wij = wxi * wy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    contrib = value * (wij * wz[k])
                    wp.atomic_add(mesh, sys_idx, gx, gy, gz, contrib)

    return kernel


# Pre-compile per-order spread specializations for orders 2-6.
_PER_ORDER_SPREAD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
_PER_ORDER_BATCH_SPREAD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _SUPPORTED_PER_ORDER:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_bspline_spread_kernel(_order, _scalar, _vec, _mat)
        _PER_ORDER_SPREAD_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),       # positions
                wp.array(dtype=_scalar),    # values
                wp.array(dtype=_mat),       # cell_inv_t  (1, 3, 3)
                wp.array3d(dtype=_scalar),  # mesh
            ],
        )
        _kb = _make_batch_bspline_spread_kernel(_order, _scalar, _vec, _mat)
        _PER_ORDER_BATCH_SPREAD_KERNELS[_scalar][_order] = wp.overload(
            _kb,
            [
                wp.array(dtype=_vec),          # positions
                wp.array(dtype=_scalar),       # values
                wp.array(dtype=wp.int32),      # batch_idx
                wp.array(dtype=_mat),          # cell_inv_t (B, 3, 3)
                wp.array(dtype=_scalar, ndim=4),  # mesh (B, nx, ny, nz)
            ],
        )


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
########################### cell_inv_t-grad backward kernels ###############################
###########################################################################################
#
# Forward (single):
#   grad_cell_inv_t[a, b] = Σ_n positions[n, b] · (-cell @ forces[n])[a]
#                       = -(cell @ forces.T @ positions)[a, b]
#
# Backward (given upstream cotangent h of grad_cell_inv_t):
#   grad_forces[n, :]    = -cell.T @ (h @ positions[n])
#   grad_positions[n, :] = -h.T   @ (cell @ forces[n])
#   grad_cell             = -Σ_n outer(h @ positions[n], forces[n])
#                       =  Σ_n outer(-h @ positions[n], forces[n])
#
# These are per-atom matmul-style operations + an atomic reduction for the
# 3x3 cell gradient.


@wp.kernel
def _spline_cell_inv_t_grad_kernel(
    forces: wp.array(dtype=Any),        # (N,) vec3
    positions: wp.array(dtype=Any),     # (N,) vec3
    cell: wp.array(dtype=Any),          # (1,) mat33 — inv(cell_inv_t.T)
    grad_cell_inv_t: wp.array(dtype=Any),  # (1,) mat33 — must be zero-initialized
):
    """Accumulate per-atom contributions to ``grad_cell_inv_t`` (single system).

    Given Cartesian per-atom "forces" (the cotangent shape from
    ``_bspline_gather_gradient_kernel``) and atomic ``positions``, compute:

    .. math::
        qgf[n] = -\\text{cell} \\cdot \\text{forces}[n] \\in \\mathbb{R}^3,
        \\quad
        \\text{grad\\_cell\\_inv\\_t}[a, b] = \\sum_n p[n,b] \\cdot qgf[n,a].

    Used by ``_SplineSpread.backward`` / ``_SplineGather.backward`` (Phase
    7b.A) to replace the prior torch.bmm + outer-product matmul chain.
    """
    n = wp.tid()
    p = positions[n]
    f = forces[n]
    qgf = -(cell[0] * f)
    wp.atomic_add(grad_cell_inv_t, 0, wp.outer(qgf, p))


@wp.kernel
def _batch_spline_cell_inv_t_grad_kernel(
    forces: wp.array(dtype=Any),                  # (N_total,) vec3
    positions: wp.array(dtype=Any),               # (N_total,) vec3
    batch_idx: wp.array(dtype=wp.int32),          # (N_total,)
    cell: wp.array(dtype=Any),                    # (B,) mat33
    grad_cell_inv_t: wp.array(dtype=Any),         # (B,) mat33 — zero-initialized
):
    """Batched version of ``_spline_cell_inv_t_grad_kernel``.

    Per atom n with system s = batch_idx[n]:
    ``grad_cell_inv_t[s, a, b] += positions[n, b] * (-cell[s] @ forces[n])[a]``.
    Atomic accumulation per (s, a, b) since multiple atoms share each system.
    """
    n = wp.tid()
    s = batch_idx[n]
    p = positions[n]
    f = forces[n]
    qgf = -(cell[s] * f)
    wp.atomic_add(grad_cell_inv_t, s, wp.outer(qgf, p))


###########################################################################################
########################### spread-with-gradient-weights ###################################
###########################################################################################
#
# Backward of ``_bspline_gather_gradient_kernel`` w.r.t. ``mesh`` requires
# accumulating per-atom 3-vec scaling factors onto the mesh using the same
# B-spline gradient weights ∇W_frac that the forward gradient kernel used.
# This is the analog of ``_bspline_spread_kernel`` but with gradient
# weights instead of value weights.
#
# Given a per-atom 3-vec ``per_atom_vec[n,:]``:
#   mesh[g] += Σ_d per_atom_vec[n, d] · ∇W_frac[d](x_n, g)
#
# Used in the backward of ``_bspline_gather_gradient_kernel`` with
# ``per_atom_vec[n] = -charges[n] · (cell_inv_t @ grad_force[n])``.


@wp.kernel
def _bspline_spread_gradient_weights_kernel(
    positions: wp.array(dtype=Any),          # (N,) vec3
    per_atom_vec: wp.array(dtype=Any),       # (N,) vec3
    cell_inv_t: wp.array(dtype=Any),         # (1,) mat33
    order: wp.int32,
    mesh: wp.array3d(dtype=Any),             # (nx, ny, nz) output (zero-initialized)
):
    """Single-system "spread-with-gradient-weights" kernel.

    For each ``(atom, support point)``:
        mesh[g] += Σ_d per_atom_vec[n, d] · ∇W_frac[d](x_n, g)
    """
    atom_idx, point_idx = wp.tid()

    mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
    position = positions[atom_idx]
    vec = per_atom_vec[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)
    grad_frac = bspline_weight_gradient_3d(theta, offset, order, mesh_dims)

    grad_mag = wp.abs(grad_frac[0]) + wp.abs(grad_frac[1]) + wp.abs(grad_frac[2])
    if grad_mag > type(vec[0])(0.0):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])
        contrib = (
            vec[0] * grad_frac[0]
            + vec[1] * grad_frac[1]
            + vec[2] * grad_frac[2]
        )
        wp.atomic_add(mesh, gx, gy, gz, contrib)


@wp.kernel
def _batch_bspline_spread_gradient_weights_kernel(
    positions: wp.array(dtype=Any),              # (N_total,) vec3
    per_atom_vec: wp.array(dtype=Any),           # (N_total,) vec3
    batch_idx: wp.array(dtype=wp.int32),         # (N_total,)
    cell_inv_t: wp.array(dtype=Any),             # (B,) mat33
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),           # (B, nx, ny, nz) output
):
    """Batched spread-with-gradient-weights kernel."""
    atom_idx, point_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]
    vec = per_atom_vec[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[system_id], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)
    grad_frac = bspline_weight_gradient_3d(theta, offset, order, mesh_dims)

    grad_mag = wp.abs(grad_frac[0]) + wp.abs(grad_frac[1]) + wp.abs(grad_frac[2])
    if grad_mag > type(vec[0])(0.0):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])
        contrib = (
            vec[0] * grad_frac[0]
            + vec[1] * grad_frac[1]
            + vec[2] * grad_frac[2]
        )
        wp.atomic_add(mesh, system_id, gx, gy, gz, contrib)


###########################################################################################
########################### gather_gradient position-Hessian kernel ########################
###########################################################################################
#
# Backward of ``_bspline_gather_gradient_kernel`` w.r.t. ``positions`` requires
# the spatial Hessian of the B-spline weight. Given an upstream cotangent
# ``grad_force`` (3-vec per atom) and the original mesh:
#
#   ∂force[n,a]/∂position[n,b] = -q · Σ_g mesh[g] · (cell_inv_t.T H_scaled cell_inv_t)[a,b]
#   grad_position[n,b] = -q · Σ_g mesh[g] · (cell_inv_t.T H_scaled v)[b]
#
# where ``H_scaled[c,d] = mesh_dims[c] · ∂²W/∂θ_c∂θ_d · mesh_dims[d]`` and
# ``v = cell_inv_t · grad_force[n]`` (precomputed per atom).


@wp.kernel
def _bspline_gather_gradient_position_hessian_kernel(
    positions: wp.array(dtype=Any),       # (N,) vec3
    charges: wp.array(dtype=Any),         # (N,)
    v_per_atom: wp.array(dtype=Any),      # (N,) vec3 — cell_inv_t @ grad_force
    cell_inv_t: wp.array(dtype=Any),      # (1,) mat33
    order: wp.int32,
    mesh: wp.array3d(dtype=Any),          # original forward-input mesh
    grad_positions: wp.array(dtype=Any),  # (N,) vec3 output — zero-initialized
):
    """Single-system position-Hessian backward of ``_bspline_gather_gradient_kernel``."""
    atom_idx, point_idx = wp.tid()

    mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
    position = positions[atom_idx]
    q = charges[atom_idx]
    v = v_per_atom[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)

    # H_scaled @ v (3-vec). Returns zero outside support.
    Hv = bspline_weight_hessian_dot_vec3(theta, offset, order, mesh_dims, v)

    mag = wp.abs(Hv[0]) + wp.abs(Hv[1]) + wp.abs(Hv[2])
    if mag > type(q)(0.0):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[gx, gy, gz]
        # contribution = -q · mesh · (cell_inv_t.T @ Hv)
        cart = wp.transpose(cell_inv_t[0]) * Hv
        scale = -q * mesh_val
        wp.atomic_add(
            grad_positions,
            atom_idx,
            type(position)(scale * cart[0], scale * cart[1], scale * cart[2]),
        )


@wp.kernel
def _batch_bspline_gather_gradient_position_hessian_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    v_per_atom: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array(dtype=Any, ndim=4),
    grad_positions: wp.array(dtype=Any),
):
    """Batched position-Hessian backward."""
    atom_idx, point_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
    position = positions[atom_idx]
    q = charges[atom_idx]
    v = v_per_atom[atom_idx]

    base_grid, theta = compute_fractional_coords(
        position, cell_inv_t[system_id], mesh_dims
    )
    offset = bspline_grid_offset(point_idx, order, theta)

    Hv = bspline_weight_hessian_dot_vec3(theta, offset, order, mesh_dims, v)

    mag = wp.abs(Hv[0]) + wp.abs(Hv[1]) + wp.abs(Hv[2])
    if mag > type(q)(0.0):
        gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
        gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
        gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])

        mesh_val = mesh[system_id, gx, gy, gz]
        cart = wp.transpose(cell_inv_t[system_id]) * Hv
        scale = -q * mesh_val
        wp.atomic_add(
            grad_positions,
            atom_idx,
            type(position)(scale * cart[0], scale * cart[1], scale * cart[2]),
        )


@wp.kernel
def _spline_cell_inv_t_grad_backward_kernel(
    h: wp.array(dtype=Any),                   # (1,) mat33 — upstream cotangent
    forces: wp.array(dtype=Any),              # (N,) vec3
    positions: wp.array(dtype=Any),           # (N,) vec3
    cell: wp.array(dtype=Any),                # (1,) mat33
    grad_forces: wp.array(dtype=Any),         # (N,) vec3 — output
    grad_positions: wp.array(dtype=Any),      # (N,) vec3 — output
    grad_cell: wp.array(dtype=Any),           # (1,) mat33 — zero-initialized
):
    """Single-system backward of ``_spline_cell_inv_t_grad_kernel``.

    Produces per-atom ``grad_forces`` / ``grad_positions`` and an
    atomically-accumulated ``grad_cell``. The chain ``cell → cell_inv_t``
    is torch (a 3x3 inverse) and stays differentiable on top of this.
    """
    n = wp.tid()
    p = positions[n]
    f = forces[n]
    h_mat = h[0]
    c_mat = cell[0]

    # 3-vec intermediates.
    h_p = h_mat * p          # h @ positions[n]
    c_f = c_mat * f          # cell @ forces[n]

    # grad_forces[n,:] = -cell.T @ (h @ positions[n])
    grad_forces[n] = -(wp.transpose(c_mat) * h_p)
    # grad_positions[n,:] = -h.T @ (cell @ forces[n])
    grad_positions[n] = -(wp.transpose(h_mat) * c_f)
    # grad_cell contribution: -outer(h @ positions[n], forces[n])
    wp.atomic_add(grad_cell, 0, -wp.outer(h_p, f))


@wp.kernel
def _batch_spline_cell_inv_t_grad_backward_kernel(
    h: wp.array(dtype=Any),                       # (B,) mat33
    forces: wp.array(dtype=Any),                  # (N_total,) vec3
    positions: wp.array(dtype=Any),               # (N_total,) vec3
    batch_idx: wp.array(dtype=wp.int32),          # (N_total,)
    cell: wp.array(dtype=Any),                    # (B,) mat33
    grad_forces: wp.array(dtype=Any),             # (N_total,) vec3
    grad_positions: wp.array(dtype=Any),          # (N_total,) vec3
    grad_cell: wp.array(dtype=Any),               # (B,) mat33 — zero-initialized
):
    """Batched backward of ``_batch_spline_cell_inv_t_grad_kernel``."""
    n = wp.tid()
    s = batch_idx[n]
    p = positions[n]
    f = forces[n]
    h_mat = h[s]
    c_mat = cell[s]

    h_p = h_mat * p
    c_f = c_mat * f

    grad_forces[n] = -(wp.transpose(c_mat) * h_p)
    grad_positions[n] = -(wp.transpose(h_mat) * c_f)
    wp.atomic_add(grad_cell, s, -wp.outer(h_p, f))


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
_bspline_weight_kernel_overload = {}
_bspline_spread_kernel_overload = {}
_bspline_gather_kernel_overload = {}
_bspline_gather_vec3_kernel_overload = {}
_bspline_gather_gradient_kernel_overload = {}
_bspline_gather_with_force_kernel_overload = {}
_spline_cell_inv_t_grad_kernel_overload = {}
_spline_cell_inv_t_grad_backward_kernel_overload = {}
_bspline_spread_gradient_weights_kernel_overload = {}
_bspline_gather_gradient_position_hessian_kernel_overload = {}

# Batch kernel overloads
_batch_bspline_spread_kernel_overload = {}
_batch_bspline_gather_kernel_overload = {}
_batch_bspline_gather_vec3_kernel_overload = {}
_batch_bspline_gather_gradient_kernel_overload = {}
_batch_spline_cell_inv_t_grad_kernel_overload = {}
_batch_spline_cell_inv_t_grad_backward_kernel_overload = {}
_batch_bspline_spread_gradient_weights_kernel_overload = {}
_batch_bspline_gather_gradient_position_hessian_kernel_overload = {}

# Multi-channel kernel overloads
_bspline_spread_channels_kernel_overload = {}
_bspline_gather_channels_kernel_overload = {}
_batch_bspline_spread_channels_kernel_overload = {}
_batch_bspline_gather_channels_kernel_overload = {}

for t, v, m in zip(_T, _V, _M):
    # Single-system kernels
    _bspline_weight_kernel_overload[t] = wp.overload(
        _bspline_weight_kernel,
        [
            wp.array(dtype=t),  # u
            wp.int32,  # order
            wp.array(dtype=t),  # weights
        ],
    )
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
    _bspline_gather_with_force_kernel_overload[t] = wp.overload(
        _bspline_gather_with_force_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell_inv_t
            wp.int32,  # order
            wp.array3d(dtype=t),  # mesh
            wp.array(dtype=t),  # output (potential)
            wp.array(dtype=v),  # forces
        ],
    )
    _spline_cell_inv_t_grad_kernel_overload[t] = wp.overload(
        _spline_cell_inv_t_grad_kernel,
        [
            wp.array(dtype=v),  # forces
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cell  (1,)
            wp.array(dtype=m),  # grad_cell_inv_t  (1,)
        ],
    )
    _spline_cell_inv_t_grad_backward_kernel_overload[t] = wp.overload(
        _spline_cell_inv_t_grad_backward_kernel,
        [
            wp.array(dtype=m),  # h (upstream cotangent, 1,)
            wp.array(dtype=v),  # forces
            wp.array(dtype=v),  # positions
            wp.array(dtype=m),  # cell  (1,)
            wp.array(dtype=v),  # grad_forces
            wp.array(dtype=v),  # grad_positions
            wp.array(dtype=m),  # grad_cell  (1,)
        ],
    )
    _bspline_spread_gradient_weights_kernel_overload[t] = wp.overload(
        _bspline_spread_gradient_weights_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # per_atom_vec
            wp.array(dtype=m),  # cell_inv_t (1,)
            wp.int32,  # order
            wp.array3d(dtype=t),  # mesh
        ],
    )
    _bspline_gather_gradient_position_hessian_kernel_overload[t] = wp.overload(
        _bspline_gather_gradient_position_hessian_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=v),  # v_per_atom
            wp.array(dtype=m),  # cell_inv_t (1,)
            wp.int32,  # order
            wp.array3d(dtype=t),  # mesh
            wp.array(dtype=v),  # grad_positions
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
    _batch_spline_cell_inv_t_grad_kernel_overload[t] = wp.overload(
        _batch_spline_cell_inv_t_grad_kernel,
        [
            wp.array(dtype=v),  # forces
            wp.array(dtype=v),  # positions
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell  (B,)
            wp.array(dtype=m),  # grad_cell_inv_t  (B,)
        ],
    )
    _batch_spline_cell_inv_t_grad_backward_kernel_overload[t] = wp.overload(
        _batch_spline_cell_inv_t_grad_backward_kernel,
        [
            wp.array(dtype=m),  # h (B,)
            wp.array(dtype=v),  # forces
            wp.array(dtype=v),  # positions
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell (B,)
            wp.array(dtype=v),  # grad_forces
            wp.array(dtype=v),  # grad_positions
            wp.array(dtype=m),  # grad_cell (B,)
        ],
    )
    _batch_bspline_spread_gradient_weights_kernel_overload[t] = wp.overload(
        _batch_bspline_spread_gradient_weights_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # per_atom_vec
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t (B,)
            wp.int32,  # order
            wp.array(dtype=t, ndim=4),  # mesh (B, nx, ny, nz)
        ],
    )
    _batch_bspline_gather_gradient_position_hessian_kernel_overload[t] = wp.overload(
        _batch_bspline_gather_gradient_position_hessian_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=v),  # v_per_atom
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=m),  # cell_inv_t (B,)
            wp.int32,  # order
            wp.array(dtype=t, ndim=4),  # mesh
            wp.array(dtype=v),  # grad_positions
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
########################### Warp Launcher Functions #######################################
###########################################################################################


def bspline_weight_launcher(
    u: wp.array,
    order: int,
    weights: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Compute B-spline weights for an array of inputs.

    Parameters
    ----------
    u : wp.array, shape (N,)
        Input values.
    order : int
        B-spline order.
    weights : wp.array, shape (N,)
        Output weights.
    wp_dtype : type
        Warp scalar dtype.
    device : str | None
        Warp device string.
    """
    num_points = u.shape[0]
    kernel = _bspline_weight_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=num_points,
        inputs=[u, wp.int32(order)],
        outputs=[weights],
        device=device,
    )


def spline_spread(
    positions: wp.array,
    values: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Spread values from atoms to mesh using B-spline interpolation.

    Framework-agnostic launcher for single-system spline spread.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    values : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Values to spread (e.g., charges).
    cell_inv_t : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (nx, ny, nz), dtype=wp.float32 or wp.float64
        OUTPUT: Mesh to accumulate values. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _bspline_spread_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, values, cell_inv_t, wp.int32(order)],
        outputs=[mesh],
        device=device,
    )


def spline_gather(
    positions: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    output: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Gather values from mesh to atoms using B-spline interpolation.

    Framework-agnostic launcher for single-system spline gather.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    cell_inv_t : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (nx, ny, nz), dtype=wp.float32 or wp.float64
        Mesh to interpolate from.
    output : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Interpolated values per atom. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _bspline_gather_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, cell_inv_t, wp.int32(order), mesh],
        outputs=[output],
        device=device,
    )


def spline_gather_vec3(
    positions: wp.array,
    charges: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    output: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Gather charge-weighted vector values from mesh using B-splines.

    Framework-agnostic launcher for single-system vec3 spline gather.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell_inv_t : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (nx, ny, nz), dtype=wp.vec3f or wp.vec3d
        Vector-valued mesh to interpolate from.
    output : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Charge-weighted interpolated vectors. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _bspline_gather_vec3_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, charges, cell_inv_t, wp.int32(order), mesh],
        outputs=[output],
        device=device,
    )


def spline_gather_gradient(
    positions: wp.array,
    charges: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    forces: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Compute forces using B-spline gradient interpolation.

    Framework-agnostic launcher for single-system spline gradient gather.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell_inv_t : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (nx, ny, nz), dtype=wp.float32 or wp.float64
        Potential mesh.
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Forces per atom. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _bspline_gather_gradient_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, charges, cell_inv_t, wp.int32(order), mesh],
        outputs=[forces],
        device=device,
    )


def spline_cell_inv_t_grad(
    forces: wp.array,
    positions: wp.array,
    cell: wp.array,
    grad_cell_inv_t: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Single-system fused launcher for ``_spline_cell_inv_t_grad_kernel``.

    Output ``grad_cell_inv_t`` must be zero-initialized.
    """
    kernel = _spline_cell_inv_t_grad_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=positions.shape[0],
        inputs=[forces, positions, cell],
        outputs=[grad_cell_inv_t],
        device=device,
    )


def spline_cell_inv_t_grad_backward(
    h: wp.array,
    forces: wp.array,
    positions: wp.array,
    cell: wp.array,
    grad_forces: wp.array,
    grad_positions: wp.array,
    grad_cell: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Single-system launcher for ``_spline_cell_inv_t_grad_backward_kernel``.

    ``grad_cell`` must be zero-initialized; the other outputs are written
    elementwise per atom.
    """
    kernel = _spline_cell_inv_t_grad_backward_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=positions.shape[0],
        inputs=[h, forces, positions, cell],
        outputs=[grad_forces, grad_positions, grad_cell],
        device=device,
    )


def spline_spread_gradient_weights(
    positions: wp.array,
    per_atom_vec: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Single-system launcher for ``_bspline_spread_gradient_weights_kernel``.

    ``mesh`` output must be zero-initialized.
    """
    kernel = _bspline_spread_gradient_weights_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(positions.shape[0], order**3),
        inputs=[positions, per_atom_vec, cell_inv_t, wp.int32(order)],
        outputs=[mesh],
        device=device,
    )


def spline_gather_gradient_position_hessian(
    positions: wp.array,
    charges: wp.array,
    v_per_atom: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    grad_positions: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Single-system launcher for the position-Hessian backward of
    ``_bspline_gather_gradient_kernel``. ``grad_positions`` must be
    zero-initialized.
    """
    kernel = _bspline_gather_gradient_position_hessian_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(positions.shape[0], order**3),
        inputs=[positions, charges, v_per_atom, cell_inv_t, wp.int32(order), mesh],
        outputs=[grad_positions],
        device=device,
    )


def spline_gather_with_force(
    positions: wp.array,
    charges: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    output: wp.array,
    forces: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Fused energy-gather + force-gather in a single kernel launch.

    Computes simultaneously, reading each mesh cell ONCE:
      - ``output[atom] = Σ_g mesh[g] * w(atom, g)``           (raw potential)
      - ``forces[atom] = -q_atom * Σ_g mesh[g] * Cell^{-T} ∇w`` (Cartesian force)

    Replaces the (``spline_gather`` → ``spline_gather_gradient``) pair when
    both outputs are needed (PME forces path). Halves the mesh DRAM traffic
    and reuses the per-thread 1D weight evaluations across both outputs.
    Output buffers must be zero-initialized.
    """
    num_atoms = positions.shape[0]

    # Per-order specialized kernel is available for orders 2-6 and uses
    # register accumulation + compile-time unrolling for a large speedup
    # over the generic runtime-order kernel (~9x at single_8k order=6).
    per_order = _PER_ORDER_GATHER_WITH_FORCE_KERNELS[wp_dtype].get(order)
    if per_order is not None:
        wp.launch(
            per_order,
            dim=num_atoms,
            inputs=[positions, charges, cell_inv_t, mesh],
            outputs=[output, forces],
            device=device,
        )
    else:
        # Fallback: generic per-(atom, stencil-point) kernel with atomics.
        kernel = _bspline_gather_with_force_kernel_overload[wp_dtype]
        wp.launch(
            kernel,
            dim=(num_atoms, order**3),
            inputs=[positions, charges, cell_inv_t, wp.int32(order), mesh],
            outputs=[output, forces],
            device=device,
        )


def batch_spline_spread(
    positions: wp.array,
    values: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Spread values from atoms to batched mesh using B-spline interpolation.

    Framework-agnostic launcher for batched spline spread.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions for all systems.
    values : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Values to spread.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    cell_inv_t : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (B, nx, ny, nz), dtype=wp.float32 or wp.float64
        OUTPUT: Batched mesh to accumulate values. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _batch_bspline_spread_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, values, batch_idx, cell_inv_t, wp.int32(order)],
        outputs=[mesh],
        device=device,
    )


def batch_spline_gather(
    positions: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    output: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Gather values from batched mesh to atoms using B-spline interpolation.

    Framework-agnostic launcher for batched spline gather.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions for all systems.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    cell_inv_t : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (B, nx, ny, nz), dtype=wp.float32 or wp.float64
        Batched mesh to interpolate from.
    output : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Interpolated values per atom. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _batch_bspline_gather_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, batch_idx, cell_inv_t, wp.int32(order), mesh],
        outputs=[output],
        device=device,
    )


def batch_spline_gather_vec3(
    positions: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    output: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Gather charge-weighted vector values from batched mesh using B-splines.

    Framework-agnostic launcher for batched vec3 spline gather.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions for all systems.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    cell_inv_t : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (B, nx, ny, nz), dtype=wp.vec3f or wp.vec3d
        Batched vector mesh to interpolate from.
    output : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Charge-weighted interpolated vectors. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _batch_bspline_gather_vec3_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, charges, batch_idx, cell_inv_t, wp.int32(order), mesh],
        outputs=[output],
        device=device,
    )


def batch_spline_gather_gradient(
    positions: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    forces: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Compute forces using B-spline gradient interpolation from batched mesh.

    Framework-agnostic launcher for batched spline gradient gather.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions for all systems.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    cell_inv_t : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system transpose of inverse cell matrix.
    order : int
        B-spline order (1-4).
    mesh : wp.array, shape (B, nx, ny, nz), dtype=wp.float32 or wp.float64
        Batched potential mesh.
    forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Forces per atom. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = positions.shape[0]
    num_points = order**3

    kernel = _batch_bspline_gather_gradient_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_atoms, num_points),
        inputs=[positions, charges, batch_idx, cell_inv_t, wp.int32(order), mesh],
        outputs=[forces],
        device=device,
    )


def batch_spline_cell_inv_t_grad(
    forces: wp.array,
    positions: wp.array,
    batch_idx: wp.array,
    cell: wp.array,
    grad_cell_inv_t: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Batched launcher for ``_batch_spline_cell_inv_t_grad_kernel``.

    ``grad_cell_inv_t`` must be zero-initialized (shape (B, 3, 3)).
    """
    kernel = _batch_spline_cell_inv_t_grad_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=positions.shape[0],
        inputs=[forces, positions, batch_idx, cell],
        outputs=[grad_cell_inv_t],
        device=device,
    )


def batch_spline_cell_inv_t_grad_backward(
    h: wp.array,
    forces: wp.array,
    positions: wp.array,
    batch_idx: wp.array,
    cell: wp.array,
    grad_forces: wp.array,
    grad_positions: wp.array,
    grad_cell: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Batched launcher for ``_batch_spline_cell_inv_t_grad_backward_kernel``."""
    kernel = _batch_spline_cell_inv_t_grad_backward_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=positions.shape[0],
        inputs=[h, forces, positions, batch_idx, cell],
        outputs=[grad_forces, grad_positions, grad_cell],
        device=device,
    )


def batch_spline_spread_gradient_weights(
    positions: wp.array,
    per_atom_vec: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Batched launcher for ``_batch_bspline_spread_gradient_weights_kernel``.

    ``mesh`` output must be zero-initialized.
    """
    kernel = _batch_bspline_spread_gradient_weights_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(positions.shape[0], order**3),
        inputs=[positions, per_atom_vec, batch_idx, cell_inv_t, wp.int32(order)],
        outputs=[mesh],
        device=device,
    )


def batch_spline_gather_gradient_position_hessian(
    positions: wp.array,
    charges: wp.array,
    v_per_atom: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    grad_positions: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Batched launcher for the position-Hessian backward of
    ``_bspline_gather_gradient_kernel``. ``grad_positions`` zero-initialized."""
    kernel = _batch_bspline_gather_gradient_position_hessian_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(positions.shape[0], order**3),
        inputs=[
            positions, charges, v_per_atom, batch_idx, cell_inv_t,
            wp.int32(order), mesh,
        ],
        outputs=[grad_positions],
        device=device,
    )


###########################################################################################
########################### Module Exports #################################################
###########################################################################################


__all__ = [
    # Warp functions (@wp.func)
    "bspline_weight",
    "bspline_derivative",
    "bspline_second_derivative",
    "bspline_weight_3d",
    "bspline_weight_gradient_3d",
    "bspline_weight_hessian_dot_vec3",
    "compute_fractional_coords",
    "bspline_grid_offset",
    "wrap_grid_index",
    # Warp kernels (single-system, scalar)
    "_bspline_weight_kernel",
    "_bspline_spread_kernel",
    "_bspline_gather_kernel",
    "_bspline_gather_vec3_kernel",
    "_bspline_gather_gradient_kernel",
    "_spline_cell_inv_t_grad_kernel",
    "_spline_cell_inv_t_grad_backward_kernel",
    "_bspline_spread_gradient_weights_kernel",
    "_bspline_gather_gradient_position_hessian_kernel",
    # Warp kernels (batch, scalar)
    "_batch_bspline_spread_kernel",
    "_batch_bspline_gather_kernel",
    "_batch_bspline_gather_vec3_kernel",
    "_batch_bspline_gather_gradient_kernel",
    "_batch_spline_cell_inv_t_grad_kernel",
    "_batch_spline_cell_inv_t_grad_backward_kernel",
    "_batch_bspline_spread_gradient_weights_kernel",
    "_batch_bspline_gather_gradient_position_hessian_kernel",
    # Warp kernels (single-system, multi-channel)
    "_bspline_spread_channels_kernel",
    "_bspline_gather_channels_kernel",
    # Warp kernels (batch, multi-channel)
    "_batch_bspline_spread_channels_kernel",
    "_batch_bspline_gather_channels_kernel",
    # Kernel overloads
    "_bspline_weight_kernel_overload",
    "_bspline_spread_kernel_overload",
    "_bspline_gather_kernel_overload",
    "_bspline_gather_vec3_kernel_overload",
    "_bspline_gather_gradient_kernel_overload",
    "_spline_cell_inv_t_grad_kernel_overload",
    "_spline_cell_inv_t_grad_backward_kernel_overload",
    "_bspline_spread_gradient_weights_kernel_overload",
    "_bspline_gather_gradient_position_hessian_kernel_overload",
    "_batch_bspline_spread_kernel_overload",
    "_batch_bspline_gather_kernel_overload",
    "_batch_bspline_gather_vec3_kernel_overload",
    "_batch_bspline_gather_gradient_kernel_overload",
    "_batch_spline_cell_inv_t_grad_kernel_overload",
    "_batch_spline_cell_inv_t_grad_backward_kernel_overload",
    "_batch_bspline_spread_gradient_weights_kernel_overload",
    "_batch_bspline_gather_gradient_position_hessian_kernel_overload",
    "_bspline_spread_channels_kernel_overload",
    "_bspline_gather_channels_kernel_overload",
    "_batch_bspline_spread_channels_kernel_overload",
    "_batch_bspline_gather_channels_kernel_overload",
    # Warp launchers
    "bspline_weight_launcher",
    "spline_spread",
    "spline_gather",
    "spline_gather_vec3",
    "spline_gather_gradient",
    "spline_cell_inv_t_grad",
    "spline_cell_inv_t_grad_backward",
    "spline_spread_gradient_weights",
    "spline_gather_gradient_position_hessian",
    "batch_spline_spread",
    "batch_spline_gather",
    "batch_spline_gather_vec3",
    "batch_spline_gather_gradient",
    "batch_spline_cell_inv_t_grad",
    "batch_spline_cell_inv_t_grad_backward",
    "batch_spline_spread_gradient_weights",
    "batch_spline_gather_gradient_position_hessian",
]
