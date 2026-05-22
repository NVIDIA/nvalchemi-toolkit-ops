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
B-Spline Interpolation PyTorch Bindings
=======================================

This module provides PyTorch bindings for B-spline interpolation functions
used in mesh-based calculations (e.g., Particle Mesh Ewald).

This module wraps the framework-agnostic Warp kernels from
``nvalchemiops.math.spline`` with PyTorch custom operators.

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
   value[atom] = sum_g mesh[g] * weight(atom, g)

3. GATHER_VEC3: Collect 3D vector field values at atom positions
   vector[atom] = sum_g mesh[g] * weight(atom, g)

4. GATHER_GRADIENT: Collect mesh values with weight gradients (forces)
   grad[atom] = sum_g mesh[g] * grad_weight(atom, g)

5. SPREAD_CHANNELS: Scatter multi-channel values (e.g., multipoles) to mesh
   mesh[c, g] += values[atom, c] * weight(atom, g)

6. GATHER_CHANNELS: Collect multi-channel values from mesh
   values[atom, c] = sum_g mesh[c, g] * weight(atom, g)

7. DECONVOLUTION: Correct B-spline approximation in Fourier space
   Used in FFT-based methods to remove B-spline smoothing artifacts.

USAGE
=====

Single-system:
    from nvalchemiops.torch.spline import spline_spread, spline_gather, spline_gather_gradient

    # Spread charges to mesh
    mesh = spline_spread(positions, charges, cell, mesh_dims, spline_order=4)

    # Gather potential from mesh
    potentials = spline_gather(positions, potential_mesh, cell, spline_order=4)

    # Gather forces
    forces = spline_gather_gradient(positions, charges, potential_mesh, cell, spline_order=4)

Multi-channel (multipoles):
    from nvalchemiops.torch.spline import spline_spread_channels, spline_gather_channels

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
    from nvalchemiops.torch.spline import compute_bspline_deconvolution

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

from nvalchemiops.math.spline import (
    # Kernel overloads (needed for custom ops)
    _batch_bspline_gather_channels_kernel_overload,
    _batch_bspline_gather_gradient_kernel_overload,
    _batch_bspline_gather_kernel_overload,
    _batch_bspline_gather_vec3_kernel_overload,
    _batch_bspline_spread_channels_kernel_overload,
    _batch_bspline_spread_kernel_overload,
    _bspline_gather_channels_kernel_overload,
    _bspline_gather_gradient_kernel_overload,
    _bspline_gather_kernel_overload,
    _bspline_gather_vec3_kernel_overload,
    _bspline_gather_with_force_kernel_overload,
    _PER_ORDER_BATCH_GATHER_WITH_FORCE_KERNELS,
    _PER_ORDER_BATCH_SPREAD_KERNELS,
    _PER_ORDER_GATHER_WITH_FORCE_KERNELS,
    _PER_ORDER_SPREAD_KERNELS,
    _bspline_spread_channels_kernel_overload,
    _bspline_spread_kernel_overload,
    _bspline_weight_kernel_overload,
    batch_spline_cell_inv_t_grad as _batch_spline_cell_inv_t_grad_launch,
    batch_spline_cell_inv_t_grad_backward as _batch_spline_cell_inv_t_grad_bwd_launch,
    batch_spline_gather_gradient_position_hessian as _batch_spline_pos_hessian_launch,
    batch_spline_spread_gradient_weights as _batch_spline_spread_grad_weights_launch,
    spline_cell_inv_t_grad as _spline_cell_inv_t_grad_launch,
    spline_cell_inv_t_grad_backward as _spline_cell_inv_t_grad_bwd_launch,
    spline_gather_gradient_position_hessian as _spline_pos_hessian_launch,
    spline_spread_gradient_weights as _spline_spread_grad_weights_launch,
)
from nvalchemiops.torch.autograd import (
    OutputSpec,
    WarpAutogradContextManager,
    attach_for_backward,
    needs_grad,
    warp_custom_op,
    warp_from_torch,
)
from nvalchemiops.torch.interactions.electrostatics._warp_op_helpers import (
    register_warp_op_chain,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

###########################################################################################
########################### Internal Custom Ops: _spline_* (Single-System) #################
###########################################################################################


@warp_custom_op(
    name="alchemiops::_spline_weight",
    outputs=[
        OutputSpec(
            "weights",
            wp.array(dtype=Any, ndim=1),
            lambda u, *_: (u.shape[0],),
        ),
    ],
    grad_arrays=[
        "weights",
        "u",
    ],
)
def _spline_weight(
    u: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Internal: B-spline weight calculation with dtype flexibility."""
    device = wp.device_from_torch(u.device)
    input_dtype = u.dtype
    wp_dtype = get_wp_dtype(input_dtype)

    num_points = u.shape[0]
    needs_grad_flag = needs_grad(u)

    wp_u = warp_from_torch(u, wp_dtype, requires_grad=needs_grad_flag)

    weights = torch.zeros_like(u)
    wp_weights = warp_from_torch(weights, wp_dtype, requires_grad=needs_grad_flag)

    kernel = _bspline_weight_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=num_points,
            inputs=[wp_u, wp.int32(spline_order)],
            outputs=[wp_weights],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            weights,
            tape=tape,
            weights=wp_weights,
            u=wp_u,
        )
    return weights


###########################################################################################
###### Explicit torch.library backward chain for single-system spread/gather ###############
###########################################################################################
# Explicit ``register_warp_op_chain + register_autograd`` wiring for
# ``_spline_spread`` and ``_spline_gather``. The two operations are
# mathematical adjoints, so the backward of one is the forward of the other:
#
#   spread:  mesh[i,j,k] = Σ_n q_n · B(x_n - r_ijk)
#   gather:  pot[n]      = Σ_{i,j,k} mesh[i,j,k] · B(x_n - r_ijk)
#
# Spread backward gives:
#   grad_values[n]   = Σ_{i,j,k} grad_mesh[i,j,k] · B(x_n - r_ijk)   (gather of grad_mesh)
#   grad_positions[n,a] = -force[n,a]  where force = gather_gradient(grad_mesh)
#   grad_cell_inv_t[a,b] = Σ_n positions[n,b] · (q_n · Σ grad_mesh · grad_frac[a] at atom n)
#                        = (qgf.T @ positions)  with qgf = -cell @ force
#
# The Cartesian "force" returned by gather_gradient is ``-q · cell_inv_t.T · qgf``,
# so we recover qgf as ``-(force @ cell.T)`` via a single matmul.


def _scoped_warp_stream(device: torch.device):
    """Bind Warp's current stream to PyTorch's current CUDA stream.

    Required for ``torch.cuda.graph`` capture: without this, warp kernel
    launches go to warp's default stream (separate from torch's capturing
    stream) and aren't recorded into the graph, so replay silently no-ops.
    """
    if device.type != "cuda":
        from contextlib import nullcontext
        return nullcontext()
    torch_stream = torch.cuda.current_stream(device)
    return wp.ScopedStream(wp.stream_from_torch(torch_stream))


def _wp_from_torch(tensor: torch.Tensor, dtype):
    """Wrap a torch tensor as a Warp array WITHOUT allocating a shadow
    gradient array.

    Default ``wp.from_torch`` inherits ``requires_grad`` from the torch
    tensor and, if True, calls ``wp_alloc_device_async`` to allocate a
    gradient buffer for Warp's tape autograd. That allocation is not
    permitted inside ``torch.cuda.graph`` capture and causes
    ``cudaErrorStreamCaptureInvalidated``. Since our autograd.Functions
    handle backward explicitly, we never need Warp's shadow gradient.
    """
    return wp.from_torch(tensor, dtype=dtype, requires_grad=False)


def _spread_forward_launch(
    positions: torch.Tensor,
    values: torch.Tensor,
    cell_inv_t: torch.Tensor,
    mesh_dims: tuple[int, int, int],
    spline_order: int,
) -> torch.Tensor:
    """Single-system spline spread forward launch. No autograd plumbing."""
    from nvalchemiops.math.spline import spline_spread as _spread_launch

    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    mesh_nx, mesh_ny, mesh_nz = mesh_dims
    mesh = torch.zeros(
        (mesh_nx, mesh_ny, mesh_nz), device=positions.device, dtype=input_dtype
    )

    wp_positions = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_values = _wp_from_torch(values.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_cell_inv_t = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat_dtype)
    wp_mesh = _wp_from_torch(mesh, dtype=wp_dtype)

    # Per-order specialized spread kernel: one-thread-per-atom layout
    # with fully-unrolled order^3 stencil and 1D weights in registers.
    # Substantially faster than the generic kernel at order=6, 128k atoms.
    per_order_kernel = _PER_ORDER_SPREAD_KERNELS[wp_dtype].get(spline_order)

    with _scoped_warp_stream(positions.device):
        if per_order_kernel is not None:
            wp.launch(
                per_order_kernel,
                dim=positions.shape[0],
                inputs=[wp_positions, wp_values, wp_cell_inv_t],
                outputs=[wp_mesh],
                device=device,
            )
        else:
            _spread_launch(
                wp_positions, wp_values, wp_cell_inv_t, spline_order, wp_mesh,
                wp_dtype=wp_dtype, device=device,
            )
    return mesh


def _gather_forward_launch(
    positions: torch.Tensor,
    mesh: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Single-system spline gather forward launch. No autograd plumbing."""
    from nvalchemiops.math.spline import spline_gather as _gather_launch

    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    output = torch.zeros(num_atoms, device=positions.device, dtype=input_dtype)

    wp_positions = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_mesh = _wp_from_torch(mesh.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_cell_inv_t = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat_dtype)
    wp_output = _wp_from_torch(output, dtype=wp_dtype)

    with _scoped_warp_stream(positions.device):
        _gather_launch(
            wp_positions, wp_cell_inv_t, spline_order, wp_mesh, wp_output,
            wp_dtype=wp_dtype, device=device,
        )
    return output


def _gather_gradient_forward_launch(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Single-system spline gather-gradient forward launch.

    Returns Cartesian "force" ``-q_n · Σ mesh · d(W)/d(position)`` per atom.
    """
    from nvalchemiops.math.spline import (
        spline_gather_gradient as _grad_launch,
    )

    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    forces = torch.zeros(
        (num_atoms, 3), device=positions.device, dtype=input_dtype
    )

    wp_positions = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_charges = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_mesh = _wp_from_torch(mesh.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_cell_inv_t = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat_dtype)
    wp_forces = _wp_from_torch(forces, dtype=wp_vec_dtype)

    with _scoped_warp_stream(positions.device):
        _grad_launch(
            wp_positions, wp_charges, wp_cell_inv_t, spline_order, wp_mesh, wp_forces,
            wp_dtype=wp_dtype, device=device,
        )
    return forces


def _spread_gradient_weights_launch(
    positions: torch.Tensor,
    per_atom_vec: torch.Tensor,
    cell_inv_t: torch.Tensor,
    mesh_dims: tuple[int, int, int],
    spline_order: int,
) -> torch.Tensor:
    """Single-system ``_bspline_spread_gradient_weights_kernel`` launch."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    mesh_nx, mesh_ny, mesh_nz = mesh_dims
    mesh = torch.zeros(
        (mesh_nx, mesh_ny, mesh_nz), device=positions.device, dtype=input_dtype
    )

    wp_positions = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_vec = _wp_from_torch(per_atom_vec.contiguous(), dtype=wp_vec_dtype)
    wp_cell_inv_t = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat_dtype)
    wp_mesh = _wp_from_torch(mesh, dtype=wp_dtype)

    with _scoped_warp_stream(positions.device):
        _spline_spread_grad_weights_launch(
            wp_positions, wp_vec, wp_cell_inv_t, spline_order, wp_mesh,
            wp_dtype=wp_dtype, device=device,
        )
    return mesh


def _pos_hessian_forward_launch(
    positions: torch.Tensor,
    charges: torch.Tensor,
    v_per_atom: torch.Tensor,
    cell_inv_t: torch.Tensor,
    mesh: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Single-system B-spline position-Hessian launch.

    Implements ``grad_pos[n] = Σ_g -q[n] · mesh[g] · ∇²W_frac(x_n, g)``
    used inside the gather_gradient / gather_with_force backward chains.
    """
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    grad_positions = torch.zeros_like(positions)
    wp_pos = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_chg = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_v = _wp_from_torch(v_per_atom.contiguous(), dtype=wp_vec_dtype)
    wp_cit = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat_dtype)
    wp_mesh = _wp_from_torch(mesh.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_grad_pos = _wp_from_torch(grad_positions, dtype=wp_vec_dtype)

    with _scoped_warp_stream(positions.device):
        _spline_pos_hessian_launch(
            wp_pos, wp_chg, wp_v, wp_cit, spline_order, wp_mesh, wp_grad_pos,
            wp_dtype=wp_dtype, device=device,
        )
    return grad_positions


# Register the two raw warp helpers used inside spline backward chains as
# forward-only custom_ops so that under torch.compile fullgraph=True, AOT
# autograd traces through gather_gradient.backward → these helpers cleanly.
# No register_autograd needed — third-order autograd through them is not
# exercised by any workflow check; AOT treats them as opaque single nodes.
register_warp_op_chain(
    name="nvalchemiops::spline_spread_gradient_weights",
    forward=_spread_gradient_weights_launch,
    forward_fake=lambda positions, per_atom_vec, cell_inv_t, mesh_dims, spline_order: (
        torch.empty(
            (mesh_dims[0], mesh_dims[1], mesh_dims[2]),
            dtype=positions.dtype, device=positions.device,
        )
    ),
)
register_warp_op_chain(
    name="nvalchemiops::spline_pos_hessian",
    forward=_pos_hessian_forward_launch,
    # Output shape == positions shape, so the default ``empty_like(positions)``
    # fake is correct.
)


# Single-system gather_gradient: forward returns Cartesian "force" per atom.
# Backward chains position-Hessian + spread-gradient-weights launches;
# grad_cell_inv_t deferred since the cell chain flows through spread/gather.
register_warp_op_chain(
    name="nvalchemiops::spline_gather_gradient",
    forward=_gather_gradient_forward_launch,
    # No backward op registered — composed manually via register_autograd below.
)


def _spline_gather_gradient_setup_ctx(ctx, inputs, output):
    positions, charges, mesh, cell_inv_t, spline_order = inputs
    ctx.save_for_backward(positions, charges, mesh, cell_inv_t)
    ctx.spline_order = spline_order
    ctx.mesh_dims = (mesh.shape[-3], mesh.shape[-2], mesh.shape[-1])
    ctx.needs_pos = positions.requires_grad
    ctx.needs_chg = charges.requires_grad
    ctx.needs_mesh = mesh.requires_grad


def _spline_gather_gradient_backward_chain(ctx, grad_force):
    positions, charges, mesh, cell_inv_t = ctx.saved_tensors
    order = ctx.spline_order

    if grad_force is None:
        return None, None, None, None, None
    grad_force_c = grad_force.contiguous()

    # grad_positions via the B-spline position-Hessian path.
    if ctx.needs_pos:
        v_per_atom = torch.bmm(
            cell_inv_t[0].unsqueeze(0).expand(positions.shape[0], -1, -1),
            grad_force_c.unsqueeze(-1),
        ).squeeze(-1)
        grad_positions = torch.ops.nvalchemiops.spline_pos_hessian(
            positions, charges, v_per_atom, cell_inv_t, mesh, order,
        )
    else:
        grad_positions = None

    # grad_charges via recursive call (q=1 path).
    if ctx.needs_chg:
        ones = torch.ones_like(charges, dtype=positions.dtype)
        force_per_unit_q = torch.ops.nvalchemiops.spline_gather_gradient(
            positions, ones, mesh, cell_inv_t, order,
        )
        grad_charges = (grad_force_c * force_per_unit_q).sum(dim=-1)
    else:
        grad_charges = None

    # grad_mesh via spread-with-gradient-weights.
    if ctx.needs_mesh:
        v = grad_force_c @ cell_inv_t[0].transpose(-1, -2)
        per_atom_vec = -(charges.to(positions.dtype).unsqueeze(-1) * v)
        grad_mesh = torch.ops.nvalchemiops.spline_spread_gradient_weights(
            positions, per_atom_vec, cell_inv_t, ctx.mesh_dims, order,
        )
    else:
        grad_mesh = None

    return grad_positions, grad_charges, grad_mesh, None, None


torch.library.register_autograd(
    "nvalchemiops::spline_gather_gradient",
    _spline_gather_gradient_backward_chain,
    setup_context=_spline_gather_gradient_setup_ctx,
)


def _cell_inv_t_grad_forward_launch(
    forces: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Pure forward launch (no autograd plumbing)."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    grad = torch.zeros_like(cell)
    wp_forces = _wp_from_torch(forces.contiguous(), dtype=wp_vec_dtype)
    wp_positions = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_cell = _wp_from_torch(cell.contiguous(), dtype=wp_mat_dtype)
    wp_grad = _wp_from_torch(grad, dtype=wp_mat_dtype)

    with _scoped_warp_stream(positions.device):
        _spline_cell_inv_t_grad_launch(
            wp_forces, wp_positions, wp_cell, wp_grad,
            wp_dtype=wp_dtype, device=device,
        )
    return grad


def _cell_inv_t_grad_backward_launch(
    h: torch.Tensor,
    forces: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure backward launch — returns (grad_forces, grad_positions, grad_cell)."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    grad_forces = torch.empty_like(forces)
    grad_positions = torch.empty_like(positions)
    grad_cell = torch.zeros_like(cell)

    wp_h = _wp_from_torch(h.contiguous(), dtype=wp_mat_dtype)
    wp_forces = _wp_from_torch(forces.contiguous(), dtype=wp_vec_dtype)
    wp_positions = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_cell = _wp_from_torch(cell.contiguous(), dtype=wp_mat_dtype)
    wp_g_forces = _wp_from_torch(grad_forces, dtype=wp_vec_dtype)
    wp_g_positions = _wp_from_torch(grad_positions, dtype=wp_vec_dtype)
    wp_g_cell = _wp_from_torch(grad_cell, dtype=wp_mat_dtype)

    with _scoped_warp_stream(positions.device):
        _spline_cell_inv_t_grad_bwd_launch(
            wp_h, wp_forces, wp_positions, wp_cell,
            wp_g_forces, wp_g_positions, wp_g_cell,
            wp_dtype=wp_dtype, device=device,
        )
    return grad_forces, grad_positions, grad_cell


# Single-system cell_inv_t_grad: both forward and backward are explicit
# warp kernels — full factory chain (no double-backward kernel; the
# launchers themselves support a single round of autograd which is all
# the workflow needs).
register_warp_op_chain(
    name="nvalchemiops::spline_cell_inv_t_grad",
    forward=_cell_inv_t_grad_forward_launch,
    forward_fake=lambda forces, positions, cell: torch.empty_like(cell),
    backward=_cell_inv_t_grad_backward_launch,
    # Backward returns (grad_forces, grad_positions, grad_cell) matching
    # forward inputs (forces@0, positions@1, cell@2) in order.
    diff_input_positions=(0, 1, 2),
    n_forward_inputs=3,
)


def _cell_inv_t_grad_from_force(
    forces: torch.Tensor,
    positions: torch.Tensor,
    cell_inv_t: torch.Tensor,
) -> torch.Tensor:
    """Compute ``grad_cell_inv_t`` (shape ``(1, 3, 3)``) via the registered
    ``spline_cell_inv_t_grad`` op. The 3x3 cell inverse stays in torch
    (preprocessing)."""
    cell = torch.linalg.inv(cell_inv_t.transpose(-1, -2)).contiguous()  # (1, 3, 3)
    return torch.ops.nvalchemiops.spline_cell_inv_t_grad(forces, positions, cell)


# Single-system spread + gather. These are mathematical adjoints, so each
# one's backward composes the OTHER's forward. We register both as forward-
# only custom_ops, then wire register_autograd manually with the composed
# backward chains (routed via torch.ops.* so they're compile-traceable).
register_warp_op_chain(
    name="nvalchemiops::spline_spread",
    forward=_spread_forward_launch,
    forward_fake=lambda positions, values, cell_inv_t, mesh_dims, spline_order: (
        torch.empty(
            (mesh_dims[0], mesh_dims[1], mesh_dims[2]),
            dtype=positions.dtype, device=positions.device,
        )
    ),
)


def _spline_spread_setup_ctx(ctx, inputs, output):
    positions, values, cell_inv_t, mesh_dims, spline_order = inputs
    ctx.save_for_backward(positions, values, cell_inv_t)
    ctx.spline_order = spline_order
    ctx.mesh_dims = tuple(mesh_dims)
    ctx.needs_pos = positions.requires_grad
    ctx.needs_val = values.requires_grad
    ctx.needs_cell = cell_inv_t.requires_grad


def _spline_spread_backward_chain(ctx, grad_mesh):
    positions, values, cell_inv_t = ctx.saved_tensors
    order = ctx.spline_order
    grad_mesh_c = grad_mesh.contiguous()

    grad_values = (
        torch.ops.nvalchemiops.spline_gather(
            positions, grad_mesh_c, cell_inv_t, order,
        ) if ctx.needs_val else None
    )

    if ctx.needs_pos or ctx.needs_cell:
        forces = torch.ops.nvalchemiops.spline_gather_gradient(
            positions, values, grad_mesh_c, cell_inv_t, order,
        )
        grad_positions = -forces if ctx.needs_pos else None
        grad_cell_inv_t = (
            _cell_inv_t_grad_from_force(forces, positions, cell_inv_t)
            if ctx.needs_cell else None
        )
    else:
        grad_positions = None
        grad_cell_inv_t = None

    return grad_positions, grad_values, grad_cell_inv_t, None, None


torch.library.register_autograd(
    "nvalchemiops::spline_spread",
    _spline_spread_backward_chain,
    setup_context=_spline_spread_setup_ctx,
)


register_warp_op_chain(
    name="nvalchemiops::spline_gather",
    forward=_gather_forward_launch,
    forward_fake=lambda positions, mesh, cell_inv_t, spline_order: torch.empty(
        positions.shape[0], dtype=positions.dtype, device=positions.device,
    ),
)


def _spline_gather_setup_ctx(ctx, inputs, output):
    positions, mesh, cell_inv_t, spline_order = inputs
    ctx.save_for_backward(positions, mesh, cell_inv_t)
    ctx.spline_order = spline_order
    ctx.mesh_dims = (mesh.shape[-3], mesh.shape[-2], mesh.shape[-1])
    ctx.needs_pos = positions.requires_grad
    ctx.needs_mesh = mesh.requires_grad
    ctx.needs_cell = cell_inv_t.requires_grad


def _spline_gather_backward_chain(ctx, grad_potentials):
    positions, mesh, cell_inv_t = ctx.saved_tensors
    order = ctx.spline_order
    grad_pot_c = grad_potentials.contiguous()

    grad_mesh = (
        torch.ops.nvalchemiops.spline_spread(
            positions, grad_pot_c, cell_inv_t, ctx.mesh_dims, order,
        ) if ctx.needs_mesh else None
    )

    if ctx.needs_pos or ctx.needs_cell:
        forces = torch.ops.nvalchemiops.spline_gather_gradient(
            positions, grad_pot_c, mesh, cell_inv_t, order,
        )
        grad_positions = -forces if ctx.needs_pos else None
        grad_cell_inv_t = (
            _cell_inv_t_grad_from_force(forces, positions, cell_inv_t)
            if ctx.needs_cell else None
        )
    else:
        grad_positions = None
        grad_cell_inv_t = None

    return grad_positions, grad_mesh, grad_cell_inv_t, None


torch.library.register_autograd(
    "nvalchemiops::spline_gather",
    _spline_gather_backward_chain,
    setup_context=_spline_gather_setup_ctx,
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
    """Internal: single-system spline spread (registered custom op)."""
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    if cell_inv_t is None:
        cell_inv = torch.linalg.inv_ex(cell)[0]
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
    return torch.ops.nvalchemiops.spline_spread(
        positions, values.to(positions.dtype), cell_inv_t,
        [mesh_nx, mesh_ny, mesh_nz], spline_order,
    )


def _spline_gather(
    positions: torch.Tensor,
    mesh: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal: single-system spline gather (registered custom op)."""
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
    return torch.ops.nvalchemiops.spline_gather(
        positions, mesh.to(positions.dtype), cell_inv_t, spline_order,
    )


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


def _spline_gather_gradient(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal: single-system spline gather-gradient (registered custom op)."""
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
    return torch.ops.nvalchemiops.spline_gather_gradient(
        positions, charges.to(positions.dtype), mesh.to(positions.dtype),
        cell_inv_t, spline_order,
    )


def _gather_with_force_forward_launch(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-system fused gather + gather-gradient forward launch.

    Selects the per-order specialized kernel for orders 2-6 (~9x faster
    than the generic runtime-order kernel) when available; falls back to
    the generic kernel otherwise.
    """
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    num_points = spline_order**3
    potential = torch.zeros(num_atoms, device=positions.device, dtype=input_dtype)
    forces = torch.zeros((num_atoms, 3), device=positions.device, dtype=input_dtype)

    wp_pos = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_chg = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_cit = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat_dtype)
    wp_mesh = _wp_from_torch(mesh.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_pot = _wp_from_torch(potential, dtype=wp_dtype)
    wp_forces = _wp_from_torch(forces, dtype=wp_vec_dtype)

    per_order_kernel = _PER_ORDER_GATHER_WITH_FORCE_KERNELS[wp_dtype].get(spline_order)

    with _scoped_warp_stream(positions.device):
        if per_order_kernel is not None:
            wp.launch(
                per_order_kernel,
                dim=num_atoms,
                inputs=[wp_pos, wp_chg, wp_cit, wp_mesh],
                outputs=[wp_pot, wp_forces],
                device=device,
            )
        else:
            kernel = _bspline_gather_with_force_kernel_overload[wp_dtype]
            wp.launch(
                kernel,
                dim=(num_atoms, num_points),
                inputs=[wp_pos, wp_chg, wp_cit, wp.int32(spline_order), wp_mesh],
                outputs=[wp_pot, wp_forces],
                device=device,
            )
    return potential, forces


# Single-system fused gather + force. Backward has two chains: grad_potential
# flows through the gather chain (spread + gather_gradient + cell_inv_t_grad)
# and grad_forces flows through the gather_gradient chain (position-Hessian,
# per-unit-q gather_gradient, spread-with-gradient-weights). Forward returns
# (potential, forces) — arity 2.
register_warp_op_chain(
    name="nvalchemiops::spline_gather_with_force",
    forward=_gather_with_force_forward_launch,
    forward_return_arity=2,
    forward_fake=lambda pos, *_: (
        torch.empty(pos.shape[0], dtype=pos.dtype, device=pos.device),
        torch.empty((pos.shape[0], 3), dtype=pos.dtype, device=pos.device),
    ),
)


def _spline_gather_with_force_setup_ctx(ctx, inputs, output):
    positions, charges, mesh, cell_inv_t, spline_order = inputs
    ctx.save_for_backward(positions, charges, mesh, cell_inv_t)
    ctx.spline_order = spline_order
    ctx.mesh_dims = (mesh.shape[-3], mesh.shape[-2], mesh.shape[-1])
    ctx.needs_pos = positions.requires_grad
    ctx.needs_chg = charges.requires_grad
    ctx.needs_mesh = mesh.requires_grad
    ctx.needs_cell = cell_inv_t.requires_grad


def _spline_gather_with_force_backward_chain(ctx, grad_potential, grad_forces):
    positions, charges, mesh, cell_inv_t = ctx.saved_tensors
    order = ctx.spline_order
    grad_pos = grad_chg = grad_mesh = grad_cell_inv_t = None

    def _add(target, contrib):
        return contrib if target is None else target + contrib

    # gather chain (grad_potential → grads)
    if grad_potential is not None:
        gp = grad_potential.contiguous()
        if ctx.needs_pos or ctx.needs_cell:
            forces_g = torch.ops.nvalchemiops.spline_gather_gradient(
                positions, gp, mesh, cell_inv_t, order,
            )
            if ctx.needs_pos:
                grad_pos = _add(grad_pos, -forces_g)
            if ctx.needs_cell:
                grad_cell_inv_t = _add(
                    grad_cell_inv_t,
                    _cell_inv_t_grad_from_force(forces_g, positions, cell_inv_t),
                )
        if ctx.needs_mesh:
            grad_mesh = _add(
                grad_mesh,
                torch.ops.nvalchemiops.spline_spread(
                    positions, gp, cell_inv_t, ctx.mesh_dims, order,
                ),
            )

    # gather_gradient chain (grad_forces → grads)
    if grad_forces is not None:
        gf = grad_forces.contiguous()

        if ctx.needs_chg:
            ones = torch.ones_like(charges, dtype=positions.dtype)
            force_per_unit_q = torch.ops.nvalchemiops.spline_gather_gradient(
                positions, ones, mesh, cell_inv_t, order,
            )
            grad_chg = _add(grad_chg, (gf * force_per_unit_q).sum(dim=-1))

        if ctx.needs_mesh:
            v = gf @ cell_inv_t[0].transpose(-1, -2)
            per_atom_vec = -(charges.to(positions.dtype).unsqueeze(-1) * v)
            grad_mesh = _add(
                grad_mesh,
                torch.ops.nvalchemiops.spline_spread_gradient_weights(
                    positions, per_atom_vec, cell_inv_t, ctx.mesh_dims, order,
                ),
            )

        if ctx.needs_pos:
            v_per_atom = torch.bmm(
                cell_inv_t[0].unsqueeze(0).expand(positions.shape[0], -1, -1),
                gf.unsqueeze(-1),
            ).squeeze(-1)
            pos_hess = torch.ops.nvalchemiops.spline_pos_hessian(
                positions, charges, v_per_atom, cell_inv_t, mesh, order,
            )
            grad_pos = _add(grad_pos, pos_hess)

    return grad_pos, grad_chg, grad_mesh, grad_cell_inv_t, None


torch.library.register_autograd(
    "nvalchemiops::spline_gather_with_force",
    _spline_gather_with_force_backward_chain,
    setup_context=_spline_gather_with_force_setup_ctx,
)


def _spline_gather_with_force(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: single-system fused gather + gather-gradient (registered op).

    Returns ``(potential, forces)``:
      - ``potential[atom] = Σ_g mesh[g] * w(atom, g)``           (raw potential)
      - ``forces[atom] = -q_atom * Σ_g mesh[g] * Cell^{-T} ∇w`` (Cartesian force)
    """
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
    return torch.ops.nvalchemiops.spline_gather_with_force(
        positions, charges.to(positions.dtype), mesh.to(positions.dtype),
        cell_inv_t, spline_order,
    )


###########################################################################################
########################### Internal Custom Ops: _batch_spline_* (Batch) ###################
###########################################################################################


###########################################################################################
###### Explicit torch.library backward chain for batched spread/gather #####################
###########################################################################################
# Same adjoint structure as the single-system case above, with batch_idx
# threading the per-system cell_inv_t through positions and forces. The
# cell_inv_t gradient is accumulated per system via index_add_.


def _batch_spread_forward_launch(
    positions: torch.Tensor,
    values: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    num_systems: int,
    mesh_dims: tuple[int, int, int],
    spline_order: int,
) -> torch.Tensor:
    """Batched spline spread forward launch. No autograd plumbing."""
    from nvalchemiops.math.spline import batch_spline_spread as _spread_launch

    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    mesh_nx, mesh_ny, mesh_nz = mesh_dims
    mesh = torch.zeros(
        (num_systems, mesh_nx, mesh_ny, mesh_nz),
        device=positions.device,
        dtype=input_dtype,
    )

    wp_positions = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_values = _wp_from_torch(values.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_batch_idx = _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32)
    wp_cell_inv_t = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat_dtype)
    wp_mesh = _wp_from_torch(mesh, dtype=wp_dtype)

    # Per-order specialized batch spread kernel.
    per_order_kernel = _PER_ORDER_BATCH_SPREAD_KERNELS[wp_dtype].get(spline_order)

    with _scoped_warp_stream(positions.device):
        if per_order_kernel is not None:
            wp.launch(
                per_order_kernel,
                dim=positions.shape[0],
                inputs=[wp_positions, wp_values, wp_batch_idx, wp_cell_inv_t],
                outputs=[wp_mesh],
                device=device,
            )
        else:
            _spread_launch(
                wp_positions, wp_values, wp_batch_idx, wp_cell_inv_t,
                spline_order, wp_mesh,
                wp_dtype=wp_dtype, device=device,
            )
    return mesh


def _batch_gather_forward_launch(
    positions: torch.Tensor,
    mesh: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Batched spline gather forward launch. No autograd plumbing."""
    from nvalchemiops.math.spline import batch_spline_gather as _gather_launch

    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    output = torch.zeros(num_atoms, device=positions.device, dtype=input_dtype)

    wp_positions = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_batch_idx = _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32)
    wp_mesh = _wp_from_torch(mesh.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_cell_inv_t = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat_dtype)
    wp_output = _wp_from_torch(output, dtype=wp_dtype)

    with _scoped_warp_stream(positions.device):
        _gather_launch(
            wp_positions, wp_batch_idx, wp_cell_inv_t, spline_order, wp_mesh, wp_output,
            wp_dtype=wp_dtype, device=device,
        )
    return output


def _batch_gather_gradient_forward_launch(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Batched spline gather-gradient forward launch.

    Returns Cartesian "force" per atom, with per-system cell_inv_t applied
    according to batch_idx.
    """
    from nvalchemiops.math.spline import (
        batch_spline_gather_gradient as _grad_launch,
    )

    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    forces = torch.zeros(
        (num_atoms, 3), device=positions.device, dtype=input_dtype
    )

    wp_positions = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_charges = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_batch_idx = _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32)
    wp_mesh = _wp_from_torch(mesh.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_cell_inv_t = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat_dtype)
    wp_forces = _wp_from_torch(forces, dtype=wp_vec_dtype)

    with _scoped_warp_stream(positions.device):
        _grad_launch(
            wp_positions, wp_charges, wp_batch_idx, wp_cell_inv_t, spline_order,
            wp_mesh, wp_forces, wp_dtype=wp_dtype, device=device,
        )
    return forces


def _batch_spread_gradient_weights_launch(
    positions: torch.Tensor,
    per_atom_vec: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    num_systems: int,
    mesh_dims: tuple[int, int, int],
    spline_order: int,
) -> torch.Tensor:
    """Batched spread-with-gradient-weights launcher."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    mesh_nx, mesh_ny, mesh_nz = mesh_dims
    mesh = torch.zeros(
        (num_systems, mesh_nx, mesh_ny, mesh_nz),
        device=positions.device, dtype=input_dtype,
    )

    wp_positions = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_vec = _wp_from_torch(per_atom_vec.contiguous(), dtype=wp_vec_dtype)
    wp_batch_idx = _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32)
    wp_cell_inv_t = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat_dtype)
    wp_mesh = _wp_from_torch(mesh, dtype=wp_dtype)

    with _scoped_warp_stream(positions.device):
        _batch_spline_spread_grad_weights_launch(
            wp_positions, wp_vec, wp_batch_idx, wp_cell_inv_t, spline_order, wp_mesh,
            wp_dtype=wp_dtype, device=device,
        )
    return mesh


def _batch_pos_hessian_forward_launch(
    positions: torch.Tensor,
    charges: torch.Tensor,
    v_per_atom: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    mesh: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Batched B-spline position-Hessian launch (mirror of the single-
    system variant). Per-system cell_inv_t is indexed via ``batch_idx``."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    grad_positions = torch.zeros_like(positions)
    wp_pos = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_chg = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_v = _wp_from_torch(v_per_atom.contiguous(), dtype=wp_vec_dtype)
    wp_bidx = _wp_from_torch(
        batch_idx.to(torch.int32).contiguous(), dtype=wp.int32,
    )
    wp_cit = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat_dtype)
    wp_mesh = _wp_from_torch(mesh.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_grad_pos = _wp_from_torch(grad_positions, dtype=wp_vec_dtype)

    with _scoped_warp_stream(positions.device):
        _batch_spline_pos_hessian_launch(
            wp_pos, wp_chg, wp_v, wp_bidx, wp_cit, spline_order,
            wp_mesh, wp_grad_pos,
            wp_dtype=wp_dtype, device=device,
        )
    return grad_positions


# Forward-only custom_ops for the two raw warp helpers used inside the
# batched spline backward chains (mirrors the single-system registrations
# above).
register_warp_op_chain(
    name="nvalchemiops::batch_spline_spread_gradient_weights",
    forward=_batch_spread_gradient_weights_launch,
    forward_fake=lambda positions, per_atom_vec, batch_idx, cell_inv_t,
                       num_systems, mesh_dims, spline_order: torch.empty(
        (num_systems, mesh_dims[0], mesh_dims[1], mesh_dims[2]),
        dtype=positions.dtype, device=positions.device,
    ),
)
register_warp_op_chain(
    name="nvalchemiops::batch_spline_pos_hessian",
    forward=_batch_pos_hessian_forward_launch,
    # Output shape == positions shape — default ``empty_like(positions)`` is right.
)


# Batched gather_gradient — same shape as single-system but with batch_idx
# (non-differentiable) in the input list and per-system cell_inv_t indexed
# via batch_idx in the position-Hessian path.
register_warp_op_chain(
    name="nvalchemiops::batch_spline_gather_gradient",
    forward=_batch_gather_gradient_forward_launch,
)


def _batch_spline_gather_gradient_setup_ctx(ctx, inputs, output):
    positions, charges, mesh, batch_idx, cell_inv_t, spline_order = inputs
    ctx.save_for_backward(positions, charges, mesh, batch_idx, cell_inv_t)
    ctx.spline_order = spline_order
    ctx.mesh_dims = (mesh.shape[-3], mesh.shape[-2], mesh.shape[-1])
    ctx.num_systems = mesh.shape[0]
    ctx.needs_pos = positions.requires_grad
    ctx.needs_chg = charges.requires_grad
    ctx.needs_mesh = mesh.requires_grad


def _batch_spline_gather_gradient_backward_chain(ctx, grad_force):
    positions, charges, mesh, batch_idx, cell_inv_t = ctx.saved_tensors
    order = ctx.spline_order
    if grad_force is None:
        return None, None, None, None, None, None
    grad_force_c = grad_force.contiguous()

    cell_inv_t_per_atom = cell_inv_t[batch_idx.to(torch.int64)]
    v_per_atom = torch.bmm(
        cell_inv_t_per_atom, grad_force_c.unsqueeze(-1),
    ).squeeze(-1)

    if ctx.needs_pos:
        grad_positions = torch.ops.nvalchemiops.batch_spline_pos_hessian(
            positions, charges, v_per_atom, batch_idx, cell_inv_t, mesh, order,
        )
    else:
        grad_positions = None

    if ctx.needs_chg:
        ones = torch.ones_like(charges, dtype=positions.dtype)
        force_per_unit_q = torch.ops.nvalchemiops.batch_spline_gather_gradient(
            positions, ones, mesh, batch_idx, cell_inv_t, order,
        )
        grad_charges = (grad_force_c * force_per_unit_q).sum(dim=-1)
    else:
        grad_charges = None

    if ctx.needs_mesh:
        per_atom_vec = -(charges.to(positions.dtype).unsqueeze(-1) * v_per_atom)
        grad_mesh = torch.ops.nvalchemiops.batch_spline_spread_gradient_weights(
            positions, per_atom_vec, batch_idx, cell_inv_t,
            ctx.num_systems, ctx.mesh_dims, order,
        )
    else:
        grad_mesh = None

    return grad_positions, grad_charges, grad_mesh, None, None, None


torch.library.register_autograd(
    "nvalchemiops::batch_spline_gather_gradient",
    _batch_spline_gather_gradient_backward_chain,
    setup_context=_batch_spline_gather_gradient_setup_ctx,
)


def _batch_cell_inv_t_grad_forward_launch(
    forces: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)
    grad = torch.zeros_like(cell)
    wp_forces = _wp_from_torch(forces.contiguous(), dtype=wp_vec_dtype)
    wp_positions = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_batch_idx = _wp_from_torch(
        batch_idx.to(torch.int32).contiguous(), dtype=wp.int32
    )
    wp_cell = _wp_from_torch(cell.contiguous(), dtype=wp_mat_dtype)
    wp_grad = _wp_from_torch(grad, dtype=wp_mat_dtype)
    with _scoped_warp_stream(positions.device):
        _batch_spline_cell_inv_t_grad_launch(
            wp_forces, wp_positions, wp_batch_idx, wp_cell, wp_grad,
            wp_dtype=wp_dtype, device=device,
        )
    return grad


def _batch_cell_inv_t_grad_backward_launch(
    h: torch.Tensor,
    forces: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    cell: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    grad_forces = torch.empty_like(forces)
    grad_positions = torch.empty_like(positions)
    grad_cell = torch.zeros_like(cell)

    wp_h = _wp_from_torch(h.contiguous(), dtype=wp_mat_dtype)
    wp_forces = _wp_from_torch(forces.contiguous(), dtype=wp_vec_dtype)
    wp_positions = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_batch_idx = _wp_from_torch(
        batch_idx.to(torch.int32).contiguous(), dtype=wp.int32
    )
    wp_cell = _wp_from_torch(cell.contiguous(), dtype=wp_mat_dtype)
    wp_g_forces = _wp_from_torch(grad_forces, dtype=wp_vec_dtype)
    wp_g_positions = _wp_from_torch(grad_positions, dtype=wp_vec_dtype)
    wp_g_cell = _wp_from_torch(grad_cell, dtype=wp_mat_dtype)

    with _scoped_warp_stream(positions.device):
        _batch_spline_cell_inv_t_grad_bwd_launch(
            wp_h, wp_forces, wp_positions, wp_batch_idx, wp_cell,
            wp_g_forces, wp_g_positions, wp_g_cell,
            wp_dtype=wp_dtype, device=device,
        )
    return grad_forces, grad_positions, grad_cell


# Batched cell_inv_t_grad — same pattern as single, with batch_idx (non-diff)
# at position 2. Backward returns 3 grads in order (forces, positions, cell).
register_warp_op_chain(
    name="nvalchemiops::batch_spline_cell_inv_t_grad",
    forward=_batch_cell_inv_t_grad_forward_launch,
    forward_fake=lambda forces, positions, batch_idx, cell: torch.empty_like(cell),
    backward=_batch_cell_inv_t_grad_backward_launch,
    diff_input_positions=(0, 1, 3),
    n_forward_inputs=4,
)


def _batch_cell_inv_t_grad_from_force(
    forces: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
) -> torch.Tensor:
    """Batched ``grad_cell_inv_t`` (registered custom op)."""
    cell = torch.linalg.inv(cell_inv_t.transpose(-1, -2)).contiguous()  # (B, 3, 3)
    return torch.ops.nvalchemiops.batch_spline_cell_inv_t_grad(
        forces, positions, batch_idx, cell,
    )


# Batched spread + gather — same adjoint pattern as single-system, with
# ``batch_idx`` (non-differentiable) at position 2. The batched spread
# additionally carries an explicit ``num_systems: int`` arg at position 4
# (used to size the output mesh).
register_warp_op_chain(
    name="nvalchemiops::batch_spline_spread",
    forward=_batch_spread_forward_launch,
    forward_fake=lambda positions, values, batch_idx, cell_inv_t,
                       num_systems, mesh_dims, spline_order: torch.empty(
        (num_systems, mesh_dims[0], mesh_dims[1], mesh_dims[2]),
        dtype=positions.dtype, device=positions.device,
    ),
)


def _batch_spline_spread_setup_ctx(ctx, inputs, output):
    (positions, values, batch_idx, cell_inv_t,
     num_systems, mesh_dims, spline_order) = inputs
    ctx.save_for_backward(positions, values, batch_idx, cell_inv_t)
    ctx.spline_order = spline_order
    ctx.num_systems = num_systems
    ctx.mesh_dims = tuple(mesh_dims)
    ctx.needs_pos = positions.requires_grad
    ctx.needs_val = values.requires_grad
    ctx.needs_cell = cell_inv_t.requires_grad


def _batch_spline_spread_backward_chain(ctx, grad_mesh):
    positions, values, batch_idx, cell_inv_t = ctx.saved_tensors
    order = ctx.spline_order
    grad_mesh_c = grad_mesh.contiguous()

    grad_values = (
        torch.ops.nvalchemiops.batch_spline_gather(
            positions, grad_mesh_c, batch_idx, cell_inv_t, order,
        ) if ctx.needs_val else None
    )

    if ctx.needs_pos or ctx.needs_cell:
        forces = torch.ops.nvalchemiops.batch_spline_gather_gradient(
            positions, values, grad_mesh_c, batch_idx, cell_inv_t, order,
        )
        grad_positions = -forces if ctx.needs_pos else None
        grad_cell_inv_t = (
            _batch_cell_inv_t_grad_from_force(
                forces, positions, batch_idx, cell_inv_t,
            ) if ctx.needs_cell else None
        )
    else:
        grad_positions = None
        grad_cell_inv_t = None

    # 7 inputs total: positions, values, batch_idx, cell_inv_t, num_systems,
    # mesh_dims, spline_order. batch_idx (2), num_systems (4), mesh_dims (5),
    # spline_order (6) are non-differentiable.
    return grad_positions, grad_values, None, grad_cell_inv_t, None, None, None


torch.library.register_autograd(
    "nvalchemiops::batch_spline_spread",
    _batch_spline_spread_backward_chain,
    setup_context=_batch_spline_spread_setup_ctx,
)


register_warp_op_chain(
    name="nvalchemiops::batch_spline_gather",
    forward=_batch_gather_forward_launch,
    forward_fake=lambda positions, mesh, batch_idx, cell_inv_t, spline_order: torch.empty(
        positions.shape[0], dtype=positions.dtype, device=positions.device,
    ),
)


def _batch_spline_gather_setup_ctx(ctx, inputs, output):
    positions, mesh, batch_idx, cell_inv_t, spline_order = inputs
    ctx.save_for_backward(positions, mesh, batch_idx, cell_inv_t)
    ctx.spline_order = spline_order
    ctx.mesh_dims = (mesh.shape[-3], mesh.shape[-2], mesh.shape[-1])
    ctx.num_systems = mesh.shape[0]
    ctx.needs_pos = positions.requires_grad
    ctx.needs_mesh = mesh.requires_grad
    ctx.needs_cell = cell_inv_t.requires_grad


def _batch_spline_gather_backward_chain(ctx, grad_potentials):
    positions, mesh, batch_idx, cell_inv_t = ctx.saved_tensors
    order = ctx.spline_order
    grad_pot_c = grad_potentials.contiguous()

    grad_mesh = (
        torch.ops.nvalchemiops.batch_spline_spread(
            positions, grad_pot_c, batch_idx, cell_inv_t,
            ctx.num_systems, ctx.mesh_dims, order,
        ) if ctx.needs_mesh else None
    )

    if ctx.needs_pos or ctx.needs_cell:
        forces = torch.ops.nvalchemiops.batch_spline_gather_gradient(
            positions, grad_pot_c, mesh, batch_idx, cell_inv_t, order,
        )
        grad_positions = -forces if ctx.needs_pos else None
        grad_cell_inv_t = (
            _batch_cell_inv_t_grad_from_force(
                forces, positions, batch_idx, cell_inv_t,
            ) if ctx.needs_cell else None
        )
    else:
        grad_positions = None
        grad_cell_inv_t = None

    # 5 inputs: positions, mesh, batch_idx, cell_inv_t, spline_order.
    return grad_positions, grad_mesh, None, grad_cell_inv_t, None


torch.library.register_autograd(
    "nvalchemiops::batch_spline_gather",
    _batch_spline_gather_backward_chain,
    setup_context=_batch_spline_gather_setup_ctx,
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
    """Internal: batch spline spread (registered custom op)."""
    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
    return torch.ops.nvalchemiops.batch_spline_spread(
        positions, values.to(positions.dtype), batch_idx, cell_inv_t,
        num_systems, [mesh_nx, mesh_ny, mesh_nz], spline_order,
    )


def _batch_spline_gather(
    positions: torch.Tensor,
    mesh: torch.Tensor,
    batch_idx: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal: batch spline gather (registered custom op)."""
    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
    return torch.ops.nvalchemiops.batch_spline_gather(
        positions, mesh.to(positions.dtype), batch_idx, cell_inv_t, spline_order,
    )


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


def _batch_spline_gather_gradient(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    batch_idx: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal: batched spline gather-gradient (registered custom op)."""
    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
    return torch.ops.nvalchemiops.batch_spline_gather_gradient(
        positions, charges.to(positions.dtype), mesh.to(positions.dtype),
        batch_idx, cell_inv_t, spline_order,
    )


def _batch_gather_with_force_forward_launch(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched fused gather + gather-gradient forward launch."""
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    wp_vec_dtype = get_wp_vec_dtype(input_dtype)
    wp_mat_dtype = get_wp_mat_dtype(input_dtype)

    num_atoms = positions.shape[0]
    potential = torch.zeros(num_atoms, device=positions.device, dtype=input_dtype)
    forces = torch.zeros((num_atoms, 3), device=positions.device, dtype=input_dtype)

    per_order_kernel = (
        _PER_ORDER_BATCH_GATHER_WITH_FORCE_KERNELS[wp_dtype].get(spline_order)
    )
    if per_order_kernel is None:
        raise NotImplementedError(
            f"Batch fused gather is only specialized for spline_order in "
            f"{tuple(_PER_ORDER_BATCH_GATHER_WITH_FORCE_KERNELS[wp_dtype])}; "
            f"got {spline_order}. The public ``spline_gather_with_force`` "
            "wrapper falls back to ``spline_gather`` + "
            "``spline_gather_gradient`` for unsupported orders."
        )

    wp_pos = _wp_from_torch(positions.contiguous(), dtype=wp_vec_dtype)
    wp_chg = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_bidx = _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32)
    wp_cit = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat_dtype)
    wp_mesh = _wp_from_torch(mesh.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_pot = _wp_from_torch(potential, dtype=wp_dtype)
    wp_forces = _wp_from_torch(forces, dtype=wp_vec_dtype)

    with _scoped_warp_stream(positions.device):
        wp.launch(
            per_order_kernel,
            dim=num_atoms,
            inputs=[wp_pos, wp_chg, wp_bidx, wp_cit, wp_mesh],
            outputs=[wp_pot, wp_forces],
            device=device,
        )
    return potential, forces


# Batched fused gather + force. Dual-chain backward (gather chain for
# grad_potential, gather_gradient chain for grad_forces), indexed by
# batch_idx.
register_warp_op_chain(
    name="nvalchemiops::batch_spline_gather_with_force",
    forward=_batch_gather_with_force_forward_launch,
    forward_return_arity=2,
    forward_fake=lambda pos, *_: (
        torch.empty(pos.shape[0], dtype=pos.dtype, device=pos.device),
        torch.empty((pos.shape[0], 3), dtype=pos.dtype, device=pos.device),
    ),
)


def _batch_spline_gather_with_force_setup_ctx(ctx, inputs, output):
    positions, charges, mesh, batch_idx, cell_inv_t, spline_order = inputs
    ctx.save_for_backward(positions, charges, mesh, batch_idx, cell_inv_t)
    ctx.spline_order = spline_order
    ctx.mesh_dims = (mesh.shape[-3], mesh.shape[-2], mesh.shape[-1])
    ctx.num_systems = mesh.shape[0]
    ctx.needs_pos = positions.requires_grad
    ctx.needs_chg = charges.requires_grad
    ctx.needs_mesh = mesh.requires_grad
    ctx.needs_cell = cell_inv_t.requires_grad


def _batch_spline_gather_with_force_backward_chain(ctx, grad_potential, grad_forces):
    positions, charges, mesh, batch_idx, cell_inv_t = ctx.saved_tensors
    order = ctx.spline_order
    grad_pos = grad_chg = grad_mesh = grad_cell_inv_t = None

    def _add(target, contrib):
        return contrib if target is None else target + contrib

    # gather chain (grad_potential → grads)
    if grad_potential is not None:
        gp = grad_potential.contiguous()
        if ctx.needs_pos or ctx.needs_cell:
            forces_g = torch.ops.nvalchemiops.batch_spline_gather_gradient(
                positions, gp, mesh, batch_idx, cell_inv_t, order,
            )
            if ctx.needs_pos:
                grad_pos = _add(grad_pos, -forces_g)
            if ctx.needs_cell:
                grad_cell_inv_t = _add(
                    grad_cell_inv_t,
                    _batch_cell_inv_t_grad_from_force(
                        forces_g, positions, batch_idx, cell_inv_t,
                    ),
                )
        if ctx.needs_mesh:
            grad_mesh = _add(
                grad_mesh,
                torch.ops.nvalchemiops.batch_spline_spread(
                    positions, gp, batch_idx, cell_inv_t,
                    ctx.num_systems, ctx.mesh_dims, order,
                ),
            )

    # gather_gradient chain (grad_forces → grads)
    if grad_forces is not None:
        gf = grad_forces.contiguous()
        cell_inv_t_per_atom = cell_inv_t[batch_idx.to(torch.int64)]
        v_per_atom = torch.bmm(
            cell_inv_t_per_atom, gf.unsqueeze(-1),
        ).squeeze(-1)

        if ctx.needs_chg:
            ones = torch.ones_like(charges, dtype=positions.dtype)
            force_per_unit_q = torch.ops.nvalchemiops.batch_spline_gather_gradient(
                positions, ones, mesh, batch_idx, cell_inv_t, order,
            )
            grad_chg = _add(grad_chg, (gf * force_per_unit_q).sum(dim=-1))

        if ctx.needs_mesh:
            per_atom_vec = -(charges.to(positions.dtype).unsqueeze(-1) * v_per_atom)
            grad_mesh = _add(
                grad_mesh,
                torch.ops.nvalchemiops.batch_spline_spread_gradient_weights(
                    positions, per_atom_vec, batch_idx, cell_inv_t,
                    ctx.num_systems, ctx.mesh_dims, order,
                ),
            )

        if ctx.needs_pos:
            pos_hess = torch.ops.nvalchemiops.batch_spline_pos_hessian(
                positions, charges, v_per_atom, batch_idx, cell_inv_t, mesh, order,
            )
            grad_pos = _add(grad_pos, pos_hess)

    # 6 inputs: positions, charges, mesh, batch_idx, cell_inv_t, spline_order
    return grad_pos, grad_chg, grad_mesh, None, grad_cell_inv_t, None


torch.library.register_autograd(
    "nvalchemiops::batch_spline_gather_with_force",
    _batch_spline_gather_with_force_backward_chain,
    setup_context=_batch_spline_gather_with_force_setup_ctx,
)


def _batch_spline_gather_with_force(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    batch_idx: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int,
    cell_inv_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: batched fused gather + gather-gradient (registered custom op)."""
    if cell_inv_t is None:
        cell_inv = torch.linalg.inv(cell)
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
    return torch.ops.nvalchemiops.batch_spline_gather_with_force(
        positions, charges.to(positions.dtype), mesh.to(positions.dtype),
        batch_idx, cell_inv_t, spline_order,
    )


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
            *_: (
                num_channels,
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
            lambda pos, mesh, *_: (pos.shape[0], mesh.shape[0]),
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


def bspline_weight(u: torch.Tensor, order: int) -> torch.Tensor:
    """Compute B-spline basis function M_n(u).

    Parameters
    ----------
    u : torch.Tensor
        Input values.
    order : int
        Spline order.

    Returns
    -------
    torch.Tensor
        Weights M_n(u).
    """
    return _spline_weight(u, order)


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
    charges : torch.Tensor, shape (N,)
        Atomic charges (or other scalar weights).
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


def spline_gather_with_force(
    positions: torch.Tensor,
    charges: torch.Tensor,
    mesh: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    cell_inv_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gather of scalar potential AND derivative-based force from one mesh.

    Returns ``(output, forces)`` where:
      - ``output[atom] = Σ_g mesh[g] * w(atom, g)``       — raw potential per atom
        (the caller multiplies by charge in the PME corrections step).
      - ``forces[atom] = -q_atom * Σ_g mesh[g] * Cell^{-T} ∇w`` — Cartesian force.

    This replaces ``spline_gather(...)`` followed by ``spline_gather_gradient(...)``
    on the same mesh: each thread reads its stencil cell ONCE and accumulates
    both outputs. Halves the mesh DRAM traffic and shares the per-thread weight
    derivative work across both channels.

    Parameters mirror ``spline_gather_gradient``. For ``spline_order`` in the
    set the per-order kernels cover (``{2, 3, 4, 5, 6}``), both single-system
    and batched inputs use the fused kernel directly. For unsupported orders,
    batched inputs fall back to the two-kernel sequence
    (``spline_gather`` + ``spline_gather_gradient``).
    """
    if batch_idx is None:
        return _spline_gather_with_force(
            positions, charges, mesh, cell, spline_order, cell_inv_t
        )

    # Batched path: use the per-order fused kernel when available, otherwise
    # fall back to the two-kernel sequence.
    wp_dtype = get_wp_dtype(positions.dtype)
    if spline_order in _PER_ORDER_BATCH_GATHER_WITH_FORCE_KERNELS[wp_dtype]:
        # Ensure cell is 3D for batched operations.
        if cell.dim() == 2:
            num_systems = int(batch_idx.max().item()) + 1
            cell = cell.unsqueeze(0).expand(num_systems, -1, -1).contiguous()
        return _batch_spline_gather_with_force(
            positions, charges, mesh, batch_idx, cell, spline_order, cell_inv_t
        )

    potential = spline_gather(
        positions, mesh, cell, spline_order=spline_order,
        batch_idx=batch_idx, cell_inv_t=cell_inv_t,
    )
    forces = spline_gather_gradient(
        positions, charges, mesh, cell, spline_order=spline_order,
        batch_idx=batch_idx, cell_inv_t=cell_inv_t,
    )
    return potential, forces


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

    The B-spline function M_n(u) has Fourier transform.

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
        |b(k)|^2 where b(k) is the B-spline Fourier coefficient.
    """
    # Compute the exponential B-spline factors
    # Following Essmann et al. (1995) Eq. 4.7
    pi = torch.tensor(math.pi, dtype=torch.float64, device=k.device)

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

    # Sum: b(k) = sum_j M_order(j+1) * exp(i w j)
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

    D(k_x, k_y, k_z) = 1 / (|b(k_x)|^2 * |b(k_y)|^2 * |b(k_z)|^2)

    where b(k) is the Fourier transform of the 1D B-spline.

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
########################### Module Exports #################################################
###########################################################################################


__all__ = [
    # Unified PyTorch API (scalar)
    "bspline_weight",
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
]
