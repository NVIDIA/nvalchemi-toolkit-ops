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

r"""
Dispersion PME (LJ-PME) - PyTorch Bindings
==========================================

PyTorch bindings for dispersion (:math:`r^{-6}`) Particle Mesh Ewald, mirroring
``nvalchemiops.torch.interactions.electrostatics.pme``. The reciprocal-space
mesh pipeline (spline spread/gather, FFT, B-spline deconvolution, k-vectors) is
reused unchanged from electrostatics; only the per-k convolution factor and the
energy corrections differ (see ``interactions/dispersion/lj_pme_kernels.py``).

Public API
----------
- ``dispersion_reciprocal_space()``: reciprocal-space term only (mesh)
- ``dispersion_pme()``: full real + reciprocal dispersion PME

Geometric combination rule: per-atom dispersion charge
``b_i = sqrt(C6_i) = 2 sqrt(eps_i) sigma_i**3`` is spread as a single mesh
channel.

.. note::
   ``alpha`` here is the dispersion splitting parameter :math:`\beta` used
   directly in the real-space screen ``S(r) = (1 + β²r² + ½β⁴r⁴) e^{-β²r²}``
   (no ``/√2`` rescaling). The same ``alpha`` must be used for the real and
   reciprocal halves; the total is invariant to its value.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from nvalchemiops.torch.interactions.dispersion.lj_dispersion import (
    lj_dispersion_energy,
    lj_dispersion_energy_forces,
    sigma_epsilon_to_dispersion_charge,
)
from nvalchemiops.torch.interactions.dispersion.parameters import (
    DispersionPMEParameters,
    estimate_dispersion_pme_parameters,
)
from nvalchemiops.torch.interactions.electrostatics._warp_op_helpers import (
    attach_simple_backward,
    register_warp_op_chain,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_pme,
)
from nvalchemiops.torch.interactions.electrostatics.pme import (
    _pme_scoped_warp_stream,
    _prepare_alpha,
    _prepare_cell,
    _vec2_wp_dtype_for,
    _wp_from_torch,
    compute_bspline_moduli_1d,
)
from nvalchemiops.torch.spline import (
    spline_gather,
    spline_gather_with_force,
    spline_spread,
)
from nvalchemiops.torch.types import get_wp_dtype

__all__ = [
    "dispersion_reciprocal_space",
    "dispersion_pme",
    "DispersionPMEParameters",
    "estimate_dispersion_pme_parameters",
]

PI = math.pi
TWOPI = 2.0 * PI
SQRT_PI = math.sqrt(math.pi)


###########################################################################################
# Fused convolution custom op (warp forward + backward)
###########################################################################################


def _lj_pme_convolve_forward(
    mesh_fft: torch.Tensor,
    k_squared: torch.Tensor,
    moduli_x: torch.Tensor,
    moduli_y: torch.Tensor,
    moduli_z: torch.Tensor,
    alpha: torch.Tensor,
    volume: torch.Tensor,
    is_batch: bool,
) -> torch.Tensor:
    """Run the dispersion convolve warp kernel on ``mesh_fft`` (no autograd)."""
    from nvalchemiops.interactions.dispersion.lj_pme_kernels import (
        batch_lj_pme_convolve as _batch_conv,
    )
    from nvalchemiops.interactions.dispersion.lj_pme_kernels import (
        lj_pme_convolve as _conv,
    )

    device = wp.device_from_torch(mesh_fft.device)
    real_dtype = torch.float32 if mesh_fft.dtype == torch.complex64 else torch.float64
    wp_dtype = wp.float32 if real_dtype == torch.float32 else wp.float64
    wp_vec2 = _vec2_wp_dtype_for(real_dtype)

    squeeze_output = False
    if is_batch and k_squared.dim() == 3:
        k_squared = k_squared.unsqueeze(0)
    if is_batch and mesh_fft.dim() == 3:
        mesh_fft = mesh_fft.unsqueeze(0)
        squeeze_output = True

    mesh_fft_real = torch.view_as_real(mesh_fft.resolve_conj()).contiguous()
    convolved_real = torch.empty_like(mesh_fft_real)

    def _as(t):
        if t.dtype != real_dtype:
            t = t.to(real_dtype)
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    wp_mesh_fft = _wp_from_torch(mesh_fft_real, dtype=wp_vec2)
    wp_convolved = _wp_from_torch(convolved_real, dtype=wp_vec2)
    wp_k_squared = _wp_from_torch(_as(k_squared), dtype=wp_dtype)
    wp_bx = _wp_from_torch(_as(moduli_x), dtype=wp_dtype)
    wp_by = _wp_from_torch(_as(moduli_y), dtype=wp_dtype)
    wp_bz = _wp_from_torch(_as(moduli_z), dtype=wp_dtype)
    alpha_in = _as(alpha)
    volume_in = _as(volume)
    if alpha_in.dim() == 0:
        alpha_in = alpha_in.reshape(1)
    if volume_in.dim() == 0:
        volume_in = volume_in.reshape(1)
    wp_alpha = _wp_from_torch(alpha_in, dtype=wp_dtype)
    wp_volume = _wp_from_torch(volume_in, dtype=wp_dtype)

    with _pme_scoped_warp_stream(mesh_fft.device):
        if is_batch:
            _batch_conv(
                wp_mesh_fft,
                wp_k_squared,
                wp_bx,
                wp_by,
                wp_bz,
                wp_alpha,
                wp_volume,
                wp_convolved,
                wp_dtype=wp_dtype,
                device=device,
            )
        else:
            _conv(
                wp_mesh_fft,
                wp_k_squared,
                wp_bx,
                wp_by,
                wp_bz,
                wp_alpha,
                wp_volume,
                wp_convolved,
                wp_dtype=wp_dtype,
                device=device,
            )

    out = torch.view_as_complex(convolved_real)
    if squeeze_output:
        out = out.squeeze(0)
    return out


def _lj_pme_convolve_backward(
    mesh_fft: torch.Tensor,
    grad_convolved: torch.Tensor,
    k_squared: torch.Tensor,
    moduli_x: torch.Tensor,
    moduli_y: torch.Tensor,
    moduli_z: torch.Tensor,
    alpha: torch.Tensor,
    volume: torch.Tensor,
    is_batch: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit backward for the dispersion convolve.

    Returns ``(grad_mesh_fft, grad_alpha, grad_volume, grad_k_squared)``.
    """
    from nvalchemiops.interactions.dispersion.lj_pme_kernels import (
        batch_lj_pme_convolve_backward as _batch_bwd,
    )
    from nvalchemiops.interactions.dispersion.lj_pme_kernels import (
        lj_pme_convolve_backward as _bwd,
    )

    device = wp.device_from_torch(mesh_fft.device)
    real_dtype = torch.float32 if mesh_fft.dtype == torch.complex64 else torch.float64
    wp_dtype = wp.float32 if real_dtype == torch.float32 else wp.float64
    wp_vec2 = _vec2_wp_dtype_for(real_dtype)

    squeeze_output = False
    if is_batch and k_squared.dim() == 3:
        k_squared = k_squared.unsqueeze(0)
    if is_batch and mesh_fft.dim() == 3:
        mesh_fft = mesh_fft.unsqueeze(0)
        squeeze_output = True
    if is_batch and grad_convolved.dim() == 3:
        grad_convolved = grad_convolved.unsqueeze(0)

    mesh_fft_real = torch.view_as_real(mesh_fft.resolve_conj()).contiguous()
    grad_conv_real = torch.view_as_real(grad_convolved.resolve_conj()).contiguous()
    grad_mesh_fft_real = torch.empty_like(mesh_fft_real)

    def _as(t):
        if t.dtype != real_dtype:
            t = t.to(real_dtype)
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    alpha_in = _as(alpha)
    volume_in = _as(volume)
    if alpha_in.dim() == 0:
        alpha_in = alpha_in.reshape(1)
    if volume_in.dim() == 0:
        volume_in = volume_in.reshape(1)
    B = alpha_in.shape[0]

    grad_alpha = torch.zeros(B, dtype=real_dtype, device=mesh_fft.device)
    grad_volume = torch.zeros(B, dtype=real_dtype, device=mesh_fft.device)
    grad_k_squared = torch.empty_like(_as(k_squared))

    wp_mesh_fft = _wp_from_torch(mesh_fft_real, dtype=wp_vec2)
    wp_grad_conv = _wp_from_torch(grad_conv_real, dtype=wp_vec2)
    wp_grad_mesh = _wp_from_torch(grad_mesh_fft_real, dtype=wp_vec2)
    wp_k_squared = _wp_from_torch(_as(k_squared), dtype=wp_dtype)
    wp_grad_k_squared = _wp_from_torch(grad_k_squared, dtype=wp_dtype)
    wp_bx = _wp_from_torch(_as(moduli_x), dtype=wp_dtype)
    wp_by = _wp_from_torch(_as(moduli_y), dtype=wp_dtype)
    wp_bz = _wp_from_torch(_as(moduli_z), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha_in, dtype=wp_dtype)
    wp_volume = _wp_from_torch(volume_in, dtype=wp_dtype)
    wp_grad_alpha = _wp_from_torch(grad_alpha, dtype=wp_dtype)
    wp_grad_volume = _wp_from_torch(grad_volume, dtype=wp_dtype)

    with _pme_scoped_warp_stream(mesh_fft.device):
        if is_batch:
            _batch_bwd(
                wp_mesh_fft,
                wp_grad_conv,
                wp_k_squared,
                wp_bx,
                wp_by,
                wp_bz,
                wp_alpha,
                wp_volume,
                wp_grad_mesh,
                wp_grad_alpha,
                wp_grad_volume,
                wp_grad_k_squared,
                wp_dtype=wp_dtype,
                device=device,
            )
        else:
            _bwd(
                wp_mesh_fft,
                wp_grad_conv,
                wp_k_squared,
                wp_bx,
                wp_by,
                wp_bz,
                wp_alpha,
                wp_volume,
                wp_grad_mesh,
                wp_grad_alpha,
                wp_grad_volume,
                wp_grad_k_squared,
                wp_dtype=wp_dtype,
                device=device,
            )

    grad_mesh_fft = torch.view_as_complex(grad_mesh_fft_real)
    if squeeze_output:
        grad_mesh_fft = grad_mesh_fft.squeeze(0)
        grad_k_squared = grad_k_squared.squeeze(0)
    return grad_mesh_fft, grad_alpha, grad_volume, grad_k_squared


def _lj_convolve_forward_fake(mesh_fft, *_):
    return torch.empty(mesh_fft.shape, dtype=mesh_fft.dtype, device=mesh_fft.device)


def _lj_convolve_backward_fake(
    mesh_fft,
    grad_convolved,
    k_squared,
    moduli_x,
    moduli_y,
    moduli_z,
    alpha,
    volume,
    is_batch,
):
    real_dtype = torch.float32 if mesh_fft.dtype == torch.complex64 else torch.float64
    B = alpha.shape[0] if alpha.dim() >= 1 else 1
    return (
        torch.empty_like(mesh_fft),
        torch.zeros(B, dtype=real_dtype, device=mesh_fft.device),
        torch.zeros(B, dtype=real_dtype, device=mesh_fft.device),
        torch.empty_like(k_squared, dtype=real_dtype),
    )


register_warp_op_chain(
    name="nvalchemiops::lj_pme_fused_convolve",
    forward=_lj_pme_convolve_forward,
    forward_fake=_lj_convolve_forward_fake,
    backward=_lj_pme_convolve_backward,
    backward_fake=_lj_convolve_backward_fake,
    backward_return_arity=4,
    # backward outputs (grad_mesh_fft, grad_alpha, grad_volume, grad_k_squared)
    # map to forward input positions (mesh_fft=0, alpha=5, volume=6, k_squared=1).
    diff_input_positions=(0, 5, 6, 1),
    n_forward_inputs=8,
    backward_args=lambda g, f: (f[0], g[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]),
)

# Convolve is linear in mesh_fft, so its backward's Jacobian w.r.t.
# grad_convolved is the same forward op applied to the cotangent.
attach_simple_backward(
    "nvalchemiops::lj_pme_fused_convolve_backward",
    torch.ops.nvalchemiops.lj_pme_fused_convolve,
    diff_input_positions=(1,),
    n_forward_inputs=9,
    propagate_outputs=(0,),
    backward_args=lambda g, f: (g[0], f[2], f[3], f[4], f[5], f[6], f[7], f[8]),
)


###########################################################################################
# Energy corrections custom op (warp forward + backward)
###########################################################################################


def _lj_corrections_forward_launch(
    raw_energies: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    from nvalchemiops.interactions.dispersion.lj_pme_kernels import (
        lj_pme_energy_corrections as _ec,
    )

    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    corrected = torch.zeros_like(raw_energies)

    wp_raw = _wp_from_torch(raw_energies.contiguous(), dtype=wp_dtype)
    wp_b = _wp_from_torch(b.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_corrected = _wp_from_torch(corrected, dtype=wp_dtype)

    with _pme_scoped_warp_stream(raw_energies.device):
        _ec(wp_raw, wp_b, wp_alpha, wp_corrected, wp_dtype=wp_dtype, device=device)
    return corrected


def _lj_corrections_backward_launch(
    grad_E: torch.Tensor,
    raw_energies: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from nvalchemiops.interactions.dispersion.lj_pme_kernels import (
        lj_pme_energy_corrections_backward as _ec_bwd,
    )

    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)

    grad_raw = torch.empty_like(raw_energies)
    grad_b = torch.empty_like(b, dtype=input_dtype)
    grad_alpha = torch.zeros(1, dtype=input_dtype, device=raw_energies.device)

    wp_gE = _wp_from_torch(grad_E.contiguous(), dtype=wp_dtype)
    wp_raw = _wp_from_torch(raw_energies.contiguous(), dtype=wp_dtype)
    wp_b = _wp_from_torch(b.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_g_raw = _wp_from_torch(grad_raw, dtype=wp_dtype)
    wp_g_b = _wp_from_torch(grad_b, dtype=wp_dtype)
    wp_g_alpha = _wp_from_torch(grad_alpha, dtype=wp_dtype)

    with _pme_scoped_warp_stream(raw_energies.device):
        _ec_bwd(
            wp_gE,
            wp_raw,
            wp_b,
            wp_alpha,
            wp_g_raw,
            wp_g_b,
            wp_g_alpha,
            wp_dtype=wp_dtype,
            device=device,
        )
    return grad_raw, grad_b, grad_alpha


register_warp_op_chain(
    name="nvalchemiops::lj_pme_energy_corrections",
    forward=_lj_corrections_forward_launch,
    backward=_lj_corrections_backward_launch,
    diff_input_positions=(0, 1, 2),
    n_forward_inputs=3,
)


def _batch_lj_corrections_forward_launch(
    raw_energies: torch.Tensor,
    b: torch.Tensor,
    batch_idx: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    from nvalchemiops.interactions.dispersion.lj_pme_kernels import (
        batch_lj_pme_energy_corrections as _ec,
    )

    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    corrected = torch.zeros_like(raw_energies)

    wp_raw = _wp_from_torch(raw_energies.contiguous(), dtype=wp_dtype)
    wp_b = _wp_from_torch(b.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_bidx = _wp_from_torch(batch_idx.to(torch.int32).contiguous(), dtype=wp.int32)
    wp_alpha = _wp_from_torch(alpha.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_corrected = _wp_from_torch(corrected, dtype=wp_dtype)

    with _pme_scoped_warp_stream(raw_energies.device):
        _ec(
            wp_raw,
            wp_b,
            wp_bidx,
            wp_alpha,
            wp_corrected,
            wp_dtype=wp_dtype,
            device=device,
        )
    return corrected


def _batch_lj_corrections_backward_launch(
    grad_E: torch.Tensor,
    raw_energies: torch.Tensor,
    b: torch.Tensor,
    batch_idx: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from nvalchemiops.interactions.dispersion.lj_pme_kernels import (
        batch_lj_pme_energy_corrections_backward as _ec_bwd,
    )

    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    B = alpha.shape[0]

    grad_raw = torch.empty_like(raw_energies)
    grad_b = torch.empty_like(b, dtype=input_dtype)
    grad_batch_idx = torch.zeros_like(batch_idx, dtype=input_dtype)
    grad_alpha = torch.zeros(B, dtype=input_dtype, device=raw_energies.device)

    wp_gE = _wp_from_torch(grad_E.contiguous(), dtype=wp_dtype)
    wp_raw = _wp_from_torch(raw_energies.contiguous(), dtype=wp_dtype)
    wp_b = _wp_from_torch(b.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_bidx = _wp_from_torch(batch_idx.to(torch.int32).contiguous(), dtype=wp.int32)
    wp_alpha = _wp_from_torch(alpha.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_g_raw = _wp_from_torch(grad_raw, dtype=wp_dtype)
    wp_g_b = _wp_from_torch(grad_b, dtype=wp_dtype)
    wp_g_alpha = _wp_from_torch(grad_alpha, dtype=wp_dtype)

    with _pme_scoped_warp_stream(raw_energies.device):
        _ec_bwd(
            wp_gE,
            wp_raw,
            wp_b,
            wp_bidx,
            wp_alpha,
            wp_g_raw,
            wp_g_b,
            wp_g_alpha,
            wp_dtype=wp_dtype,
            device=device,
        )
    # grad for batch_idx (int) is a structural zero.
    return grad_raw, grad_b, grad_batch_idx, grad_alpha


register_warp_op_chain(
    name="nvalchemiops::batch_lj_pme_energy_corrections",
    forward=_batch_lj_corrections_forward_launch,
    backward=_batch_lj_corrections_backward_launch,
    diff_input_positions=(0, 1, 3),  # raw, b, alpha (batch_idx is non-diff)
    n_forward_inputs=4,
    backward_return_arity=4,
)


def _apply_corrections(raw_energies, b, alpha, batch_idx):
    """Dispatch to single/batch corrections custom op."""
    b = b.to(raw_energies.dtype)
    alpha = alpha.to(raw_energies.dtype)
    if batch_idx is None:
        return torch.ops.nvalchemiops.lj_pme_energy_corrections(raw_energies, b, alpha)
    return torch.ops.nvalchemiops.batch_lj_pme_energy_corrections(
        raw_energies, b, batch_idx.to(torch.int32), alpha
    )


###########################################################################################
# Reciprocal-space virial (pure torch; mirrors electrostatics _compute_pme_reciprocal_virial)
###########################################################################################


def _dispersion_reciprocal_virial(
    mesh_fft_raw: torch.Tensor,
    convolved_mesh: torch.Tensor,
    k_vectors: torch.Tensor,
    k_squared: torch.Tensor,
    alpha: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    is_batch: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    r"""Reciprocal-space dispersion virial ``W = -dE/dε``.

    Per-k energy density ``E_k = Re(mesh_fft_raw · conv*)`` (= ``|S|² G_disp``).
    The strain derivative gives

    .. math::
        W_{ab} = \sum_k E_k\left[\delta_{ab} + 2 k_a k_b
                 \frac{\partial \ln G_{\text{disp}}}{\partial k^2}\right]

    with ``∂ln G/∂k² = g1/(4α² f)`` (``g1 = df/db²``). The k=0 term contributes
    only its isotropic ``E_0 δ_ab`` (k_a k_b = 0) and is **kept** (dispersion
    background is non-zero), unlike Coulomb where k=0 is masked.
    """
    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    acc_dtype = dtype

    fft_raw_cast = mesh_fft_raw.to(complex_dtype)
    conv_cast = convolved_mesh.to(complex_dtype)
    energy_density = (fft_raw_cast * conv_cast.conj()).real

    weight = torch.full_like(energy_density, 2.0)
    weight[..., 0] = 1.0
    if mesh_nz % 2 == 0:
        weight[..., -1] = 1.0
    weighted_energy = weight * energy_density

    k_sq_acc = k_squared.to(acc_dtype)
    alpha_acc = alpha.to(acc_dtype)
    if is_batch and k_sq_acc.dim() == 3:
        k_sq_acc = k_sq_acc.unsqueeze(0)
    if is_batch and alpha_acc.dim() == 1:
        alpha_view = alpha_acc.view(-1, 1, 1, 1)
    else:
        alpha_view = alpha_acc.view(-1) if alpha_acc.dim() == 0 else alpha_acc

    # b² = k²/(4α²); f(b) and g1 = df/db² (torch). The k=0 point has b=0,
    # f=1, g1=-3 (finite) but k_a k_b = 0 so the kk term vanishes there.
    b2 = k_sq_acc * (0.25 / (alpha_view**2))
    b = torch.sqrt(b2.clamp(min=0.0))
    e = torch.exp(-b2)
    erfcb = torch.erfc(b)
    f = (1.0 - 2.0 * b2) * e + 2.0 * SQRT_PI * b2 * b * erfcb
    g1 = 3.0 * (SQRT_PI * b * erfcb - e)
    # k_factor multiplies k_a k_b in the kk term; defined so that
    # virial = eye·trace - kk_term, i.e. k_factor = -2 ∂lnG/∂k² = -g1/(2α² f).
    f_safe = torch.where(f.abs() > 1e-300, f, torch.ones_like(f))
    k_factor = -g1 / (2.0 * (alpha_view**2) * f_safe)

    k_vecs_acc = k_vectors.to(acc_dtype)
    if is_batch and k_vecs_acc.dim() == 4:
        k_vecs_acc = k_vecs_acc.unsqueeze(0)

    masked_energy_kf = weighted_energy * k_factor

    sum_dims = (1, 2, 3) if is_batch else (0, 1, 2)
    trace_term = weighted_energy.sum(dim=sum_dims)

    kx = k_vecs_acc[..., 0]
    ky = k_vecs_acc[..., 1]
    kz = k_vecs_acc[..., 2]
    xx = (kx * kx * masked_energy_kf).sum(dim=sum_dims)
    yy = (ky * ky * masked_energy_kf).sum(dim=sum_dims)
    zz = (kz * kz * masked_energy_kf).sum(dim=sum_dims)
    xy = (kx * ky * masked_energy_kf).sum(dim=sum_dims)
    xz = (kx * kz * masked_energy_kf).sum(dim=sum_dims)
    yz = (ky * kz * masked_energy_kf).sum(dim=sum_dims)

    eye = torch.eye(3, device=device, dtype=acc_dtype)
    if is_batch:
        kk_term = torch.stack(
            [
                torch.stack([xx, xy, xz], dim=-1),
                torch.stack([xy, yy, yz], dim=-1),
                torch.stack([xz, yz, zz], dim=-1),
            ],
            dim=-2,
        )
        virial = eye * trace_term[:, None, None] - kk_term
    else:
        kk_term = torch.stack(
            [
                torch.stack([xx, xy, xz]),
                torch.stack([xy, yy, yz]),
                torch.stack([xz, yz, zz]),
            ]
        )
        virial = (eye * trace_term - kk_term).unsqueeze(0)

    return virial.to(dtype)


###########################################################################################
# Reciprocal-space implementation
###########################################################################################


def _dispersion_reciprocal_space_impl(
    positions: torch.Tensor,
    b: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    spline_order: int,
    batch_idx: torch.Tensor | None,
    compute_forces: bool = False,
    compute_virial: bool = False,
    k_vectors: torch.Tensor | None = None,
    k_squared: torch.Tensor | None = None,
    volume: torch.Tensor | None = None,
    cell_inv_t: torch.Tensor | None = None,
    moduli_x: torch.Tensor | None = None,
    moduli_y: torch.Tensor | None = None,
    moduli_z: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Mesh reciprocal-space dispersion: returns (energies, forces, virial)."""
    device = positions.device
    input_dtype = positions.dtype
    num_atoms = positions.shape[0]
    is_batch = batch_idx is not None
    fft_dims = (1, 2, 3) if is_batch else (0, 1, 2)

    if num_atoms == 0:
        energies = torch.zeros(num_atoms, device=device, dtype=input_dtype)
        forces = (
            torch.zeros(num_atoms, 3, device=device, dtype=input_dtype)
            if compute_forces
            else None
        )
        num_systems = cell.shape[0] if is_batch else 1
        virial = (
            torch.zeros(num_systems, 3, 3, device=device, dtype=input_dtype)
            if compute_virial
            else None
        )
        return energies, forces, virial

    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions

    if cell_inv_t is None:
        cell_inv = torch.linalg.inv_ex(cell)[0]
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
    reciprocal_cell = TWOPI * cell_inv_t.transpose(-1, -2)

    mesh_grid = spline_spread(
        positions,
        b,
        cell,
        mesh_dims=(mesh_nx, mesh_ny, mesh_nz),
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t,
    )

    if k_vectors is None or k_squared is None:
        k_vectors, k_squared = generate_k_vectors_pme(
            cell,
            mesh_dimensions=mesh_dimensions,
            reciprocal_cell=reciprocal_cell,
        )

    if moduli_x is None or moduli_y is None or moduli_z is None:
        miller_x = torch.fft.fftfreq(
            mesh_nx, d=1.0 / mesh_nx, device=device, dtype=input_dtype
        )
        miller_y = torch.fft.fftfreq(
            mesh_ny, d=1.0 / mesh_ny, device=device, dtype=input_dtype
        )
        miller_z = torch.fft.rfftfreq(
            mesh_nz, d=1.0 / mesh_nz, device=device, dtype=input_dtype
        )
        moduli_x = compute_bspline_moduli_1d(miller_x, mesh_nx, spline_order)
        moduli_y = compute_bspline_moduli_1d(miller_y, mesh_ny, spline_order)
        moduli_z = compute_bspline_moduli_1d(miller_z, mesh_nz, spline_order)

    if volume is None:
        cell_for_vol = cell if cell.dim() == 3 else cell.unsqueeze(0)
        volume = torch.abs(torch.linalg.det(cell_for_vol)).to(input_dtype)

    mesh_fft = torch.fft.rfftn(mesh_grid, norm="backward", dim=fft_dims)
    mesh_fft_raw = mesh_fft if compute_virial else None
    convolved_mesh = torch.ops.nvalchemiops.lj_pme_fused_convolve(
        mesh_fft,
        k_squared,
        moduli_x,
        moduli_y,
        moduli_z,
        alpha,
        volume,
        is_batch,
    )
    potential_mesh = torch.fft.irfftn(
        convolved_mesh, norm="forward", s=mesh_dimensions, dim=fft_dims
    ).to(input_dtype)

    if compute_forces:
        raw_energies, gathered_force = spline_gather_with_force(
            positions,
            b,
            potential_mesh,
            cell,
            spline_order=spline_order,
            batch_idx=batch_idx,
            cell_inv_t=cell_inv_t,
        )
    else:
        raw_energies = spline_gather(
            positions,
            potential_mesh,
            cell,
            spline_order=spline_order,
            batch_idx=batch_idx,
            cell_inv_t=cell_inv_t,
        )
        gathered_force = None

    reciprocal_energies = _apply_corrections(raw_energies, b, alpha, batch_idx)

    virial = None
    if compute_virial:
        virial = _dispersion_reciprocal_virial(
            mesh_fft_raw=mesh_fft_raw,
            convolved_mesh=convolved_mesh,
            k_vectors=k_vectors,
            k_squared=k_squared,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            is_batch=is_batch,
            device=device,
            dtype=input_dtype,
        )
        del mesh_fft_raw

    # E = Σ_k G|S|² is quadratic in the mesh, so the explicit gather force
    # (−b ∇Φ) is half the energy gradient; 2× recovers the full force.
    forces = 2.0 * gathered_force if compute_forces else None

    return reciprocal_energies, forces, virial


def dispersion_reciprocal_space(
    positions: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    cell: torch.Tensor,
    alpha: float | torch.Tensor,
    mesh_dimensions: tuple[int, int, int] | None = None,
    mesh_spacing: float | None = None,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    k_vectors: torch.Tensor | None = None,
    k_squared: torch.Tensor | None = None,
    volume: torch.Tensor | None = None,
    cell_inv_t: torch.Tensor | None = None,
    moduli_x: torch.Tensor | None = None,
    moduli_y: torch.Tensor | None = None,
    moduli_z: torch.Tensor | None = None,
    compute_forces: bool = False,
    compute_virial: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""Reciprocal-space (mesh) dispersion PME energy and optional forces/virial.

    See module docstring for the influence function and conventions. ``alpha``
    is the dispersion splitting parameter :math:`\beta`.

    Returns per-atom reciprocal energies (input dtype), optionally followed by
    forces ``(N, 3)`` and/or virial ``(B, 3, 3)`` (appended in that order).
    """
    cell, num_systems = _prepare_cell(cell)
    alpha_tensor = _prepare_alpha(alpha, num_systems, positions.dtype, positions.device)

    if mesh_dimensions is None:
        if mesh_spacing is None:
            raise ValueError("Either mesh_dimensions or mesh_spacing must be provided")
        cell_lengths = torch.norm(cell[0], dim=1)
        mesh_dimensions = tuple(
            int(torch.ceil(length / mesh_spacing).item()) for length in cell_lengths
        )

    b = sigma_epsilon_to_dispersion_charge(sigma, epsilon).to(positions.dtype)

    energies, forces, virial = _dispersion_reciprocal_space_impl(
        positions,
        b,
        cell,
        alpha_tensor,
        mesh_dimensions,
        spline_order,
        batch_idx,
        compute_forces=compute_forces,
        compute_virial=compute_virial,
        k_vectors=k_vectors,
        k_squared=k_squared,
        volume=volume,
        cell_inv_t=cell_inv_t,
        moduli_x=moduli_x,
        moduli_y=moduli_y,
        moduli_z=moduli_z,
    )

    match (compute_forces, compute_virial):
        case (True, True):
            return energies, forces, virial
        case (True, False):
            return energies, forces
        case (False, True):
            return energies, virial
        case _:
            return energies


###########################################################################################
# Full dispersion PME (real + reciprocal)
###########################################################################################


def dispersion_pme(
    positions: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    cell: torch.Tensor,
    alpha: float | torch.Tensor | None = None,
    mesh_spacing: float | None = None,
    mesh_dimensions: tuple[int, int, int] | None = None,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    k_vectors: torch.Tensor | None = None,
    k_squared: torch.Tensor | None = None,
    cell_inv_t: torch.Tensor | None = None,
    volume: torch.Tensor | None = None,
    moduli_x: torch.Tensor | None = None,
    moduli_y: torch.Tensor | None = None,
    moduli_z: torch.Tensor | None = None,
    real_space_cutoff: float | None = None,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    neighbor_shifts: torch.Tensor | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    fill_value: int | None = None,
    compute_forces: bool = False,
    compute_virial: bool = False,
    accuracy: float = 1e-6,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""Complete dispersion (LJ-PME) calculation: real + reciprocal :math:`r^{-6}`.

    Parameters mirror ``particle_mesh_ewald`` but take per-atom LJ
    ``sigma``/``epsilon`` (converted to ``b_i = sqrt(C6_i)``). ``alpha`` is the
    dispersion splitting parameter :math:`\beta`; if None it (and the mesh /
    cutoff) is estimated via :func:`estimate_dispersion_pme_parameters`.

    The real-space half (damped :math:`r^{-6}`) is evaluated over the supplied
    neighbor list with cutoff ``real_space_cutoff``. Returns per-atom total
    energies, optionally followed by forces ``(N, 3)`` and/or virial
    ``(B, 3, 3)``.

    Notes
    -----
    The reciprocal virial is computed with a dedicated k-space expression;
    the real-space virial is obtained from the differentiable pairwise energy
    via a strain derivative.
    """
    cell, num_systems = _prepare_cell(cell)

    if alpha is None:
        params = estimate_dispersion_pme_parameters(
            positions, cell, batch_idx, accuracy, real_space_cutoff
        )
        alpha = params.alpha
        if real_space_cutoff is None:
            real_space_cutoff = float(params.real_space_cutoff.max().item())
        if mesh_dimensions is None and mesh_spacing is None:
            mesh_dimensions = tuple(params.mesh_dimensions)

    alpha_tensor = _prepare_alpha(alpha, num_systems, positions.dtype, positions.device)
    # Scalar beta for the (per-pair) real-space kernel.
    beta = float(alpha_tensor.reshape(-1)[0].item())

    if real_space_cutoff is None:
        raise ValueError(
            "real_space_cutoff must be provided (or let alpha=None auto-estimate it)"
        )

    use_list = neighbor_list is not None and neighbor_shifts is not None
    use_matrix = neighbor_matrix is not None and neighbor_matrix_shifts is not None
    if not use_list and not use_matrix:
        raise ValueError(
            "Must provide a neighbor list/matrix for the real-space dispersion term"
        )

    # --- Real-space damped r^-6 (reuses Phase-1 pairwise kernels) ---
    real_kwargs = dict(
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        fill_value=fill_value,
        batch_idx=batch_idx,
    )
    if compute_forces:
        e_real, f_real = lj_dispersion_energy_forces(
            positions, sigma, epsilon, cell, real_space_cutoff, beta, **real_kwargs
        )
    else:
        e_real = lj_dispersion_energy(
            positions, sigma, epsilon, cell, real_space_cutoff, beta, **real_kwargs
        )
        f_real = None

    # --- Reciprocal-space (mesh) ---
    rec = dispersion_reciprocal_space(
        positions,
        sigma,
        epsilon,
        cell,
        alpha_tensor,
        mesh_dimensions=mesh_dimensions,
        mesh_spacing=mesh_spacing,
        spline_order=spline_order,
        batch_idx=batch_idx,
        k_vectors=k_vectors,
        k_squared=k_squared,
        volume=volume,
        cell_inv_t=cell_inv_t,
        moduli_x=moduli_x,
        moduli_y=moduli_y,
        moduli_z=moduli_z,
        compute_forces=compute_forces,
        compute_virial=compute_virial,
    )

    if compute_forces and compute_virial:
        e_rec, f_rec, v_rec = rec
    elif compute_forces:
        e_rec, f_rec = rec
        v_rec = None
    elif compute_virial:
        e_rec, v_rec = rec
        f_rec = None
    else:
        e_rec, f_rec, v_rec = rec, None, None

    energies = e_real.to(positions.dtype) + e_rec.to(positions.dtype)

    outputs: list[torch.Tensor] = [energies]
    if compute_forces:
        outputs.append(f_real + f_rec)
    if compute_virial:
        v_real = _real_space_virial(
            positions,
            sigma,
            epsilon,
            cell,
            real_space_cutoff,
            beta,
            num_systems,
            batch_idx,
            real_kwargs,
        )
        outputs.append(v_real + v_rec)

    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _real_space_virial(
    positions, sigma, epsilon, cell, cutoff, beta, num_systems, batch_idx, real_kwargs
):
    r"""Real-space dispersion virial ``W = -dE_real/dε`` via a strain derivative.

    Applies a symmetric virtual strain ``ε`` to positions and cell, then
    differentiates the (autograd-capable) pairwise energy. Forward-only
    (the returned virial itself is detached).
    """
    device = positions.device
    dtype = positions.dtype
    if num_systems == 1:
        strain = torch.zeros(3, 3, device=device, dtype=dtype, requires_grad=True)
        eye = torch.eye(3, device=device, dtype=dtype)
        pos_s = positions @ (eye + strain).T
        cell_s = cell @ (eye + strain).T
        e = lj_dispersion_energy(
            pos_s, sigma, epsilon, cell_s, cutoff, beta, **real_kwargs
        ).sum()
        (grad,) = torch.autograd.grad(e, strain)
        # W = -dE/dε, symmetrized.
        w = -0.5 * (grad + grad.T)
        return w.unsqueeze(0).detach()
    # Batched: per-system strain.
    strain = torch.zeros(
        num_systems, 3, 3, device=device, dtype=dtype, requires_grad=True
    )
    eye = torch.eye(3, device=device, dtype=dtype)
    per_atom_strain = strain[batch_idx]  # (N, 3, 3)
    pos_s = torch.einsum("nij,nj->ni", eye + per_atom_strain, positions)
    cell_s = torch.einsum("bij,bkj->bki", eye + strain, cell)
    e = lj_dispersion_energy(
        pos_s, sigma, epsilon, cell_s, cutoff, beta, **real_kwargs
    ).sum()
    (grad,) = torch.autograd.grad(e, strain)
    w = -0.5 * (grad + grad.transpose(-1, -2))
    return w.detach()
