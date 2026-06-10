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
PyTorch bindings for Multipole Particle-Mesh Ewald (PME).

Per-Warp-kernel ``torch.autograd.Function`` wrappers built on top of the
launchers in
``nvalchemiops/interactions/electrostatics/pme_multipole_kernels.py``.

Design:

* One ``torch.autograd.Function`` per Warp kernel — keeps autograd
  boundaries small and bisectable, exposes intermediates as torch
  tensors so distributed utilities can allreduce them, and gives
  ``torch.compile`` discrete custom-op edges.
* Analytical backward kernels (not Warp tape replay) — composes cleanly
  for double-backward via standard torch composition.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics.pme_multipole_kernels import (
    batch_multipole_pme_convolve_backward_launch,
    batch_multipole_pme_convolve_double_backward_launch,
    batch_multipole_pme_convolve_launch,
    batch_multipole_pme_corrections_backward_launch,
    batch_multipole_pme_corrections_double_backward_launch,
    batch_multipole_pme_corrections_launch,
    batch_multipole_pme_gather_gradient_launch,
    batch_multipole_pme_green_structure_factor_launch,
    batch_multipole_pme_octupole_backward_launch,
    batch_multipole_pme_octupole_spread_launch,
    batch_multipole_pme_spread_backward_launch,
    batch_multipole_pme_spread_backward_unified_launch,
    batch_multipole_pme_spread_launch,
    batch_multipole_pme_spread_unified_launch,
    bspline_moduli_1d_launch,
    multipole_pme_convolve_backward_launch,
    multipole_pme_convolve_double_backward_launch,
    multipole_pme_convolve_launch,
    multipole_pme_corrections_backward_launch,
    multipole_pme_corrections_double_backward_launch,
    multipole_pme_corrections_launch,
    multipole_pme_gather_gradient_launch,
    multipole_pme_gather_hessian_launch,
    multipole_pme_green_structure_factor_launch,
    multipole_pme_mesh_inner_product_backward_launch,
    multipole_pme_mesh_inner_product_double_backward_launch,
    multipole_pme_mesh_inner_product_launch,
    multipole_pme_octupole_backward_launch,
    multipole_pme_octupole_spread_launch,
    multipole_pme_spread_backward_unified_launch,
    multipole_pme_spread_launch,
    multipole_pme_spread_unified_launch,
    multipole_reciprocal_rho_energy_backward_launch,
    multipole_reciprocal_rho_energy_double_backward_launch,
    multipole_reciprocal_rho_energy_launch,
    multipole_self_energy_backward_launch,
    multipole_self_energy_double_backward_launch,
    multipole_self_energy_launch,
    pme_effective_moments_launch,
    pme_fractionalize_backward_launch,
    pme_fractionalize_double_backward_launch,
    pme_fractionalize_launch,
    pme_k_squared_backward_launch,
    pme_k_squared_launch,
    pme_spread_dbwd_readout_launch,
)
from nvalchemiops.math.spline import (
    batch_spline_gather,
    batch_spline_gather_gradient,
    spline_gather,
    spline_gather_gradient,
)
from nvalchemiops.torch._warp_op_helpers import (
    register_warp_op_chain,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    infer_l_max,
    split_multipole_moments,
)
from nvalchemiops.torch.math import FIELD_CONSTANT
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype


def _scoped_warp_stream(device: torch.device):
    """Bind Warp's current stream to PyTorch's current CUDA stream.

    Required for ``torch.cuda.graph`` capture so Warp kernel launches end
    up on the stream being captured rather than Warp's default stream.
    Same as the monopole branch's ``_pme_scoped_warp_stream``.
    """
    if device.type != "cuda":
        return nullcontext()
    torch_stream = torch.cuda.current_stream(device)
    return wp.ScopedStream(wp.stream_from_torch(torch_stream))


def _wp_from_torch(tensor: torch.Tensor, dtype):
    """``wp.from_torch`` with shadow-gradient allocation disabled.

    Default ``wp.from_torch`` inherits ``requires_grad`` from the source
    tensor and allocates a Warp-side gradient buffer when True. That
    allocation breaks ``torch.cuda.graph`` capture
    (``cudaErrorStreamCaptureInvalidated``). Our backward custom_ops own
    the analytical backward, so the shadow grad is unused — force
    ``requires_grad=False``.
    """
    return wp.from_torch(tensor, dtype=dtype, requires_grad=False)


@torch.library.custom_op(
    "nvalchemiops::multipole_pme_bspline_moduli_1d",
    mutates_args=(),
)
def _bspline_moduli_1d_op(miller: torch.Tensor, n: int, order: int) -> torch.Tensor:
    r"""``b[i] = sinc(miller[i] / n)^order`` via the Warp ``bspline_moduli_1d`` kernel.

    Opaque (``torch.compile``-safe) wrapper around the Warp moduli kernel —
    keeps the ``sinc^order`` transcendental in Warp (mirroring the monopole
    ``compute_sinc``) instead of host ``torch.sinc``. Non-differentiable: the
    moduli depend only on integer Miller indices and the mesh dim, so there is
    no gradient to register.
    """
    device = miller.device
    wp_device = str(wp.device_from_torch(device))
    miller64 = miller.to(torch.float64).contiguous()
    out = torch.empty(miller.shape[0], dtype=torch.float64, device=device)
    bspline_moduli_1d_launch(
        wp.from_torch(miller64, dtype=wp.float64),
        1.0 / float(n),
        int(order),
        wp.from_torch(out, dtype=wp.float64),
        device=wp_device,
    )
    return out


@torch.library.register_fake("nvalchemiops::multipole_pme_bspline_moduli_1d")
def _bspline_moduli_1d_fake(
    miller: torch.Tensor, n: int, order: int
) -> torch.Tensor:  # pragma: no cover
    """Shape/dtype metadata: ``(n_axis,)`` float64."""
    return miller.new_empty((miller.shape[0],), dtype=torch.float64)


def _compute_bspline_moduli(
    mesh_dimensions: tuple[int, int, int],
    spline_order: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Precompute 1-D B-spline modulus LUTs ``b_a[i] = sinc(mi/N_a)^order``.

    Hoisting the per-axis ``sinc^order`` to 1-D LUTs reduces the Green's
    kernel cost to 3 reads + 2 multiplies per thread. The ``sinc^order`` math
    runs in the opaque Warp ``multipole_pme_bspline_moduli_1d`` op (Warp
    ``compute_sinc``); only the integer Miller-index generation stays in torch.
    The returned LUTs cover the rfft layout: ``b_z`` is length
    ``Nz_rfft = Nz // 2 + 1``.
    """
    nx, ny, nz = mesh_dimensions
    nz_rfft = nz // 2 + 1
    # Integer Miller indices (fftfreq layout for the full axes, 0..Nz_rfft-1 for
    # the rfft axis). Index generation only; the sinc^order is the Warp op.
    miller_x = torch.fft.fftfreq(nx, d=1.0 / nx, device=device, dtype=torch.float64)
    miller_y = torch.fft.fftfreq(ny, d=1.0 / ny, device=device, dtype=torch.float64)
    miller_z = torch.arange(nz_rfft, device=device, dtype=torch.float64)
    op = torch.ops.nvalchemiops.multipole_pme_bspline_moduli_1d
    bx = op(miller_x, nx, spline_order)
    by = op(miller_y, ny, spline_order)
    bz = op(miller_z, nz, spline_order)
    return (
        bx.to(dtype).contiguous(),
        by.to(dtype).contiguous(),
        bz.to(dtype).contiguous(),
    )


def _resolve_pme_moduli(
    mesh_dimensions: tuple[int, int, int],
    spline_order: int,
    dtype: torch.dtype,
    device: torch.device,
    moduli: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the moduli triplet, computing on the fly if the caller did not
    pass a precomputed one."""
    if moduli is not None:
        bx, by, bz = moduli
        return (
            bx.to(dtype=dtype, device=device).contiguous(),
            by.to(dtype=dtype, device=device).contiguous(),
            bz.to(dtype=dtype, device=device).contiguous(),
        )
    return _compute_bspline_moduli(mesh_dimensions, spline_order, dtype, device)


def _resolve_pme_k_squared(
    cell: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    dtype: torch.dtype,
    k_squared: torch.Tensor | None,
) -> torch.Tensor:
    """Return the precomputed ``k_squared`` rfft grid, or compute on the fly.

    Caching ``k_squared`` is a large reciprocal-space-side win for MD
    steady-state callers: it depends only on ``cell`` and
    ``mesh_dimensions`` (both constant once equilibrated).
    """
    if k_squared is not None:
        nx, ny, nz = mesh_dimensions
        expected = (nx, ny, nz // 2 + 1)
        if tuple(k_squared.shape) != expected:
            raise ValueError(
                f"k_squared shape {tuple(k_squared.shape)} != expected {expected} "
                f"(rfft grid for mesh_dimensions={mesh_dimensions})"
            )
        return k_squared.to(dtype=dtype, device=cell.device).contiguous()
    return _build_pme_k_grids(cell, mesh_dimensions, dtype)


def _resolve_batch_pme_k_squared(
    cells: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    dtype: torch.dtype,
    k_squared: torch.Tensor | None,
) -> torch.Tensor:
    """Batched companion to :func:`_resolve_pme_k_squared`."""
    if k_squared is not None:
        nx, ny, nz = mesh_dimensions
        expected = (cells.shape[0], nx, ny, nz // 2 + 1)
        if tuple(k_squared.shape) != expected:
            raise ValueError(
                f"k_squared shape {tuple(k_squared.shape)} != expected {expected} "
                f"(B={cells.shape[0]}, rfft grid for mesh_dimensions={mesh_dimensions})"
            )
        return k_squared.to(dtype=dtype, device=cells.device).contiguous()
    return _build_batch_pme_k_grids(cells, mesh_dimensions, dtype)


def _multipole_pme_convolve_run(
    mesh_fft: torch.Tensor,
    k_squared: torch.Tensor,
    moduli_x: torch.Tensor,
    moduli_y: torch.Tensor,
    moduli_z: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    volume: torch.Tensor,
    is_batch: bool,
) -> torch.Tensor:
    """Run the fused multipole-PME convolve kernel (factored out for
    forward+backward custom_ops; both apply the same multiplicative
    factor since it's real)."""
    device = mesh_fft.device
    real_dtype = k_squared.dtype
    wp_scalar = get_wp_dtype(real_dtype)
    wp_vec2 = wp.vec2d if wp_scalar == wp.float64 else wp.vec2f

    mesh_fft_real = torch.view_as_real(mesh_fft.resolve_conj()).contiguous()
    convolved_mesh_real = torch.empty_like(mesh_fft_real)

    launch = (
        batch_multipole_pme_convolve_launch
        if is_batch
        else multipole_pme_convolve_launch
    )

    def _as(t):
        return t.detach().to(real_dtype).contiguous()

    with _scoped_warp_stream(device):
        launch(
            _wp_from_torch(mesh_fft_real, dtype=wp_vec2),
            _wp_from_torch(k_squared.contiguous(), dtype=wp_scalar),
            _wp_from_torch(moduli_x.contiguous(), dtype=wp_scalar),
            _wp_from_torch(moduli_y.contiguous(), dtype=wp_scalar),
            _wp_from_torch(moduli_z.contiguous(), dtype=wp_scalar),
            _wp_from_torch(_as(alpha), dtype=wp_scalar),
            _wp_from_torch(_as(sigma), dtype=wp_scalar),
            _wp_from_torch(_as(volume), dtype=wp_scalar),
            _wp_from_torch(convolved_mesh_real, dtype=wp_vec2),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return torch.view_as_complex(convolved_mesh_real)


def _multipole_pme_convolve_forward(
    mesh_fft: torch.Tensor,
    k_squared: torch.Tensor,
    moduli_x: torch.Tensor,
    moduli_y: torch.Tensor,
    moduli_z: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Forward op for the single-system fused multipole-PME convolve."""
    return _multipole_pme_convolve_run(
        mesh_fft,
        k_squared,
        moduli_x,
        moduli_y,
        moduli_z,
        alpha,
        sigma,
        volume,
        is_batch=False,
    )


def _multipole_pme_convolve_backward(  # pragma: no cover
    grad_convolved: torch.Tensor,
    mesh_fft: torch.Tensor,
    k_squared: torch.Tensor,
    moduli_x: torch.Tensor,
    moduli_y: torch.Tensor,
    moduli_z: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    volume: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Backward op — returns ``(grad_mesh_fft, grad_k_squared, grad_volume)``.

    With ``convolved = mesh_fft · factor`` and
    :math:`\text{factor} = 2\pi \exp(-(1/(4\alpha^2) + \sigma^2) k^2) / (V k^2 \mathrm{sf}^2)`:

    - :math:`\partial L/\partial \text{mesh\_fft} = \text{grad\_convolved} \cdot \text{factor}`.
    - :math:`\partial L/\partial k^2 = \mathrm{Re}(\text{grad\_convolved} \cdot \overline{\text{mesh\_fft}}) \cdot \text{factor} \cdot (-(1/(4\alpha^2)+\sigma^2) - 1/k^2)` per cell.
    - :math:`\partial L/\partial V = -\sum_g \mathrm{Re}(\text{grad\_convolved} \cdot \overline{\text{mesh\_fft}}) \cdot \text{factor} / V` (scalar).
    """
    if not grad_convolved.is_complex():
        grad_convolved = grad_convolved.to(
            torch.complex64 if k_squared.dtype == torch.float32 else torch.complex128
        )
    device = mesh_fft.device
    real_dtype = k_squared.dtype
    wp_scalar = get_wp_dtype(real_dtype)
    wp_vec2 = wp.vec2d if wp_scalar == wp.float64 else wp.vec2f

    mesh_fft_real = torch.view_as_real(mesh_fft.resolve_conj()).contiguous()
    grad_conv_real = torch.view_as_real(grad_convolved.resolve_conj()).contiguous()
    grad_mesh_fft_real = torch.empty_like(mesh_fft_real)
    grad_k_squared = torch.zeros_like(k_squared)
    grad_volume = torch.zeros((1,), dtype=real_dtype, device=device)

    def _as(t):
        return t.detach().to(real_dtype).contiguous()

    with _scoped_warp_stream(device):
        multipole_pme_convolve_backward_launch(
            _wp_from_torch(mesh_fft_real, dtype=wp_vec2),
            _wp_from_torch(k_squared.contiguous(), dtype=wp_scalar),
            _wp_from_torch(moduli_x.contiguous(), dtype=wp_scalar),
            _wp_from_torch(moduli_y.contiguous(), dtype=wp_scalar),
            _wp_from_torch(moduli_z.contiguous(), dtype=wp_scalar),
            _wp_from_torch(_as(alpha), dtype=wp_scalar),
            _wp_from_torch(_as(sigma), dtype=wp_scalar),
            _wp_from_torch(_as(volume), dtype=wp_scalar),
            _wp_from_torch(grad_conv_real, dtype=wp_vec2),
            _wp_from_torch(grad_mesh_fft_real, dtype=wp_vec2),
            _wp_from_torch(grad_k_squared, dtype=wp_scalar),
            _wp_from_torch(grad_volume, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    grad_mesh_fft = torch.view_as_complex(grad_mesh_fft_real)
    return grad_mesh_fft, grad_k_squared, grad_volume


def _batch_multipole_pme_convolve_forward(
    mesh_fft: torch.Tensor,
    k_squared: torch.Tensor,
    moduli_x: torch.Tensor,
    moduli_y: torch.Tensor,
    moduli_z: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Forward op for the batched fused multipole-PME convolve."""
    return _multipole_pme_convolve_run(
        mesh_fft,
        k_squared,
        moduli_x,
        moduli_y,
        moduli_z,
        alpha,
        sigma,
        volume,
        is_batch=True,
    )


def _batch_multipole_pme_convolve_backward(  # pragma: no cover
    grad_convolved: torch.Tensor,
    mesh_fft: torch.Tensor,
    k_squared: torch.Tensor,
    moduli_x: torch.Tensor,
    moduli_y: torch.Tensor,
    moduli_z: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    volume: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward op for the batched fused multipole-PME convolve.

    Returns ``(grad_mesh_fft, grad_k_squared, grad_volume)`` per system, the
    batched analog of :func:`_multipole_pme_convolve_backward`."""
    if not grad_convolved.is_complex():
        grad_convolved = grad_convolved.to(
            torch.complex64 if k_squared.dtype == torch.float32 else torch.complex128
        )
    device = mesh_fft.device
    real_dtype = k_squared.dtype
    wp_scalar = get_wp_dtype(real_dtype)
    wp_vec2 = wp.vec2d if wp_scalar == wp.float64 else wp.vec2f
    B = mesh_fft.shape[0]

    mesh_fft_real = torch.view_as_real(mesh_fft.resolve_conj()).contiguous()
    grad_conv_real = torch.view_as_real(grad_convolved.resolve_conj()).contiguous()
    grad_mesh_fft_real = torch.empty_like(mesh_fft_real)
    grad_k_squared = torch.zeros_like(k_squared)
    grad_volume = torch.zeros((B,), dtype=real_dtype, device=device)

    def _as(t):
        return t.detach().to(real_dtype).contiguous()

    with _scoped_warp_stream(device):
        batch_multipole_pme_convolve_backward_launch(
            _wp_from_torch(mesh_fft_real, dtype=wp_vec2),
            _wp_from_torch(k_squared.contiguous(), dtype=wp_scalar),
            _wp_from_torch(moduli_x.contiguous(), dtype=wp_scalar),
            _wp_from_torch(moduli_y.contiguous(), dtype=wp_scalar),
            _wp_from_torch(moduli_z.contiguous(), dtype=wp_scalar),
            _wp_from_torch(_as(alpha), dtype=wp_scalar),
            _wp_from_torch(_as(sigma), dtype=wp_scalar),
            _wp_from_torch(_as(volume), dtype=wp_scalar),
            _wp_from_torch(grad_conv_real, dtype=wp_vec2),
            _wp_from_torch(grad_mesh_fft_real, dtype=wp_vec2),
            _wp_from_torch(grad_k_squared, dtype=wp_scalar),
            _wp_from_torch(grad_volume, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    grad_mesh_fft = torch.view_as_complex(grad_mesh_fft_real)
    return grad_mesh_fft, grad_k_squared, grad_volume


def _convolve_forward_fake(mesh_fft, *_):  # pragma: no cover
    """Fake for the convolve forward: contiguous output matching the real op.

    The real forward returns ``view_as_complex`` of a freshly-allocated
    contiguous buffer, so the meta must be contiguous too — ``empty_like`` would
    inherit ``mesh_fft``'s (possibly non-contiguous FFT) strides and trip
    inductor's ``assert_size_stride``.
    """
    return mesh_fft.new_empty(mesh_fft.shape)


def _convolve_backward_fake(
    grad_convolved, mesh_fft, k_squared, *_
):  # pragma: no cover
    """Fake for the convolve backward: 3-tuple
    (grad_mesh_fft, grad_k_squared, grad_volume). ``grad_mesh_fft`` is
    contiguous (the real op allocates a fresh contiguous buffer)."""
    del grad_convolved
    return (
        mesh_fft.new_empty(mesh_fft.shape),
        torch.empty_like(k_squared),
        torch.empty((1,), dtype=k_squared.dtype, device=k_squared.device),
    )


def _convolve_double_backward_run(
    gg_mesh_fft,
    gg_k_squared,
    gg_volume,
    grad_convolved,
    mesh_fft,
    k_squared,
    moduli_x,
    moduli_y,
    moduli_z,
    alpha,
    sigma,
    volume,
    launch_fn,
    grad_volume_shape,
):
    r"""Shared single/batched convolve double-backward.

    Returns grads w.r.t. the backward's inputs at
    ``second_order_diff_positions = (0, 1, 2, 8)`` =
    ``(grad_convolved, mesh_fft, k_squared, volume)``. The ``gg_mesh_fft``
    cotangent is the force-loss path; ``gg_k_squared``/``gg_volume`` are the
    cell-coupled stress-loss paths (``create_graph`` through ``dE/dcell``).
    """
    device = mesh_fft.device
    real_dtype = k_squared.dtype
    wp_scalar = get_wp_dtype(real_dtype)
    wp_vec2 = wp.vec2d if wp_scalar == wp.float64 else wp.vec2f
    cdtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128

    if gg_mesh_fft is None:
        gg_mesh_fft = torch.zeros_like(mesh_fft)
    elif not gg_mesh_fft.is_complex():
        gg_mesh_fft = gg_mesh_fft.to(cdtype)
    if gg_k_squared is None:
        gg_k_squared = torch.zeros_like(k_squared)
    if gg_volume is None:
        gg_volume = torch.zeros(grad_volume_shape, dtype=real_dtype, device=device)

    mesh_fft_real = torch.view_as_real(mesh_fft.resolve_conj()).contiguous()
    grad_conv_real = torch.view_as_real(grad_convolved.resolve_conj()).contiguous()
    gg_mesh_real = torch.view_as_real(gg_mesh_fft.resolve_conj()).contiguous()

    d_gc_real = torch.empty_like(mesh_fft_real)
    d_mf_real = torch.empty_like(mesh_fft_real)
    d_ksq = torch.empty_like(k_squared)
    d_vol = torch.zeros(grad_volume_shape, dtype=real_dtype, device=device)

    def _as(t):
        return t.detach().to(real_dtype).contiguous()

    with _scoped_warp_stream(device):
        launch_fn(
            _wp_from_torch(mesh_fft_real, dtype=wp_vec2),
            _wp_from_torch(k_squared.contiguous(), dtype=wp_scalar),
            _wp_from_torch(moduli_x.contiguous(), dtype=wp_scalar),
            _wp_from_torch(moduli_y.contiguous(), dtype=wp_scalar),
            _wp_from_torch(moduli_z.contiguous(), dtype=wp_scalar),
            _wp_from_torch(_as(alpha), dtype=wp_scalar),
            _wp_from_torch(_as(sigma), dtype=wp_scalar),
            _wp_from_torch(_as(volume), dtype=wp_scalar),
            _wp_from_torch(grad_conv_real, dtype=wp_vec2),
            _wp_from_torch(gg_mesh_real, dtype=wp_vec2),
            _wp_from_torch(gg_k_squared.contiguous(), dtype=wp_scalar),
            _wp_from_torch(_as(gg_volume), dtype=wp_scalar),
            _wp_from_torch(d_gc_real, dtype=wp_vec2),
            _wp_from_torch(d_mf_real, dtype=wp_vec2),
            _wp_from_torch(d_ksq, dtype=wp_scalar),
            _wp_from_torch(d_vol, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return (
        torch.view_as_complex(d_gc_real),
        torch.view_as_complex(d_mf_real),
        d_ksq,
        d_vol,
    )


def _multipole_pme_convolve_double_backward(  # pragma: no cover
    gg_mesh_fft,
    gg_k_squared,
    gg_volume,
    grad_convolved,
    mesh_fft,
    k_squared,
    moduli_x,
    moduli_y,
    moduli_z,
    alpha,
    sigma,
    volume,
):
    """Single-system convolve double-backward → (grad_convolved, mesh_fft,
    k_squared, volume) grads."""
    return _convolve_double_backward_run(
        gg_mesh_fft,
        gg_k_squared,
        gg_volume,
        grad_convolved,
        mesh_fft,
        k_squared,
        moduli_x,
        moduli_y,
        moduli_z,
        alpha,
        sigma,
        volume,
        multipole_pme_convolve_double_backward_launch,
        (1,),
    )


def _batch_multipole_pme_convolve_double_backward(  # pragma: no cover
    gg_mesh_fft,
    gg_k_squared,
    gg_volume,
    grad_convolved,
    mesh_fft,
    k_squared,
    moduli_x,
    moduli_y,
    moduli_z,
    alpha,
    sigma,
    volume,
):
    """Batched convolve double-backward → (grad_convolved, mesh_fft, k_squared,
    volume) grads; ``volume`` grad is per-system ``(B,)``."""
    return _convolve_double_backward_run(
        gg_mesh_fft,
        gg_k_squared,
        gg_volume,
        grad_convolved,
        mesh_fft,
        k_squared,
        moduli_x,
        moduli_y,
        moduli_z,
        alpha,
        sigma,
        volume,
        batch_multipole_pme_convolve_double_backward_launch,
        (mesh_fft.shape[0],),
    )


def _batch_convolve_backward_fake(
    grad_convolved, mesh_fft, k_squared, *_
):  # pragma: no cover
    """Fake for the batched convolve backward: per-system grad_volume (B,).

    ``grad_mesh_fft`` is contiguous (fresh buffer in the real op)."""
    return (
        mesh_fft.new_empty(mesh_fft.shape),
        torch.empty_like(k_squared),
        torch.empty(
            (mesh_fft.shape[0],), dtype=k_squared.dtype, device=k_squared.device
        ),
    )


def _convolve_double_backward_fake(  # pragma: no cover
    gg_mesh_fft, gg_k_squared, gg_volume, grad_convolved, mesh_fft, k_squared, *_args
):
    """Fake: grads of (grad_convolved, mesh_fft, k_squared, volume).

    The first two are complex (fresh contiguous buffers); k_squared real;
    volume ``(1,)`` single-system."""
    return (
        mesh_fft.new_empty(mesh_fft.shape),
        mesh_fft.new_empty(mesh_fft.shape),
        torch.empty_like(k_squared),
        torch.empty((1,), dtype=k_squared.dtype, device=k_squared.device),
    )


def _batch_convolve_double_backward_fake(  # pragma: no cover
    gg_mesh_fft, gg_k_squared, gg_volume, grad_convolved, mesh_fft, k_squared, *_args
):
    """Batched fake: ``volume`` grad is per-system ``(B,)``."""
    return (
        mesh_fft.new_empty(mesh_fft.shape),
        mesh_fft.new_empty(mesh_fft.shape),
        torch.empty_like(k_squared),
        torch.empty(
            (mesh_fft.shape[0],), dtype=k_squared.dtype, device=k_squared.device
        ),
    )


_CONVOLVE_DBWD_SCHEMA = (
    "(Tensor? gg_mesh_fft, Tensor? gg_k_squared, Tensor? gg_volume, "
    "Tensor grad_convolved, Tensor mesh_fft, Tensor k_squared, "
    "Tensor moduli_x, Tensor moduli_y, Tensor moduli_z, Tensor alpha, "
    "Tensor sigma, Tensor volume) -> (Tensor, Tensor, Tensor, Tensor)"
)
_BATCH_CONVOLVE_DBWD_SCHEMA = _CONVOLVE_DBWD_SCHEMA

register_warp_op_chain(
    name="nvalchemiops::multipole_pme_convolve",
    forward=_multipole_pme_convolve_forward,
    forward_fake=_convolve_forward_fake,
    backward=_multipole_pme_convolve_backward,
    backward_fake=_convolve_backward_fake,
    backward_return_arity=3,
    # mesh_fft (0), k_squared (1), volume (7) — cell-grad.
    diff_input_positions=(0, 1, 7),
    n_forward_inputs=8,
    # create_graph (force-loss + stress-loss). Backward inputs:
    # grad_convolved=0, mesh_fft=1, k_squared=2, ..., volume=8. The double-back
    # covers grad_convolved (force-loss) AND the k_squared/volume cotangent
    # directions (stress-loss: create_graph through dE/dcell).
    double_backward=_multipole_pme_convolve_double_backward,
    double_backward_fake=_convolve_double_backward_fake,
    double_backward_schema=_CONVOLVE_DBWD_SCHEMA,
    second_order_diff_positions=(0, 1, 2, 8),
    n_backward_inputs=9,
    double_backward_return_arity=4,
)


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_convolve_batch",
    forward=_batch_multipole_pme_convolve_forward,
    forward_fake=_convolve_forward_fake,
    backward=_batch_multipole_pme_convolve_backward,
    backward_fake=_batch_convolve_backward_fake,
    backward_return_arity=3,
    # mesh_fft (0), k_squared (1), volume (7) — batched stress (per-system
    # cell-grad) through the convolve.
    diff_input_positions=(0, 1, 7),
    n_forward_inputs=8,
    # create_graph force-loss + stress-loss (batched).
    double_backward=_batch_multipole_pme_convolve_double_backward,
    double_backward_fake=_batch_convolve_double_backward_fake,
    double_backward_schema=_BATCH_CONVOLVE_DBWD_SCHEMA,
    second_order_diff_positions=(0, 1, 2, 8),
    n_backward_inputs=9,
    double_backward_return_arity=4,
)


def multipole_pme_convolve(
    mesh_fft: torch.Tensor,
    k_squared: torch.Tensor,
    moduli_x: torch.Tensor,
    moduli_y: torch.Tensor,
    moduli_z: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    r"""Fused multipole-PME reciprocal-space convolution.

    Applies the Green's function, the GTO ``exp(-\sigma^2 k^2)`` factor and
    the squared B-spline modulus deconvolution to the Fourier-space density
    in a single Warp kernel pass, returning the convolved (potential) mesh
    in Fourier space.

    The per-cell multiplicative factor is real:

    .. math::

        \tilde{\phi}(k) = \tilde{\rho}(k)\,
        \frac{2\pi}{V\,k^2}\,
        \frac{\exp\!\big(-(1/(4\alpha^2) + \sigma^2)\,k^2\big)}
             {|C(k)|^2}

    where :math:`|C(k)|^2 = (b_x b_y b_z)^2` is the B-spline modulus built
    from ``moduli_x/y/z``. The :math:`k = 0` cell is set to zero to avoid the
    Coulomb singularity.

    Dispatches to the single-system or batched custom_op based on the rank of
    ``mesh_fft`` (3D → single, 4D → batched).

    Parameters
    ----------
    mesh_fft : torch.Tensor
        Complex Fourier-space density :math:`\tilde{\rho}(k)` from
        ``torch.fft.rfftn``. Shape ``(Nx, Ny, Nz_rfft)`` (single-system) or
        ``(B, Nx, Ny, Nz_rfft)`` (batched), complex dtype matching the real
        precision (``complex64`` for float32 inputs, ``complex128`` for
        float64).
    k_squared : torch.Tensor, shape ``(Nx, Ny, Nz_rfft)``
        :math:`|k|^2` at each rfft grid point. Real dtype (float32/float64);
        sets the working precision for the whole convolve.
    moduli_x : torch.Tensor, shape ``(Nx,)``
        1-D B-spline modulus LUT :math:`b_x = \mathrm{sinc}(m_x/N_x)^p` along
        x. Real dtype.
    moduli_y : torch.Tensor, shape ``(Ny,)``
        B-spline modulus LUT along y. Real dtype.
    moduli_z : torch.Tensor, shape ``(Nz_rfft,)``
        B-spline modulus LUT along z (rfft half-space). Real dtype.
    alpha : torch.Tensor, shape ``(1,)`` or ``(B,)``
        Ewald splitting parameter; ``(1,)`` single-system, ``(B,)`` batched.
        Real dtype.
    sigma : torch.Tensor, shape ``(1,)`` or ``(B,)``
        GTO basis width. Real dtype.
    volume : torch.Tensor, shape ``(1,)`` or ``(B,)``
        Cell volume :math:`V`. Real dtype.

    Returns
    -------
    convolved : torch.Tensor
        Convolved Fourier-space potential mesh :math:`\tilde{\phi}(k)`. Same
        shape and complex dtype as ``mesh_fft``.

    Notes
    -----
    The forward is an elementwise multiply by a real factor that depends only
    on ``k_squared``, the moduli, ``alpha``, ``sigma`` and ``volume`` (all
    position-independent), so the map is a real, diagonal, self-adjoint linear
    operator. Autograd flows to ``mesh_fft``, ``k_squared`` and ``volume``
    (the latter two enabling reciprocal-space stress / cell gradients), and
    ``create_graph=True`` (force-loss) is supported: the double-backward
    w.r.t. the cotangent is the same forward convolve applied to the upstream
    cotangent.
    """
    if mesh_fft.dim() == 4:
        return torch.ops.nvalchemiops.multipole_pme_convolve_batch(
            mesh_fft,
            k_squared,
            moduli_x,
            moduli_y,
            moduli_z,
            alpha,
            sigma,
            volume,
        )
    return torch.ops.nvalchemiops.multipole_pme_convolve(
        mesh_fft,
        k_squared,
        moduli_x,
        moduli_y,
        moduli_z,
        alpha,
        sigma,
        volume,
    )


def _resolve_cell_inv_t(
    cell: torch.Tensor, cell_inv_t: torch.Tensor | None
) -> torch.Tensor:
    """Return ``transpose(inv(cell))`` shaped ``(1, 3, 3)``.

    Mirrors the convention in ``nvalchemiops.torch.spline._spline_spread``:
    callers may pass either ``cell`` (shape ``(3, 3)`` or ``(1, 3, 3)``)
    or a precomputed ``cell_inv_t`` for MD steady-state where the cell
    is fixed.
    """
    if cell_inv_t is not None:
        if cell_inv_t.dim() == 2:
            cell_inv_t = cell_inv_t.unsqueeze(0)
        return cell_inv_t.contiguous()
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    cell_inv = torch.linalg.inv_ex(cell)[0]
    return cell_inv.transpose(-1, -2).contiguous()


# ---------------------------------------------------------------------------
# Unified (charges + dipoles + quadrupoles) spread custom_op — l_max = 2
# ---------------------------------------------------------------------------


def _multipole_pme_spread_unified_forward(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    cell_inv_t: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    lmax: int,
) -> torch.Tensor:
    """Unified spread (q + μ + Q) — picks (ORDER, LMAX, dtype) overload."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    mesh = torch.zeros((mesh_nx, mesh_ny, mesh_nz), dtype=input_dtype, device=device)
    with _scoped_warp_stream(device):
        ok = multipole_pme_spread_unified_launch(
            _wp_from_torch(positions.detach().contiguous(), dtype=wp_vec),
            _wp_from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            _wp_from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
            _wp_from_torch(quadrupoles.detach().contiguous(), dtype=wp_mat),
            _wp_from_torch(cell_inv_t.detach().contiguous(), dtype=wp_mat),
            order=spline_order,
            lmax=lmax,
            mesh=_wp_from_torch(mesh, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
        if not ok:
            raise NotImplementedError(
                f"Unified spread for (order={spline_order}, lmax={lmax}, "
                f"dtype={input_dtype}) is not registered."
            )
    return mesh


def _multipole_pme_spread_unified_backward(  # pragma: no cover
    grad_mesh: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    cell_inv_t: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unified spread backward — gradients w.r.t. positions, charges,
    dipoles, quadrupoles, cell_inv_t."""
    del mesh_nx, mesh_ny, mesh_nz
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    grad_positions = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    grad_charges = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    grad_dipoles = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    grad_quadrupoles = torch.zeros((n_atoms, 3, 3), dtype=input_dtype, device=device)
    grad_cell_inv_t = torch.zeros((3, 3), dtype=input_dtype, device=device)
    with _scoped_warp_stream(device):
        ok = multipole_pme_spread_backward_unified_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
            _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
            _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            order=spline_order,
            lmax=lmax,
            grad_mesh=_wp_from_torch(grad_mesh.contiguous(), dtype=wp_scalar),
            grad_positions=_wp_from_torch(grad_positions, dtype=wp_vec),
            grad_charges=_wp_from_torch(grad_charges, dtype=wp_scalar),
            grad_dipoles=_wp_from_torch(grad_dipoles, dtype=wp_vec),
            grad_quadrupoles=_wp_from_torch(grad_quadrupoles, dtype=wp_mat),
            grad_cell_inv_t=_wp_from_torch(grad_cell_inv_t, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
        if not ok:
            raise NotImplementedError(
                f"Unified spread-backward for (order={spline_order}, "
                f"lmax={lmax}, dtype={input_dtype}) is not registered."
            )
    # Custom_op expects ``cell_inv_t`` shape (1, 3, 3); the kernel writes a
    # (3, 3) buffer, so reshape.
    return (
        grad_positions,
        grad_charges,
        grad_dipoles,
        grad_quadrupoles,
        grad_cell_inv_t.unsqueeze(0),
    )


def _spread_unified_forward_fake(positions, *_args):  # pragma: no cover
    """Fake: output mesh shape ``(mesh_nx, mesh_ny, mesh_nz)``."""
    # positions=0, charges=1, dipoles=2, quadrupoles=3, cell_inv_t=4,
    # mesh_nx=5, mesh_ny=6, mesh_nz=7, spline_order=8, lmax=9.
    mesh_nx, mesh_ny, mesh_nz = _args[4], _args[5], _args[6]
    return torch.zeros(
        (mesh_nx, mesh_ny, mesh_nz),
        dtype=positions.dtype,
        device=positions.device,
    )


# ---------------------------------------------------------------------------
# Spread double-backward (create_graph=True through PME)
# ---------------------------------------------------------------------------


def _multipole_pme_spread_unified_double_backward(  # pragma: no cover
    gg_positions: torch.Tensor | None,
    gg_charges: torch.Tensor | None,
    gg_dipoles: torch.Tensor | None,
    gg_quadrupoles: torch.Tensor | None,
    gg_cell_inv_t: torch.Tensor | None,
    grad_mesh: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    cell_inv_t: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Double-backward of the unified spread (l_max 0/1/2).

    l_max<=1 is pure effective-moment reuse; l_max=2 adds the input-Q octupole
    ∇³/∇⁴ terms. Returns grads w.r.t. the backward op's inputs at
    ``second_order_diff_positions = (0, 1, 2, 3, 4)`` —
    ``(grad_mesh, positions, charges, dipoles, quadrupoles)``.
    """
    device = positions.device
    dtype = positions.dtype
    n = positions.shape[0]

    def _z(shape):
        return torch.zeros(shape, dtype=dtype, device=device)

    ggpos = gg_positions if gg_positions is not None else _z((n, 3))
    ggc = gg_charges if gg_charges is not None else _z((n,))
    ggd = gg_dipoles if gg_dipoles is not None else _z((n, 3))
    ggQ = gg_quadrupoles if gg_quadrupoles is not None else _z((n, 3, 3))
    ggpos = ggpos.to(dtype)

    # Effective moments (promote to an l_max=2 effective spread) — built by a
    # per-atom Warp kernel (no torch einsum/transpose).
    wp_scalar = get_wp_dtype(dtype)
    wp_vec = get_wp_vec_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    eff_c = ggc.to(dtype).contiguous()
    eff_d = torch.empty((n, 3), dtype=dtype, device=device)
    eff_Q = torch.empty((n, 3, 3), dtype=dtype, device=device)
    with _scoped_warp_stream(device):
        pme_effective_moments_launch(
            _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
            _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
            _wp_from_torch(ggpos.contiguous(), dtype=wp_vec),
            _wp_from_torch(ggd.to(dtype).contiguous(), dtype=wp_vec),
            _wp_from_torch(ggQ.to(dtype).contiguous(), dtype=wp_mat),
            _wp_from_torch(eff_d, dtype=wp_vec),
            _wp_from_torch(eff_Q, dtype=wp_mat),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )

    # ∂L/∂grad_mesh = forward spread of the effective moments (l_max=2).
    d_grad_mesh = _multipole_pme_spread_unified_forward(
        positions,
        eff_c,
        eff_d,
        eff_Q,
        cell_inv_t,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        spline_order,
        2,
    )

    # One backward spread with the effective moments yields the position-Hessian
    # (grad_positions) AND the moment-independent field readouts grad_dipoles
    # (= Mᵀ acc_f) and grad_quadrupoles (= ½ Mᵀ acc_H M).
    gpos2, _gc2, gd2, gQ2, _gcell2 = _multipole_pme_spread_unified_backward(
        grad_mesh,
        positions,
        eff_c,
        eff_d,
        eff_Q,
        cell_inv_t,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        spline_order,
        2,
    )

    d_positions = gpos2
    d_charges = torch.empty((n,), dtype=dtype, device=device)
    d_dipoles = torch.empty((n, 3), dtype=dtype, device=device)
    with _scoped_warp_stream(device):
        pme_spread_dbwd_readout_launch(
            _wp_from_torch(ggpos.contiguous(), dtype=wp_vec),
            _wp_from_torch(gd2.contiguous(), dtype=wp_vec),
            _wp_from_torch(gQ2.contiguous(), dtype=wp_mat),
            _wp_from_torch(d_charges, dtype=wp_scalar),
            _wp_from_torch(d_dipoles, dtype=wp_vec),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    d_quadrupoles = torch.zeros_like(quadrupoles)

    if lmax >= 2:
        # l_max=2 input-Q octupole terms (the only pieces the effective-moment
        # reuse does not cover): the directional-derivative Q-channel needs ∇³
        # (∂L/∂grad_mesh, ∂L/∂Q) and ∇⁴ (∂L/∂positions).
        wp_scalar = get_wp_dtype(dtype)
        wp_vec = get_wp_vec_dtype(dtype)
        wp_mat = get_wp_mat_dtype(dtype)
        ggpos_c = ggpos.contiguous()
        with _scoped_warp_stream(device):
            # (a) ∂L/∂grad_mesh += ½ Σ_ijk Qe_ij (M gg_pos)_k ∂³_frac B.
            octu_mesh = torch.zeros(
                (mesh_nx, mesh_ny, mesh_nz), dtype=dtype, device=device
            )
            multipole_pme_octupole_spread_launch(
                _wp_from_torch(positions.contiguous(), dtype=wp_vec),
                _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
                _wp_from_torch(ggpos_c, dtype=wp_vec),
                _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
                spline_order,
                _wp_from_torch(octu_mesh, dtype=wp_scalar),
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )
            # (b) ∂L/∂positions (∇⁴) + (c) ∂L/∂Q (∇³), both from grad_mesh.
            octu_pos = torch.zeros((n, 3), dtype=dtype, device=device)
            octu_q = torch.zeros((n, 3, 3), dtype=dtype, device=device)
            multipole_pme_octupole_backward_launch(
                _wp_from_torch(positions.contiguous(), dtype=wp_vec),
                _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
                _wp_from_torch(ggpos_c, dtype=wp_vec),
                _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
                _wp_from_torch(grad_mesh.contiguous(), dtype=wp_scalar),
                _wp_from_torch(octu_pos, dtype=wp_vec),
                _wp_from_torch(octu_q, dtype=wp_mat),
                spline_order,
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )
        d_grad_mesh = d_grad_mesh + octu_mesh
        d_positions = d_positions + octu_pos
        d_quadrupoles = d_quadrupoles + octu_q

    return d_grad_mesh, d_positions, d_charges, d_dipoles, d_quadrupoles


def _spread_unified_double_backward_fake(  # pragma: no cover
    gg_positions,
    gg_charges,
    gg_dipoles,
    gg_quadrupoles,
    gg_cell_inv_t,
    grad_mesh,
    positions,
    charges,
    dipoles,
    quadrupoles,
    cell_inv_t,
    *_args,
):
    """Fake: grads of (grad_mesh, positions, charges, dipoles, quadrupoles)."""
    return (
        torch.empty_like(grad_mesh),
        torch.empty_like(positions),
        torch.empty_like(charges),
        torch.empty_like(dipoles),
        torch.empty_like(quadrupoles),
    )


_SPREAD_DBWD_SCHEMA = (
    "(Tensor? gg_positions, Tensor? gg_charges, Tensor? gg_dipoles, "
    "Tensor? gg_quadrupoles, Tensor? gg_cell_inv_t, Tensor grad_mesh, "
    "Tensor positions, Tensor charges, Tensor dipoles, Tensor quadrupoles, "
    "Tensor cell_inv_t, int mesh_nx, int mesh_ny, int mesh_nz, "
    "int spline_order, int lmax) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_spread_unified",
    forward=_multipole_pme_spread_unified_forward,
    forward_fake=_spread_unified_forward_fake,
    backward=_multipole_pme_spread_unified_backward,
    backward_return_arity=5,
    # Differentiate w.r.t. positions, charges, dipoles, quadrupoles,
    # cell_inv_t — full forward-input coverage.
    diff_input_positions=(0, 1, 2, 3, 4),
    n_forward_inputs=10,
    # double-backward for create_graph=True. Backward inputs: grad_mesh=0,
    # positions=1, charges=2, dipoles=3, quadrupoles=4, cell_inv_t=5,
    # mesh_nx=6, ... (11 total).
    double_backward=_multipole_pme_spread_unified_double_backward,
    double_backward_fake=_spread_unified_double_backward_fake,
    double_backward_schema=_SPREAD_DBWD_SCHEMA,
    second_order_diff_positions=(0, 1, 2, 3, 4),
    n_backward_inputs=11,
)


def _batch_multipole_pme_spread_forward(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    B: int,
    spline_order: int,
) -> torch.Tensor:
    """Batched spread of (charges + dipoles) onto per-system meshes."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    mesh = torch.zeros((B, mesh_nx, mesh_ny, mesh_nz), dtype=input_dtype, device=device)
    with _scoped_warp_stream(device):
        batch_multipole_pme_spread_launch(
            _wp_from_torch(positions.detach().contiguous(), dtype=wp_vec),
            _wp_from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            _wp_from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(cell_inv_t.detach().contiguous(), dtype=wp_mat),
            order=spline_order,
            mesh=_wp_from_torch(mesh, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return mesh


def _batch_multipole_pme_spread_backward(  # pragma: no cover
    grad_mesh: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    B: int,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched analytical backward of the spread."""
    del mesh_nx, mesh_ny, mesh_nz, B
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    grad_positions = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    grad_charges = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    grad_dipoles = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    with _scoped_warp_stream(device):
        batch_multipole_pme_spread_backward_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
            _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            order=spline_order,
            grad_mesh=_wp_from_torch(grad_mesh.contiguous(), dtype=wp_scalar),
            grad_positions=_wp_from_torch(grad_positions, dtype=wp_vec),
            grad_charges=_wp_from_torch(grad_charges, dtype=wp_scalar),
            grad_dipoles=_wp_from_torch(grad_dipoles, dtype=wp_vec),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return grad_positions, grad_charges, grad_dipoles


def _batch_spread_forward_fake(positions, *_args):  # pragma: no cover
    """Fake for batched spread — derive from mesh_nx/y/z and B kwargs."""
    # positions=0, charges=1, dipoles=2, batch_idx=3, cell_inv_t=4,
    # mesh_nx=5, mesh_ny=6, mesh_nz=7, B=8, spline_order=9.
    mesh_nx, mesh_ny, mesh_nz, B = _args[4], _args[5], _args[6], _args[7]
    return torch.zeros(
        (B, mesh_nx, mesh_ny, mesh_nz),
        dtype=positions.dtype,
        device=positions.device,
    )


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_spread_batch",
    forward=_batch_multipole_pme_spread_forward,
    forward_fake=_batch_spread_forward_fake,
    backward=_batch_multipole_pme_spread_backward,
    backward_return_arity=3,
    diff_input_positions=(0, 1, 2),
    n_forward_inputs=10,
)


def _batch_multipole_pme_spread_unified_forward(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    B: int,
    spline_order: int,
    lmax: int,
) -> torch.Tensor:
    """Batched unified spread (q + μ + Q) onto per-system meshes."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    mesh = torch.zeros((B, mesh_nx, mesh_ny, mesh_nz), dtype=input_dtype, device=device)
    with _scoped_warp_stream(device):
        ok = batch_multipole_pme_spread_unified_launch(
            _wp_from_torch(positions.detach().contiguous(), dtype=wp_vec),
            _wp_from_torch(charges.detach().contiguous(), dtype=wp_scalar),
            _wp_from_torch(dipoles.detach().contiguous(), dtype=wp_vec),
            _wp_from_torch(quadrupoles.detach().contiguous(), dtype=wp_mat),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(cell_inv_t.detach().contiguous(), dtype=wp_mat),
            order=spline_order,
            lmax=lmax,
            mesh=_wp_from_torch(mesh, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
        if not ok:
            raise NotImplementedError(
                f"Batched unified spread for (order={spline_order}, lmax={lmax}, "
                f"dtype={input_dtype}) is not registered."
            )
    return mesh


def _batch_multipole_pme_spread_unified_backward(  # pragma: no cover
    grad_mesh: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    B: int,
    spline_order: int,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched unified spread backward — grads w.r.t. positions, charges,
    dipoles, quadrupoles, cell_inv_t (per-system, (B, 3, 3))."""
    del mesh_nx, mesh_ny, mesh_nz
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    grad_positions = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    grad_charges = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    grad_dipoles = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    grad_quadrupoles = torch.zeros((n_atoms, 3, 3), dtype=input_dtype, device=device)
    grad_cell_inv_t = torch.zeros((B, 3, 3), dtype=input_dtype, device=device)
    with _scoped_warp_stream(device):
        ok = batch_multipole_pme_spread_backward_unified_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
            _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
            _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            order=spline_order,
            lmax=lmax,
            grad_mesh=_wp_from_torch(grad_mesh.contiguous(), dtype=wp_scalar),
            grad_positions=_wp_from_torch(grad_positions, dtype=wp_vec),
            grad_charges=_wp_from_torch(grad_charges, dtype=wp_scalar),
            grad_dipoles=_wp_from_torch(grad_dipoles, dtype=wp_vec),
            grad_quadrupoles=_wp_from_torch(grad_quadrupoles, dtype=wp_mat),
            grad_cell_inv_t=_wp_from_torch(grad_cell_inv_t, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
        if not ok:
            raise NotImplementedError(
                f"Batched unified spread-backward for (order={spline_order}, "
                f"lmax={lmax}, dtype={input_dtype}) is not registered."
            )
    return (
        grad_positions,
        grad_charges,
        grad_dipoles,
        grad_quadrupoles,
        grad_cell_inv_t,
    )


def _batch_spread_unified_forward_fake(positions, *_args):  # pragma: no cover
    """Fake: output mesh shape ``(B, mesh_nx, mesh_ny, mesh_nz)``."""
    # positions=0, charges=1, dipoles=2, quadrupoles=3, batch_idx=4,
    # cell_inv_t=5, mesh_nx=6, mesh_ny=7, mesh_nz=8, B=9, spline_order=10,
    # lmax=11.
    mesh_nx, mesh_ny, mesh_nz, B = _args[5], _args[6], _args[7], _args[8]
    return torch.zeros(
        (B, mesh_nx, mesh_ny, mesh_nz),
        dtype=positions.dtype,
        device=positions.device,
    )


def _batch_multipole_pme_spread_unified_double_backward(  # pragma: no cover
    gg_positions: torch.Tensor | None,
    gg_charges: torch.Tensor | None,
    gg_dipoles: torch.Tensor | None,
    gg_quadrupoles: torch.Tensor | None,
    gg_cell_inv_t: torch.Tensor | None,
    grad_mesh: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    B: int,
    spline_order: int,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched double-backward of the unified spread (l_max 0/1/2).

    Batched analog of :func:`_multipole_pme_spread_unified_double_backward`:
    l_max<=1 is pure effective-moment reuse (batched fwd + bwd spread); l_max=2
    adds the input-Q ∇³/∇⁴ octupole terms via the batched octupole kernels.
    Returns grads w.r.t. the backward op's inputs at
    ``second_order_diff_positions = (0, 1, 2, 3, 4)`` —
    ``(grad_mesh, positions, charges, dipoles, quadrupoles)``.
    """
    device = positions.device
    dtype = positions.dtype
    n = positions.shape[0]

    def _z(shape):
        return torch.zeros(shape, dtype=dtype, device=device)

    ggpos = gg_positions if gg_positions is not None else _z((n, 3))
    ggc = gg_charges if gg_charges is not None else _z((n,))
    ggd = gg_dipoles if gg_dipoles is not None else _z((n, 3))
    ggQ = gg_quadrupoles if gg_quadrupoles is not None else _z((n, 3, 3))
    ggpos = ggpos.to(dtype)

    # Effective moments (promote to an l_max=2 effective spread) — built by a
    # per-atom Warp kernel (no torch einsum/transpose).
    wp_scalar = get_wp_dtype(dtype)
    wp_vec = get_wp_vec_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    eff_c = ggc.to(dtype).contiguous()
    eff_d = torch.empty((n, 3), dtype=dtype, device=device)
    eff_Q = torch.empty((n, 3, 3), dtype=dtype, device=device)
    with _scoped_warp_stream(device):
        pme_effective_moments_launch(
            _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
            _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
            _wp_from_torch(ggpos.contiguous(), dtype=wp_vec),
            _wp_from_torch(ggd.to(dtype).contiguous(), dtype=wp_vec),
            _wp_from_torch(ggQ.to(dtype).contiguous(), dtype=wp_mat),
            _wp_from_torch(eff_d, dtype=wp_vec),
            _wp_from_torch(eff_Q, dtype=wp_mat),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )

    # ∂L/∂grad_mesh = batched forward spread of the effective moments (l_max=2).
    d_grad_mesh = _batch_multipole_pme_spread_unified_forward(
        positions,
        eff_c,
        eff_d,
        eff_Q,
        batch_idx,
        cell_inv_t,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        B,
        spline_order,
        2,
    )

    # One batched backward spread with the effective moments yields the
    # position-Hessian (grad_positions) AND the moment-independent field
    # readouts grad_dipoles (= Mᵀ acc_f) and grad_quadrupoles (= ½ Mᵀ acc_H M).
    gpos2, _gc2, gd2, gQ2, _gcell2 = _batch_multipole_pme_spread_unified_backward(
        grad_mesh,
        positions,
        eff_c,
        eff_d,
        eff_Q,
        batch_idx,
        cell_inv_t,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        B,
        spline_order,
        2,
    )

    d_positions = gpos2
    d_charges = torch.empty((n,), dtype=dtype, device=device)
    d_dipoles = torch.empty((n, 3), dtype=dtype, device=device)
    with _scoped_warp_stream(device):
        pme_spread_dbwd_readout_launch(
            _wp_from_torch(ggpos.contiguous(), dtype=wp_vec),
            _wp_from_torch(gd2.contiguous(), dtype=wp_vec),
            _wp_from_torch(gQ2.contiguous(), dtype=wp_mat),
            _wp_from_torch(d_charges, dtype=wp_scalar),
            _wp_from_torch(d_dipoles, dtype=wp_vec),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    d_quadrupoles = torch.zeros_like(quadrupoles)

    if lmax >= 2:
        # l_max=2 input-Q octupole terms (the only pieces the effective-moment
        # reuse does NOT cover): ∇³ (∂L/∂grad_mesh, ∂L/∂Q) + ∇⁴ (∂L/∂positions).
        wp_scalar = get_wp_dtype(dtype)
        wp_vec = get_wp_vec_dtype(dtype)
        wp_mat = get_wp_mat_dtype(dtype)
        ggpos_c = ggpos.contiguous()
        with _scoped_warp_stream(device):
            octu_mesh = torch.zeros(
                (B, mesh_nx, mesh_ny, mesh_nz), dtype=dtype, device=device
            )
            batch_multipole_pme_octupole_spread_launch(
                _wp_from_torch(positions.contiguous(), dtype=wp_vec),
                _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
                _wp_from_torch(ggpos_c, dtype=wp_vec),
                _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
                _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
                spline_order,
                _wp_from_torch(octu_mesh, dtype=wp_scalar),
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )
            octu_pos = torch.zeros((n, 3), dtype=dtype, device=device)
            octu_q = torch.zeros((n, 3, 3), dtype=dtype, device=device)
            batch_multipole_pme_octupole_backward_launch(
                _wp_from_torch(positions.contiguous(), dtype=wp_vec),
                _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
                _wp_from_torch(ggpos_c, dtype=wp_vec),
                _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
                _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
                _wp_from_torch(grad_mesh.contiguous(), dtype=wp_scalar),
                _wp_from_torch(octu_pos, dtype=wp_vec),
                _wp_from_torch(octu_q, dtype=wp_mat),
                spline_order,
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )
        d_grad_mesh = d_grad_mesh + octu_mesh
        d_positions = d_positions + octu_pos
        d_quadrupoles = d_quadrupoles + octu_q

    return d_grad_mesh, d_positions, d_charges, d_dipoles, d_quadrupoles


def _batch_spread_unified_double_backward_fake(  # pragma: no cover
    gg_positions,
    gg_charges,
    gg_dipoles,
    gg_quadrupoles,
    gg_cell_inv_t,
    grad_mesh,
    positions,
    charges,
    dipoles,
    quadrupoles,
    *_args,
):
    """Fake: grads of (grad_mesh, positions, charges, dipoles, quadrupoles)."""
    return (
        torch.empty_like(grad_mesh),
        torch.empty_like(positions),
        torch.empty_like(charges),
        torch.empty_like(dipoles),
        torch.empty_like(quadrupoles),
    )


_BATCH_SPREAD_DBWD_SCHEMA = (
    "(Tensor? gg_positions, Tensor? gg_charges, Tensor? gg_dipoles, "
    "Tensor? gg_quadrupoles, Tensor? gg_cell_inv_t, Tensor grad_mesh, "
    "Tensor positions, Tensor charges, Tensor dipoles, Tensor quadrupoles, "
    "Tensor batch_idx, Tensor cell_inv_t, int mesh_nx, int mesh_ny, "
    "int mesh_nz, int B, int spline_order, int lmax) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_spread_unified_batch",
    forward=_batch_multipole_pme_spread_unified_forward,
    forward_fake=_batch_spread_unified_forward_fake,
    backward=_batch_multipole_pme_spread_unified_backward,
    backward_return_arity=5,
    # positions, charges, dipoles, quadrupoles, cell_inv_t (batch stress).
    diff_input_positions=(0, 1, 2, 3, 5),
    n_forward_inputs=12,
    batch_match=True,
    # batched create_graph force-loss. Backward inputs: grad_mesh=0,
    # positions=1, charges=2, dipoles=3, quadrupoles=4, batch_idx=5,
    # cell_inv_t=6, mesh_nx=7, ... (13 total).
    double_backward=_batch_multipole_pme_spread_unified_double_backward,
    double_backward_fake=_batch_spread_unified_double_backward_fake,
    double_backward_schema=_BATCH_SPREAD_DBWD_SCHEMA,
    second_order_diff_positions=(0, 1, 2, 3, 4),
    n_backward_inputs=13,
)


# ===========================================================================
# Gather-via-spread-transpose — the per-atom reciprocal energy primitive
# ===========================================================================
#
# The mesh-side reciprocal energy ``<ρ, φ>`` is *exactly* the moment-weighted
# sum of the spread-transpose ``Sᵀφ`` gathered to atoms:
#
#   <ρ, φ> = <S·m, φ> = <m, Sᵀφ> = Σ_i [ q_i g_q_i + d_i·g_d_i + Q_i:g_Q_i ]
#
# where ``(g_q, g_d, g_Q) = Sᵀφ`` are the gathered potential / field / hessian
# (the ½ on the quad channel matches the spread's own ½). Assembling the energy
# this way gives a genuine PER-ATOM decomposition (``E_i`` is atom i's share)
# while staying *bit-identical* to the mesh-mesh inner product — because ``Sᵀ``
# here is the spread's OWN backward, not the discretization-mismatched
# ``spline_gather``. Forward/backward/double-backward all reuse the unified
# spread's three Python launchers, so 2nd-order training composes for free.


def _mask_gather_cotangents(
    cg_d: torch.Tensor, cg_Q: torch.Tensor, lmax: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero the dipole/quad cotangents above ``lmax`` and symmetrize the quad.

    The spread ignores channels above ``lmax`` (their gathered outputs are
    constant zero), so feeding their cotangents into the spread's
    double-backward — which always promotes to an ``lmax=2`` effective spread —
    would synthesize spurious gradients. Masking keeps every spread primitive
    consistent with the op's actual ``lmax``.
    """
    if lmax < 1:
        cg_d = torch.zeros_like(cg_d)
    if lmax < 2:
        cg_Q = torch.zeros_like(cg_Q)
    else:
        cg_Q = 0.5 * (cg_Q + cg_Q.transpose(-1, -2))
    return cg_d, cg_Q


def _multipole_pme_gather_via_spread_t_forward(
    phi_grid: torch.Tensor,
    p_frac: torch.Tensor,
    identity_cell: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Gather ``Sᵀφ`` to atoms → ``(g_q (N,), g_d (N,3), g_Q (N,3,3))``.

    ``Sᵀ`` is the unified-spread backward: the spread ``ρ = S·m`` is linear in
    the moments, so its moment-gradients evaluated at *zero* moments are the
    gathered potential (``g_q``), field (``g_d``) and ½·hessian (``g_Q``).
    """
    n = p_frac.shape[0]
    dtype = p_frac.dtype
    device = p_frac.device
    zc = torch.zeros(n, dtype=dtype, device=device)
    zd = torch.zeros((n, 3), dtype=dtype, device=device)
    zq = torch.zeros((n, 3, 3), dtype=dtype, device=device)
    _gpos, g_q, g_d, g_Q, _gcell = _multipole_pme_spread_unified_backward(
        phi_grid,
        p_frac,
        zc,
        zd,
        zq,
        identity_cell,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        spline_order,
        lmax,
    )
    return g_q, g_d, g_Q


def _gather_via_spread_t_forward_fake(phi_grid, p_frac, *_args):  # pragma: no cover
    """Fake: ``(g_q (N,), g_d (N,3), g_Q (N,3,3))``."""
    del phi_grid
    n = p_frac.shape[0]
    return (
        torch.empty(n, dtype=p_frac.dtype, device=p_frac.device),
        torch.empty((n, 3), dtype=p_frac.dtype, device=p_frac.device),
        torch.empty((n, 3, 3), dtype=p_frac.dtype, device=p_frac.device),
    )


def _multipole_pme_gather_via_spread_t_backward(  # pragma: no cover
    cg_q: torch.Tensor,
    cg_d: torch.Tensor,
    cg_Q: torch.Tensor,
    phi_grid: torch.Tensor,
    p_frac: torch.Tensor,
    identity_cell: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Adjoint of ``g = Sᵀφ`` → grads w.r.t. ``(phi_grid, p_frac)``.

    With cotangent ``cg = (cg_q, cg_d, cg_Q)`` on ``g``:

    .. math::

        \partial/\partial\varphi &= S\cdot cg \quad
            (\text{forward spread of the cotangent-as-moments}) \\
        \partial/\partial p &= \partial\langle S\cdot cg, \varphi\rangle/\partial p
            \quad (\text{spread-backward grad\_positions, moments}=cg)

    Channels above ``lmax`` are inert in the spread (``g_d``/``g_Q`` are constant
    zero there), so their cotangents are masked out; the quad channel is also
    symmetrized to match the spread's symmetric storage (a no-op for real use).
    """
    cg_d, cg_Q = _mask_gather_cotangents(cg_d, cg_Q, lmax)
    d_phi = _multipole_pme_spread_unified_forward(
        p_frac,
        cg_q,
        cg_d,
        cg_Q,
        identity_cell,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        spline_order,
        lmax,
    )
    d_p, _gc, _gd, _gQ, _gcell = _multipole_pme_spread_unified_backward(
        phi_grid,
        p_frac,
        cg_q,
        cg_d,
        cg_Q,
        identity_cell,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        spline_order,
        lmax,
    )
    return d_phi, d_p


def _gather_via_spread_t_backward_fake(
    cg_q, cg_d, cg_Q, phi_grid, p_frac, *_args
):  # pragma: no cover
    """Fake: grads of ``(phi_grid, p_frac)``."""
    del cg_q, cg_d, cg_Q
    return torch.empty_like(phi_grid), torch.empty_like(p_frac)


def _multipole_pme_gather_via_spread_t_double_backward(  # pragma: no cover
    gg_phi: torch.Tensor | None,
    gg_p: torch.Tensor | None,
    cg_q: torch.Tensor,
    cg_d: torch.Tensor,
    cg_Q: torch.Tensor,
    phi_grid: torch.Tensor,
    p_frac: torch.Tensor,
    identity_cell: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Second-order backward of :func:`_multipole_pme_gather_via_spread_t_backward`.

    The backward maps ``cg`` to ``d_phi = S·cg`` and ``d_p = ∂<S·cg, φ>/∂p``.
    With cotangents ``gg_phi`` (on ``d_phi``) and ``gg_p`` (on ``d_p``), the
    scalar ``Φ = <gg_phi, S·cg> + <gg_p, ∂<S·cg, φ>/∂p>`` splits into two groups
    whose derivatives are *exactly* the unified spread's own backward and
    double-backward:

    * group A ``<gg_phi, S·cg>`` → ``∂/∂cg = Sᵀ·gg_phi`` (= gather of ``gg_phi``)
      and ``∂/∂p = ∂<S·cg, gg_phi>/∂p`` (spread-backward grad_positions).
    * group B ``<gg_p, Bpos(p, cg, φ)>`` is exactly the spread double-backward
      with ``gg_positions = gg_p``, ``grad_mesh = φ``, moments ``= cg`` —
      returning ``∂/∂φ``, ``∂/∂p`` and ``∂/∂cg``.

    Returns grads for the backward's diff inputs at
    ``second_order_diff_positions = (0, 1, 2, 3, 4)`` =
    ``(cg_q, cg_d, cg_Q, phi_grid, p_frac)``.
    """
    if gg_phi is None:
        gg_phi = torch.zeros_like(phi_grid)
    if gg_p is None:
        gg_p = torch.zeros_like(p_frac)
    # Mask inert channels above ``lmax`` (and symmetrize the quad) so every
    # spread primitive below agrees with the op's actual ``lmax``.
    cg_d, cg_Q = _mask_gather_cotangents(cg_d, cg_Q, lmax)

    # Group A: gather(gg_phi) = Sᵀ·gg_phi  (the op's own forward applied to gg_phi).
    gth_q, gth_d, gth_Q = _multipole_pme_gather_via_spread_t_forward(
        gg_phi,
        p_frac,
        identity_cell,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        spline_order,
        lmax,
    )
    # Group A: ∂<S·cg, gg_phi>/∂p  (spread-backward grad_positions, grad_mesh=gg_phi).
    a_dp, _gc, _gd, _gQ, _gcell = _multipole_pme_spread_unified_backward(
        gg_phi,
        p_frac,
        cg_q,
        cg_d,
        cg_Q,
        identity_cell,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        spline_order,
        lmax,
    )
    # Group B: spread double-backward with gg_positions = gg_p, grad_mesh = φ,
    # moments = cg → (∂/∂φ, ∂/∂p, ∂/∂cg_q, ∂/∂cg_d, ∂/∂cg_Q).
    d_phi_b, d_p_b, d_cg_q_b, d_cg_d_b, d_cg_Q_b = (
        _multipole_pme_spread_unified_double_backward(
            gg_p,
            None,
            None,
            None,
            None,
            phi_grid,
            p_frac,
            cg_q,
            cg_d,
            cg_Q,
            identity_cell,
            mesh_nx,
            mesh_ny,
            mesh_nz,
            spline_order,
            lmax,
        )
    )
    d_cg_q = gth_q + d_cg_q_b
    d_cg_d = gth_d + d_cg_d_b
    d_cg_Q = gth_Q + d_cg_Q_b
    d_phi = d_phi_b
    d_p = a_dp + d_p_b
    # ``Φ`` is independent of the cotangents on inert channels above ``lmax``;
    # zero their grads (the spread double-back's l=2 promotion would leak here).
    if lmax < 1:
        d_cg_d = torch.zeros_like(d_cg_d)
    if lmax < 2:
        d_cg_Q = torch.zeros_like(d_cg_Q)
    return d_cg_q, d_cg_d, d_cg_Q, d_phi, d_p


def _gather_via_spread_t_double_backward_fake(  # pragma: no cover
    gg_phi, gg_p, cg_q, cg_d, cg_Q, phi_grid, p_frac, *_args
):
    """Fake: grads of ``(cg_q, cg_d, cg_Q, phi_grid, p_frac)``."""
    del gg_phi, gg_p
    return (
        torch.empty_like(cg_q),
        torch.empty_like(cg_d),
        torch.empty_like(cg_Q),
        torch.empty_like(phi_grid),
        torch.empty_like(p_frac),
    )


_GATHER_VST_FWD_SCHEMA = (
    "(Tensor phi_grid, Tensor p_frac, Tensor identity_cell, int mesh_nx, "
    "int mesh_ny, int mesh_nz, int spline_order, int lmax) "
    "-> (Tensor, Tensor, Tensor)"
)

_GATHER_VST_BWD_SCHEMA = (
    "(Tensor cg_q, Tensor cg_d, Tensor cg_Q, Tensor phi_grid, Tensor p_frac, "
    "Tensor identity_cell, int mesh_nx, int mesh_ny, int mesh_nz, "
    "int spline_order, int lmax) -> (Tensor, Tensor)"
)

_GATHER_VST_DBWD_SCHEMA = (
    "(Tensor? gg_phi, Tensor? gg_p, Tensor cg_q, Tensor cg_d, Tensor cg_Q, "
    "Tensor phi_grid, Tensor p_frac, Tensor identity_cell, int mesh_nx, "
    "int mesh_ny, int mesh_nz, int spline_order, int lmax) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_gather_via_spread_t",
    forward=_multipole_pme_gather_via_spread_t_forward,
    forward_fake=_gather_via_spread_t_forward_fake,
    forward_return_arity=3,
    forward_schema=_GATHER_VST_FWD_SCHEMA,
    backward=_multipole_pme_gather_via_spread_t_backward,
    backward_fake=_gather_via_spread_t_backward_fake,
    backward_schema=_GATHER_VST_BWD_SCHEMA,
    backward_return_arity=2,
    # Differentiate w.r.t. phi_grid (0) and p_frac (1); identity_cell is the
    # fixed identity (cell-grad flows through fractionalize, not here).
    diff_input_positions=(0, 1),
    n_forward_inputs=8,
    # Backward inputs: cg_q=0, cg_d=1, cg_Q=2, phi_grid=3, p_frac=4,
    # identity_cell=5, mesh_nx=6, ... (11 total).
    double_backward=_multipole_pme_gather_via_spread_t_double_backward,
    double_backward_fake=_gather_via_spread_t_double_backward_fake,
    double_backward_schema=_GATHER_VST_DBWD_SCHEMA,
    second_order_diff_positions=(0, 1, 2, 3, 4),
    n_backward_inputs=11,
    double_backward_return_arity=5,
)


# --- Batched gather-via-spread-transpose -----------------------------------


def _batch_multipole_pme_gather_via_spread_t_forward(
    phi_grid: torch.Tensor,
    p_frac: torch.Tensor,
    batch_idx: torch.Tensor,
    identity_cell: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    B: int,
    spline_order: int,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched ``Sᵀφ`` gather → ``(g_q (N,), g_d (N,3), g_Q (N,3,3))``."""
    n = p_frac.shape[0]
    dtype = p_frac.dtype
    device = p_frac.device
    zc = torch.zeros(n, dtype=dtype, device=device)
    zd = torch.zeros((n, 3), dtype=dtype, device=device)
    zq = torch.zeros((n, 3, 3), dtype=dtype, device=device)
    _gpos, g_q, g_d, g_Q, _gcell = _batch_multipole_pme_spread_unified_backward(
        phi_grid,
        p_frac,
        zc,
        zd,
        zq,
        batch_idx,
        identity_cell,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        B,
        spline_order,
        lmax,
    )
    return g_q, g_d, g_Q


def _batch_gather_via_spread_t_forward_fake(
    phi_grid, p_frac, *_args
):  # pragma: no cover
    """Fake: ``(g_q (N,), g_d (N,3), g_Q (N,3,3))``."""
    del phi_grid
    n = p_frac.shape[0]
    return (
        torch.empty(n, dtype=p_frac.dtype, device=p_frac.device),
        torch.empty((n, 3), dtype=p_frac.dtype, device=p_frac.device),
        torch.empty((n, 3, 3), dtype=p_frac.dtype, device=p_frac.device),
    )


def _batch_multipole_pme_gather_via_spread_t_backward(  # pragma: no cover
    cg_q: torch.Tensor,
    cg_d: torch.Tensor,
    cg_Q: torch.Tensor,
    phi_grid: torch.Tensor,
    p_frac: torch.Tensor,
    batch_idx: torch.Tensor,
    identity_cell: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    B: int,
    spline_order: int,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched adjoint of ``g = Sᵀφ`` → grads w.r.t. ``(phi_grid, p_frac)``."""
    cg_d, cg_Q = _mask_gather_cotangents(cg_d, cg_Q, lmax)
    d_phi = _batch_multipole_pme_spread_unified_forward(
        p_frac,
        cg_q,
        cg_d,
        cg_Q,
        batch_idx,
        identity_cell,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        B,
        spline_order,
        lmax,
    )
    d_p, _gc, _gd, _gQ, _gcell = _batch_multipole_pme_spread_unified_backward(
        phi_grid,
        p_frac,
        cg_q,
        cg_d,
        cg_Q,
        batch_idx,
        identity_cell,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        B,
        spline_order,
        lmax,
    )
    return d_phi, d_p


def _batch_gather_via_spread_t_backward_fake(  # pragma: no cover
    cg_q, cg_d, cg_Q, phi_grid, p_frac, *_args
):
    """Fake: grads of ``(phi_grid, p_frac)``."""
    del cg_q, cg_d, cg_Q
    return torch.empty_like(phi_grid), torch.empty_like(p_frac)


def _batch_multipole_pme_gather_via_spread_t_double_backward(  # pragma: no cover
    gg_phi: torch.Tensor | None,
    gg_p: torch.Tensor | None,
    cg_q: torch.Tensor,
    cg_d: torch.Tensor,
    cg_Q: torch.Tensor,
    phi_grid: torch.Tensor,
    p_frac: torch.Tensor,
    batch_idx: torch.Tensor,
    identity_cell: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    B: int,
    spline_order: int,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched second-order backward (see the single-system docstring)."""
    if gg_phi is None:
        gg_phi = torch.zeros_like(phi_grid)
    if gg_p is None:
        gg_p = torch.zeros_like(p_frac)
    cg_d, cg_Q = _mask_gather_cotangents(cg_d, cg_Q, lmax)

    gth_q, gth_d, gth_Q = _batch_multipole_pme_gather_via_spread_t_forward(
        gg_phi,
        p_frac,
        batch_idx,
        identity_cell,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        B,
        spline_order,
        lmax,
    )
    a_dp, _gc, _gd, _gQ, _gcell = _batch_multipole_pme_spread_unified_backward(
        gg_phi,
        p_frac,
        cg_q,
        cg_d,
        cg_Q,
        batch_idx,
        identity_cell,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        B,
        spline_order,
        lmax,
    )
    d_phi_b, d_p_b, d_cg_q_b, d_cg_d_b, d_cg_Q_b = (
        _batch_multipole_pme_spread_unified_double_backward(
            gg_p,
            None,
            None,
            None,
            None,
            phi_grid,
            p_frac,
            cg_q,
            cg_d,
            cg_Q,
            batch_idx,
            identity_cell,
            mesh_nx,
            mesh_ny,
            mesh_nz,
            B,
            spline_order,
            lmax,
        )
    )
    d_cg_q = gth_q + d_cg_q_b
    d_cg_d = gth_d + d_cg_d_b
    d_cg_Q = gth_Q + d_cg_Q_b
    d_phi = d_phi_b
    d_p = a_dp + d_p_b
    if lmax < 1:
        d_cg_d = torch.zeros_like(d_cg_d)
    if lmax < 2:
        d_cg_Q = torch.zeros_like(d_cg_Q)
    return d_cg_q, d_cg_d, d_cg_Q, d_phi, d_p


def _batch_gather_via_spread_t_double_backward_fake(  # pragma: no cover
    gg_phi, gg_p, cg_q, cg_d, cg_Q, phi_grid, p_frac, *_args
):
    """Fake: grads of ``(cg_q, cg_d, cg_Q, phi_grid, p_frac)``."""
    del gg_phi, gg_p
    return (
        torch.empty_like(cg_q),
        torch.empty_like(cg_d),
        torch.empty_like(cg_Q),
        torch.empty_like(phi_grid),
        torch.empty_like(p_frac),
    )


_BATCH_GATHER_VST_FWD_SCHEMA = (
    "(Tensor phi_grid, Tensor p_frac, Tensor batch_idx, Tensor identity_cell, "
    "int mesh_nx, int mesh_ny, int mesh_nz, int B, int spline_order, int lmax) "
    "-> (Tensor, Tensor, Tensor)"
)

_BATCH_GATHER_VST_BWD_SCHEMA = (
    "(Tensor cg_q, Tensor cg_d, Tensor cg_Q, Tensor phi_grid, Tensor p_frac, "
    "Tensor batch_idx, Tensor identity_cell, int mesh_nx, int mesh_ny, "
    "int mesh_nz, int B, int spline_order, int lmax) -> (Tensor, Tensor)"
)

_BATCH_GATHER_VST_DBWD_SCHEMA = (
    "(Tensor? gg_phi, Tensor? gg_p, Tensor cg_q, Tensor cg_d, Tensor cg_Q, "
    "Tensor phi_grid, Tensor p_frac, Tensor batch_idx, Tensor identity_cell, "
    "int mesh_nx, int mesh_ny, int mesh_nz, int B, int spline_order, int lmax) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_gather_via_spread_t_batch",
    forward=_batch_multipole_pme_gather_via_spread_t_forward,
    forward_fake=_batch_gather_via_spread_t_forward_fake,
    forward_return_arity=3,
    forward_schema=_BATCH_GATHER_VST_FWD_SCHEMA,
    backward=_batch_multipole_pme_gather_via_spread_t_backward,
    backward_fake=_batch_gather_via_spread_t_backward_fake,
    backward_schema=_BATCH_GATHER_VST_BWD_SCHEMA,
    backward_return_arity=2,
    diff_input_positions=(0, 1),
    n_forward_inputs=10,
    batch_match=True,
    # Backward inputs: cg_q=0, cg_d=1, cg_Q=2, phi_grid=3, p_frac=4,
    # batch_idx=5, identity_cell=6, mesh_nx=7, ... (13 total).
    double_backward=_batch_multipole_pme_gather_via_spread_t_double_backward,
    double_backward_fake=_batch_gather_via_spread_t_double_backward_fake,
    double_backward_schema=_BATCH_GATHER_VST_DBWD_SCHEMA,
    second_order_diff_positions=(0, 1, 2, 3, 4),
    n_backward_inputs=13,
    double_backward_return_arity=5,
)


def _resolve_batch_cell_inv_t(
    cell: torch.Tensor, cell_inv_t: torch.Tensor | None
) -> torch.Tensor:
    """Return per-system ``transpose(inv(cell))`` shaped ``(B, 3, 3)``.

    For the batched path, ``cell`` must be ``(B, 3, 3)``; the helper
    inverts and transposes per-system. ``cell_inv_t`` may be passed
    pre-computed (same shape).
    """
    if cell_inv_t is not None:
        if cell_inv_t.dim() != 3 or cell_inv_t.shape[-2:] != (3, 3):
            raise ValueError(
                "batched cell_inv_t must be shape (B, 3, 3); got "
                f"{tuple(cell_inv_t.shape)}"
            )
        return cell_inv_t.contiguous()
    if cell.dim() != 3 or cell.shape[-2:] != (3, 3):
        raise ValueError(
            f"batched cell must be shape (B, 3, 3); got {tuple(cell.shape)}"
        )
    cell_inv = torch.linalg.inv_ex(cell)[0]
    return cell_inv.transpose(-1, -2).contiguous()


# =============================================================================
# Green's function + structure factor
# =============================================================================


def _multipole_pme_green_struct_forward(
    k_squared: torch.Tensor,
    miller_x: torch.Tensor,
    miller_y: torch.Tensor,
    miller_z: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    volume: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(G̃(k), |C(k)|²)`` from the multipole-PME Green's kernel."""
    device = k_squared.device
    input_dtype = k_squared.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    nx, ny, nz_rfft = k_squared.shape
    green_function = torch.zeros((nx, ny, nz_rfft), dtype=input_dtype, device=device)
    structure_factor_sq = torch.zeros(
        (nx, ny, nz_rfft), dtype=input_dtype, device=device
    )

    def _as(t):
        return t.detach().to(input_dtype).contiguous()

    with _scoped_warp_stream(device):
        multipole_pme_green_structure_factor_launch(
            _wp_from_torch(k_squared.contiguous(), dtype=wp_scalar),
            _wp_from_torch(_as(miller_x), dtype=wp_scalar),
            _wp_from_torch(_as(miller_y), dtype=wp_scalar),
            _wp_from_torch(_as(miller_z), dtype=wp_scalar),
            _wp_from_torch(_as(alpha), dtype=wp_scalar),
            _wp_from_torch(_as(sigma), dtype=wp_scalar),
            _wp_from_torch(_as(volume), dtype=wp_scalar),
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=spline_order,
            green_function=_wp_from_torch(green_function, dtype=wp_scalar),
            structure_factor_sq=_wp_from_torch(structure_factor_sq, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return green_function, structure_factor_sq


def _green_struct_forward_fake(k_squared, *_args):  # pragma: no cover
    """Fake: ``(G̃, |C|²)`` both match k_squared shape (single-system)."""
    return torch.empty_like(k_squared), torch.empty_like(k_squared)


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_green_struct",
    forward=_multipole_pme_green_struct_forward,
    forward_fake=_green_struct_forward_fake,
    forward_return_arity=2,
    # No backward — scalar parameters aren't differentiated in PME use.
)


def _batch_multipole_pme_green_struct_forward(
    k_squared: torch.Tensor,
    miller_x: torch.Tensor,
    miller_y: torch.Tensor,
    miller_z: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    volume: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched ``(G̃_b(k), |C(k)|²)`` — per-system Green's, shared struct."""
    device = k_squared.device
    input_dtype = k_squared.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    B, nx, ny, nz_rfft = k_squared.shape
    green_function = torch.zeros((B, nx, ny, nz_rfft), dtype=input_dtype, device=device)
    structure_factor_sq = torch.zeros(
        (nx, ny, nz_rfft), dtype=input_dtype, device=device
    )

    def _as(t):
        return t.detach().to(input_dtype).contiguous()

    with _scoped_warp_stream(device):
        batch_multipole_pme_green_structure_factor_launch(
            _wp_from_torch(k_squared.contiguous(), dtype=wp_scalar),
            _wp_from_torch(_as(miller_x), dtype=wp_scalar),
            _wp_from_torch(_as(miller_y), dtype=wp_scalar),
            _wp_from_torch(_as(miller_z), dtype=wp_scalar),
            _wp_from_torch(_as(alpha), dtype=wp_scalar),
            _wp_from_torch(_as(sigma), dtype=wp_scalar),
            _wp_from_torch(_as(volume), dtype=wp_scalar),
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=spline_order,
            green_function=_wp_from_torch(green_function, dtype=wp_scalar),
            structure_factor_sq=_wp_from_torch(structure_factor_sq, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return green_function, structure_factor_sq


def _batch_green_struct_forward_fake(k_squared, *_args):  # pragma: no cover
    """Fake: ``green`` matches ``(B, Nx, Ny, Nz_rfft)``, ``|C|²`` is shared."""
    _, nx, ny, nz_rfft = k_squared.shape
    return (
        torch.empty_like(k_squared),
        torch.zeros(
            (nx, ny, nz_rfft),
            dtype=k_squared.dtype,
            device=k_squared.device,
        ),
    )


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_green_struct_batch",
    forward=_batch_multipole_pme_green_struct_forward,
    forward_fake=_batch_green_struct_forward_fake,
    forward_return_arity=2,
    # No backward — scalar parameters aren't differentiated in PME use.
)


def multipole_pme_green_structure_factor(
    k_squared: torch.Tensor,
    miller_x: torch.Tensor,
    miller_y: torch.Tensor,
    miller_z: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    volume: torch.Tensor,
    *,
    mesh_dimensions: tuple[int, int, int],
    spline_order: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Green's function + structure factor for multipole PME.

    Dispatches between single-system and batched paths based on the
    ``k_squared`` rank: 3D ``(Nx, Ny, Nz_rfft)`` selects single-system,
    4D ``(B, Nx, Ny, Nz_rfft)`` selects batched. ``alpha``, ``sigma``,
    ``volume`` shapes must match: ``(1,)`` for single-system, ``(B,)``
    for batched.

    Parameters
    ----------
    k_squared : torch.Tensor
        - Single-system: shape ``(Nx, Ny, Nz_rfft)``.
        - Batched: shape ``(B, Nx, Ny, Nz_rfft)``.
        ``|k|²`` at each grid point (rfft half-space).
    miller_x, miller_y, miller_z : torch.Tensor
        Miller indices from ``fftfreq`` / ``rfftfreq`` — shared across
        batch (mesh geometry is the same for all systems).
    alpha, sigma, volume : torch.Tensor
        Ewald splitting parameter, GTO width, cell volume. Per-system
        ``(B,)`` in batched mode, ``(1,)`` single-system.
    mesh_dimensions : tuple[int, int, int]
        Full mesh dimensions ``(Nx, Ny, Nz)`` (note ``Nz``, not
        ``Nz_rfft``).
    spline_order : int, default 4
        B-spline order. ``|C|² = (sinc_x sinc_y sinc_z)^(2·spline_order)``.

    Returns
    -------
    green_function : torch.Tensor
        :math:`\tilde{G}(k)` with the GTO factor baked in. Shape matches
        ``k_squared``.
    structure_factor_sq : torch.Tensor, shape ``(Nx, Ny, Nz_rfft)``
        :math:`|C(k)|^2`. Shared across batch (mesh-geometry only).
    """
    nx, ny, nz = mesh_dimensions
    if k_squared.dim() == 4:
        return torch.ops.nvalchemiops.multipole_pme_green_struct_batch(
            k_squared,
            miller_x,
            miller_y,
            miller_z,
            alpha,
            sigma,
            volume,
            nx,
            ny,
            nz,
            spline_order,
        )
    return torch.ops.nvalchemiops.multipole_pme_green_struct(
        k_squared,
        miller_x,
        miller_y,
        miller_z,
        alpha,
        sigma,
        volume,
        nx,
        ny,
        nz,
        spline_order,
    )


# =============================================================================
# Gather potential φ(r_i) from the PME potential grid
# =============================================================================


def _multipole_pme_gather_potential_forward(
    mesh: torch.Tensor,
    positions: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Gather ``φ(r_i) = Σ_g B_p(r_i, g) · φ_grid[g]`` (single-system)."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    output = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    with _scoped_warp_stream(device):
        spline_gather(
            _wp_from_torch(positions.detach().contiguous(), dtype=wp_vec),
            _wp_from_torch(cell_inv_t.detach().contiguous(), dtype=wp_mat),
            spline_order,
            _wp_from_torch(mesh.detach().contiguous(), dtype=wp_scalar),
            _wp_from_torch(output, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return output


def _multipole_pme_gather_potential_backward(  # pragma: no cover
    grad_output: torch.Tensor,
    mesh: torch.Tensor,
    positions: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward: spread grad_output onto mesh + gather_gradient for ∂L/∂r."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    grad_mesh = torch.zeros_like(mesh)
    n_atoms = positions.shape[0]
    zero_dipoles = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    force_as_neg_grad_pos = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    grad_charges = grad_output.to(input_dtype).contiguous()
    with _scoped_warp_stream(device):
        multipole_pme_spread_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(grad_charges, dtype=wp_scalar),
            _wp_from_torch(zero_dipoles, dtype=wp_vec),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            order=spline_order,
            mesh=_wp_from_torch(grad_mesh, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
        if not multipole_pme_gather_gradient_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(grad_charges, dtype=wp_scalar),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            spline_order,
            _wp_from_torch(mesh.contiguous(), dtype=wp_scalar),
            _wp_from_torch(force_as_neg_grad_pos, dtype=wp_vec),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        ):
            spline_gather_gradient(
                _wp_from_torch(positions.contiguous(), dtype=wp_vec),
                _wp_from_torch(grad_charges, dtype=wp_scalar),
                _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
                spline_order,
                _wp_from_torch(mesh.contiguous(), dtype=wp_scalar),
                _wp_from_torch(force_as_neg_grad_pos, dtype=wp_vec),
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )
    return grad_mesh, -force_as_neg_grad_pos


def _gather_potential_forward_fake(mesh, positions, *_args):  # pragma: no cover
    """Fake: output ``(N,)`` matching positions dtype/device."""
    del mesh
    return torch.empty(
        positions.shape[0], dtype=positions.dtype, device=positions.device
    )


def _gather_grad_field(  # pragma: no cover
    positions: torch.Tensor,
    weight: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
    mesh: torch.Tensor,
) -> torch.Tensor:
    r"""``Σ_g weight_i · ∇B(r_i, g) · mesh[g]`` per atom ``(N, 3)`` (Warp, with CSR fallback).

    Returns the gathered gradient (NOT negated) — the gather-gradient launcher
    emits the force convention ``-Σ_g weight ∇B mesh``, so this negates it. The
    ``multipole_pme_gather_gradient_launch`` tiled kernel falls back to the CSR
    ``spline_gather_gradient`` on CPU / unsupported configs
    ([[feedback_warp_launch_tiled_cpu_silent]]).
    """
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    force_buf = torch.zeros((positions.shape[0], 3), dtype=input_dtype, device=device)
    pos_wp = _wp_from_torch(positions.contiguous(), dtype=wp_vec)
    w_wp = _wp_from_torch(weight.to(input_dtype).contiguous(), dtype=wp_scalar)
    cit_wp = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat)
    mesh_wp = _wp_from_torch(mesh.contiguous(), dtype=wp_scalar)
    out_wp = _wp_from_torch(force_buf, dtype=wp_vec)
    wp_device = str(wp.device_from_torch(device))
    if not multipole_pme_gather_gradient_launch(
        pos_wp,
        w_wp,
        cit_wp,
        spline_order,
        mesh_wp,
        out_wp,
        wp_dtype=wp_scalar,
        device=wp_device,
    ):
        spline_gather_gradient(
            pos_wp,
            w_wp,
            cit_wp,
            spline_order,
            mesh_wp,
            out_wp,
            wp_dtype=wp_scalar,
            device=wp_device,
        )
    return -force_buf


def _hessian_contract(  # pragma: no cover
    positions: torch.Tensor,
    direction: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
    mesh: torch.Tensor,
) -> torch.Tensor:
    r"""``(Σ_g ∇²B(r_i, g) mesh[g]) · direction_i`` per atom ``(N, 3)`` (Warp).

    The directional Hessian-gather ``Σ_α direction[α] Σ_g mesh(g) ∂²B/∂r_α∂r_γ``,
    computed via the unified-spread backward (``lmax=1``, ``dipoles=direction``,
    ``grad_mesh=mesh``) — the same primitive ``gather_field``'s position-backward
    uses. Stencil math stays in Warp.
    """
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    grad_positions = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    zero_charges = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    zero_Q = torch.zeros((n_atoms, 3, 3), dtype=input_dtype, device=device)
    scratch_q = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    scratch_d = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    scratch_Q = torch.zeros((n_atoms, 3, 3), dtype=input_dtype, device=device)
    scratch_M = torch.zeros((3, 3), dtype=input_dtype, device=device)
    ok = multipole_pme_spread_backward_unified_launch(
        _wp_from_torch(positions.contiguous(), dtype=wp_vec),
        _wp_from_torch(zero_charges, dtype=wp_scalar),
        _wp_from_torch(direction.to(input_dtype).contiguous(), dtype=wp_vec),
        _wp_from_torch(zero_Q, dtype=wp_mat),
        _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
        order=spline_order,
        lmax=1,
        grad_mesh=_wp_from_torch(mesh.contiguous(), dtype=wp_scalar),
        grad_positions=_wp_from_torch(grad_positions, dtype=wp_vec),
        grad_charges=_wp_from_torch(scratch_q, dtype=wp_scalar),
        grad_dipoles=_wp_from_torch(scratch_d, dtype=wp_vec),
        grad_quadrupoles=_wp_from_torch(scratch_Q, dtype=wp_mat),
        grad_cell_inv_t=_wp_from_torch(scratch_M, dtype=wp_scalar),
        wp_dtype=wp_scalar,
        device=str(wp.device_from_torch(device)),
    )
    if not ok:
        raise NotImplementedError(
            f"gather_potential double-backward needs unified spread backward "
            f"for (order={spline_order}, lmax=1, dtype={input_dtype})."
        )
    return grad_positions


def _multipole_pme_gather_potential_double_backward(  # pragma: no cover
    gg_grad_mesh: torch.Tensor,
    gg_grad_positions: torch.Tensor,
    grad_output: torch.Tensor,
    mesh: torch.Tensor,
    positions: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Second-order backward of :func:`_multipole_pme_gather_potential_backward`.

    The first backward maps ``c = grad_output`` to ``gm[g] = Σ_i c_i B(r_i, g)``
    (spread) and ``gp_i = c_i · F_i`` with ``F_i = Σ_g ∇B(r_i, g) mesh[g]``. With
    cotangents ``gg_m`` (on ``gm``) and ``gg_p`` (on ``gp``):

    .. math::

        \partial/\partial c_i &= \text{gather}(gg_m)_i + gg_{p,i} \cdot F_i \\
        \partial/\partial mesh[g] &= \sum_i (c_i\, gg_{p,i}) \cdot \nabla B(r_i, g)
            \quad (\text{dipole-spread of } d_i = c_i\, gg_{p,i}) \\
        \partial/\partial r_i &= c_i\, \text{gatherGrad}(gg_m)_i
            + c_i\, (H_i \cdot gg_{p,i}),\quad H_i = \sum_g \nabla^2 B(r_i, g)\, mesh[g]

    All stencil sums run in Warp launchers (gather / gather-gradient / spread /
    unified-spread-backward); only O(N) per-atom vector combines are torch.
    Returns grads for the backward's diff inputs ``(grad_output, mesh, positions)``.
    """
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    c = grad_output.to(input_dtype)
    gg_p = gg_grad_positions.to(input_dtype)
    gg_m = gg_grad_mesh.to(input_dtype)

    with _scoped_warp_stream(device):
        # gather(gg_m)_i = Σ_g B(r_i, g) gg_m[g]  (forward gather of gg_m).
        gather_ggm = torch.zeros(n_atoms, dtype=input_dtype, device=device)
        spline_gather(
            _wp_from_torch(positions.detach().contiguous(), dtype=wp_vec),
            _wp_from_torch(cell_inv_t.detach().contiguous(), dtype=wp_mat),
            spline_order,
            _wp_from_torch(gg_m.contiguous(), dtype=wp_scalar),
            _wp_from_torch(gather_ggm, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
        # F_i = Σ_g ∇B(r_i, g) mesh[g]  and  gatherGrad(gg_m)_i.
        field = _gather_grad_field(
            positions,
            torch.ones(n_atoms, dtype=input_dtype, device=device),
            cell_inv_t,
            spline_order,
            mesh,
        )
        ggrad_field = _gather_grad_field(
            positions,
            c,
            cell_inv_t,
            spline_order,
            gg_m,
        )  # = c_i · gatherGrad(gg_m)_i
        # H_i · gg_p_i  (directional Hessian-gather of mesh along gg_p).
        hess_dir = _hessian_contract(positions, gg_p, cell_inv_t, spline_order, mesh)

    # ∂/∂grad_output (N,): gather(gg_m) + gg_p·F.
    d_grad_output = (gather_ggm + (gg_p * field).sum(-1)).to(grad_output.dtype)

    # ∂/∂mesh: dipole-spread of d_i = c_i · gg_p_i.
    d_mesh = torch.zeros_like(mesh)
    with _scoped_warp_stream(device):
        multipole_pme_spread_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(
                torch.zeros(n_atoms, dtype=input_dtype, device=device), dtype=wp_scalar
            ),
            _wp_from_torch((c.unsqueeze(-1) * gg_p).contiguous(), dtype=wp_vec),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            order=spline_order,
            mesh=_wp_from_torch(d_mesh, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )

    # ∂/∂positions (N,3): c·gatherGrad(gg_m) + c·(H·gg_p).
    d_positions = (ggrad_field + c.unsqueeze(-1) * hess_dir).to(input_dtype)
    return d_grad_output, d_mesh, d_positions


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_gather_potential",
    forward=_multipole_pme_gather_potential_forward,
    forward_fake=_gather_potential_forward_fake,
    backward=_multipole_pme_gather_potential_backward,
    backward_return_arity=2,
    diff_input_positions=(0, 1),  # mesh, positions
    n_forward_inputs=4,
    double_backward=_multipole_pme_gather_potential_double_backward,
    # backward op inputs: (grad_output, mesh, positions, cell_inv_t, spline_order)
    second_order_diff_positions=(0, 1, 2),
    n_backward_inputs=5,
    double_backward_return_arity=3,
)


def _batch_multipole_pme_gather_potential_forward(
    mesh: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Batched companion of :func:`_multipole_pme_gather_potential_forward`."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    output = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    with _scoped_warp_stream(device):
        batch_spline_gather(
            _wp_from_torch(positions.detach().contiguous(), dtype=wp_vec),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(cell_inv_t.detach().contiguous(), dtype=wp_mat),
            spline_order,
            _wp_from_torch(mesh.detach().contiguous(), dtype=wp_scalar),
            _wp_from_torch(output, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return output


def _batch_multipole_pme_gather_potential_backward(  # pragma: no cover
    grad_output: torch.Tensor,
    mesh: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched backward — same composition as the single-system path."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    grad_mesh = torch.zeros_like(mesh)
    n_atoms = positions.shape[0]
    zero_dipoles = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    force_as_neg_grad_pos = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    grad_charges = grad_output.to(input_dtype).contiguous()
    with _scoped_warp_stream(device):
        batch_multipole_pme_spread_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(grad_charges, dtype=wp_scalar),
            _wp_from_torch(zero_dipoles, dtype=wp_vec),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            order=spline_order,
            mesh=_wp_from_torch(grad_mesh, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
        if not batch_multipole_pme_gather_gradient_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(grad_charges, dtype=wp_scalar),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            spline_order,
            _wp_from_torch(mesh.contiguous(), dtype=wp_scalar),
            _wp_from_torch(force_as_neg_grad_pos, dtype=wp_vec),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        ):
            batch_spline_gather_gradient(
                _wp_from_torch(positions.contiguous(), dtype=wp_vec),
                _wp_from_torch(grad_charges, dtype=wp_scalar),
                _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
                _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
                spline_order,
                _wp_from_torch(mesh.contiguous(), dtype=wp_scalar),
                _wp_from_torch(force_as_neg_grad_pos, dtype=wp_vec),
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )
    return grad_mesh, -force_as_neg_grad_pos


def _batch_gather_grad_field(  # pragma: no cover
    positions: torch.Tensor,
    weight: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
    mesh: torch.Tensor,
) -> torch.Tensor:
    """Batched companion of :func:`_gather_grad_field` (``Σ_g w·∇B·mesh``, ``(N,3)``)."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    force_buf = torch.zeros((positions.shape[0], 3), dtype=input_dtype, device=device)
    pos_wp = _wp_from_torch(positions.contiguous(), dtype=wp_vec)
    w_wp = _wp_from_torch(weight.to(input_dtype).contiguous(), dtype=wp_scalar)
    bidx_wp = _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32)
    cit_wp = _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat)
    mesh_wp = _wp_from_torch(mesh.contiguous(), dtype=wp_scalar)
    out_wp = _wp_from_torch(force_buf, dtype=wp_vec)
    wp_device = str(wp.device_from_torch(device))
    if not batch_multipole_pme_gather_gradient_launch(
        pos_wp,
        w_wp,
        bidx_wp,
        cit_wp,
        spline_order,
        mesh_wp,
        out_wp,
        wp_dtype=wp_scalar,
        device=wp_device,
    ):
        batch_spline_gather_gradient(
            pos_wp,
            w_wp,
            bidx_wp,
            cit_wp,
            spline_order,
            mesh_wp,
            out_wp,
            wp_dtype=wp_scalar,
            device=wp_device,
        )
    return -force_buf


def _batch_hessian_contract(  # pragma: no cover
    positions: torch.Tensor,
    direction: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
    mesh: torch.Tensor,
) -> torch.Tensor:
    """Batched companion of :func:`_hessian_contract` (``(Σ_g ∇²B·mesh)·dir``, ``(N,3)``)."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    grad_positions = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    zero_charges = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    zero_Q = torch.zeros((n_atoms, 3, 3), dtype=input_dtype, device=device)
    scratch_q = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    scratch_d = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    scratch_Q = torch.zeros((n_atoms, 3, 3), dtype=input_dtype, device=device)
    n_sys = mesh.shape[0]
    scratch_M = torch.zeros((n_sys, 3, 3), dtype=input_dtype, device=device)
    ok = batch_multipole_pme_spread_backward_unified_launch(
        _wp_from_torch(positions.contiguous(), dtype=wp_vec),
        _wp_from_torch(zero_charges, dtype=wp_scalar),
        _wp_from_torch(direction.to(input_dtype).contiguous(), dtype=wp_vec),
        _wp_from_torch(zero_Q, dtype=wp_mat),
        _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
        _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
        order=spline_order,
        lmax=1,
        grad_mesh=_wp_from_torch(mesh.contiguous(), dtype=wp_scalar),
        grad_positions=_wp_from_torch(grad_positions, dtype=wp_vec),
        grad_charges=_wp_from_torch(scratch_q, dtype=wp_scalar),
        grad_dipoles=_wp_from_torch(scratch_d, dtype=wp_vec),
        grad_quadrupoles=_wp_from_torch(scratch_Q, dtype=wp_mat),
        grad_cell_inv_t=_wp_from_torch(scratch_M, dtype=wp_scalar),
        wp_dtype=wp_scalar,
        device=str(wp.device_from_torch(device)),
    )
    if not ok:
        raise NotImplementedError(
            f"batched gather_potential double-backward needs unified spread "
            f"backward for (order={spline_order}, lmax=1, dtype={input_dtype})."
        )
    return grad_positions


def _batch_multipole_pme_gather_potential_double_backward(  # pragma: no cover
    gg_grad_mesh: torch.Tensor,
    gg_grad_positions: torch.Tensor,
    grad_output: torch.Tensor,
    mesh: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched second-order backward — mirrors :func:`_multipole_pme_gather_potential_double_backward`."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    c = grad_output.to(input_dtype)
    gg_p = gg_grad_positions.to(input_dtype)
    gg_m = gg_grad_mesh.to(input_dtype)

    with _scoped_warp_stream(device):
        gather_ggm = torch.zeros(n_atoms, dtype=input_dtype, device=device)
        batch_spline_gather(
            _wp_from_torch(positions.detach().contiguous(), dtype=wp_vec),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(cell_inv_t.detach().contiguous(), dtype=wp_mat),
            spline_order,
            _wp_from_torch(gg_m.contiguous(), dtype=wp_scalar),
            _wp_from_torch(gather_ggm, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
        field = _batch_gather_grad_field(
            positions,
            torch.ones(n_atoms, dtype=input_dtype, device=device),
            batch_idx,
            cell_inv_t,
            spline_order,
            mesh,
        )
        ggrad_field = _batch_gather_grad_field(
            positions,
            c,
            batch_idx,
            cell_inv_t,
            spline_order,
            gg_m,
        )
        hess_dir = _batch_hessian_contract(
            positions,
            gg_p,
            batch_idx,
            cell_inv_t,
            spline_order,
            mesh,
        )

    d_grad_output = (gather_ggm + (gg_p * field).sum(-1)).to(grad_output.dtype)

    d_mesh = torch.zeros_like(mesh)
    with _scoped_warp_stream(device):
        batch_multipole_pme_spread_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(
                torch.zeros(n_atoms, dtype=input_dtype, device=device), dtype=wp_scalar
            ),
            _wp_from_torch((c.unsqueeze(-1) * gg_p).contiguous(), dtype=wp_vec),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            order=spline_order,
            mesh=_wp_from_torch(d_mesh, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )

    d_positions = (ggrad_field + c.unsqueeze(-1) * hess_dir).to(input_dtype)
    return d_grad_output, d_mesh, d_positions


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_gather_potential_batch",
    forward=_batch_multipole_pme_gather_potential_forward,
    forward_fake=_gather_potential_forward_fake,
    backward=_batch_multipole_pme_gather_potential_backward,
    backward_return_arity=2,
    diff_input_positions=(0, 1),  # mesh, positions
    n_forward_inputs=5,
    double_backward=_batch_multipole_pme_gather_potential_double_backward,
    # backward inputs: (grad_output, mesh, positions, batch_idx, cell_inv_t, spline_order)
    second_order_diff_positions=(0, 1, 2),
    n_backward_inputs=6,
    double_backward_return_arity=3,
)


def multipole_pme_gather_potential(
    mesh: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    *,
    spline_order: int = 4,
    cell_inv_t: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Gather the potential :math:`\phi(r_i)` at atom positions from a grid.

    Interpolates the grid potential back to each atom via the B-spline
    weights:

    .. math::

        \phi(r_i) = \sum_g B_p(r_i, g)\, \phi_\text{grid}(g)

    Single-system or batched (selected by ``batch_idx``). Autograd-aware:
    analytical backward via the existing spread and gather-gradient
    primitives.

    Parameters
    ----------
    mesh : torch.Tensor
        Potential grid :math:`\phi(g)`. Shape ``(nx, ny, nz)``
        (single-system) or ``(B, nx, ny, nz)`` (batched). float32/float64.
    positions : torch.Tensor, shape ``(N, 3)`` or ``(N_total, 3)``
        Cartesian atom positions; ``N_total`` in batched mode. Same dtype
        as ``mesh``.
    cell : torch.Tensor, shape ``(3, 3)``, ``(1, 3, 3)``, or ``(B, 3, 3)``
        Unit-cell matrix (rows are lattice vectors); ``(B, 3, 3)`` batched.
    spline_order : int, default 4
        B-spline interpolation order ``p`` (cardinal B-spline).
    cell_inv_t : torch.Tensor, optional
        Pre-computed ``transpose(inv(cell))`` for MD steady-state. Shape
        ``(3, 3)`` / ``(1, 3, 3)`` single-system, ``(B, 3, 3)`` batched.
        When ``None`` it is derived from ``cell``.
    batch_idx : torch.Tensor, optional
        ``(N_total,)`` int32 — system index per atom. Triggers the
        batched path when provided.

    Returns
    -------
    phi : torch.Tensor, shape ``(N,)`` or ``(N_total,)``
        Per-atom potential values, same dtype as ``mesh``.

    Notes
    -----
    Backward implements both ``∂L/∂mesh`` (spread of the cotangent) and
    ``∂L/∂positions`` (gather-gradient against the grid).
    """
    if batch_idx is None:
        cell_inv_t_resolved = _resolve_cell_inv_t(cell, cell_inv_t)
        return torch.ops.nvalchemiops.multipole_pme_gather_potential(
            mesh, positions, cell_inv_t_resolved, spline_order
        )
    cell_inv_t_resolved = _resolve_batch_cell_inv_t(cell, cell_inv_t)
    return torch.ops.nvalchemiops.multipole_pme_gather_potential_batch(
        mesh, positions, batch_idx, cell_inv_t_resolved, spline_order
    )


# =============================================================================
# Gather field ∇φ(r_i) from the PME potential grid
# =============================================================================


def _multipole_pme_gather_field_forward(
    mesh: torch.Tensor,
    positions: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Gather ``∇_{cart} φ(r_i)`` (single-system) — gradient-gather + sign flip."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    ones = torch.ones(n_atoms, dtype=input_dtype, device=device)
    force_buf = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    with _scoped_warp_stream(device):
        if not multipole_pme_gather_gradient_launch(
            _wp_from_torch(positions.detach().contiguous(), dtype=wp_vec),
            _wp_from_torch(ones, dtype=wp_scalar),
            _wp_from_torch(cell_inv_t.detach().contiguous(), dtype=wp_mat),
            spline_order,
            _wp_from_torch(mesh.detach().contiguous(), dtype=wp_scalar),
            _wp_from_torch(force_buf, dtype=wp_vec),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        ):
            spline_gather_gradient(
                _wp_from_torch(positions.detach().contiguous(), dtype=wp_vec),
                _wp_from_torch(ones, dtype=wp_scalar),
                _wp_from_torch(cell_inv_t.detach().contiguous(), dtype=wp_mat),
                spline_order,
                _wp_from_torch(mesh.detach().contiguous(), dtype=wp_scalar),
                _wp_from_torch(force_buf, dtype=wp_vec),
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )
    return -force_buf


def _multipole_pme_gather_field_backward(  # pragma: no cover
    grad_field: torch.Tensor,
    mesh: torch.Tensor,
    positions: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Two-slot backward — ``∂L/∂mesh`` and ``∂L/∂positions``.

    - ``∂L/∂mesh(g) = Σ_i Σ_α grad_field[i, α] · ∂B/∂r_α(r_i, g)``
      — same as "spread ``grad_field`` as dipoles" (``multipole_pme_spread_launch``
      with charges=0).
    - ``∂L/∂r_γ[i] = Σ_α grad_field[i, α] · Σ_g mesh(g) ·
      ∂²B/∂r_α∂r_γ(r_i, g)``
      — a rank-2 Hessian gather at the atom. Implemented via the
      unified-spread backward kernel with ``dipoles = grad_field``,
      ``charges = 0``, ``quadrupoles = 0``, ``lmax = 1``,
      ``grad_mesh = mesh``; its ``grad_positions`` output reduces to the
      desired Hessian contraction.
    """
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    grad_mesh = torch.zeros_like(mesh)
    n_atoms = positions.shape[0]
    zero_charges = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    grad_field_c = grad_field.to(input_dtype).contiguous()
    with _scoped_warp_stream(device):
        # Mesh-side adjoint.
        multipole_pme_spread_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(zero_charges, dtype=wp_scalar),
            _wp_from_torch(grad_field_c, dtype=wp_vec),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            order=spline_order,
            mesh=_wp_from_torch(grad_mesh, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
        # Position-side adjoint via the unified spread backward
        # (LMAX=1, dipoles=grad_field, grad_mesh=mesh); only the
        # ``grad_positions`` output is needed.
        grad_positions = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
        scratch_q = torch.zeros(n_atoms, dtype=input_dtype, device=device)
        scratch_d = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
        scratch_Q = torch.zeros((n_atoms, 3, 3), dtype=input_dtype, device=device)
        scratch_M = torch.zeros((3, 3), dtype=input_dtype, device=device)
        zero_Q_input = torch.zeros((n_atoms, 3, 3), dtype=input_dtype, device=device)
        ok = multipole_pme_spread_backward_unified_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(zero_charges, dtype=wp_scalar),
            _wp_from_torch(grad_field_c, dtype=wp_vec),
            _wp_from_torch(zero_Q_input, dtype=wp_mat),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            order=spline_order,
            lmax=1,
            grad_mesh=_wp_from_torch(mesh.contiguous(), dtype=wp_scalar),
            grad_positions=_wp_from_torch(grad_positions, dtype=wp_vec),
            grad_charges=_wp_from_torch(scratch_q, dtype=wp_scalar),
            grad_dipoles=_wp_from_torch(scratch_d, dtype=wp_vec),
            grad_quadrupoles=_wp_from_torch(scratch_Q, dtype=wp_mat),
            grad_cell_inv_t=_wp_from_torch(scratch_M, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
        if not ok:
            raise NotImplementedError(
                f"gather_field positions-backward needs unified spread "
                f"backward for (order={spline_order}, lmax=1, "
                f"dtype={input_dtype}); not registered."
            )
    return grad_mesh, grad_positions


def _gather_field_forward_fake(mesh, positions, *_args):  # pragma: no cover
    """Fake: ``(N, 3)`` per-atom field matching positions dtype/device."""
    del mesh
    return torch.empty(
        (positions.shape[0], 3),
        dtype=positions.dtype,
        device=positions.device,
    )


def _quad_gradpos(  # pragma: no cover
    positions: torch.Tensor,
    quadrupoles: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
    mesh: torch.Tensor,
) -> torch.Tensor:
    r"""``Σ_{αβ} Q_i[α,β] Σ_g mesh[g] ∂³B/∂r_α∂r_β∂r_γ(r_i,g)`` per atom ``(N,3)``.

    The ∂³B contraction of a per-atom Cartesian quadrupole against the mesh —
    the position-derivative of the quadrupole spread. Computed via the unified
    spread backward (``lmax=2``, ``quadrupoles=Q``, ``grad_mesh=mesh``); its
    ``grad_positions`` output is exactly this third-derivative gather. Stencil
    math stays in Warp.
    """
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    grad_positions = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    zero_charges = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    zero_d = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    scratch_q = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    scratch_d = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    scratch_Q = torch.zeros((n_atoms, 3, 3), dtype=input_dtype, device=device)
    scratch_M = torch.zeros((3, 3), dtype=input_dtype, device=device)
    ok = multipole_pme_spread_backward_unified_launch(
        _wp_from_torch(positions.contiguous(), dtype=wp_vec),
        _wp_from_torch(zero_charges, dtype=wp_scalar),
        _wp_from_torch(zero_d, dtype=wp_vec),
        _wp_from_torch(quadrupoles.to(input_dtype).contiguous(), dtype=wp_mat),
        _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
        order=spline_order,
        lmax=2,
        grad_mesh=_wp_from_torch(mesh.contiguous(), dtype=wp_scalar),
        grad_positions=_wp_from_torch(grad_positions, dtype=wp_vec),
        grad_charges=_wp_from_torch(scratch_q, dtype=wp_scalar),
        grad_dipoles=_wp_from_torch(scratch_d, dtype=wp_vec),
        grad_quadrupoles=_wp_from_torch(scratch_Q, dtype=wp_mat),
        grad_cell_inv_t=_wp_from_torch(scratch_M, dtype=wp_scalar),
        wp_dtype=wp_scalar,
        device=str(wp.device_from_torch(device)),
    )
    if not ok:
        raise NotImplementedError(
            f"gather_field double-backward needs unified spread backward "
            f"for (order={spline_order}, lmax=2, dtype={input_dtype})."
        )
    return grad_positions


def _quad_spread(  # pragma: no cover
    positions: torch.Tensor,
    quadrupoles: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
    mesh_shape: tuple[int, ...],
    device: torch.device,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    r"""``Σ_i Σ_{αβ} Q_i[α,β] ∂²B/∂r_α∂r_β(r_i,g)`` on the mesh ``(quad-spread)``.

    The ∂²B (quadrupole) channel of the unified spread (``lmax=2``, ``charges=0``,
    ``dipoles=0``) — the mesh-side adjoint of the Hessian gather.
    """
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    out = torch.zeros(mesh_shape, dtype=input_dtype, device=device)
    ok = multipole_pme_spread_unified_launch(
        _wp_from_torch(positions.contiguous(), dtype=wp_vec),
        _wp_from_torch(
            torch.zeros(n_atoms, dtype=input_dtype, device=device), dtype=wp_scalar
        ),
        _wp_from_torch(
            torch.zeros((n_atoms, 3), dtype=input_dtype, device=device), dtype=wp_vec
        ),
        _wp_from_torch(quadrupoles.to(input_dtype).contiguous(), dtype=wp_mat),
        _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
        order=spline_order,
        lmax=2,
        mesh=_wp_from_torch(out, dtype=wp_scalar),
        wp_dtype=wp_scalar,
        device=str(wp.device_from_torch(device)),
    )
    if not ok:
        raise NotImplementedError(
            f"gather_field double-backward needs unified spread (quad) for "
            f"(order={spline_order}, lmax=2, dtype={input_dtype})."
        )
    return out


def _multipole_pme_gather_field_double_backward(  # pragma: no cover
    gg_grad_mesh: torch.Tensor,
    gg_grad_positions: torch.Tensor,
    grad_field: torch.Tensor,
    mesh: torch.Tensor,
    positions: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Second-order backward of :func:`_multipole_pme_gather_field_backward`.

    Forward ``field_i = Σ_g ∇B(r_i,g) mesh[g]`` (N,3). Backward (cotangent
    ``gf``): ``grad_mesh = dipole-spread(gf)``, ``grad_pos_i = H^m_i·gf_i``
    (``H^m = Σ_g ∇²B·mesh``). With cotangents ``gg_m`` (grad_mesh), ``gg_p``
    (grad_pos), and ``Q_i = gf_i ⊗ gg_p_i``:

    .. math::

        \partial/\partial gf_i &= \text{gatherGrad}(gg_m)_i + H^m_i \cdot gg_{p,i} \\
        \partial/\partial mesh[g] &= \text{quadSpread}(Q)[g]
            \quad (\sum_i Q_i : \nabla^2 B(r_i,g)) \\
        \partial/\partial r_i &= H^{gg_m}_i \cdot gf_i
            + \text{quadGradPos}(Q, mesh)_i \quad (\nabla^3 B\ \text{term})

    All stencil sums run in Warp launchers; only O(N) per-atom combines + the
    ``gf ⊗ gg_p`` outer product are torch. Returns grads for the backward's diff
    inputs ``(grad_field, mesh, positions)``.
    """
    device = positions.device
    input_dtype = positions.dtype
    n_atoms = positions.shape[0]
    gf = grad_field.to(input_dtype)
    gg_m = gg_grad_mesh.to(input_dtype)
    gg_p = gg_grad_positions.to(input_dtype)
    # Q_fake_i = gf_i⊗gg_p_i + gg_p_i⊗gf_i  (= 2·sym(gf⊗gg_p)). The unified-spread
    # Q-channel carries a ½ factor AND reads only the symmetric storage (see
    # gather_hessian backward: it uses Q = 2·grad_H), so the bare contraction
    # ``Σ_αβ gf_α gg_p_β ∂²B_αβ`` needs the symmetrized, doubled tensor.
    quad = gf.unsqueeze(-1) * gg_p.unsqueeze(-2) + gg_p.unsqueeze(-1) * gf.unsqueeze(-2)

    with _scoped_warp_stream(device):
        ones = torch.ones(n_atoms, dtype=input_dtype, device=device)
        ggrad_ggm = _gather_grad_field(positions, ones, cell_inv_t, spline_order, gg_m)
        hm_ggp = _hessian_contract(positions, gg_p, cell_inv_t, spline_order, mesh)
        hggm_gf = _hessian_contract(positions, gf, cell_inv_t, spline_order, gg_m)
        quad_gp = _quad_gradpos(positions, quad, cell_inv_t, spline_order, mesh)

    # ∂/∂grad_field (N,3): gatherGrad(gg_m) + H^m·gg_p.
    d_grad_field = (ggrad_ggm + hm_ggp).to(grad_field.dtype)
    # ∂/∂mesh: quad-spread of Q = gf ⊗ gg_p.
    d_mesh = _quad_spread(
        positions,
        quad,
        cell_inv_t,
        spline_order,
        tuple(mesh.shape),
        device,
        input_dtype,
    )
    # ∂/∂positions (N,3): H^{gg_m}·gf + ∇³B contraction.
    d_positions = (hggm_gf + quad_gp).to(input_dtype)
    return d_grad_field, d_mesh, d_positions


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_gather_field",
    forward=_multipole_pme_gather_field_forward,
    forward_fake=_gather_field_forward_fake,
    backward=_multipole_pme_gather_field_backward,
    backward_return_arity=2,
    diff_input_positions=(0, 1),  # mesh + positions
    n_forward_inputs=4,
    double_backward=_multipole_pme_gather_field_double_backward,
    # backward inputs: (grad_field, mesh, positions, cell_inv_t, spline_order)
    second_order_diff_positions=(0, 1, 2),
    n_backward_inputs=5,
    double_backward_return_arity=3,
)


def _batch_multipole_pme_gather_field_forward(
    mesh: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Batched ``∇_{cart} φ(r_i)`` gather."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    ones = torch.ones(n_atoms, dtype=input_dtype, device=device)
    force_buf = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    with _scoped_warp_stream(device):
        if not batch_multipole_pme_gather_gradient_launch(
            _wp_from_torch(positions.detach().contiguous(), dtype=wp_vec),
            _wp_from_torch(ones, dtype=wp_scalar),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(cell_inv_t.detach().contiguous(), dtype=wp_mat),
            spline_order,
            _wp_from_torch(mesh.detach().contiguous(), dtype=wp_scalar),
            _wp_from_torch(force_buf, dtype=wp_vec),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        ):
            batch_spline_gather_gradient(
                _wp_from_torch(positions.detach().contiguous(), dtype=wp_vec),
                _wp_from_torch(ones, dtype=wp_scalar),
                _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
                _wp_from_torch(cell_inv_t.detach().contiguous(), dtype=wp_mat),
                spline_order,
                _wp_from_torch(mesh.detach().contiguous(), dtype=wp_scalar),
                _wp_from_torch(force_buf, dtype=wp_vec),
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )
    return -force_buf


def _batch_multipole_pme_gather_field_backward(  # pragma: no cover
    grad_field: torch.Tensor,
    mesh: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-slot batched backward — ``∂L/∂mesh`` AND ``∂L/∂positions``.

    Same math as the single-system version (``∂L/∂r_γ[i] = Σ_α
    grad_field[i, α] · Σ_g mesh(g) · ∂²B/∂r_α∂r_γ(r_i, g)``), routed
    through ``batch_multipole_pme_spread_backward_launch`` with
    charges=0 and dipoles=grad_field.
    """
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    grad_mesh = torch.zeros_like(mesh)
    n_atoms = positions.shape[0]
    zero_charges = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    grad_field_c = grad_field.to(input_dtype).contiguous()
    with _scoped_warp_stream(device):
        # Mesh-side adjoint: spread grad_field as dipoles.
        batch_multipole_pme_spread_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(zero_charges, dtype=wp_scalar),
            _wp_from_torch(grad_field_c, dtype=wp_vec),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            order=spline_order,
            mesh=_wp_from_torch(grad_mesh, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
        # Position-side adjoint via the batched spread backward.
        # Charges=0, dipoles=grad_field → ``grad_positions`` equals the
        # desired Hessian contraction.
        grad_positions = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
        scratch_q = torch.zeros(n_atoms, dtype=input_dtype, device=device)
        scratch_d = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
        batch_multipole_pme_spread_backward_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(zero_charges, dtype=wp_scalar),
            _wp_from_torch(grad_field_c, dtype=wp_vec),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            order=spline_order,
            grad_mesh=_wp_from_torch(mesh.contiguous(), dtype=wp_scalar),
            grad_positions=_wp_from_torch(grad_positions, dtype=wp_vec),
            grad_charges=_wp_from_torch(scratch_q, dtype=wp_scalar),
            grad_dipoles=_wp_from_torch(scratch_d, dtype=wp_vec),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return grad_mesh, grad_positions


def _batch_quad_gradpos(  # pragma: no cover
    positions: torch.Tensor,
    quadrupoles: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
    mesh: torch.Tensor,
) -> torch.Tensor:
    """Batched companion of :func:`_quad_gradpos` (∇³B contraction, ``(N,3)``)."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    grad_positions = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    zero_charges = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    zero_d = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    scratch_q = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    scratch_d = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    scratch_Q = torch.zeros((n_atoms, 3, 3), dtype=input_dtype, device=device)
    scratch_M = torch.zeros((mesh.shape[0], 3, 3), dtype=input_dtype, device=device)
    ok = batch_multipole_pme_spread_backward_unified_launch(
        _wp_from_torch(positions.contiguous(), dtype=wp_vec),
        _wp_from_torch(zero_charges, dtype=wp_scalar),
        _wp_from_torch(zero_d, dtype=wp_vec),
        _wp_from_torch(quadrupoles.to(input_dtype).contiguous(), dtype=wp_mat),
        _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
        _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
        order=spline_order,
        lmax=2,
        grad_mesh=_wp_from_torch(mesh.contiguous(), dtype=wp_scalar),
        grad_positions=_wp_from_torch(grad_positions, dtype=wp_vec),
        grad_charges=_wp_from_torch(scratch_q, dtype=wp_scalar),
        grad_dipoles=_wp_from_torch(scratch_d, dtype=wp_vec),
        grad_quadrupoles=_wp_from_torch(scratch_Q, dtype=wp_mat),
        grad_cell_inv_t=_wp_from_torch(scratch_M, dtype=wp_scalar),
        wp_dtype=wp_scalar,
        device=str(wp.device_from_torch(device)),
    )
    if not ok:
        raise NotImplementedError(
            f"batched gather_field double-backward needs unified spread backward "
            f"for (order={spline_order}, lmax=2, dtype={input_dtype})."
        )
    return grad_positions


def _batch_quad_spread(  # pragma: no cover
    positions: torch.Tensor,
    quadrupoles: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
    mesh_shape: tuple[int, ...],
    device: torch.device,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    """Batched companion of :func:`_quad_spread` (∂²B quad channel → mesh)."""
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    out = torch.zeros(mesh_shape, dtype=input_dtype, device=device)
    ok = batch_multipole_pme_spread_unified_launch(
        _wp_from_torch(positions.contiguous(), dtype=wp_vec),
        _wp_from_torch(
            torch.zeros(n_atoms, dtype=input_dtype, device=device), dtype=wp_scalar
        ),
        _wp_from_torch(
            torch.zeros((n_atoms, 3), dtype=input_dtype, device=device), dtype=wp_vec
        ),
        _wp_from_torch(quadrupoles.to(input_dtype).contiguous(), dtype=wp_mat),
        _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
        _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
        order=spline_order,
        lmax=2,
        mesh=_wp_from_torch(out, dtype=wp_scalar),
        wp_dtype=wp_scalar,
        device=str(wp.device_from_torch(device)),
    )
    if not ok:
        raise NotImplementedError(
            f"batched gather_field double-backward needs unified spread (quad) "
            f"for (order={spline_order}, lmax=2, dtype={input_dtype})."
        )
    return out


def _batch_multipole_pme_gather_field_double_backward(  # pragma: no cover
    gg_grad_mesh: torch.Tensor,
    gg_grad_positions: torch.Tensor,
    grad_field: torch.Tensor,
    mesh: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched second-order backward — mirrors :func:`_multipole_pme_gather_field_double_backward`."""
    device = positions.device
    input_dtype = positions.dtype
    n_atoms = positions.shape[0]
    gf = grad_field.to(input_dtype)
    gg_m = gg_grad_mesh.to(input_dtype)
    gg_p = gg_grad_positions.to(input_dtype)
    quad = gf.unsqueeze(-1) * gg_p.unsqueeze(-2) + gg_p.unsqueeze(-1) * gf.unsqueeze(-2)

    with _scoped_warp_stream(device):
        ones = torch.ones(n_atoms, dtype=input_dtype, device=device)
        ggrad_ggm = _batch_gather_grad_field(
            positions, ones, batch_idx, cell_inv_t, spline_order, gg_m
        )
        hm_ggp = _batch_hessian_contract(
            positions, gg_p, batch_idx, cell_inv_t, spline_order, mesh
        )
        hggm_gf = _batch_hessian_contract(
            positions, gf, batch_idx, cell_inv_t, spline_order, gg_m
        )
        quad_gp = _batch_quad_gradpos(
            positions, quad, batch_idx, cell_inv_t, spline_order, mesh
        )

    d_grad_field = (ggrad_ggm + hm_ggp).to(grad_field.dtype)
    d_mesh = _batch_quad_spread(
        positions,
        quad,
        batch_idx,
        cell_inv_t,
        spline_order,
        tuple(mesh.shape),
        device,
        input_dtype,
    )
    d_positions = (hggm_gf + quad_gp).to(input_dtype)
    return d_grad_field, d_mesh, d_positions


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_gather_field_batch",
    forward=_batch_multipole_pme_gather_field_forward,
    forward_fake=_gather_field_forward_fake,
    backward=_batch_multipole_pme_gather_field_backward,
    backward_return_arity=2,
    diff_input_positions=(0, 1),  # mesh + positions
    n_forward_inputs=5,
    double_backward=_batch_multipole_pme_gather_field_double_backward,
    # backward inputs: (grad_field, mesh, positions, batch_idx, cell_inv_t, spline_order)
    second_order_diff_positions=(0, 1, 2),
    n_backward_inputs=6,
    double_backward_return_arity=3,
)


def multipole_pme_gather_field(
    mesh: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    *,
    spline_order: int = 4,
    cell_inv_t: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Gather the field :math:`\nabla_\text{cart} \phi(r_i)` at atom positions.

    Interpolates the Cartesian gradient of the grid potential back to each
    atom (sign-flipped relative to the grid-side gradient because
    :math:`\nabla_{r_\text{atom}} = -\nabla_\text{grid}`):

    .. math::

        \nabla_\text{cart} \phi(r_i) = \sum_g \nabla_{r_i} B_p(r_i, g)\,
        \phi_\text{grid}(g)

    Single-system or batched (selected by ``batch_idx``). Used by the PME
    composite to build the dipole-channel contribution
    :math:`-\sum_i \boldsymbol{\mu}_i \cdot \nabla\phi(r_i)` to the
    reciprocal energy.

    Parameters
    ----------
    mesh : torch.Tensor
        Potential grid :math:`\phi(g)`. Shape ``(nx, ny, nz)``
        (single-system) or ``(B, nx, ny, nz)`` (batched). float32/float64.
    positions : torch.Tensor, shape ``(N, 3)`` or ``(N_total, 3)``
        Cartesian atom positions; ``N_total`` in batched mode. Same dtype
        as ``mesh``.
    cell : torch.Tensor, shape ``(3, 3)``, ``(1, 3, 3)``, or ``(B, 3, 3)``
        Unit-cell matrix (rows are lattice vectors); ``(B, 3, 3)`` batched.
    spline_order : int, default 4
        B-spline interpolation order ``p``.
    cell_inv_t : torch.Tensor, optional
        Pre-computed ``transpose(inv(cell))`` for MD steady-state. Shape
        ``(3, 3)`` / ``(1, 3, 3)`` single-system, ``(B, 3, 3)`` batched.
        When ``None`` it is derived from ``cell``.
    batch_idx : torch.Tensor, optional
        ``(N_total,)`` int32 — system index per atom. Triggers the
        batched path when provided.

    Returns
    -------
    field : torch.Tensor, shape ``(N, 3)`` or ``(N_total, 3)``
        Per-atom Cartesian field gradient of :math:`\phi`, same dtype as
        ``mesh``.

    Notes
    -----
    Backward implements both ``∂L/∂φ_grid`` and ``∂L/∂positions`` (the
    latter via a rank-2 Hessian gather at the atom), so the field gather is
    fully position-autograd-aware.
    """
    if batch_idx is None:
        cell_inv_t_resolved = _resolve_cell_inv_t(cell, cell_inv_t)
        return torch.ops.nvalchemiops.multipole_pme_gather_field(
            mesh, positions, cell_inv_t_resolved, spline_order
        )
    cell_inv_t_resolved = _resolve_batch_cell_inv_t(cell, cell_inv_t)
    return torch.ops.nvalchemiops.multipole_pme_gather_field_batch(
        mesh, positions, batch_idx, cell_inv_t_resolved, spline_order
    )


# =============================================================================
# Gather Hessian ∇²φ(r_i) from the PME potential grid
# =============================================================================


def _multipole_pme_gather_hessian_forward(
    mesh: torch.Tensor,
    positions: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Gather the symmetric ``(N, 3, 3)`` Cartesian Hessian of ``φ``."""
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    diag = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    off = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    with _scoped_warp_stream(device):
        multipole_pme_gather_hessian_launch(
            _wp_from_torch(positions.detach().contiguous(), dtype=wp_vec),
            _wp_from_torch(cell_inv_t.detach().contiguous(), dtype=wp_mat),
            order=spline_order,
            mesh=_wp_from_torch(mesh.detach().contiguous(), dtype=wp_scalar),
            hessian_diag=_wp_from_torch(diag, dtype=wp_vec),
            hessian_off=_wp_from_torch(off, dtype=wp_vec),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )

    H = torch.zeros((n_atoms, 3, 3), dtype=input_dtype, device=device)
    H[:, 0, 0] = diag[:, 0]
    H[:, 1, 1] = diag[:, 1]
    H[:, 2, 2] = diag[:, 2]
    H[:, 0, 1] = off[:, 0]
    H[:, 1, 0] = off[:, 0]
    H[:, 0, 2] = off[:, 1]
    H[:, 2, 0] = off[:, 1]
    H[:, 1, 2] = off[:, 2]
    H[:, 2, 1] = off[:, 2]
    return H


def _multipole_pme_gather_hessian_backward(  # pragma: no cover
    grad_H: torch.Tensor,
    mesh: torch.Tensor,
    positions: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Two-slot backward for ``gather_hessian``.

    Given upstream cotangent ``grad_H[i, α, β]`` (treated as a symmetric
    ``(N, 3, 3)`` tensor since the forward output is symmetric):

    - ``∂L/∂mesh(g) = Σ_i Σ_αβ grad_H[i, α, β] · ∂²B/∂r_α∂r_β(r_i, g)``
    - ``∂L/∂r_γ[i] = Σ_g mesh(g) · Σ_αβ grad_H[i, α, β] ·
      ∂³B/∂r_α∂r_β∂r_γ(r_i, g)``

    Implementation: reuse the unified-spread Q-channel infrastructure.
    The unified spread forward with ``Q = 2·grad_H`` (charges=0,
    dipoles=0, lmax=2) produces exactly the mesh-side adjoint
    (the ``(1/2) Qe : H_frac`` per-cell contribution doubles to
    ``grad_H : H_cart`` per atom). The unified spread backward with
    the same fake-Q and ``grad_mesh = mesh`` produces the rank-3 ∂³B
    position contraction.
    """
    device = positions.device
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    n_atoms = positions.shape[0]
    # Fake-Q = 2 · grad_H compensates the (1/2) factor in the unified
    # spread Q-channel.
    Q_fake = (grad_H.to(input_dtype) * 2.0).contiguous()
    zero_charges = torch.zeros(n_atoms, dtype=input_dtype, device=device)
    zero_dipoles = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
    grad_mesh = torch.zeros_like(mesh)
    with _scoped_warp_stream(device):
        ok_fwd = multipole_pme_spread_unified_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(zero_charges, dtype=wp_scalar),
            _wp_from_torch(zero_dipoles, dtype=wp_vec),
            _wp_from_torch(Q_fake, dtype=wp_mat),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            order=spline_order,
            lmax=2,
            mesh=_wp_from_torch(grad_mesh, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
        if not ok_fwd:
            raise NotImplementedError(
                f"gather_hessian mesh-side backward needs unified spread "
                f"(order={spline_order}, lmax=2, dtype={input_dtype})."
            )
        # Position-side adjoint via unified spread backward.
        grad_positions = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
        scratch_q = torch.zeros(n_atoms, dtype=input_dtype, device=device)
        scratch_d = torch.zeros((n_atoms, 3), dtype=input_dtype, device=device)
        scratch_Q = torch.zeros((n_atoms, 3, 3), dtype=input_dtype, device=device)
        scratch_M = torch.zeros((3, 3), dtype=input_dtype, device=device)
        ok_bwd = multipole_pme_spread_backward_unified_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(zero_charges, dtype=wp_scalar),
            _wp_from_torch(zero_dipoles, dtype=wp_vec),
            _wp_from_torch(Q_fake, dtype=wp_mat),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            order=spline_order,
            lmax=2,
            grad_mesh=_wp_from_torch(mesh.contiguous(), dtype=wp_scalar),
            grad_positions=_wp_from_torch(grad_positions, dtype=wp_vec),
            grad_charges=_wp_from_torch(scratch_q, dtype=wp_scalar),
            grad_dipoles=_wp_from_torch(scratch_d, dtype=wp_vec),
            grad_quadrupoles=_wp_from_torch(scratch_Q, dtype=wp_mat),
            grad_cell_inv_t=_wp_from_torch(scratch_M, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
        if not ok_bwd:
            raise NotImplementedError(
                f"gather_hessian positions-side backward needs unified "
                f"spread backward (order={spline_order}, lmax=2, "
                f"dtype={input_dtype})."
            )
    return grad_mesh, grad_positions


def _gather_hessian_forward_fake(mesh, positions, *_args):  # pragma: no cover
    """Fake: symmetric ``(N, 3, 3)`` Hessian per atom."""
    del mesh
    return torch.empty(
        (positions.shape[0], 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_gather_hessian",
    forward=_multipole_pme_gather_hessian_forward,
    forward_fake=_gather_hessian_forward_fake,
    backward=_multipole_pme_gather_hessian_backward,
    backward_return_arity=2,
    diff_input_positions=(0, 1),  # mesh + positions
    n_forward_inputs=4,
)


def multipole_pme_gather_hessian(
    mesh: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    *,
    spline_order: int = 4,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Gather the symmetric Cartesian Hessian :math:`\nabla^2_\text{cart} \phi(r_i)`.

    Interpolates the second Cartesian derivative of the grid potential back to
    each atom:

    .. math::

        \big[\nabla^2_\text{cart} \phi(r_i)\big]_{\alpha\beta}
        = \sum_g \partial^2_{r_\alpha r_\beta} B_p(r_i, g)\, \phi_\text{grid}(g)

    Single-system only. Used by the PME composite for the quadrupole channel
    :math:`\tfrac{1}{2} Q_i : \nabla^2\phi(r_i)`.

    Parameters
    ----------
    mesh : torch.Tensor, shape ``(nx, ny, nz)``
        Potential grid :math:`\phi(g)`. float32/float64.
    positions : torch.Tensor, shape ``(N, 3)``
        Cartesian atom positions, same dtype as ``mesh``.
    cell : torch.Tensor, shape ``(3, 3)`` or ``(1, 3, 3)``
        Unit-cell matrix (rows are lattice vectors).
    spline_order : int, default 4
        B-spline interpolation order ``p``.
    cell_inv_t : torch.Tensor, optional
        Pre-computed ``transpose(inv(cell))`` for MD steady-state, shape
        ``(3, 3)`` or ``(1, 3, 3)``. When ``None`` it is derived from
        ``cell``.

    Returns
    -------
    H : torch.Tensor, shape ``(N, 3, 3)``
        Per-atom symmetric Hessian of :math:`\phi` at each atom position,
        same dtype as ``mesh``.

    Notes
    -----
    Backward implements both ``∂L/∂mesh`` and ``∂L/∂positions`` by
    reusing the unified-spread Q-channel infrastructure.
    """
    cell_inv_t_resolved = _resolve_cell_inv_t(cell, cell_inv_t)
    return torch.ops.nvalchemiops.multipole_pme_gather_hessian(
        mesh, positions, cell_inv_t_resolved, spline_order
    )


# =============================================================================
# Energy corrections — self + background
# =============================================================================


def _corr_scalar_array(value: float, device: torch.device) -> torch.Tensor:
    """A length-1 fp64 array holding a precomputed coefficient."""
    return torch.tensor([value], dtype=torch.float64, device=device)


def _multipole_pme_corrections_forward(
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    volume: torch.Tensor,
    total_charge: torch.Tensor,
    batch_idx: torch.Tensor | None,
    c_self_q: torch.Tensor,
    c_self_mu: torch.Tensor,
    c_self_q2: torch.Tensor,
    c_bg_no_v: torch.Tensor,
    has_dipoles: bool,
    has_quadrupoles: bool,
) -> torch.Tensor:
    """Forward: per-atom Warp correction kernel + reduction.

    Returns a scalar (``batch_idx is None``) or per-system ``(B,)``
    (via ``scatter_add``). All math is fp64 to match the reference path.
    """
    device = charges.device
    wp_scalar = get_wp_dtype(torch.float64)
    wp_vec = get_wp_vec_dtype(torch.float64)
    n_atoms = charges.shape[0]
    per_atom = torch.empty(n_atoms, dtype=torch.float64, device=device)
    with _scoped_warp_stream(device):
        if batch_idx is None:
            multipole_pme_corrections_launch(
                _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
                _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
                _wp_from_torch(
                    quadrupoles.contiguous(), dtype=get_wp_mat_dtype(torch.float64)
                ),
                _wp_from_torch(volume.contiguous(), dtype=wp_scalar),
                _wp_from_torch(total_charge.contiguous(), dtype=wp_scalar),
                _wp_from_torch(c_self_q, dtype=wp_scalar),
                _wp_from_torch(c_self_mu, dtype=wp_scalar),
                _wp_from_torch(c_self_q2, dtype=wp_scalar),
                _wp_from_torch(c_bg_no_v, dtype=wp_scalar),
                has_dipoles,
                has_quadrupoles,
                _wp_from_torch(per_atom, dtype=wp_scalar),
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )
        else:
            batch_i32 = batch_idx.to(torch.int32).contiguous()
            batch_multipole_pme_corrections_launch(
                _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
                _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
                _wp_from_torch(
                    quadrupoles.contiguous(), dtype=get_wp_mat_dtype(torch.float64)
                ),
                _wp_from_torch(batch_i32, dtype=wp.int32),
                _wp_from_torch(volume.contiguous(), dtype=wp_scalar),
                _wp_from_torch(total_charge.contiguous(), dtype=wp_scalar),
                _wp_from_torch(c_self_q, dtype=wp_scalar),
                _wp_from_torch(c_self_mu, dtype=wp_scalar),
                _wp_from_torch(c_self_q2, dtype=wp_scalar),
                _wp_from_torch(c_bg_no_v, dtype=wp_scalar),
                has_dipoles,
                has_quadrupoles,
                _wp_from_torch(per_atom, dtype=wp_scalar),
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )
    if batch_idx is None:
        return per_atom.sum()
    n_systems = volume.shape[0]
    out = torch.zeros(n_systems, dtype=torch.float64, device=device)
    out.scatter_add_(0, batch_idx, per_atom)
    return out


def _multipole_pme_corrections_backward(  # pragma: no cover
    grad_out: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    volume: torch.Tensor,
    total_charge: torch.Tensor,
    batch_idx: torch.Tensor | None,
    c_self_q: torch.Tensor,
    c_self_mu: torch.Tensor,
    c_self_q2: torch.Tensor,
    c_bg_no_v: torch.Tensor,
    has_dipoles: bool,
    has_quadrupoles: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward: analytic ``∂L/∂{q, μ, Q, V}`` via the Warp backward kernel.

    Returns grads for ``charges``, ``dipoles``, ``quadrupoles``, ``volume``
    (the four ``diff_input_positions``). ``grad_out`` is the upstream scalar
    (single) or per-system ``(B,)`` cotangent.
    """
    device = charges.device
    wp_scalar = get_wp_dtype(torch.float64)
    wp_vec = get_wp_vec_dtype(torch.float64)
    wp_mat = get_wp_mat_dtype(torch.float64)
    n_atoms = charges.shape[0]
    grad_charges = torch.empty(n_atoms, dtype=torch.float64, device=device)
    grad_dipoles = torch.empty((n_atoms, 3), dtype=torch.float64, device=device)
    grad_quadrupoles = torch.empty((n_atoms, 3, 3), dtype=torch.float64, device=device)
    grad_volume = torch.zeros_like(volume, dtype=torch.float64)
    # ``grad_out`` is 0-d (single) or (B,); the kernels read it as a (1,) /
    # (B,) array, so reshape the single-system scalar to (1,).
    grad_out_arr = grad_out.reshape(-1).to(torch.float64).contiguous()
    with _scoped_warp_stream(device):
        if batch_idx is None:
            multipole_pme_corrections_backward_launch(
                _wp_from_torch(grad_out_arr, dtype=wp_scalar),
                _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
                _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
                _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
                _wp_from_torch(volume.contiguous(), dtype=wp_scalar),
                _wp_from_torch(total_charge.contiguous(), dtype=wp_scalar),
                _wp_from_torch(c_self_q, dtype=wp_scalar),
                _wp_from_torch(c_self_mu, dtype=wp_scalar),
                _wp_from_torch(c_self_q2, dtype=wp_scalar),
                _wp_from_torch(c_bg_no_v, dtype=wp_scalar),
                has_dipoles,
                has_quadrupoles,
                _wp_from_torch(grad_charges, dtype=wp_scalar),
                _wp_from_torch(grad_dipoles, dtype=wp_vec),
                _wp_from_torch(grad_quadrupoles, dtype=wp_mat),
                _wp_from_torch(grad_volume.reshape(-1), dtype=wp_scalar),
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )
        else:
            batch_i32 = batch_idx.to(torch.int32).contiguous()
            batch_multipole_pme_corrections_backward_launch(
                _wp_from_torch(grad_out_arr, dtype=wp_scalar),
                _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
                _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
                _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
                _wp_from_torch(batch_i32, dtype=wp.int32),
                _wp_from_torch(volume.contiguous(), dtype=wp_scalar),
                _wp_from_torch(total_charge.contiguous(), dtype=wp_scalar),
                _wp_from_torch(c_self_q, dtype=wp_scalar),
                _wp_from_torch(c_self_mu, dtype=wp_scalar),
                _wp_from_torch(c_self_q2, dtype=wp_scalar),
                _wp_from_torch(c_bg_no_v, dtype=wp_scalar),
                has_dipoles,
                has_quadrupoles,
                _wp_from_torch(grad_charges, dtype=wp_scalar),
                _wp_from_torch(grad_dipoles, dtype=wp_vec),
                _wp_from_torch(grad_quadrupoles, dtype=wp_mat),
                _wp_from_torch(grad_volume, dtype=wp_scalar),
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )
    return grad_charges, grad_dipoles, grad_quadrupoles, grad_volume


def _multipole_pme_corrections_double_backward(  # pragma: no cover
    gg_charges: torch.Tensor | None,
    gg_dipoles: torch.Tensor | None,
    gg_quadrupoles: torch.Tensor | None,
    gg_volume: torch.Tensor | None,
    grad_out: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    volume: torch.Tensor,
    total_charge: torch.Tensor,
    batch_idx: torch.Tensor | None,
    c_self_q: torch.Tensor,
    c_self_mu: torch.Tensor,
    c_self_q2: torch.Tensor,
    c_bg_no_v: torch.Tensor,
    has_dipoles: bool,
    has_quadrupoles: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Double-backward: propagate ``gg_*`` cotangents (one per backward
    output) to ``(grad_out, charges, dipoles, quadrupoles, volume)``.

    Needed for moment-moment HVPs (e.g. the l=2 Q-Q Hessian) through the
    PME composite force-loss. The first-order backward is linear in its
    differentiable inputs, so this is a fixed bilinear contraction.
    """
    device = charges.device
    wp_scalar = get_wp_dtype(torch.float64)
    wp_vec = get_wp_vec_dtype(torch.float64)
    wp_mat = get_wp_mat_dtype(torch.float64)
    n_atoms = charges.shape[0]
    n_sys = total_charge.shape[0]

    # Missing upstream cotangents → zeros (this op output didn't feed loss).
    if gg_charges is None:
        gg_charges = torch.zeros(n_atoms, dtype=torch.float64, device=device)
    if gg_dipoles is None:
        gg_dipoles = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    if gg_quadrupoles is None:
        gg_quadrupoles = torch.zeros(
            (n_atoms, 3, 3), dtype=torch.float64, device=device
        )
    if gg_volume is None:
        gg_volume = torch.zeros(n_sys, dtype=torch.float64, device=device)

    gg_charges = gg_charges.to(torch.float64).contiguous()
    gg_dipoles = gg_dipoles.to(torch.float64).contiguous()
    gg_quadrupoles = gg_quadrupoles.to(torch.float64).contiguous()
    gg_volume = gg_volume.to(torch.float64).reshape(-1).contiguous()

    # Per-system Σ_i gg_charges_i (a plain reduction).
    if batch_idx is None:
        sum_gg = gg_charges.sum().reshape(1)
    else:
        sum_gg = torch.zeros(n_sys, dtype=torch.float64, device=device)
        sum_gg.scatter_add_(0, batch_idx, gg_charges)

    grad_grad_out = torch.zeros(n_sys, dtype=torch.float64, device=device)
    grad_charges = torch.empty(n_atoms, dtype=torch.float64, device=device)
    grad_dipoles = torch.empty((n_atoms, 3), dtype=torch.float64, device=device)
    grad_quadrupoles = torch.empty((n_atoms, 3, 3), dtype=torch.float64, device=device)
    grad_volume = torch.zeros(n_sys, dtype=torch.float64, device=device)
    grad_out_arr = grad_out.reshape(-1).to(torch.float64).contiguous()

    with _scoped_warp_stream(device):
        if batch_idx is None:
            multipole_pme_corrections_double_backward_launch(
                _wp_from_torch(gg_charges, dtype=wp_scalar),
                _wp_from_torch(gg_dipoles, dtype=wp_vec),
                _wp_from_torch(gg_quadrupoles, dtype=wp_mat),
                _wp_from_torch(gg_volume, dtype=wp_scalar),
                _wp_from_torch(grad_out_arr, dtype=wp_scalar),
                _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
                _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
                _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
                _wp_from_torch(total_charge.contiguous(), dtype=wp_scalar),
                _wp_from_torch(sum_gg, dtype=wp_scalar),
                _wp_from_torch(volume.contiguous(), dtype=wp_scalar),
                _wp_from_torch(c_self_q, dtype=wp_scalar),
                _wp_from_torch(c_self_mu, dtype=wp_scalar),
                _wp_from_torch(c_self_q2, dtype=wp_scalar),
                _wp_from_torch(c_bg_no_v, dtype=wp_scalar),
                has_dipoles,
                has_quadrupoles,
                _wp_from_torch(grad_grad_out, dtype=wp_scalar),
                _wp_from_torch(grad_charges, dtype=wp_scalar),
                _wp_from_torch(grad_dipoles, dtype=wp_vec),
                _wp_from_torch(grad_quadrupoles, dtype=wp_mat),
                _wp_from_torch(grad_volume, dtype=wp_scalar),
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )
        else:
            batch_i32 = batch_idx.to(torch.int32).contiguous()
            batch_multipole_pme_corrections_double_backward_launch(
                _wp_from_torch(gg_charges, dtype=wp_scalar),
                _wp_from_torch(gg_dipoles, dtype=wp_vec),
                _wp_from_torch(gg_quadrupoles, dtype=wp_mat),
                _wp_from_torch(gg_volume, dtype=wp_scalar),
                _wp_from_torch(grad_out_arr, dtype=wp_scalar),
                _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
                _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
                _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
                _wp_from_torch(batch_i32, dtype=wp.int32),
                _wp_from_torch(total_charge.contiguous(), dtype=wp_scalar),
                _wp_from_torch(sum_gg, dtype=wp_scalar),
                _wp_from_torch(volume.contiguous(), dtype=wp_scalar),
                _wp_from_torch(c_self_q, dtype=wp_scalar),
                _wp_from_torch(c_self_mu, dtype=wp_scalar),
                _wp_from_torch(c_self_q2, dtype=wp_scalar),
                _wp_from_torch(c_bg_no_v, dtype=wp_scalar),
                has_dipoles,
                has_quadrupoles,
                _wp_from_torch(grad_grad_out, dtype=wp_scalar),
                _wp_from_torch(grad_charges, dtype=wp_scalar),
                _wp_from_torch(grad_dipoles, dtype=wp_vec),
                _wp_from_torch(grad_quadrupoles, dtype=wp_mat),
                _wp_from_torch(grad_volume, dtype=wp_scalar),
                wp_dtype=wp_scalar,
                device=str(wp.device_from_torch(device)),
            )

    # grad_out grad matches the original grad_out shape (0-d single, (B,)
    # batched); ``_match_shape`` collapses the (1,) single-system result.
    return grad_grad_out, grad_charges, grad_dipoles, grad_quadrupoles, grad_volume


def _corrections_forward_fake(  # pragma: no cover
    charges, dipoles, quadrupoles, volume, total_charge, batch_idx, *_
):
    """Fake: scalar (single) or per-system ``(B,)`` (batched)."""
    if batch_idx is None:
        return torch.empty((), dtype=torch.float64, device=charges.device)
    return torch.empty_like(volume, dtype=torch.float64)


def _corrections_backward_fake(  # pragma: no cover
    grad_out, charges, dipoles, quadrupoles, volume, total_charge, batch_idx, *_
):
    """Fake: 4-tuple (grad_charges, grad_dipoles, grad_quadrupoles, grad_volume)."""
    del grad_out, total_charge, batch_idx
    return (
        torch.empty_like(charges, dtype=torch.float64),
        torch.empty_like(dipoles, dtype=torch.float64),
        torch.empty_like(quadrupoles, dtype=torch.float64),
        torch.empty_like(volume, dtype=torch.float64),
    )


_CORRECTIONS_FWD_SCHEMA = (
    "(Tensor charges, Tensor dipoles, Tensor quadrupoles, Tensor volume, "
    "Tensor total_charge, Tensor? batch_idx, Tensor c_self_q, "
    "Tensor c_self_mu, Tensor c_self_q2, Tensor c_bg_no_v, bool has_dipoles, "
    "bool has_quadrupoles) -> Tensor"
)
_CORRECTIONS_BWD_SCHEMA = (
    "(Tensor grad_out, Tensor charges, Tensor dipoles, Tensor quadrupoles, "
    "Tensor volume, Tensor total_charge, Tensor? batch_idx, Tensor c_self_q, "
    "Tensor c_self_mu, Tensor c_self_q2, Tensor c_bg_no_v, bool has_dipoles, "
    "bool has_quadrupoles) -> (Tensor, Tensor, Tensor, Tensor)"
)
_CORRECTIONS_DBWD_SCHEMA = (
    "(Tensor? gg_charges, Tensor? gg_dipoles, Tensor? gg_quadrupoles, "
    "Tensor? gg_volume, Tensor grad_out, Tensor charges, Tensor dipoles, "
    "Tensor quadrupoles, Tensor volume, Tensor total_charge, Tensor? batch_idx, "
    "Tensor c_self_q, Tensor c_self_mu, Tensor c_self_q2, Tensor c_bg_no_v, "
    "bool has_dipoles, bool has_quadrupoles) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)


def _corrections_double_backward_fake(  # pragma: no cover
    gg_charges,
    gg_dipoles,
    gg_quadrupoles,
    gg_volume,
    grad_out,
    charges,
    dipoles,
    quadrupoles,
    volume,
    total_charge,
    *_,
):
    """Fake: grads for (grad_out, charges, dipoles, quadrupoles, volume)."""
    del gg_charges, gg_dipoles, gg_quadrupoles, gg_volume
    return (
        torch.empty_like(grad_out, dtype=torch.float64),
        torch.empty_like(charges, dtype=torch.float64),
        torch.empty_like(dipoles, dtype=torch.float64),
        torch.empty_like(quadrupoles, dtype=torch.float64),
        torch.empty_like(volume, dtype=torch.float64),
    )


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_corrections",
    forward=_multipole_pme_corrections_forward,
    forward_schema=_CORRECTIONS_FWD_SCHEMA,
    forward_fake=_corrections_forward_fake,
    backward=_multipole_pme_corrections_backward,
    backward_schema=_CORRECTIONS_BWD_SCHEMA,
    backward_fake=_corrections_backward_fake,
    backward_return_arity=4,
    diff_input_positions=(0, 1, 2, 3),
    n_forward_inputs=12,
    double_backward=_multipole_pme_corrections_double_backward,
    double_backward_schema=_CORRECTIONS_DBWD_SCHEMA,
    double_backward_fake=_corrections_double_backward_fake,
    double_backward_return_arity=5,
    # Backward inputs: grad_out(0), charges(1), dipoles(2), quadrupoles(3),
    # volume(4) are the differentiable slots the double-back returns grads for.
    second_order_diff_positions=(0, 1, 2, 3, 4),
    n_backward_inputs=13,
)


def multipole_pme_energy_corrections(
    charges: torch.Tensor,
    dipoles: torch.Tensor | None,
    sigma: float,
    alpha: float,
    volume: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None = None,
    quadrupoles: torch.Tensor | None = None,
    n_systems: int | None = None,
) -> torch.Tensor:
    r"""GTO-Ewald multipole self + background energy corrections.

    Returns the per-atom correction terms that must be subtracted from
    the raw reciprocal-space PME energy to recover the physical pair
    energy. Matches the convention from the direct-k
    ``_multipole_ewald_self_energy_per_atom`` in ``multipole_ewald.py``,
    so the reciprocal composite drops in this wrapper without unit
    conversion.

    Self-energy per atom (subtract from reciprocal):

    .. math::

        E_\text{self}(l = 0)_i &= \frac{F\, q_i^2}{8 \pi^{3/2}\,\sigma_c}, \\
        E_\text{self}(l = 1)_i &= \frac{F\, |\boldsymbol{\mu}_i|^2}{48 \pi^{3/2}\,\sigma_c^3}.

    where :math:`F = \mathrm{FIELD\_CONSTANT}` (Coulomb prefactor in
    nvalchemiops's Hartree-like units) and
    :math:`\sigma_c = \sqrt{\sigma^2 + 1/(4 \alpha^2)}` is the
    GTO-Ewald combined width.

    Background correction (non-neutral systems only):

    .. math::

        E_\text{background} = \frac{F \pi}{2 \alpha^2 V}\, Q_\text{total}^2

    For neutral systems (``Q_total = 0``) the background term vanishes;
    for non-neutral systems it is included via the standard
    "uniform neutralizing background" PME convention.

    The per-element physics (per-atom squares, the quadrupole Frobenius
    norm, the per-atom background share, and the analytic input gradients)
    runs in Warp kernels via the ``multipole_pme_corrections`` custom op;
    this wrapper only precomputes the scalar coefficients (folding in
    σ_c, α, F), feeds fp64 inputs, and the op reduces the per-atom output
    (``.sum()`` single / ``scatter_add`` batched). The result is
    autograd-connected to ``charges``, ``dipoles``, ``quadrupoles`` and
    ``volume``. Constants ``sigma``, ``alpha`` are passed as Python floats.

    Parameters
    ----------
    charges : torch.Tensor, shape ``(N,)``
        Atomic charges.
    dipoles : torch.Tensor or None, shape ``(N, 3)``
        Cartesian dipole moments, or ``None`` for a pure-monopole call.
        Passing ``None`` skips the dipole self-energy term entirely;
        bit-for-bit equivalent to passing ``torch.zeros((N, 3))``.
    sigma : float
        GTO width.
    alpha : float
        Ewald splitting parameter.
    volume : torch.Tensor, shape ``()`` or ``(1,)`` or ``(B,)``
        Cell volume. Must be a tensor (not a Python float) so autograd
        through cell parameters works in MD-with-cell-flexion contexts.
    batch_idx : torch.Tensor, shape ``(N,)``, dtype int32, optional
        Per-atom system index. If provided, returns per-system
        corrections of shape ``(B,)`` via ``scatter_add``; if ``None``,
        returns a scalar.
    quadrupoles : torch.Tensor or None, shape ``(N, 3, 3)``, optional
        Cartesian symmetric quadrupole moments, or ``None`` to skip the
        l=2 self-energy term. When provided, adds the per-atom term
        :math:`E_\text{self}(l = 2)_i = F\,|Q_i|_F^2 / (320\,\pi^{3/2}\,\sigma_c^5)`,
        where :math:`|Q_i|_F^2 = \sum_{\alpha\beta} Q_{i,\alpha\beta}^2` is
        the squared Frobenius norm; matches the direct-k
        ``_multipole_ewald_self_energy_per_atom`` convention.

    Returns
    -------
    correction : torch.Tensor
        ``E_self - E_background`` per system (or scalar). Caller
        SUBTRACTS this from the raw reciprocal energy: ``E_recip_corr
        = E_recip_raw - correction``.
    """
    if dipoles is not None and dipoles.shape != (charges.shape[0], 3):
        raise ValueError(
            f"dipoles must have shape (N, 3) = ({charges.shape[0]}, 3); "
            f"got {tuple(dipoles.shape)}"
        )

    sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
    pi32 = math.pi**1.5
    F = FIELD_CONSTANT
    device = charges.device

    # Precompute the scalar coefficients (fold in σ_c, α, F). The
    # background coefficient excludes the per-system volume so the kernel
    # divides by ``V`` at runtime (keeps ``∂/∂V`` autograd-aware).
    c_self_q = _corr_scalar_array(F / (8.0 * pi32 * sigma_c), device)
    c_self_mu = _corr_scalar_array(F / (48.0 * pi32 * sigma_c**3), device)
    c_self_q2 = _corr_scalar_array(F / (320.0 * pi32 * sigma_c**5), device)
    c_bg_no_v = _corr_scalar_array(F * math.pi / (2.0 * alpha**2), device)

    charges_f64 = charges.to(torch.float64)
    # Optional moments → zeros (bit-for-bit equal to the explicit-zero call)
    # plus a flag so the kernel skips the channel entirely.
    has_dipoles = dipoles is not None
    has_quadrupoles = quadrupoles is not None
    dipoles_f64 = (
        dipoles.to(torch.float64)
        if has_dipoles
        else torch.zeros((charges.shape[0], 3), dtype=torch.float64, device=device)
    )
    quadrupoles_f64 = (
        quadrupoles.to(torch.float64)
        if has_quadrupoles
        else torch.zeros((charges.shape[0], 3, 3), dtype=torch.float64, device=device)
    )

    if batch_idx is None:
        total_charge = charges_f64.sum().reshape(1)
        # ``volume`` may be shape () or (1,); the kernel reads it as (1,).
        vol_arg = volume.to(torch.float64).reshape(1)
        return torch.ops.nvalchemiops.multipole_pme_corrections(
            charges_f64,
            dipoles_f64,
            quadrupoles_f64,
            vol_arg,
            total_charge,
            None,
            c_self_q,
            c_self_mu,
            c_self_q2,
            c_bg_no_v,
            has_dipoles,
            has_quadrupoles,
        )

    # Batched: per-system total charge + per-system volume (B,).
    if n_systems is None:
        # Eager fallback only; the batched composite passes n_systems to
        # avoid this device sync (a torch.compile graph break) on the hot path.
        n_systems = int(batch_idx.max().item()) + 1
    total_charge = torch.zeros(n_systems, dtype=torch.float64, device=device)
    total_charge.scatter_add_(0, batch_idx, charges_f64)
    vol_per_system = volume.to(torch.float64).reshape(-1)
    if vol_per_system.numel() == 1 and n_systems > 1:
        vol_per_system = vol_per_system.expand(n_systems)
    return torch.ops.nvalchemiops.multipole_pme_corrections(
        charges_f64,
        dipoles_f64,
        quadrupoles_f64,
        vol_per_system,
        total_charge,
        batch_idx.to(torch.int32),
        c_self_q,
        c_self_mu,
        c_self_q2,
        c_bg_no_v,
        has_dipoles,
        has_quadrupoles,
    )


def multipole_pme_energy_corrections_per_atom(
    charges: torch.Tensor,
    dipoles: torch.Tensor | None,
    sigma: float,
    alpha: float,
    volume: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None = None,
    quadrupoles: torch.Tensor | None = None,
    n_systems: int | None = None,
) -> torch.Tensor:
    r"""Per-atom GTO-Ewald multipole self + background corrections.

    Per-atom ``(N,)`` (single) / ``(N_total,)`` (batched) counterpart of
    :func:`multipole_pme_energy_corrections`. The reduced wrapper returns
    ``Σ_i correction_i`` (scalar / ``(B,)``); this returns the per-atom
    ``correction_i`` so the per-atom-energy composite can assemble
    ``E_i = E_real_i + E_recip_i − correction_i`` and let the caller own the
    reduction (the PR #96 energy-autograd contract — ``Σ_i`` matches the reduced
    op bit-for-bit, and ``grad(Σ_i, ·)`` matches the reduced op's gradients).

    Closed forms (all per-atom, ``F = FIELD_CONSTANT``,
    ``σ_c = sqrt(σ² + 1/(4α²))``):

    .. math::

        \text{self}_i &= \frac{F q_i^2}{8\pi^{3/2}\sigma_c}
          + \frac{F |\mu_i|^2}{48\pi^{3/2}\sigma_c^3}
          + \frac{F |Q_i|_F^2}{320\pi^{3/2}\sigma_c^5}, \\
        \text{bg}_i &= \frac{F\pi}{2\alpha^2 V}\, q_i\, Q_\text{total}, \\
        \text{correction}_i &= \text{self}_i - \text{bg}_i.

    The background is split per atom as ``q_i · Q_total`` so the per-atom sum
    recovers the collective ``(F\pi/2\alpha^2 V) Q_\text{total}^2``. Pure-torch
    elementwise (twice-differentiable for free) — matches the
    ``multipole_pme_corrections`` Warp op's reduced value/grads to machine eps.

    Parameters mirror :func:`multipole_pme_energy_corrections`. Returns a
    ``float64`` tensor of shape ``(N,)`` (single) or ``(N_total,)`` (batched).
    """
    if dipoles is not None and dipoles.shape != (charges.shape[0], 3):
        raise ValueError(
            f"dipoles must have shape (N, 3) = ({charges.shape[0]}, 3); "
            f"got {tuple(dipoles.shape)}"
        )

    sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
    pi32 = math.pi**1.5
    F = FIELD_CONSTANT
    c_self_q = F / (8.0 * pi32 * sigma_c)
    c_self_mu = F / (48.0 * pi32 * sigma_c**3)
    c_self_q2 = F / (320.0 * pi32 * sigma_c**5)
    c_bg_no_v = F * math.pi / (2.0 * alpha**2)

    charges_f64 = charges.to(torch.float64)
    self_e = c_self_q * charges_f64.square()
    if dipoles is not None:
        self_e = self_e + c_self_mu * dipoles.to(torch.float64).square().sum(-1)
    if quadrupoles is not None:
        self_e = self_e + c_self_q2 * quadrupoles.to(torch.float64).square().sum(
            (-1, -2)
        )

    # Background: per-atom share q_i · Q_total / V so Σ_i = Q_total² / V (× c_bg).
    if batch_idx is None:
        total_charge = charges_f64.sum()
        inv_v = (c_bg_no_v / volume.to(torch.float64).reshape(())).reshape(())
        bg = inv_v * charges_f64 * total_charge
    else:
        if n_systems is None:
            n_systems = int(batch_idx.max().item()) + 1
        total_charge = torch.zeros(
            n_systems, dtype=torch.float64, device=charges.device
        )
        total_charge = total_charge.scatter_add(0, batch_idx, charges_f64)
        vol_ps = volume.to(torch.float64).reshape(-1)
        if vol_ps.numel() == 1 and n_systems > 1:
            vol_ps = vol_ps.expand(n_systems)
        per_atom_v = vol_ps.index_select(0, batch_idx)
        per_atom_qtot = total_charge.index_select(0, batch_idx)
        bg = c_bg_no_v * charges_f64 * per_atom_qtot / per_atom_v

    return self_e + bg


# =============================================================================
# Reciprocal-space per-k energy from rho (shared by direct k-space + Ewald reciprocal)
# =============================================================================


def _multipole_reciprocal_rho_energy_forward(
    rho: torch.Tensor,
    per_k_factor: torch.Tensor,
) -> torch.Tensor:
    """Forward: per-k energy ``2 f_k |rho_k|^2`` (flat ``(M,)`` output).

    ``rho`` is ``(M, 2)`` (real/imag) and ``per_k_factor`` is ``(M,)``;
    single- and batched-system callers flatten the leading axes
    (``M = K`` or ``M = B * K``) and reshape the output back themselves.
    """
    device = rho.device
    wp_scalar = get_wp_dtype(torch.float64)
    m = rho.shape[0]
    per_k_energy = torch.empty(m, dtype=torch.float64, device=device)
    wp_vec2 = wp.vec2d
    with _scoped_warp_stream(device):
        multipole_reciprocal_rho_energy_launch(
            _wp_from_torch(rho.contiguous(), dtype=wp_vec2),
            _wp_from_torch(per_k_factor.contiguous(), dtype=wp_scalar),
            _wp_from_torch(per_k_energy, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return per_k_energy


def _multipole_reciprocal_rho_energy_backward(  # pragma: no cover
    grad_out: torch.Tensor,
    rho: torch.Tensor,
    per_k_factor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Backward: ``∂L/∂rho`` and ``∂L/∂per_k_factor``.

    .. math::

        grad\_rho_{k,c} = g_k\,4\,f_k\,rho_{k,c}, \quad
        grad\_f_k = g_k\,2\,|rho_k|^2 .

    ``per_k_factor`` is differentiable because it depends on the cell
    (cell-grad / stress path).
    """
    device = rho.device
    wp_scalar = get_wp_dtype(torch.float64)
    wp_vec2 = wp.vec2d
    grad_rho = torch.empty_like(rho, dtype=torch.float64)
    grad_pkf = torch.empty(rho.shape[0], dtype=torch.float64, device=device)
    with _scoped_warp_stream(device):
        multipole_reciprocal_rho_energy_backward_launch(
            _wp_from_torch(grad_out.contiguous().to(torch.float64), dtype=wp_scalar),
            _wp_from_torch(rho.contiguous(), dtype=wp_vec2),
            _wp_from_torch(per_k_factor.contiguous(), dtype=wp_scalar),
            _wp_from_torch(grad_rho, dtype=wp_vec2),
            _wp_from_torch(grad_pkf, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return grad_rho, grad_pkf


def _multipole_reciprocal_rho_energy_double_backward(  # pragma: no cover
    gg_rho: torch.Tensor | None,
    gg_per_k_factor: torch.Tensor | None,
    grad_out: torch.Tensor,
    rho: torch.Tensor,
    per_k_factor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Double-backward: propagate ``(gg_rho, gg_per_k_factor)`` to
    ``(grad_out, rho, per_k_factor)``.

    The first-order backward is bilinear in ``(grad_out, rho, per_k_factor)``;
    needed for position / moment / cell HVPs (force-loss) since ``rho``
    depends on positions and the assembly is quadratic in ``rho``.
    """
    device = rho.device
    wp_scalar = get_wp_dtype(torch.float64)
    wp_vec2 = wp.vec2d
    m = rho.shape[0]
    if gg_rho is None:
        gg_rho = torch.zeros_like(rho, dtype=torch.float64)
    if gg_per_k_factor is None:
        gg_per_k_factor = torch.zeros(m, dtype=torch.float64, device=device)
    gg_rho = gg_rho.to(torch.float64).contiguous()
    gg_per_k_factor = gg_per_k_factor.to(torch.float64).contiguous()
    grad_grad_out = torch.empty(m, dtype=torch.float64, device=device)
    grad_rho = torch.empty_like(rho, dtype=torch.float64)
    grad_pkf = torch.empty(m, dtype=torch.float64, device=device)
    with _scoped_warp_stream(device):
        multipole_reciprocal_rho_energy_double_backward_launch(
            _wp_from_torch(gg_rho, dtype=wp_vec2),
            _wp_from_torch(gg_per_k_factor, dtype=wp_scalar),
            _wp_from_torch(grad_out.contiguous().to(torch.float64), dtype=wp_scalar),
            _wp_from_torch(rho.contiguous(), dtype=wp_vec2),
            _wp_from_torch(per_k_factor.contiguous(), dtype=wp_scalar),
            _wp_from_torch(grad_grad_out, dtype=wp_scalar),
            _wp_from_torch(grad_rho, dtype=wp_vec2),
            _wp_from_torch(grad_pkf, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return grad_grad_out, grad_rho, grad_pkf


def _rho_energy_forward_fake(rho, per_k_factor):  # pragma: no cover
    """Fake: per-k energy ``(M,)``."""
    del per_k_factor
    return torch.empty(rho.shape[0], dtype=torch.float64, device=rho.device)


def _rho_energy_backward_fake(grad_out, rho, per_k_factor):  # pragma: no cover
    """Fake: (grad_rho, grad_per_k_factor)."""
    del grad_out
    return (
        torch.empty_like(rho, dtype=torch.float64),
        torch.empty_like(per_k_factor, dtype=torch.float64),
    )


def _rho_energy_double_backward_fake(  # pragma: no cover
    gg_rho, gg_per_k_factor, grad_out, rho, per_k_factor
):
    """Fake: grads for ``(grad_out, rho, per_k_factor)``."""
    del gg_rho, gg_per_k_factor
    return (
        torch.empty_like(grad_out, dtype=torch.float64),
        torch.empty_like(rho, dtype=torch.float64),
        torch.empty_like(per_k_factor, dtype=torch.float64),
    )


_RHO_ENERGY_DBWD_SCHEMA = (
    "(Tensor? gg_rho, Tensor? gg_per_k_factor, Tensor grad_out, Tensor rho, "
    "Tensor per_k_factor) -> (Tensor, Tensor, Tensor)"
)
register_warp_op_chain(
    name="nvalchemiops::multipole_reciprocal_rho_energy",
    forward=_multipole_reciprocal_rho_energy_forward,
    forward_fake=_rho_energy_forward_fake,
    backward=_multipole_reciprocal_rho_energy_backward,
    backward_fake=_rho_energy_backward_fake,
    backward_return_arity=2,
    diff_input_positions=(0, 1),  # rho, per_k_factor
    n_forward_inputs=2,
    double_backward=_multipole_reciprocal_rho_energy_double_backward,
    double_backward_schema=_RHO_ENERGY_DBWD_SCHEMA,
    double_backward_fake=_rho_energy_double_backward_fake,
    double_backward_return_arity=3,
    # Backward inputs: grad_out(0), rho(1), per_k_factor(2) are the diff slots
    # the double-backward returns grads for.
    second_order_diff_positions=(0, 1, 2),
    n_backward_inputs=3,
)


def multipole_reciprocal_rho_energy(
    rho: torch.Tensor,
    per_k_factor: torch.Tensor,
) -> torch.Tensor:
    r"""Per-k reciprocal energy ``2 f_k |rho_k|^2`` (autograd-connected to rho).

    Moves the per-element ``|rho|^2`` physics into a Warp kernel shared by
    the direct-k-space SCF step, the feature step, and the reciprocal fused-scalar
    Functions. Accepts ``rho`` of any leading shape ending in a length-2
    (real, imag) axis; flattens to ``(M, 2)`` and returns the per-k energy
    with the SAME leading shape (minus the trailing 2). The caller applies
    the ``0.5 V / (2 pi)^6`` scale + reduction in torch (keeping the volume
    cell-grad torch-side).

    Parameters
    ----------
    rho : torch.Tensor
        Reciprocal density, shape ``(..., 2)`` (real/imag last axis).
    per_k_factor : torch.Tensor
        Position-independent per-k factor, broadcastable to ``rho[..., 0]``.

    Returns
    -------
    torch.Tensor
        Per-k energy ``2 f_k |rho_k|^2``, shape ``rho.shape[:-1]``, float64.
    """
    lead = rho.shape[:-1]
    rho_flat = rho.reshape(-1, 2).to(torch.float64)
    pkf_flat = per_k_factor.reshape(-1).to(torch.float64)
    e_flat = torch.ops.nvalchemiops.multipole_reciprocal_rho_energy(rho_flat, pkf_flat)
    return e_flat.reshape(lead)


# =============================================================================
# Multipole self-energy: weighted sum of per-atom moment squares
# =============================================================================


def _multipole_self_energy_forward(
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    c_self_q: torch.Tensor,
    c_self_mu: torch.Tensor,
    c_self_q2: torch.Tensor,
    has_dipoles: bool,
    has_quadrupoles: bool,
) -> torch.Tensor:
    """Forward: per-atom ``c0 q^2 + c1 |mu|^2 + c2 |Q|_F^2`` (``(N,)``)."""
    device = charges.device
    wp_scalar = get_wp_dtype(torch.float64)
    wp_vec = get_wp_vec_dtype(torch.float64)
    wp_mat = get_wp_mat_dtype(torch.float64)
    n_atoms = charges.shape[0]
    self_energy = torch.empty(n_atoms, dtype=torch.float64, device=device)
    with _scoped_warp_stream(device):
        multipole_self_energy_launch(
            _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
            _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
            _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
            _wp_from_torch(c_self_q, dtype=wp_scalar),
            _wp_from_torch(c_self_mu, dtype=wp_scalar),
            _wp_from_torch(c_self_q2, dtype=wp_scalar),
            has_dipoles,
            has_quadrupoles,
            _wp_from_torch(self_energy, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return self_energy


def _multipole_self_energy_backward(  # pragma: no cover
    grad_out: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    c_self_q: torch.Tensor,
    c_self_mu: torch.Tensor,
    c_self_q2: torch.Tensor,
    has_dipoles: bool,
    has_quadrupoles: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward: ``∂L/∂{q, mu, Q}`` (per-atom cotangent ``grad_out``)."""
    device = charges.device
    wp_scalar = get_wp_dtype(torch.float64)
    wp_vec = get_wp_vec_dtype(torch.float64)
    wp_mat = get_wp_mat_dtype(torch.float64)
    n_atoms = charges.shape[0]
    grad_charges = torch.empty(n_atoms, dtype=torch.float64, device=device)
    grad_dipoles = torch.empty((n_atoms, 3), dtype=torch.float64, device=device)
    grad_quadrupoles = torch.empty((n_atoms, 3, 3), dtype=torch.float64, device=device)
    with _scoped_warp_stream(device):
        multipole_self_energy_backward_launch(
            _wp_from_torch(grad_out.contiguous().to(torch.float64), dtype=wp_scalar),
            _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
            _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
            _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
            _wp_from_torch(c_self_q, dtype=wp_scalar),
            _wp_from_torch(c_self_mu, dtype=wp_scalar),
            _wp_from_torch(c_self_q2, dtype=wp_scalar),
            has_dipoles,
            has_quadrupoles,
            _wp_from_torch(grad_charges, dtype=wp_scalar),
            _wp_from_torch(grad_dipoles, dtype=wp_vec),
            _wp_from_torch(grad_quadrupoles, dtype=wp_mat),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return grad_charges, grad_dipoles, grad_quadrupoles


def _multipole_self_energy_double_backward(  # pragma: no cover
    gg_charges: torch.Tensor | None,
    gg_dipoles: torch.Tensor | None,
    gg_quadrupoles: torch.Tensor | None,
    grad_out: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    c_self_q: torch.Tensor,
    c_self_mu: torch.Tensor,
    c_self_q2: torch.Tensor,
    has_dipoles: bool,
    has_quadrupoles: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Double-backward: propagate ``gg_*`` to ``(grad_out, q, mu, Q)``.

    The first-order backward is bilinear in ``(grad_out, moments)``; needed
    for moment-moment HVPs (force-loss) through the reciprocal composite.
    """
    device = charges.device
    wp_scalar = get_wp_dtype(torch.float64)
    wp_vec = get_wp_vec_dtype(torch.float64)
    wp_mat = get_wp_mat_dtype(torch.float64)
    n_atoms = charges.shape[0]

    if gg_charges is None:
        gg_charges = torch.zeros(n_atoms, dtype=torch.float64, device=device)
    if gg_dipoles is None:
        gg_dipoles = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    if gg_quadrupoles is None:
        gg_quadrupoles = torch.zeros(
            (n_atoms, 3, 3), dtype=torch.float64, device=device
        )
    gg_charges = gg_charges.to(torch.float64).contiguous()
    gg_dipoles = gg_dipoles.to(torch.float64).contiguous()
    gg_quadrupoles = gg_quadrupoles.to(torch.float64).contiguous()

    grad_grad_out = torch.empty(n_atoms, dtype=torch.float64, device=device)
    grad_charges = torch.empty(n_atoms, dtype=torch.float64, device=device)
    grad_dipoles = torch.empty((n_atoms, 3), dtype=torch.float64, device=device)
    grad_quadrupoles = torch.empty((n_atoms, 3, 3), dtype=torch.float64, device=device)
    with _scoped_warp_stream(device):
        multipole_self_energy_double_backward_launch(
            _wp_from_torch(gg_charges, dtype=wp_scalar),
            _wp_from_torch(gg_dipoles, dtype=wp_vec),
            _wp_from_torch(gg_quadrupoles, dtype=wp_mat),
            _wp_from_torch(grad_out.contiguous().to(torch.float64), dtype=wp_scalar),
            _wp_from_torch(charges.contiguous(), dtype=wp_scalar),
            _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
            _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
            _wp_from_torch(c_self_q, dtype=wp_scalar),
            _wp_from_torch(c_self_mu, dtype=wp_scalar),
            _wp_from_torch(c_self_q2, dtype=wp_scalar),
            has_dipoles,
            has_quadrupoles,
            _wp_from_torch(grad_grad_out, dtype=wp_scalar),
            _wp_from_torch(grad_charges, dtype=wp_scalar),
            _wp_from_torch(grad_dipoles, dtype=wp_vec),
            _wp_from_torch(grad_quadrupoles, dtype=wp_mat),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return grad_grad_out, grad_charges, grad_dipoles, grad_quadrupoles


def _self_energy_forward_fake(charges, *_):  # pragma: no cover
    """Fake: per-atom self-energy ``(N,)``."""
    return torch.empty_like(charges, dtype=torch.float64)


def _self_energy_backward_fake(
    grad_out, charges, dipoles, quadrupoles, *_
):  # pragma: no cover
    """Fake: 3-tuple (grad_charges, grad_dipoles, grad_quadrupoles)."""
    del grad_out
    return (
        torch.empty_like(charges, dtype=torch.float64),
        torch.empty_like(dipoles, dtype=torch.float64),
        torch.empty_like(quadrupoles, dtype=torch.float64),
    )


def _self_energy_double_backward_fake(  # pragma: no cover
    gg_charges, gg_dipoles, gg_quadrupoles, grad_out, charges, dipoles, quadrupoles, *_
):
    """Fake: grads for (grad_out, charges, dipoles, quadrupoles)."""
    del gg_charges, gg_dipoles, gg_quadrupoles
    return (
        torch.empty_like(grad_out, dtype=torch.float64),
        torch.empty_like(charges, dtype=torch.float64),
        torch.empty_like(dipoles, dtype=torch.float64),
        torch.empty_like(quadrupoles, dtype=torch.float64),
    )


# Diff inputs: charges(0), dipoles(1), quadrupoles(2). The op is quadratic
# in the moments → double-backward (moment-moment HVP / force-loss).
_SELF_ENERGY_DBWD_SCHEMA = (
    "(Tensor? gg_charges, Tensor? gg_dipoles, Tensor? gg_quadrupoles, "
    "Tensor grad_out, Tensor charges, Tensor dipoles, Tensor quadrupoles, "
    "Tensor c_self_q, Tensor c_self_mu, Tensor c_self_q2, bool has_dipoles, "
    "bool has_quadrupoles) -> (Tensor, Tensor, Tensor, Tensor)"
)
register_warp_op_chain(
    name="nvalchemiops::multipole_self_energy",
    forward=_multipole_self_energy_forward,
    forward_fake=_self_energy_forward_fake,
    backward=_multipole_self_energy_backward,
    backward_fake=_self_energy_backward_fake,
    backward_return_arity=3,
    diff_input_positions=(0, 1, 2),
    n_forward_inputs=8,
    double_backward=_multipole_self_energy_double_backward,
    double_backward_schema=_SELF_ENERGY_DBWD_SCHEMA,
    double_backward_fake=_self_energy_double_backward_fake,
    double_backward_return_arity=4,
    # Backward inputs: grad_out(0), charges(1), dipoles(2), quadrupoles(3).
    second_order_diff_positions=(0, 1, 2, 3),
    n_backward_inputs=9,
)


def multipole_self_energy(
    charges: torch.Tensor,
    dipoles: torch.Tensor | None,
    quadrupoles: torch.Tensor | None,
    c_self_q: torch.Tensor,
    c_self_mu: torch.Tensor,
    c_self_q2: torch.Tensor,
) -> torch.Tensor:
    r"""Per-atom self-energy ``c0 q^2 + c1 |mu|^2 + c2 |Q|_F^2``.

    Moves the per-element moment-square physics into a Warp kernel shared by
    the direct-k-space SCF energy/feature steps and the reciprocal fused-scalar
    Functions. ``dipoles`` / ``quadrupoles`` may be ``None`` to skip the
    l=1 / l=2 channel (the kernel gates them by flag). The overlap-constant
    coefficients are precomputed by the caller; the reduction
    (``.sum()`` / ``scatter_add``) also stays caller-side.

    Parameters
    ----------
    charges : torch.Tensor, shape ``(N,)``
    dipoles : torch.Tensor or None, shape ``(N, 3)``
    quadrupoles : torch.Tensor or None, shape ``(N, 3, 3)``
    c_self_q, c_self_mu, c_self_q2 : torch.Tensor
        Length-1 fp64 overlap-constant coefficients (l=0/1/2).

    Returns
    -------
    torch.Tensor
        Per-atom self-energy, shape ``(N,)``, float64.
    """
    device = charges.device
    n_atoms = charges.shape[0]
    has_dipoles = dipoles is not None
    has_quadrupoles = quadrupoles is not None
    dip = (
        dipoles.to(torch.float64)
        if has_dipoles
        else torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    )
    quad = (
        quadrupoles.to(torch.float64)
        if has_quadrupoles
        else torch.zeros((n_atoms, 3, 3), dtype=torch.float64, device=device)
    )
    return torch.ops.nvalchemiops.multipole_self_energy(
        charges.to(torch.float64),
        dip,
        quad,
        c_self_q,
        c_self_mu,
        c_self_q2,
        has_dipoles,
        has_quadrupoles,
    )


# =============================================================================
# PME mesh inner product: ``Σ_g rho_grid · phi_grid``
# =============================================================================


def _multipole_pme_mesh_inner_product_forward(
    rho_grid: torch.Tensor,
    phi_grid: torch.Tensor,
) -> torch.Tensor:
    """Forward: per-grid-point product ``rho_grid * phi_grid`` (flat ``(M,)``).

    ``rho_grid`` / ``phi_grid`` are flat ``(M,)`` fp64 (``M = Nx*Ny*Nz`` single
    / ``M = B*Nx*Ny*Nz`` batched); the caller reshapes + reduces.
    """
    device = rho_grid.device
    wp_scalar = get_wp_dtype(torch.float64)
    m = rho_grid.shape[0]
    per_grid_energy = torch.empty(m, dtype=torch.float64, device=device)
    with _scoped_warp_stream(device):
        multipole_pme_mesh_inner_product_launch(
            _wp_from_torch(rho_grid.contiguous(), dtype=wp_scalar),
            _wp_from_torch(phi_grid.contiguous(), dtype=wp_scalar),
            _wp_from_torch(per_grid_energy, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return per_grid_energy


def _multipole_pme_mesh_inner_product_backward(  # pragma: no cover
    grad_out: torch.Tensor,
    rho_grid: torch.Tensor,
    phi_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Backward: ``∂L/∂rho_grid`` and ``∂L/∂phi_grid`` (bilinear product).

    .. math::

        grad\_rho_g = g_g\,phi_g, \qquad grad\_phi_g = g_g\,rho_g .

    Both grids are autograd-connected (they depend on positions / moments via
    spread + convolve), so both gradients are returned.
    """
    device = rho_grid.device
    wp_scalar = get_wp_dtype(torch.float64)
    grad_rho_grid = torch.empty_like(rho_grid, dtype=torch.float64)
    grad_phi_grid = torch.empty_like(phi_grid, dtype=torch.float64)
    with _scoped_warp_stream(device):
        multipole_pme_mesh_inner_product_backward_launch(
            _wp_from_torch(grad_out.contiguous().to(torch.float64), dtype=wp_scalar),
            _wp_from_torch(rho_grid.contiguous(), dtype=wp_scalar),
            _wp_from_torch(phi_grid.contiguous(), dtype=wp_scalar),
            _wp_from_torch(grad_rho_grid, dtype=wp_scalar),
            _wp_from_torch(grad_phi_grid, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return grad_rho_grid, grad_phi_grid


def _multipole_pme_mesh_inner_product_double_backward(  # pragma: no cover
    gg_rho_grid: torch.Tensor | None,
    gg_phi_grid: torch.Tensor | None,
    grad_out: torch.Tensor,
    rho_grid: torch.Tensor,
    phi_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Double-backward: propagate ``(gg_rho_grid, gg_phi_grid)`` to
    ``(grad_out, rho_grid, phi_grid)``.

    The first-order backward is bilinear in ``(grad_out, rho_grid, phi_grid)``;
    needed for the PME force-loss / stress HVPs since both grids depend on
    positions / moments / cell and the assembly is bilinear in them.
    """
    device = rho_grid.device
    wp_scalar = get_wp_dtype(torch.float64)
    m = rho_grid.shape[0]
    if gg_rho_grid is None:
        gg_rho_grid = torch.zeros(m, dtype=torch.float64, device=device)
    if gg_phi_grid is None:
        gg_phi_grid = torch.zeros(m, dtype=torch.float64, device=device)
    gg_rho_grid = gg_rho_grid.to(torch.float64).contiguous()
    gg_phi_grid = gg_phi_grid.to(torch.float64).contiguous()
    grad_grad_out = torch.empty(m, dtype=torch.float64, device=device)
    grad_rho_grid = torch.empty_like(rho_grid, dtype=torch.float64)
    grad_phi_grid = torch.empty_like(phi_grid, dtype=torch.float64)
    with _scoped_warp_stream(device):
        multipole_pme_mesh_inner_product_double_backward_launch(
            _wp_from_torch(gg_rho_grid, dtype=wp_scalar),
            _wp_from_torch(gg_phi_grid, dtype=wp_scalar),
            _wp_from_torch(grad_out.contiguous().to(torch.float64), dtype=wp_scalar),
            _wp_from_torch(rho_grid.contiguous(), dtype=wp_scalar),
            _wp_from_torch(phi_grid.contiguous(), dtype=wp_scalar),
            _wp_from_torch(grad_grad_out, dtype=wp_scalar),
            _wp_from_torch(grad_rho_grid, dtype=wp_scalar),
            _wp_from_torch(grad_phi_grid, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return grad_grad_out, grad_rho_grid, grad_phi_grid


def _mesh_inner_product_forward_fake(rho_grid, phi_grid):  # pragma: no cover
    """Fake: per-grid-point energy ``(M,)``."""
    del phi_grid
    return torch.empty(rho_grid.shape[0], dtype=torch.float64, device=rho_grid.device)


def _mesh_inner_product_backward_fake(grad_out, rho_grid, phi_grid):  # pragma: no cover
    """Fake: (grad_rho_grid, grad_phi_grid)."""
    del grad_out
    return (
        torch.empty_like(rho_grid, dtype=torch.float64),
        torch.empty_like(phi_grid, dtype=torch.float64),
    )


def _mesh_inner_product_double_backward_fake(  # pragma: no cover
    gg_rho_grid, gg_phi_grid, grad_out, rho_grid, phi_grid
):
    """Fake: grads for ``(grad_out, rho_grid, phi_grid)``."""
    del gg_rho_grid, gg_phi_grid
    return (
        torch.empty_like(grad_out, dtype=torch.float64),
        torch.empty_like(rho_grid, dtype=torch.float64),
        torch.empty_like(phi_grid, dtype=torch.float64),
    )


_MESH_INNER_PRODUCT_DBWD_SCHEMA = (
    "(Tensor? gg_rho_grid, Tensor? gg_phi_grid, Tensor grad_out, "
    "Tensor rho_grid, Tensor phi_grid) -> (Tensor, Tensor, Tensor)"
)
register_warp_op_chain(
    name="nvalchemiops::multipole_pme_mesh_inner_product",
    forward=_multipole_pme_mesh_inner_product_forward,
    forward_fake=_mesh_inner_product_forward_fake,
    backward=_multipole_pme_mesh_inner_product_backward,
    backward_fake=_mesh_inner_product_backward_fake,
    backward_return_arity=2,
    diff_input_positions=(0, 1),  # rho_grid, phi_grid
    n_forward_inputs=2,
    double_backward=_multipole_pme_mesh_inner_product_double_backward,
    double_backward_schema=_MESH_INNER_PRODUCT_DBWD_SCHEMA,
    double_backward_fake=_mesh_inner_product_double_backward_fake,
    double_backward_return_arity=3,
    # Backward inputs: grad_out(0), rho_grid(1), phi_grid(2) are the diff slots
    # the double-backward returns grads for.
    second_order_diff_positions=(0, 1, 2),
    n_backward_inputs=3,
)


def multipole_pme_mesh_inner_product(
    rho_grid: torch.Tensor,
    phi_grid: torch.Tensor,
) -> torch.Tensor:
    r"""Mesh inner product ``Σ_g rho_grid · phi_grid`` (autograd-connected).

    Moves the per-element ``rho_grid * phi_grid`` product physics into a Warp
    kernel; the caller owns the reduction + ``F/(4 pi)`` scale. Accepts the 3D
    single-system mesh ``(Nx, Ny, Nz)`` or the 4D batched mesh
    ``(B, Nx, Ny, Nz)``: single-system flattens to one ``(M,)`` grid and
    reduces to a scalar; batched flattens per system and reduces over the
    spatial axes to ``(B,)``. Inputs are cast to ``float64`` (preserving the
    pre-existing ``.to(float64)`` accumulation convention). Both ``rho_grid``
    and ``phi_grid`` are differentiable.

    Parameters
    ----------
    rho_grid : torch.Tensor
        Spread density mesh, shape ``(Nx, Ny, Nz)`` or ``(B, Nx, Ny, Nz)``.
    phi_grid : torch.Tensor
        Convolved potential mesh, same shape as ``rho_grid``.

    Returns
    -------
    torch.Tensor
        Single-system: scalar ``()``. Batched: ``(B,)`` per-system inner
        product. Both ``float64``.
    """
    rho_flat = rho_grid.reshape(-1).to(torch.float64)
    phi_flat = phi_grid.reshape(-1).to(torch.float64)
    e_flat = torch.ops.nvalchemiops.multipole_pme_mesh_inner_product(rho_flat, phi_flat)
    if rho_grid.dim() == 4:
        # Batched ``(B, Nx, Ny, Nz)``: per-system sum over the spatial axes.
        b = rho_grid.shape[0]
        return e_flat.reshape(b, -1).sum(dim=1)
    return e_flat.sum()


# =============================================================================
# Reciprocal-space composite
# =============================================================================


def _pme_k_squared_forward(
    inv_cell_t: torch.Tensor,
    miller_x: torch.Tensor,
    miller_y: torch.Tensor,
    miller_z: torch.Tensor,
    nx: int,
    ny: int,
    nz_rfft: int,
) -> torch.Tensor:
    """Wrap :func:`pme_k_squared_launch` as a torch custom_op forward."""
    device = inv_cell_t.device
    dtype = inv_cell_t.dtype
    wp_scalar = get_wp_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    k_squared = torch.empty((nx, ny, nz_rfft), dtype=dtype, device=device)
    with _scoped_warp_stream(device):
        pme_k_squared_launch(
            _wp_from_torch(miller_x.contiguous(), dtype=wp_scalar),
            _wp_from_torch(miller_y.contiguous(), dtype=wp_scalar),
            _wp_from_torch(miller_z.contiguous(), dtype=wp_scalar),
            _wp_from_torch(inv_cell_t.contiguous(), dtype=wp_mat),
            _wp_from_torch(k_squared, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return k_squared


def _pme_k_squared_backward(  # pragma: no cover
    grad_k_squared: torch.Tensor,
    inv_cell_t: torch.Tensor,
    miller_x: torch.Tensor,
    miller_y: torch.Tensor,
    miller_z: torch.Tensor,
    nx: int,
    ny: int,
    nz_rfft: int,
) -> torch.Tensor:
    r"""``∂L/∂inv_cell_t`` given ``∂L/∂k_squared``.

    :math:`\partial k^2/\partial M[c, d] = (2\pi)^2 \cdot 2 \cdot k_c \cdot m_d`
    where :math:`k = M (m_x, m_y, m_z)`. Per-cell contributions are
    ``atomic_add``ed into a ``(3, 3)`` accumulator inside the Warp kernel.
    """
    del nx, ny, nz_rfft
    device = inv_cell_t.device
    dtype = inv_cell_t.dtype
    wp_scalar = get_wp_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    grad_M = torch.zeros((3, 3), dtype=dtype, device=device)
    with _scoped_warp_stream(device):
        pme_k_squared_backward_launch(
            _wp_from_torch(miller_x.contiguous(), dtype=wp_scalar),
            _wp_from_torch(miller_y.contiguous(), dtype=wp_scalar),
            _wp_from_torch(miller_z.contiguous(), dtype=wp_scalar),
            _wp_from_torch(inv_cell_t.contiguous(), dtype=wp_mat),
            _wp_from_torch(grad_k_squared.contiguous(), dtype=wp_scalar),
            _wp_from_torch(grad_M, dtype=wp_scalar),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    # Custom_op expects (1, 3, 3) to match the inv_cell_t input shape.
    return grad_M.unsqueeze(0)


def _pme_k_squared_forward_fake(
    inv_cell_t, _mx, _my, _mz, nx, ny, nz_rfft
):  # pragma: no cover
    """Fake: ``(nx, ny, nz_rfft)`` k_squared grid."""
    return torch.empty(
        (nx, ny, nz_rfft),
        dtype=inv_cell_t.dtype,
        device=inv_cell_t.device,
    )


def _pme_k_squared_double_backward(  # pragma: no cover
    v_grad_m: torch.Tensor,
    grad_k_squared: torch.Tensor,
    inv_cell_t: torch.Tensor,
    miller_x: torch.Tensor,
    miller_y: torch.Tensor,
    miller_z: torch.Tensor,
    nx: int,
    ny: int,
    nz_rfft: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Second-order backward for ``multipole_pme_k_squared`` (stress-loss).

    The first backward is :math:`G[c,d] = \sum_g \text{grad\_k}^2[g]\,J[g][c,d]`
    with per-grid Jacobian :math:`J[g][c,d] = 2(2\pi)^2 k_\text{red}[g][c]\,m[g][d]`
    and :math:`k_\text{red} = M m` (:math:`M=\text{inv\_cell\_t}`,
    :math:`m=(m_x,m_y,m_z)`). Given the cotangent :math:`v` on :math:`G`, returns
    the exact closed-form :math:`\partial(v\cdot G)/\partial\text{grad\_k}^2` and
    :math:`\partial(v\cdot G)/\partial M` (the constant ``2(2\pi)^2`` matches the
    kernel's ``eightpi_sq``).
    """
    const = 2.0 * (2.0 * math.pi) ** 2
    m_mat = inv_cell_t[0]
    v = v_grad_m[0]
    dtype = m_mat.dtype
    mx = miller_x.to(dtype).view(nx, 1, 1)
    my = miller_y.to(dtype).view(1, ny, 1)
    mz = miller_z.to(dtype).view(1, 1, nz_rfft)
    m_axes = (mx, my, mz)
    # k_red[c] = Σ_d M[c, d] m[d];  w[e] = Σ_d v[e, d] m[d]  (broadcast over grid)
    kred = [m_mat[c, 0] * mx + m_mat[c, 1] * my + m_mat[c, 2] * mz for c in range(3)]
    w = [v[e, 0] * mx + v[e, 1] * my + v[e, 2] * mz for e in range(3)]
    # ∂(v·G)/∂grad_k² = const · Σ_c k_red[c]·w[c]
    grad_grad_k_squared = (
        (const * (kred[0] * w[0] + kred[1] * w[1] + kred[2] * w[2]))
        .expand(nx, ny, nz_rfft)
        .contiguous()
    )
    # ∂(v·G)/∂M[e,f] = const · Σ_g grad_k²[g]·w[e][g]·m[f][g]
    grad_m = torch.stack(
        [
            torch.stack(
                [const * (grad_k_squared * w[e] * m_axes[f]).sum() for f in range(3)]
            )
            for e in range(3)
        ]
    )
    return grad_grad_k_squared, grad_m.unsqueeze(0).contiguous()


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_k_squared",
    forward=_pme_k_squared_forward,
    forward_fake=_pme_k_squared_forward_fake,
    backward=_pme_k_squared_backward,
    backward_return_arity=1,
    diff_input_positions=(0,),  # inv_cell_t only (miller indices constant)
    n_forward_inputs=7,
    double_backward=_pme_k_squared_double_backward,
    second_order_diff_positions=(0, 1),  # grad_k_squared, inv_cell_t
    n_backward_inputs=8,
    double_backward_return_arity=2,
)


def _pme_fractionalize_forward(
    positions: torch.Tensor,
    cell_inv_t: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Map Cartesian (positions, moments) to the cell-fractional frame.

    Returns ``(p, d_frac, Q_frac)`` with ``p = M·r``, ``d_frac = M·μ``,
    ``Q_frac = M·Q·Mᵀ`` (``M = cell_inv_t[batch_idx[i]]``). Feeding these to the
    spread kernel with an identity ``cell_inv_t`` reproduces the cell-coupled
    spread bit-for-bit, so the cell-stress 2nd-order composes through this
    multilinear map's autograd.
    """
    device = positions.device
    dtype = positions.dtype
    wp_scalar = get_wp_dtype(dtype)
    wp_vec = get_wp_vec_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    n = positions.shape[0]
    p = torch.empty((n, 3), dtype=dtype, device=device)
    df = torch.empty((n, 3), dtype=dtype, device=device)
    qf = torch.empty((n, 3, 3), dtype=dtype, device=device)
    with _scoped_warp_stream(device):
        pme_fractionalize_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
            _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
            _wp_from_torch(p, dtype=wp_vec),
            _wp_from_torch(df, dtype=wp_vec),
            _wp_from_torch(qf, dtype=wp_mat),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return p, df, qf


def _pme_fractionalize_backward(  # pragma: no cover
    grad_p: torch.Tensor,
    grad_df: torch.Tensor,
    grad_qf: torch.Tensor,
    positions: torch.Tensor,
    cell_inv_t: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adjoint of the fractionalize map → (grad_pos, grad_cell, grad_dip, grad_quad)."""
    device = positions.device
    dtype = positions.dtype
    wp_scalar = get_wp_dtype(dtype)
    wp_vec = get_wp_vec_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    n = positions.shape[0]
    b = cell_inv_t.shape[0]
    grad_pos = torch.empty((n, 3), dtype=dtype, device=device)
    grad_cell = torch.zeros((b, 3, 3), dtype=dtype, device=device)
    grad_dip = torch.empty((n, 3), dtype=dtype, device=device)
    grad_quad = torch.empty((n, 3, 3), dtype=dtype, device=device)
    with _scoped_warp_stream(device):
        pme_fractionalize_backward_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
            _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
            _wp_from_torch(grad_p.contiguous(), dtype=wp_vec),
            _wp_from_torch(grad_df.contiguous(), dtype=wp_vec),
            _wp_from_torch(grad_qf.contiguous(), dtype=wp_mat),
            _wp_from_torch(grad_pos, dtype=wp_vec),
            _wp_from_torch(grad_cell, dtype=wp_mat),
            _wp_from_torch(grad_dip, dtype=wp_vec),
            _wp_from_torch(grad_quad, dtype=wp_mat),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return grad_pos, grad_cell, grad_dip, grad_quad


def _pme_fractionalize_double_backward(  # pragma: no cover
    g_pos: torch.Tensor,
    g_cell: torch.Tensor,
    g_dip: torch.Tensor,
    g_quad: torch.Tensor,
    grad_p: torch.Tensor,
    grad_df: torch.Tensor,
    grad_qf: torch.Tensor,
    positions: torch.Tensor,
    cell_inv_t: torch.Tensor,
    dipoles: torch.Tensor,
    quadrupoles: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""Second-order backward (stress-loss). Cotangents ``(g_pos, g_cell, g_dip,
    g_quad)`` on the backward outputs → grads w.r.t. backward inputs at
    ``second_order_diff_positions = (0, 1, 2, 3, 4, 5, 6)`` =
    ``(grad_p, grad_df, grad_qf, positions, cell_inv_t, dipoles, quadrupoles)``.
    """
    device = positions.device
    dtype = positions.dtype
    wp_scalar = get_wp_dtype(dtype)
    wp_vec = get_wp_vec_dtype(dtype)
    wp_mat = get_wp_mat_dtype(dtype)
    n = positions.shape[0]
    b = cell_inv_t.shape[0]
    d_gp = torch.empty((n, 3), dtype=dtype, device=device)
    d_gdf = torch.empty((n, 3), dtype=dtype, device=device)
    d_gqf = torch.empty((n, 3, 3), dtype=dtype, device=device)
    d_pos = torch.empty((n, 3), dtype=dtype, device=device)
    d_cell = torch.zeros((b, 3, 3), dtype=dtype, device=device)
    d_dip = torch.empty((n, 3), dtype=dtype, device=device)
    d_quad = torch.empty((n, 3, 3), dtype=dtype, device=device)
    with _scoped_warp_stream(device):
        pme_fractionalize_double_backward_launch(
            _wp_from_torch(positions.contiguous(), dtype=wp_vec),
            _wp_from_torch(cell_inv_t.contiguous(), dtype=wp_mat),
            _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32),
            _wp_from_torch(dipoles.contiguous(), dtype=wp_vec),
            _wp_from_torch(quadrupoles.contiguous(), dtype=wp_mat),
            _wp_from_torch(grad_p.contiguous(), dtype=wp_vec),
            _wp_from_torch(grad_df.contiguous(), dtype=wp_vec),
            _wp_from_torch(grad_qf.contiguous(), dtype=wp_mat),
            _wp_from_torch(g_pos.contiguous(), dtype=wp_vec),
            _wp_from_torch(g_cell.contiguous(), dtype=wp_mat),
            _wp_from_torch(g_dip.contiguous(), dtype=wp_vec),
            _wp_from_torch(g_quad.contiguous(), dtype=wp_mat),
            _wp_from_torch(d_gp, dtype=wp_vec),
            _wp_from_torch(d_gdf, dtype=wp_vec),
            _wp_from_torch(d_gqf, dtype=wp_mat),
            _wp_from_torch(d_pos, dtype=wp_vec),
            _wp_from_torch(d_cell, dtype=wp_mat),
            _wp_from_torch(d_dip, dtype=wp_vec),
            _wp_from_torch(d_quad, dtype=wp_mat),
            wp_dtype=wp_scalar,
            device=str(wp.device_from_torch(device)),
        )
    return d_gp, d_gdf, d_gqf, d_pos, d_cell, d_dip, d_quad


def _pme_fractionalize_forward_fake(  # pragma: no cover
    positions, cell_inv_t, dipoles, quadrupoles, *_args
):
    """Fake: ``(p, d_frac, Q_frac)`` shaped like the moment inputs."""
    return (
        torch.empty_like(positions),
        torch.empty_like(dipoles),
        torch.empty_like(quadrupoles),
    )


def _pme_fractionalize_backward_fake(  # pragma: no cover
    grad_p, grad_df, grad_qf, positions, cell_inv_t, dipoles, quadrupoles, *_args
):
    """Fake: grads of (positions, cell_inv_t, dipoles, quadrupoles)."""
    return (
        torch.empty_like(positions),
        torch.empty_like(cell_inv_t),
        torch.empty_like(dipoles),
        torch.empty_like(quadrupoles),
    )


def _pme_fractionalize_double_backward_fake(  # pragma: no cover
    g_pos,
    g_cell,
    g_dip,
    g_quad,
    grad_p,
    grad_df,
    grad_qf,
    positions,
    cell_inv_t,
    dipoles,
    quadrupoles,
    *_args,
):
    """Fake: grads of (grad_p, grad_df, grad_qf, pos, cell_inv_t, dip, quad)."""
    return (
        torch.empty_like(grad_p),
        torch.empty_like(grad_df),
        torch.empty_like(grad_qf),
        torch.empty_like(positions),
        torch.empty_like(cell_inv_t),
        torch.empty_like(dipoles),
        torch.empty_like(quadrupoles),
    )


_FRACTIONALIZE_DBWD_SCHEMA = (
    "(Tensor g_pos, Tensor g_cell, Tensor g_dip, Tensor g_quad, "
    "Tensor grad_p, Tensor grad_df, Tensor grad_qf, Tensor positions, "
    "Tensor cell_inv_t, Tensor dipoles, Tensor quadrupoles, Tensor batch_idx) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)


register_warp_op_chain(
    name="nvalchemiops::multipole_pme_fractionalize",
    forward=_pme_fractionalize_forward,
    forward_fake=_pme_fractionalize_forward_fake,
    forward_return_arity=3,
    backward=_pme_fractionalize_backward,
    backward_fake=_pme_fractionalize_backward_fake,
    backward_return_arity=4,
    # positions=0, cell_inv_t=1, dipoles=2, quadrupoles=3 (batch_idx=4 const).
    diff_input_positions=(0, 1, 2, 3),
    n_forward_inputs=5,
    double_backward=_pme_fractionalize_double_backward,
    double_backward_fake=_pme_fractionalize_double_backward_fake,
    double_backward_schema=_FRACTIONALIZE_DBWD_SCHEMA,
    # Backward inputs: grad_p=0, grad_df=1, grad_qf=2, positions=3,
    # cell_inv_t=4, dipoles=5, quadrupoles=6 (batch_idx=7 const).
    second_order_diff_positions=(0, 1, 2, 3, 4, 5, 6),
    n_backward_inputs=8,
    double_backward_return_arity=7,
)


def _build_pme_k_grids(
    cell: torch.Tensor, mesh_dimensions: tuple[int, int, int], dtype: torch.dtype
) -> torch.Tensor:
    """Build the ``k_squared`` rfft grid for the multipole PME pipeline.

    Returns ``k_squared`` shaped ``(nx, ny, nz_rfft)``, autograd-aware
    through ``cell`` via the registered ``multipole_pme_k_squared``
    custom_op + torch's ``linalg.inv_ex``.
    """
    nx, ny, nz = mesh_dimensions
    device = cell.device

    miller_x = torch.fft.fftfreq(nx, d=1.0 / nx, device=device, dtype=dtype)
    miller_y = torch.fft.fftfreq(ny, d=1.0 / ny, device=device, dtype=dtype)
    miller_z = torch.fft.rfftfreq(nz, d=1.0 / nz, device=device, dtype=dtype)

    cell_2d = cell if cell.dim() == 2 else cell.squeeze(0)
    inv_cell_t = torch.linalg.inv_ex(cell_2d.T)[0].to(dtype).unsqueeze(0).contiguous()

    return torch.ops.nvalchemiops.multipole_pme_k_squared(
        inv_cell_t,
        miller_x,
        miller_y,
        miller_z,
        nx,
        ny,
        nz // 2 + 1,
    )


def multipole_pme_reciprocal_space(
    positions: torch.Tensor,
    multipole_moments: torch.Tensor,
    cell: torch.Tensor,
    *,
    sigma: float,
    alpha: float,
    mesh_dimensions: tuple[int, int, int],
    spline_order: int = 4,
    cell_inv_t: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    volume: torch.Tensor | None = None,
    moduli: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    k_squared: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Compute the multipole-PME reciprocal-space energy.

    Single-system or batched, dispatched by ``batch_idx``. Matches the
    convention of ``multipole_ewald_summation`` (single function with an
    optional ``batch_idx`` rather than a separate ``batch_*`` symbol).

    Composes the spread / Green's / gather wrappers with ``torch.fft``
    into the canonical PME pipeline:

    1. Spread ``ρ_grid = q · B + μ · ∇B``.
    2. Forward rfftn (``norm="backward"``).
    3. Apply ``ρ̃(k) · G̃(k) / |C(k)|²`` (G̃ has the GTO ``exp(-σ²k²)``
       factor baked in — the full σ², not σ²/2, because the pair-sum
       convolution multiplies the per-side ``exp(-σ²k²/2)`` Fourier
       transform).
    4. Inverse irfftn (``norm="forward"``) — same FFT-norm convention as
       monopole PME so the 2π-normalized Green's function absorbs the
       round-trip scaling.
    5. Gather φ and ∇φ at atom positions.
    6. Per-atom raw energy ``q_i · φ(r_i) + μ_i · ∇φ(r_i)``; sum and
       multiply by ``F/(4π) = FIELD_CONSTANT/(4π)`` to convert from
       natural Coulomb units (the Green's function uses ``2π/V``) into
       F-scaled units.
    7. Subtract the self + background corrections (already F-scaled — they
       match ``_multipole_ewald_self_energy_per_atom``).

    The dipole sign in step 6 is positive: Parseval on the multipole
    density (``ρ_grid = q·B + μ·∇_r_atom B``) yields
    ``E_recip = ⟨ρ_grid, φ_grid⟩ = Σ_i [q_i φ(r_i) + μ_i · ∇φ]``; the
    apparent ``-μ·∇B`` in the standard multipole-density formula becomes
    ``+μ·∇_r_atom B`` after the chain-rule sign flip
    (``∇_r_atom = -∇_grid``).

    Forward only — autograd flows through the wrapper-level ``torch.fft``
    ops plus the analytical backwards of the spread / gather pieces.

    Parameters
    ----------
    positions : torch.Tensor, shape ``(N, 3)``
        Cartesian atom positions.
    multipole_moments : torch.Tensor, shape ``(N, 1)``, ``(N, 4)``, or
        ``(N, 9)``. e3nn spherical-harmonic packing: ``[q]`` (l_max=0),
        ``[q, μ_y, μ_z, μ_x]`` (l_max=1), or the l_max=1 block plus the
        five traceless l=2 channels (l_max=2). The trailing dim selects
        the path; quadrupoles are expanded to the Cartesian symmetric
        ``(N, 3, 3)`` form internally. A pure-monopole ``(N, 1)`` call
        skips the field gather and the dipole self-energy correction term.
    cell : torch.Tensor, shape ``(3, 3)``, ``(1, 3, 3)``, or ``(B, 3, 3)``
        Unit-cell matrix (rows are lattice vectors); ``(B, 3, 3)`` batched.
    sigma : float
        GTO width.
    alpha : float
        Ewald splitting parameter (positive).
    mesh_dimensions : tuple[int, int, int]
        FFT mesh dimensions ``(Nx, Ny, Nz)`` — shared across batch.
    spline_order : int, default 4
        B-spline interpolation order ``p``.
    cell_inv_t : torch.Tensor, optional
        Pre-computed ``transpose(inv(cell))`` for MD steady-state. Shape
        ``(3, 3)`` / ``(1, 3, 3)`` single-system, ``(B, 3, 3)`` batched.
        When ``None`` it is derived from ``cell``.
    batch_idx : torch.Tensor, optional
        ``(N_total,)`` int32 — system index per atom. Triggers the batched
        path when provided.
    volume : torch.Tensor or None, optional
        Cell volume(s); ``()`` / ``(1,)`` single-system, ``(B,)`` batched.
        When ``None`` it is computed from ``cell`` via ``det``. Passing a
        tensor keeps cell autograd (stress) alive.
    moduli : tuple[torch.Tensor, torch.Tensor, torch.Tensor] or None, optional
        Pre-computed 1-D B-spline modulus LUTs ``(b_x, b_y, b_z)`` of shapes
        ``(Nx,)``, ``(Ny,)``, ``(Nz_rfft,)`` for MD steady-state reuse. When
        ``None`` they are computed from ``mesh_dimensions`` and
        ``spline_order``.
    k_squared : torch.Tensor or None, optional
        Pre-computed :math:`|k|^2` rfft grid of shape ``(Nx, Ny, Nz_rfft)``
        (single-system) or ``(B, Nx, Ny, Nz_rfft)`` (batched). Cacheable
        across MD steps when the cell is fixed; when ``None`` it is built
        from ``cell`` and ``mesh_dimensions``.

    Returns
    -------
    energy : torch.Tensor
        Per-atom :math:`(N,)` :math:`\text{float64}` (single) or
        :math:`(N_\text{total},)` (batched, flat across systems), in
        ``FIELD_CONSTANT`` units. Reciprocal energy minus self/background
        corrections. Call ``.sum()`` for the total or
        ``torch.zeros(B).scatter_add(0, batch_idx, E)`` for per-system totals;
        forces/stress/charge-grads flow from ``grad(E.sum(), ...)``.
    """
    # Split the packed e3nn ``multipole_moments`` into the Cartesian
    # channels the spread/correction kernels consume: charges ``(N,)``,
    # dipoles ``(N, 3)`` or None, quadrupoles ``(N, 3, 3)`` or None.
    charges, dipoles, quadrupoles, _l_max = split_multipole_moments(multipole_moments)

    if batch_idx is not None:
        return _batch_multipole_pme_reciprocal_space_impl(
            positions,
            charges,
            dipoles,
            cell,
            batch_idx,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            spline_order=spline_order,
            cell_inv_t=cell_inv_t,
            volumes=volume,
            moduli=moduli,
            k_squared=k_squared,
            quadrupoles=quadrupoles,
        )

    cell_inv_t_resolved = _resolve_cell_inv_t(cell, cell_inv_t)
    cell_2d = cell if cell.dim() == 2 else cell.squeeze(0)
    if volume is None:
        volume = torch.abs(torch.det(cell_2d.to(torch.float64)))
    nx, ny, nz = mesh_dimensions

    # Unified ``(positions, charges, dipoles, quadrupoles)`` spread
    # custom_op; selects the LMAX-specialized kernel at compile time
    # (0/1/2 based on inputs) with full autograd through all channels.
    if quadrupoles is not None:
        lmax = 2
    elif dipoles is not None:
        lmax = 1
    else:
        lmax = 0
    dipoles_in = (
        dipoles
        if dipoles is not None
        else torch.zeros(
            (positions.shape[0], 3),
            dtype=positions.dtype,
            device=positions.device,
        )
    )
    quadrupoles_in = (
        quadrupoles
        if quadrupoles is not None
        else torch.zeros(
            (positions.shape[0], 3, 3),
            dtype=positions.dtype,
            device=positions.device,
        )
    )
    # B-warp stress-loss path: factor ALL cell_inv_t coupling into the
    # ``fractionalize`` op (p=M·r, df=Mμ, Qf=MQMᵀ) so the spread runs
    # cell-free with an identity cell — bit-identical forward, but the
    # cell autograd (1st + 2nd order, i.e. stress + stress-loss) now flows
    # through fractionalize's analytic multilinear-map backward instead of
    # the spread's missing cell double-backward.
    batch_idx_single = torch.zeros(
        positions.shape[0], dtype=torch.int32, device=positions.device
    )
    p_frac, df_frac, qf_frac = torch.ops.nvalchemiops.multipole_pme_fractionalize(
        positions,
        cell_inv_t_resolved,
        dipoles_in,
        quadrupoles_in,
        batch_idx_single,
    )
    identity_cell = torch.eye(
        3, dtype=positions.dtype, device=positions.device
    ).unsqueeze(0)
    rho_grid = torch.ops.nvalchemiops.multipole_pme_spread_unified(
        p_frac,
        charges,
        df_frac,
        qf_frac,
        identity_cell,
        nx,
        ny,
        nz,
        spline_order,
        lmax,
    )

    # Force contiguous: under torch.compile, inductor otherwise propagates
    # the rfft output's "natural" (non-contiguous) layout into the convolve
    # custom op's planned output, while the real op returns a contiguous
    # mesh -> assert_size_stride mismatch at large meshes (e.g. 128^3).
    mesh_fft = torch.fft.rfftn(rho_grid, norm="backward").contiguous()

    # Build k-grid + precomputed B-spline modulus LUTs.
    dtype = positions.dtype
    k_squared = _resolve_pme_k_squared(cell_2d, mesh_dimensions, dtype, k_squared)
    alpha_t = torch.tensor([alpha], dtype=dtype, device=positions.device)
    sigma_t = torch.tensor([sigma], dtype=dtype, device=positions.device)
    # ``volume`` may be ``()`` or ``(1,)``; reshape to a definite ``(1,)`` to
    # match the ``alpha_t``/``sigma_t`` convention (unsqueeze turns ``(1,)``
    # into ``(1, 1)``). reshape keeps the cell-autograd graph for stress.
    volume_t = volume.to(dtype).reshape(1)
    moduli_x, moduli_y, moduli_z = _resolve_pme_moduli(
        mesh_dimensions, spline_order, dtype, positions.device, moduli
    )

    # Fused Green's + B-spline deconvolution + complex multiply in a
    # single Warp kernel pass.
    convolved = multipole_pme_convolve(
        mesh_fft,
        k_squared,
        moduli_x,
        moduli_y,
        moduli_z,
        alpha_t,
        sigma_t,
        volume_t,
    )

    phi_grid = torch.fft.irfftn(convolved, s=(nx, ny, nz), norm="forward").to(dtype)

    # Per-atom reciprocal energy via the spread-transpose gather
    # ``g = Sᵀφ``: ``E_i = q_i g_q_i + d_i·g_d_i + Q_i:g_Q_i`` (fractional
    # moments, since ``g`` lives in the spread's identity-cell fractional frame;
    # the ``1/2`` pair factor is baked into ``G̃ = 2π/Vk²`` and the quad ``g_Q``
    # channel). ``Σ_i E_i`` equals the mesh-mesh ``<ρ, φ>`` *bit-for-bit*
    # (``Sᵀ`` is the spread's own backward, not the discretization-mismatched
    # ``spline_gather``), but yields a genuine per-atom decomposition. The op is
    # twice-differentiable (force-loss + stress-loss compose through its
    # double-backward), so it routes the whole 2nd-order graph cleanly.
    #
    # Unit conversion: the Green's function ``2π/V · exp(...)/k²`` lands the
    # gathered energy in natural Coulomb units → multiply by ``F/(4π)`` to reach
    # F-scaled units; corrections are already F-scaled.
    coulomb_scale = FIELD_CONSTANT / (4.0 * math.pi)
    g_q, g_d, g_Q = torch.ops.nvalchemiops.multipole_pme_gather_via_spread_t(
        phi_grid, p_frac, identity_cell, nx, ny, nz, spline_order, lmax
    )
    e_per_atom = charges * g_q + (df_frac * g_d).sum(-1) + (qf_frac * g_Q).sum((-1, -2))
    # Per-atom reciprocal energy ``(N,)``; the caller owns the reduction
    # (PR #96 energy-autograd contract — ``grad(E.sum(), .)`` recovers the
    # collective forces/stress/charge-grads).
    e_recip_per_atom = coulomb_scale * e_per_atom

    corrections = multipole_pme_energy_corrections_per_atom(
        charges,
        dipoles,
        sigma,
        alpha,
        volume,
        quadrupoles=quadrupoles,
    )
    return e_recip_per_atom - corrections


# =============================================================================
# Top-level composite
# =============================================================================


def multipole_particle_mesh_ewald(
    positions: torch.Tensor,
    multipole_moments: torch.Tensor,
    cell: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    unit_shifts: torch.Tensor,
    *,
    sigma: float,
    alpha: float | None = None,
    mesh_dimensions: tuple[int, int, int] | None = None,
    spline_order: int = 4,
    cell_inv_t: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    accuracy: float = 1e-6,
    cost_ratio: float = 1.0,
    volume: torch.Tensor | None = None,
    moduli: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    k_squared: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Total multipole PME energy (real + reciprocal − self − background).

    Single-system or batched, dispatched by ``batch_idx``. Mirrors the
    API of ``multipole_ewald_summation`` (single function with optional
    ``batch_idx``) but routes the O(N · N_k) reciprocal half through
    PME, dropping the asymptotic cost to O(M log M + N · p³) at the
    cost of a small spline-truncation residual.

    The composition matches the direct-k ``multipole_ewald_summation``:

    1. Real-space pair sum via
       :func:`~nvalchemiops.torch.interactions.electrostatics.multipole_ewald.multipole_real_space_energy`
       (single, or batched through its ``batch_idx=`` path). Per-atom output
       is multiplied by ``coulomb_scale = F/(4π)``.
    2. Reciprocal piece via :func:`multipole_pme_reciprocal_space` —
       returns ``E_recip - E_self - E_bg`` already in F units.
    3. ``E_total = E_real + E_recip - E_self - E_bg``.

    Direct-k parity holds at the spline-truncation floor (``rtol ≈ 1e-4``
    at ``mesh = 60³``, ``L = 10``).

    Parameters
    ----------
    positions : torch.Tensor, shape ``(N, 3)`` or ``(N_total, 3)``
        Cartesian atom positions; ``N_total`` in batched mode.
    multipole_moments : torch.Tensor, shape ``(N, 1)``, ``(N, 4)``, or
        ``(N, 9)`` (or the ``N_total`` analog in batched mode). e3nn
        spherical-harmonic packing: ``[q]`` (l_max=0),
        ``[q, μ_y, μ_z, μ_x]`` (l_max=1), or the l_max=1 block plus the
        five traceless l=2 channels (l_max=2). The trailing dim selects
        the l_max path.
    cell : torch.Tensor, shape ``(3, 3)``, ``(1, 3, 3)``, or ``(B, 3, 3)``
        Unit-cell matrix (rows are lattice vectors); ``(B, 3, 3)`` batched.
    idx_j, neighbor_ptr, unit_shifts : torch.Tensor
        Real-space CSR neighbor list. In batched mode the list is flat
        across systems; each atom's neighbors must live in the same
        system (caller's responsibility).
    sigma : float
        GTO basis width — uniform across batch.
    alpha : float, optional
        Ewald splitting parameter (positive). When ``None`` (default) it
        is auto-estimated from ``sigma`` and the system geometry via
        :func:`estimate_multipole_pme_parameters` at the requested
        ``accuracy``. The caller is still responsible for having built
        the neighbor list with the matching real-space cutoff (also
        available via the same estimator).
    mesh_dimensions : tuple[int, int, int], optional
        FFT mesh dimensions — shared across batch. Auto-estimated from
        the same Kolafa-Perram balance when ``None``. Override this if
        you need to lock the mesh resolution (e.g. for kernel reuse).
    spline_order : int, default 4
        B-spline interpolation order ``p`` (used for both spread and gather).
    cell_inv_t : torch.Tensor, optional
        Pre-computed ``transpose(inv(cell))`` — shape ``(3, 3)`` /
        ``(1, 3, 3)`` for single-system, ``(B, 3, 3)`` for batched.
    batch_idx : torch.Tensor, optional
        ``(N_total,)`` int32. Triggers the batched path when provided.
    accuracy : float, default 1e-6
        Target relative-energy accuracy used by the auto-estimator when
        ``alpha`` and/or ``mesh_dimensions`` are ``None``. Same semantics
        as the monopole :func:`particle_mesh_ewald`.
    cost_ratio : float, default 1.0
        Hardware-empirical per-real-space-pair vs per-reciprocal cost
        ratio passed through to
        :func:`estimate_multipole_pme_parameters`. ``1.0`` reproduces
        canonical Kolafa-Perram. Note that PME's true reciprocal cost
        (FFT + spread/gather) doesn't have the same shape as Ewald's
        per-k cost — the Ewald optimum may not transfer directly. See
        :func:`estimate_multipole_pme_parameters` for details. Ignored
        if ``alpha`` and ``mesh_dimensions`` are supplied.
    volume : torch.Tensor or None, optional
        Cell volume(s) forwarded to the reciprocal half; ``()`` / ``(1,)``
        single-system, ``(B,)`` batched. When ``None`` it is computed from
        ``cell``. Passing a tensor keeps cell autograd (stress) alive.
    moduli : tuple[torch.Tensor, torch.Tensor, torch.Tensor] or None, optional
        Pre-computed 1-D B-spline modulus LUTs ``(b_x, b_y, b_z)`` of shapes
        ``(Nx,)``, ``(Ny,)``, ``(Nz_rfft,)`` for MD steady-state reuse;
        forwarded to :func:`multipole_pme_reciprocal_space`. When ``None``
        they are computed from ``mesh_dimensions`` and ``spline_order``.
    k_squared : torch.Tensor or None, optional
        Pre-computed :math:`|k|^2` rfft grid of shape ``(Nx, Ny, Nz_rfft)``
        (single-system) or ``(B, Nx, Ny, Nz_rfft)`` (batched), cacheable
        across MD steps when the cell is fixed. When ``None`` it is built
        from ``cell`` and ``mesh_dimensions``.

    Returns
    -------
    energy : torch.Tensor, ``float64``
        Per-atom :math:`(N,)` (single) or :math:`(N_\text{total},)` (batched,
        flat across systems). Call ``.sum()`` for the total Coulomb energy or
        ``torch.zeros(B).scatter_add(0, batch_idx, E)`` for per-system totals;
        forces/stress/charge-grads flow from ``grad(E.sum(), ...)``.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    if alpha is None or mesh_dimensions is None:
        from nvalchemiops.torch.interactions.electrostatics.parameters import (
            estimate_multipole_pme_parameters,
        )

        params = estimate_multipole_pme_parameters(
            positions,
            cell,
            sigma=sigma,
            batch_idx=batch_idx,
            accuracy=accuracy,
            cost_ratio=cost_ratio,
        )
        if alpha is None:
            alpha_tensor = params.alpha
            if alpha_tensor.numel() == 1:
                alpha = float(alpha_tensor.item())
            else:
                alpha_min = float(alpha_tensor.min().item())
                alpha_max = float(alpha_tensor.max().item())
                if alpha_max - alpha_min > 1e-12 * max(alpha_max, 1.0):
                    raise ValueError(
                        "Auto-estimated alpha differs across batch systems "
                        f"({alpha_min} vs {alpha_max}). The current PME "
                        "kernel takes a single scalar alpha."
                    )
                alpha = alpha_min
        if mesh_dimensions is None:
            mesh_dimensions = params.mesh_dimensions

    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    # Delayed import to avoid circular module graph.
    from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
        multipole_real_space_energy_with_stress,
    )

    device = positions.device
    input_dtype = positions.dtype
    coulomb_scale = FIELD_CONSTANT / (4.0 * math.pi)

    # The packed ``multipole_moments`` trailing dim (1 / 4 / 9) selects the
    # l_max path; ``infer_l_max`` validates it. Both halves take the packed
    # tensor directly: the real-space entry and the PME reciprocal (which split
    # internally and fold the l=2 self-energy into the returned per-atom
    # ``E_recip − E_self − E_bg``).
    infer_l_max(multipole_moments)

    if batch_idx is not None:
        if cell.dim() != 3 or cell.shape[-2:] != (3, 3):
            raise ValueError(
                f"batched cell must be shape (B, 3, 3); got {tuple(cell.shape)}"
            )
        if batch_idx.shape != (positions.shape[0],):
            raise ValueError(
                "batch_idx must have shape (N_total,) matching positions[0]; "
                f"got {tuple(batch_idx.shape)}"
            )
        B = cell.shape[0]
        sigmas = torch.full((B,), sigma, dtype=input_dtype, device=device)
        alphas = torch.full((B,), alpha, dtype=input_dtype, device=device)

        # Per-atom real-space ``(N_total,)``, cell-grad aware for all l_max.
        e_real = coulomb_scale * multipole_real_space_energy_with_stress(
            positions,
            multipole_moments,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            sigmas,
            alphas,
            batch_idx=batch_idx,
        )

        e_recip_minus_corr = multipole_pme_reciprocal_space(
            positions,
            multipole_moments,
            cell,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            spline_order=spline_order,
            cell_inv_t=cell_inv_t,
            batch_idx=batch_idx,
            volume=volume,
            moduli=moduli,
            k_squared=k_squared,
        )

        return (e_real + e_recip_minus_corr).to(torch.float64)

    sigma_t = torch.tensor([sigma], dtype=input_dtype, device=device)
    alpha_t = torch.tensor([alpha], dtype=input_dtype, device=device)

    # Per-atom real-space ``(N,)``, cell-grad aware for all l_max.
    e_real = coulomb_scale * multipole_real_space_energy_with_stress(
        positions,
        multipole_moments,
        cell,
        idx_j,
        neighbor_ptr,
        unit_shifts,
        sigma_t,
        alpha_t,
    )

    # Reciprocal half. Returns per-atom E_recip - E_self - E_bg.
    e_recip_minus_corr = multipole_pme_reciprocal_space(
        positions,
        multipole_moments,
        cell,
        sigma=sigma,
        alpha=alpha,
        mesh_dimensions=mesh_dimensions,
        spline_order=spline_order,
        cell_inv_t=cell_inv_t,
        volume=volume,
        moduli=moduli,
        k_squared=k_squared,
    )

    return (e_real + e_recip_minus_corr).to(torch.float64)


# =============================================================================
# Batched reciprocal-space + top-level helpers
# =============================================================================


def _build_batch_pme_k_grids(
    cells: torch.Tensor, mesh_dimensions: tuple[int, int, int], dtype: torch.dtype
) -> torch.Tensor:
    """Per-system ``k_squared`` rfft grid for the batched multipole-PME pipeline.

    ``cells`` is ``(B, 3, 3)``; returns ``k_squared`` shaped
    ``(B, nx, ny, nz_rfft)``, autograd-aware through ``cells`` (per-system
    stress). Reuses the cell-differentiable single-system
    ``multipole_pme_k_squared`` custom_op per system and stacks — B is the
    (small) number of systems, so the Python loop is cheap relative to the
    spread/FFT cost and gives exact batched cell-grad without a separate
    batched k_squared backward kernel.
    """
    nx, ny, nz = mesh_dimensions
    device = cells.device
    B = cells.shape[0]

    miller_x = torch.fft.fftfreq(nx, d=1.0 / nx, device=device, dtype=dtype)
    miller_y = torch.fft.fftfreq(ny, d=1.0 / ny, device=device, dtype=dtype)
    miller_z = torch.fft.rfftfreq(nz, d=1.0 / nz, device=device, dtype=dtype)
    nz_rfft = nz // 2 + 1

    per_system = []
    for b in range(B):
        inv_cell_t = (
            torch.linalg.inv_ex(cells[b].transpose(-1, -2))[0]
            .to(dtype)
            .unsqueeze(0)
            .contiguous()
        )
        per_system.append(
            torch.ops.nvalchemiops.multipole_pme_k_squared(
                inv_cell_t,
                miller_x,
                miller_y,
                miller_z,
                nx,
                ny,
                nz_rfft,
            )
        )
    return torch.stack(per_system, dim=0)  # (B, nx, ny, nz_rfft)


def _batch_multipole_pme_reciprocal_space_impl(
    positions: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor | None,
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    sigma: float,
    alpha: float,
    mesh_dimensions: tuple[int, int, int],
    spline_order: int = 4,
    cell_inv_t: torch.Tensor | None = None,
    volumes: torch.Tensor | None = None,
    moduli: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    k_squared: torch.Tensor | None = None,
    quadrupoles: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Batched body of :func:`multipole_pme_reciprocal_space`.

    Composes the batched spread / Green's / gather primitives plus
    corrections (already batched) with ``torch.fft.rfftn`` along the
    spatial dims:

    1. Batched spread → ``ρ_grid: (B, nx, ny, nz)``.
    2. ``torch.fft.rfftn(rho_grid, dim=(1, 2, 3), norm="backward")``.
    3. Batched Green's function → per-system ``G̃_b(k)``.
    4. ``mesh_fft / struct_sq * green`` — per-system multiply.
    5. ``torch.fft.irfftn(..., dim=(1, 2, 3), norm="forward")``.
    6. Batched gather φ + ∇φ.
    7. Per-atom ``q_i φ + μ_i · ∇φ``, scaled by ``F/(4π)``,
       scatter-summed by ``batch_idx`` → per-system raw recip energy.
    8. Subtract per-system self/background corrections.

    Sigma and alpha are scalar floats — same convention as
    ``multipole_ewald_summation``'s batched path.
    """
    if cell.dim() != 3 or cell.shape[-2:] != (3, 3):
        raise ValueError(
            f"batched cell must be shape (B, 3, 3); got {tuple(cell.shape)}"
        )
    cell_inv_t_resolved = _resolve_batch_cell_inv_t(cell, cell_inv_t)
    B = cell.shape[0]
    nx, ny, nz = mesh_dimensions
    if volumes is None:
        volumes = torch.abs(torch.det(cell.to(torch.float64)))  # (B,)

    # Step 1: batched spread via the unified batched custom_op (handles l=0/1/2
    # by the ``lmax`` codegen gate). The custom_op requires concrete dipoles /
    # quadrupoles tensors (no None), so materialize zeros for lower orders.
    # Routing through the unified op gives batched create_graph (force-loss)
    # at every l_max via its double-back.
    # l_max from which channels are actually supplied (mirror the single path):
    # quadrupoles -> 2, dipoles -> 1, charges-only -> 0. Materialize zeros for the
    # absent higher channels (the unified op needs concrete tensors).
    if quadrupoles is not None:
        lmax = 2
    elif dipoles is not None:
        lmax = 1
    else:
        lmax = 0
    if dipoles is None:
        dipoles = torch.zeros(
            (positions.shape[0], 3), dtype=positions.dtype, device=positions.device
        )
    if quadrupoles is None:
        quadrupoles_in = torch.zeros(
            (positions.shape[0], 3, 3), dtype=positions.dtype, device=positions.device
        )
    else:
        quadrupoles_in = quadrupoles
    # B-warp stress-loss path (mirror the single composite): factor all cell
    # coupling into fractionalize (p=M·r, df=Mμ, Qf=MQMᵀ) so the batched spread
    # runs cell-free with a per-system identity cell — bit-identical forward,
    # cell autograd (stress + stress-loss) flows through fractionalize's
    # analytic backward + the convolve k²/volume double-back.
    p_frac, df_frac, qf_frac = torch.ops.nvalchemiops.multipole_pme_fractionalize(
        positions,
        cell_inv_t_resolved,
        dipoles,
        quadrupoles_in,
        batch_idx,
    )
    identity_cell = (
        torch.eye(3, dtype=positions.dtype, device=positions.device)
        .unsqueeze(0)
        .expand(B, 3, 3)
    )
    rho_grid = torch.ops.nvalchemiops.multipole_pme_spread_unified_batch(
        p_frac,
        charges,
        df_frac,
        qf_frac,
        batch_idx,
        identity_cell,
        nx,
        ny,
        nz,
        B,
        spline_order,
        lmax,
    )  # (B, nx, ny, nz)

    # Step 2: forward FFT along spatial dims. ``.contiguous()`` so torch.compile
    # plans the convolve custom op's output contiguously (matching the real op);
    # see the single-system note above.
    mesh_fft = torch.fft.rfftn(rho_grid, dim=(1, 2, 3), norm="backward").contiguous()

    # Step 3: per-system k-grid + shared B-spline modulus LUTs.
    dtype = positions.dtype
    k_squared = _resolve_batch_pme_k_squared(cell, mesh_dimensions, dtype, k_squared)
    alpha_t = torch.full((B,), alpha, dtype=dtype, device=positions.device)
    sigma_t = torch.full((B,), sigma, dtype=dtype, device=positions.device)
    volumes_t = volumes.to(dtype)
    moduli_x, moduli_y, moduli_z = _resolve_pme_moduli(
        mesh_dimensions, spline_order, dtype, positions.device, moduli
    )

    # Step 4: fused batched convolve (Green's + deconvolution + multiply).
    convolved = multipole_pme_convolve(
        mesh_fft,
        k_squared,
        moduli_x,
        moduli_y,
        moduli_z,
        alpha_t,
        sigma_t,
        volumes_t,
    )

    # Step 5: inverse FFT.
    phi_grid = torch.fft.irfftn(
        convolved, s=(nx, ny, nz), dim=(1, 2, 3), norm="forward"
    ).to(dtype)

    # Steps 6/7: per-atom reciprocal energy via the batched spread-transpose
    # gather ``g = Sᵀφ`` — ``E_i = q_i g_q_i + d_i·g_d_i + Q_i:g_Q_i`` in the
    # identity-cell fractional frame (the convolve's ``G̃ = 2π/Vk²`` bakes in the
    # pair 1/2 factor; ``g_Q`` carries its own 1/2). ``Σ_i E_i`` per system is
    # bit-identical to the mesh-mesh ``<ρ, φ>`` but yields a per-atom
    # decomposition that is twice-differentiable through the op's
    # double-backward (force-loss + stress-loss).
    coulomb_scale = FIELD_CONSTANT / (4.0 * math.pi)
    g_q, g_d, g_Q = torch.ops.nvalchemiops.multipole_pme_gather_via_spread_t_batch(
        phi_grid, p_frac, batch_idx, identity_cell, nx, ny, nz, B, spline_order, lmax
    )
    e_per_atom = charges * g_q + (df_frac * g_d).sum(-1) + (qf_frac * g_Q).sum((-1, -2))
    # Per-atom reciprocal energy ``(N_total,)``; the caller owns the reduction
    # (PR #96 energy-autograd contract).
    e_recip_per_atom = coulomb_scale * e_per_atom

    # Step 8: subtract per-atom corrections (incl. l=2 self-energy + background).
    corrections = multipole_pme_energy_corrections_per_atom(
        charges,
        dipoles,
        sigma,
        alpha,
        volumes,
        batch_idx=batch_idx,
        quadrupoles=quadrupoles,
        n_systems=B,
    )
    return e_recip_per_atom - corrections
