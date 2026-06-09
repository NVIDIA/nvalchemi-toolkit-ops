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
PyTorch Bindings for Particle Mesh Ewald (PME)
==============================================

This module provides PyTorch bindings for the Particle Mesh Ewald algorithm,
wrapping Warp kernels with PyTorch custom operators for autograd support.

The PME module has unique challenges - it requires FFT operations that Warp
doesn't support. The Warp layer provides building blocks (Green's function,
energy corrections), but the complete PME workflow must remain in framework
bindings due to FFT dependency on PyTorch.

This module provides a unified GPU-accelerated API for Particle Mesh Ewald that
handles both single-system and batched calculations transparently. PME achieves
:math:`O(N \\log N)` scaling compared to :math:`O(N^2)` for direct summation, making it efficient
for large systems.

The output dtype convention follows ewald.py: public energy, force, and virial
outputs preserve input precision while selected internal reductions use float64
for numerical stability.

API STRUCTURE
=============

Primary APIs (public, with autograd support):
    particle_mesh_ewald(): Complete PME calculation (real + reciprocal)
    pme_reciprocal_space(): Reciprocal-space FFT-based component only

Helper APIs:
    pme_energy_corrections(): Self-energy and background corrections

The batch_idx parameter determines kernel dispatch:
    batch_idx=None → Single-system kernels
    batch_idx provided → Batch kernels (multiple independent systems)

``particle_mesh_ewald`` treats energy autograd as the differentiable training
contract; its direct-output flags warn and are deprecated. The
``pme_reciprocal_space`` component intentionally retains direct forces as
no-autograd MD/inference escape hatches. Component charge-gradient, virial,
and hybrid direct outputs are legacy training-style outputs and warn.

MATHEMATICAL FORMULATION
========================

PME uses B-spline interpolation to assign charges to a mesh, computes the
convolution with the Coulomb kernel efficiently via FFT, then interpolates
back to get energies and forces.

.. math::

    E_{\\text{total}} = E_{\\text{real}} + E_{\\text{reciprocal}} - E_{\\text{self}} - E_{\\text{background}}

Reciprocal-Space Steps:

1. Charge assignment:

.. math::

    Q(x) = \\sum_i q_i M_p(x - r_i)

where :math:`M_p` is the pth-order cardinal B-spline

2. FFT:

.. math::

    \\tilde{Q}(k) = \\text{FFT}[Q(x)]

3. Convolution in k-space:

.. math::

    \\tilde{\\Phi}(k) = \\frac{G(k)}{C^2(k)} \\tilde{Q}(k)

where :math:`G(k) = \\frac{2\\pi}{V} \\frac{\\exp(-k^2/(4\\alpha^2))}{k^2}` and :math:`C(k) = [\\text{sinc products}]^p` is the B-spline correction

4. Inverse FFT for potential and field:

.. math::

    \\begin{aligned}
    \\Phi(x) &= \\text{IFFT}[\\tilde{\\Phi}(k)] \\\\
    E(x) &= \\text{IFFT}[-ik \\tilde{\\Phi}(k)]
    \\end{aligned}

5. Energy and force interpolation:

.. math::

    \\begin{aligned}
    E_i &= q_i \\cdot \\text{interpolate}(\\Phi, r_i) \\\\
    F_i &= q_i \\cdot \\text{interpolate}(E, r_i)
    \\end{aligned}

Corrections:

.. math::

    \\begin{aligned}
    E_{\\text{self}} &= \\sum_i \\frac{\\alpha}{\\sqrt{\\pi}} q_i^2 \\\\
    E_{\\text{background}} &= \\sum_i \\frac{\\pi}{2\\alpha^2 V} q_i Q_{\\text{total}}
    \\end{aligned}

Examples
--------

Automatic parameter estimation::

    >>> from nvalchemiops.torch.interactions.electrostatics import particle_mesh_ewald
    >>> energies = particle_mesh_ewald(
    ...     positions, charges, cell,
    ...     neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    ...     accuracy=1e-6,  # alpha and mesh estimated automatically
    ... )
    >>> forces = -torch.autograd.grad(energies.sum(), positions, create_graph=True)[0]

Explicit parameters::

    >>> energies = particle_mesh_ewald(
    ...     positions, charges, cell,
    ...     alpha=0.3,
    ...     mesh_dimensions=(32, 32, 32),
    ...     spline_order=4,
    ...     neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    ... )

Batched systems::

    >>> energies = particle_mesh_ewald(
    ...     positions, charges, cells,  # cells shape (B, 3, 3)
    ...     alpha=torch.tensor([0.3, 0.35]),
    ...     batch_idx=batch_idx,
    ...     mesh_dimensions=(32, 32, 32),
    ...     neighbor_list=nl, neighbor_ptr=nl_ptr, neighbor_shifts=shifts,
    ... )

Reciprocal-space only (no real-space)::

    >>> energies = pme_reciprocal_space(
    ...     positions, charges, cell,
    ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
    ... )
References
----------

- Essmann et al. (1995). J. Chem. Phys. 103, 8577 (SPME paper)
- Darden et al. (1993). J. Chem. Phys. 98, 10089 (Original PME)
- torchpme: https://github.com/lab-cosmo/torch-pme (Reference implementation)
"""

import math
import warnings

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics.pme_kernels import (
    batch_pme_energy_corrections_with_charge_grad as _batch_pme_energy_corrections_with_charge_grad_warp,
)
from nvalchemiops.interactions.electrostatics.pme_kernels import (
    pme_energy_corrections_with_charge_grad as _pme_energy_corrections_with_charge_grad_warp,
)
from nvalchemiops.interactions.electrostatics.pme_kernels import (
    pme_virial_bg_correction as _pme_virial_bg_correction_warp,
)
from nvalchemiops.interactions.electrostatics.pme_kernels import (
    pme_virial_bg_correction_backward as _pme_virial_bg_correction_backward_warp,
)
from nvalchemiops.torch._warp_op_helpers import (
    attach_simple_backward,
    register_warp_op_chain,
)
from nvalchemiops.torch.interactions.electrostatics._registration import (
    ensure_electrostatics_ops_registered,
)
from nvalchemiops.torch.interactions.electrostatics._util import (
    _build_electrostatic_result,
    _combine_electrostatic_outputs,
    _compiled_direct_output_deprecation_signal,
    _component_direct_output_deprecation_msg,
    _detach_setup_tensor,
    _direct_output_deprecation_msg,
    _InjectCachedEvalGrad,
    _InjectCachedEvalGradWithFallback,
    _InjectChargeGrad,
    _is_uniform_cotangent,
    _unpack_electrostatic_outputs,
)
from nvalchemiops.torch.interactions.electrostatics.ewald import (
    ewald_real_space,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_pme,
)
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    estimate_pme_mesh_dimensions,
    estimate_pme_parameters,
    mesh_spacing_to_dimensions,
)
from nvalchemiops.torch.interactions.electrostatics.slab import (
    _prepare_pbc_for_slab,
)
from nvalchemiops.torch.interactions.electrostatics.slab import (
    compute_slab_correction as _compute_slab_correction,
)
from nvalchemiops.torch.spline import (
    spline_gather,
    spline_gather_with_force,
    spline_spread,
)
from nvalchemiops.torch.types import get_wp_dtype

_PME_OPS_REGISTERED = False

# Mathematical constants
PI = math.pi
TWOPI = 2.0 * PI
FOURPI = 4.0 * PI


###########################################################################################
########################### Helper Functions ##############################################
###########################################################################################


def _prepare_alpha(
    alpha: float | torch.Tensor,
    num_systems: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Convert alpha to a per-system tensor.

    Parameters
    ----------
    alpha : float or torch.Tensor
        Ewald splitting parameter. Can be:
        - A scalar float (broadcast to all systems)
        - A 0-d tensor (broadcast to all systems)
        - A 1-d tensor of shape (num_systems,) for per-system values
    num_systems : int
        Number of systems in the batch.
    dtype : torch.dtype
        Target dtype (typically float64).
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor, shape (num_systems,)
        Per-system alpha values.
    """
    if isinstance(alpha, (int, float)):
        return torch.full((num_systems,), float(alpha), dtype=dtype, device=device)
    elif isinstance(alpha, torch.Tensor):
        if alpha.dim() == 0:
            return alpha.expand(num_systems).to(dtype=dtype, device=device)
        elif alpha.shape[0] != num_systems:
            raise ValueError(
                f"alpha has {alpha.shape[0]} values but there are {num_systems} systems"
            )
        return alpha.to(dtype=dtype, device=device)
    else:
        raise TypeError(f"alpha must be float or torch.Tensor, got {type(alpha)}")


def _prepare_cell(cell: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Ensure cell is 3D (B, 3, 3) and return number of systems.

    Parameters
    ----------
    cell : torch.Tensor
        Unit cell matrix. Shape (3, 3) for single system or (B, 3, 3) for batch.

    Returns
    -------
    cell : torch.Tensor, shape (B, 3, 3)
        Cell with batch dimension.
    num_systems : int
        Number of systems (B).
    """
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    return cell, cell.shape[0]


def _materialize_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Force a fresh complex tensor for compiled FFT consumers."""
    if not tensor.is_complex():
        return tensor
    return torch.complex(tensor.real, tensor.imag)


def _vec2_wp_dtype_for(real_dtype: torch.dtype):
    """Map torch real dtype to the corresponding Warp vec2 type."""
    import warp as _wp

    return _wp.vec2f if real_dtype == torch.float32 else _wp.vec2d


def _pme_scoped_warp_stream(device: torch.device):
    """Bind Warp's current stream to PyTorch's current CUDA stream.

    Required for ``torch.cuda.graph`` capture so Warp kernel launches end
    up on the stream being captured rather than Warp's default stream.
    """
    if device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    torch_stream = torch.cuda.current_stream(device)
    return wp.ScopedStream(wp.stream_from_torch(torch_stream))


def _wp_from_torch(tensor: torch.Tensor, dtype):
    """``wp.from_torch`` with shadow-gradient allocation disabled.

    Default ``wp.from_torch`` inherits ``requires_grad`` from the source
    tensor and allocates a Warp-side gradient buffer when True. That
    allocation breaks ``torch.cuda.graph`` capture
    (``cudaErrorStreamCaptureInvalidated``). Our autograd.Functions own
    the backward, so the shadow grad is unused — force ``requires_grad=False``.
    """
    return wp.from_torch(tensor, dtype=dtype, requires_grad=False)


def compute_bspline_moduli_1d(
    miller_indices: torch.Tensor,
    mesh_N: int,
    spline_order: int,
) -> torch.Tensor:
    """Precompute the 1D B-spline modulus LUT for one PME mesh axis.

    Returns ``b[i] = sinc(m_i / N)^spline_order`` for each miller index
    ``m_i`` (with ``sinc(x) = sin(pi*x)/(pi*x)``, ``sinc(0) = 1``). The
    three-axis product ``b_x[i] * b_y[j] * b_z[k]`` is the B-spline
    structure factor consumed by the factory-backed convolve kernel after a
    1e-10 clamp + square. Precomputing the LUT lets the convolve kernel
    replace three sinc transcendentals + an order-dependent power loop
    per (i, j, k) thread with three reads + two multiplies.
    """
    # sinc(x) for x in [-0.5, 0.5] is bounded in [2/pi, 1], so s^spline_order
    # (for orders 2-6) stays well within fp32 range. Stay in the input dtype
    # to avoid an fp32 -> fp64 -> fp32 round-trip every call.
    arg = miller_indices / float(mesh_N)
    s = torch.special.sinc(arg)
    return s**spline_order


def pme_green_structure_factor(
    k_squared: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    alpha: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the PME Green's function and B-spline structure-factor correction.

    Compatibility entry point retained for the 0.3.1 public API. Returns the
    volume-normalized Coulomb Green's function and the squared B-spline structure
    factor used for PME deconvolution:

    .. math::

        G(k) = \frac{2\pi}{V} \frac{\exp(-k^2/(4\alpha^2))}{k^2}, \qquad
        C^2(k) = \left[\operatorname{sinc}(m_x/N_x)\,\operatorname{sinc}(m_y/N_y)\,
        \operatorname{sinc}(m_z/N_z)\right]^{2p}

    with :math:`G(0)=0` (tin-foil boundary) and ``p = spline_order``.

    Parameters
    ----------
    k_squared : torch.Tensor
        ``|k|^2`` at each rfft grid point: ``(Nx, Ny, Nz_rfft)`` (single) or
        ``(B, Nx, Ny, Nz_rfft)`` (batch).
    mesh_dimensions : tuple[int, int, int]
        Full mesh ``(Nx, Ny, Nz)`` before rfft.
    alpha : torch.Tensor
        Ewald splitting parameter, shape ``(1,)`` or ``(B,)``.
    cell : torch.Tensor
        Unit cell(s): ``(3, 3)``, ``(1, 3, 3)``, or ``(B, 3, 3)``.
    spline_order : int, default=4
        B-spline interpolation order.
    batch_idx : torch.Tensor | None, default=None
        When provided, ``k_squared``/``alpha``/``cell`` are treated as batched.

    Returns
    -------
    green_function : torch.Tensor
        Volume-normalized :math:`G(k)`, same shape as ``k_squared``.
    structure_factor_sq : torch.Tensor
        :math:`C^2(k)`, shape ``(Nx, Ny, Nz_rfft)`` (mesh-only, shared across batch).
    """
    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
    device = k_squared.device
    input_dtype = k_squared.dtype

    cell3 = cell if cell.dim() == 3 else cell.unsqueeze(0)
    volume = torch.abs(torch.linalg.det(cell3)).to(input_dtype)
    alpha_flat = alpha.reshape(-1).to(input_dtype)
    ksq_safe = torch.where(k_squared < 1e-10, torch.ones_like(k_squared), k_squared)
    if batch_idx is None:
        inv_4a2 = 1.0 / (4.0 * alpha_flat[0] * alpha_flat[0])
        green = (2.0 * torch.pi / volume[0]) * torch.exp(-ksq_safe * inv_4a2) / ksq_safe
    else:
        b = k_squared.shape[0]
        inv_4a2 = (1.0 / (4.0 * alpha_flat * alpha_flat)).view(b, 1, 1, 1)
        vol_b = volume.view(b, 1, 1, 1)
        green = (2.0 * torch.pi / vol_b) * torch.exp(-ksq_safe * inv_4a2) / ksq_safe
    green = torch.where(k_squared < 1e-10, torch.zeros_like(green), green)

    miller_x = torch.fft.fftfreq(
        mesh_nx, d=1.0 / mesh_nx, device=device, dtype=input_dtype
    )
    miller_y = torch.fft.fftfreq(
        mesh_ny, d=1.0 / mesh_ny, device=device, dtype=input_dtype
    )
    miller_z = torch.fft.rfftfreq(
        mesh_nz, d=1.0 / mesh_nz, device=device, dtype=input_dtype
    )
    c = (
        compute_bspline_moduli_1d(miller_x, mesh_nx, spline_order)[:, None, None]
        * compute_bspline_moduli_1d(miller_y, mesh_ny, spline_order)[None, :, None]
        * compute_bspline_moduli_1d(miller_z, mesh_nz, spline_order)[None, None, :]
    )
    structure_factor_sq = c.clamp_min(1e-10) ** 2
    return green, structure_factor_sq


def _pme_convolve_forward(
    mesh_fft: torch.Tensor,
    k_squared: torch.Tensor,
    moduli_x: torch.Tensor,
    moduli_y: torch.Tensor,
    moduli_z: torch.Tensor,
    alpha: torch.Tensor,
    volume: torch.Tensor,
    is_batch: bool,
) -> torch.Tensor:
    """Run the fused Warp convolve kernel on ``mesh_fft``. No autograd here —
    callers wrap this in ``_PMEFusedConvolve`` for the autograd-aware version.

    ``moduli_x/y/z`` are precomputed 1D B-spline modulus LUTs
    (``sinc(m/N)^spline_order`` per axis); see ``compute_bspline_moduli_1d``.
    """
    from nvalchemiops.interactions.electrostatics.pme_kernels import (
        batch_pme_convolve as _batch_pme_convolve,
    )
    from nvalchemiops.interactions.electrostatics.pme_kernels import (
        pme_convolve as _pme_convolve,
    )

    device = wp.device_from_torch(mesh_fft.device)
    real_dtype = torch.float32 if mesh_fft.dtype == torch.complex64 else torch.float64
    wp_dtype = wp.float32 if real_dtype == torch.float32 else wp.float64
    wp_vec2 = _vec2_wp_dtype_for(real_dtype)

    # generate_k_vectors_pme squeezes the batch dim when B=1 — restore it for
    # the batch kernel, which expects (B, nx, ny, nz_r). We track whether we
    # had to add a dim so we can squeeze the output back to the caller's shape.
    squeeze_output = False
    if is_batch and k_squared.dim() == 3:
        k_squared = k_squared.unsqueeze(0)
    if is_batch and mesh_fft.dim() == 3:
        mesh_fft = mesh_fft.unsqueeze(0)
        squeeze_output = True

    # `.resolve_conj()` materializes any pending lazy conjugation (autograd of
    # complex ops can hand us such tensors), which `view_as_real` doesn't
    # accept directly.
    mesh_fft_real = torch.view_as_real(mesh_fft.resolve_conj()).contiguous()
    convolved_real = torch.empty_like(mesh_fft_real)

    # Skip redundant .to()/.contiguous() when inputs are already in the right
    # form. At small N these calls dominate CPU dispatch time (~25 aten::to per
    # iter contribute ~0.9 ms at N=8k mesh=64^3 before this change).
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
    # alpha / volume: 0-d scalars or 1-d (1,) for single-system; (B,) for batch.
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
            _batch_pme_convolve(
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
            _pme_convolve(
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


def _pme_convolve_backward(
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
    """Explicit backward for the fused PME convolve.

    Returns ``(grad_mesh_fft, grad_alpha, grad_volume, grad_k_squared)``
    produced by a single Warp kernel that walks the k-space mesh once.
    See the kernel docstring in ``pme_kernels.py`` for the analytical
    derivatives. ``grad_k_squared`` is required because the Green's
    function uses k² (which itself depends on cell via the reciprocal
    lattice) and the per-cell gradient chain needs to flow through k².

    Layout: ``alpha`` and ``volume`` may be scalar (0-d) or shape ``(1,)``
    for single-system; shape ``(B,)`` for batch. Grad shape matches input.
    """
    from nvalchemiops.interactions.electrostatics.pme_kernels import (
        batch_pme_convolve_backward as _batch_pme_convolve_backward,
    )
    from nvalchemiops.interactions.electrostatics.pme_kernels import (
        pme_convolve_backward as _pme_convolve_backward_launch,
    )

    device = wp.device_from_torch(mesh_fft.device)
    real_dtype = torch.float32 if mesh_fft.dtype == torch.complex64 else torch.float64
    wp_dtype = wp.float32 if real_dtype == torch.float32 else wp.float64
    wp_vec2 = _vec2_wp_dtype_for(real_dtype)

    # Match shape conventions from _pme_convolve_forward (batch + B=1 squeeze).
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

    # alpha / volume always passed as length>=1 arrays (kernel reads index 0
    # or batch_idx). grad_alpha / grad_volume zero-initialized to match.
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

    # grad_k_squared has the same shape as k_squared (already unsqueezed above).
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
            _batch_pme_convolve_backward(
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
            _pme_convolve_backward_launch(
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


def _pme_convolve_double_backward(
    k_squared: torch.Tensor,
    h_grad_mesh: torch.Tensor,
    h_grad_alpha: torch.Tensor,
    h_grad_volume: torch.Tensor,
    h_grad_ksq: torch.Tensor,
    mesh_fft: torch.Tensor,
    grad_convolved: torch.Tensor,
    moduli_x: torch.Tensor,
    moduli_y: torch.Tensor,
    moduli_z: torch.Tensor,
    alpha: torch.Tensor,
    volume: torch.Tensor,
    is_batch: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Second-order node for the fused PME convolve backward.

    The convolve is LINEAR in ``mesh_fft``; every first-backward output is
    bilinear in ``(mesh_fft, grad_convolved)`` with constant per-k coefficients,
    so the position-relevant second-order terms are themselves linear. Given the
    cotangents on the four backward outputs (``h_grad_mesh``, ``h_grad_alpha``,
    ``h_grad_volume``, ``h_grad_ksq``), returns grads w.r.t. the backward op's
    five differentiable inputs (``mesh_fft``, ``grad_convolved``, ``k_squared``,
    ``alpha``, ``volume``) — backward positions ``(0, 1, 2, 6, 7)``.

    The real ``k_squared`` is placed at arg-0 (ahead of the complex
    ``h_grad_mesh`` cotangent) to mirror the backward op's complex-arg-0
    torch.compile/inductor workaround.

    ``grad_mesh_fft_out`` (dL/dmesh_fft) and ``grad_grad_convolved``
    (dL/dgrad_convolved) carry the force-loss second order. The ``k_squared`` /
    ``alpha`` / ``volume`` second-order grads carry the cell/stress second order
    (k² and V are functions of the cell; PyTorch maps them → cell outside this
    op).
    """
    from nvalchemiops.interactions.electrostatics.pme_kernels import (
        batch_pme_convolve_double_backward as _batch_dbwd_launch,
    )
    from nvalchemiops.interactions.electrostatics.pme_kernels import (
        pme_convolve_double_backward as _dbwd_launch,
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
    if is_batch and h_grad_mesh.dim() == 3:
        h_grad_mesh = h_grad_mesh.unsqueeze(0)
    if is_batch and h_grad_ksq.dim() == 3:
        h_grad_ksq = h_grad_ksq.unsqueeze(0)

    def _as(t):
        if t.dtype != real_dtype:
            t = t.to(real_dtype)
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    mesh_fft_real = torch.view_as_real(mesh_fft.resolve_conj()).contiguous()
    grad_conv_real = torch.view_as_real(grad_convolved.resolve_conj()).contiguous()
    h_grad_mesh_real = torch.view_as_real(h_grad_mesh.resolve_conj()).contiguous()

    grad_mesh_out_real = torch.empty_like(mesh_fft_real)
    grad_grad_conv_real = torch.empty_like(mesh_fft_real)
    grad_k_squared_out = torch.zeros_like(_as(k_squared))

    alpha_in = _as(alpha)
    volume_in = _as(volume)
    h_a_in = _as(h_grad_alpha)
    h_v_in = _as(h_grad_volume)
    if alpha_in.dim() == 0:
        alpha_in = alpha_in.reshape(1)
    if volume_in.dim() == 0:
        volume_in = volume_in.reshape(1)
    if h_a_in.dim() == 0:
        h_a_in = h_a_in.reshape(1)
    if h_v_in.dim() == 0:
        h_v_in = h_v_in.reshape(1)
    B = alpha_in.shape[0]
    grad_alpha_out = torch.zeros(B, dtype=real_dtype, device=mesh_fft.device)
    grad_volume_out = torch.zeros(B, dtype=real_dtype, device=mesh_fft.device)

    wp_h_grad_mesh = _wp_from_torch(h_grad_mesh_real, dtype=wp_vec2)
    wp_h_alpha = _wp_from_torch(h_a_in, dtype=wp_dtype)
    wp_h_volume = _wp_from_torch(h_v_in, dtype=wp_dtype)
    wp_h_grad_ksq = _wp_from_torch(_as(h_grad_ksq), dtype=wp_dtype)
    wp_mesh_fft = _wp_from_torch(mesh_fft_real, dtype=wp_vec2)
    wp_grad_conv = _wp_from_torch(grad_conv_real, dtype=wp_vec2)
    wp_k_squared = _wp_from_torch(_as(k_squared), dtype=wp_dtype)
    wp_bx = _wp_from_torch(_as(moduli_x), dtype=wp_dtype)
    wp_by = _wp_from_torch(_as(moduli_y), dtype=wp_dtype)
    wp_bz = _wp_from_torch(_as(moduli_z), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha_in, dtype=wp_dtype)
    wp_volume = _wp_from_torch(volume_in, dtype=wp_dtype)
    wp_grad_mesh_out = _wp_from_torch(grad_mesh_out_real, dtype=wp_vec2)
    wp_grad_grad_conv = _wp_from_torch(grad_grad_conv_real, dtype=wp_vec2)
    wp_grad_ksq_out = _wp_from_torch(grad_k_squared_out, dtype=wp_dtype)
    wp_grad_alpha_out = _wp_from_torch(grad_alpha_out, dtype=wp_dtype)
    wp_grad_volume_out = _wp_from_torch(grad_volume_out, dtype=wp_dtype)

    launch = _batch_dbwd_launch if is_batch else _dbwd_launch
    with _pme_scoped_warp_stream(mesh_fft.device):
        launch(
            wp_h_grad_mesh,
            wp_h_alpha,
            wp_h_volume,
            wp_h_grad_ksq,
            wp_mesh_fft,
            wp_grad_conv,
            wp_k_squared,
            wp_bx,
            wp_by,
            wp_bz,
            wp_alpha,
            wp_volume,
            wp_grad_mesh_out,
            wp_grad_grad_conv,
            wp_grad_ksq_out,
            wp_grad_alpha_out,
            wp_grad_volume_out,
            wp_dtype=wp_dtype,
            device=device,
        )

    grad_mesh_fft_out = torch.view_as_complex(grad_mesh_out_real)
    grad_grad_convolved = torch.view_as_complex(grad_grad_conv_real)
    if squeeze_output:
        grad_mesh_fft_out = grad_mesh_fft_out.squeeze(0)
        grad_grad_convolved = grad_grad_convolved.squeeze(0)
        grad_k_squared_out = grad_k_squared_out.squeeze(0)
    return (
        grad_mesh_fft_out,
        grad_grad_convolved,
        grad_k_squared_out,
        grad_alpha_out,
        grad_volume_out,
    )


def _convolve_double_backward_fake(
    k_squared,
    h_grad_mesh,
    h_grad_alpha,
    h_grad_volume,
    h_grad_ksq,
    mesh_fft,
    grad_convolved,
    moduli_x,
    moduli_y,
    moduli_z,
    alpha,
    volume,
    is_batch,
):
    real_dtype = torch.float32 if mesh_fft.dtype == torch.complex64 else torch.float64
    B = alpha.shape[0] if alpha.dim() >= 1 else 1
    # Meta/shape function: only shapes+dtypes matter for tracing, so the values
    # are placeholders (the real values come from the kernel launch).
    return (
        torch.empty_like(mesh_fft),  # grad_mesh_fft (dL/dmesh_fft)
        torch.empty_like(grad_convolved),  # grad_grad_convolved (dL/dgrad_convolved)
        torch.zeros_like(k_squared, dtype=real_dtype),  # grad_k_squared (dL/ds)
        torch.zeros(B, dtype=real_dtype, device=mesh_fft.device),  # grad_alpha
        torch.zeros(B, dtype=real_dtype, device=mesh_fft.device),  # grad_volume
    )


# Backward signature is ``(mesh_fft, grad_convolved, ...)`` (mesh_fft first,
# not cotangents-first) to work around an AOT-autograd/inductor complex
# codegen bug that produces ~1% wrong grads when a complex cotangent is in
# arg-0 under torch.compile fullgraph=True.


def _convolve_backward_fake(
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
        torch.empty_like(mesh_fft),  # grad_mesh_fft
        torch.zeros(B, dtype=real_dtype, device=mesh_fft.device),  # grad_alpha
        torch.zeros(B, dtype=real_dtype, device=mesh_fft.device),  # grad_volume
        torch.empty_like(k_squared, dtype=real_dtype),  # grad_k_squared
    )


def _convolve_forward_fake(mesh_fft, *_):
    # Launcher returns natural-contiguous; caller is responsible for passing
    # a contiguous mesh_fft so the fake stride matches the real call.
    return torch.empty(
        mesh_fft.shape,
        dtype=mesh_fft.dtype,
        device=mesh_fft.device,
    )


# Second-order autograd for the convolve. The convolve is LINEAR in mesh_fft and
# every first-backward output is bilinear in (mesh_fft, grad_convolved) with
# constant per-k coefficients, so the position-relevant second-order terms
# (dL/dmesh_fft, dL/dgrad_convolved) are linear — see ``_pme_convolve_double_backward``.
# A dedicated double-backward kernel handles all four backward-output cotangents.
# The alpha/volume/k_squared second-order gradients carry the cell/stress terms
# because k² and V are functions of the cell.
#
# The double-backward op signature leads with the real ``k_squared`` (arg-0)
# ahead of the complex ``h_grad_mesh`` cotangent, mirroring the backward op's
# complex-arg-0 inductor workaround. ``second_order_backward_args`` maps the
# backward node's (cotangents g, full inputs f) to that ordering; the backward
# op's inputs are f = (mesh_fft, grad_convolved, k_squared, moduli_x, moduli_y,
# moduli_z, alpha, volume, is_batch) and its outputs' cotangents are
# g = (h_grad_mesh, h_grad_alpha, h_grad_volume, h_grad_ksq).
###########################################################################################
########################### PME Energy Corrections Custom Ops #############################
###########################################################################################


###########################################################################################
###### Explicit Warp-backed backward chain for energy_corrections ##########################
###########################################################################################
#
# Forward/backward kernels come from the factory-backed PME corrections
# component for both single-system and batched launches.
#
# Wiring forward+backward via ``register_warp_op_chain`` +
# ``register_autograd``:
#   * is CUDA-graph-capture safe (no token tensor);
#   * gives torch a registered backward formula needed for
#     ``create_graph=True`` chains.
#
# Double-backward is registered on top of this via the second-order
# Warp kernel further down.


def _energy_corrections_forward_launch(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> torch.Tensor:
    """Single-system forward launch only (no autograd plumbing)."""
    from nvalchemiops.interactions.electrostatics.pme_kernels import (
        pme_energy_corrections as _ec_launch,
    )

    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    num_atoms = raw_energies.shape[0]

    corrected = torch.zeros(num_atoms, dtype=input_dtype, device=raw_energies.device)

    wp_raw = _wp_from_torch(raw_energies.contiguous(), dtype=wp_dtype)
    wp_charges = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_volume = _wp_from_torch(volume.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_qtot = _wp_from_torch(total_charge.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_corrected = _wp_from_torch(corrected, dtype=wp_dtype)

    with _pme_scoped_warp_stream(raw_energies.device):
        _ec_launch(
            wp_raw,
            wp_charges,
            wp_volume,
            wp_alpha,
            wp_qtot,
            wp_corrected,
            wp_dtype=wp_dtype,
            device=device,
        )
    return corrected


def _energy_corrections_backward_launch(
    grad_E: torch.Tensor,
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Single-system backward launch — returns the 5 input grads."""
    from nvalchemiops.interactions.electrostatics.pme_kernels import (
        pme_energy_corrections_backward as _ec_backward_launch,
    )

    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)

    grad_raw = torch.empty_like(raw_energies)
    grad_charges = torch.empty_like(charges, dtype=input_dtype)
    grad_volume = torch.zeros(1, dtype=input_dtype, device=raw_energies.device)
    grad_alpha = torch.zeros(1, dtype=input_dtype, device=raw_energies.device)
    grad_qtot = torch.zeros(1, dtype=input_dtype, device=raw_energies.device)

    wp_gE = _wp_from_torch(grad_E.contiguous(), dtype=wp_dtype)
    wp_raw = _wp_from_torch(raw_energies.contiguous(), dtype=wp_dtype)
    wp_chg = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_vol = _wp_from_torch(volume.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_qtot_in = _wp_from_torch(
        total_charge.to(input_dtype).contiguous(), dtype=wp_dtype
    )

    wp_g_raw = _wp_from_torch(grad_raw, dtype=wp_dtype)
    wp_g_chg = _wp_from_torch(grad_charges, dtype=wp_dtype)
    wp_g_vol = _wp_from_torch(grad_volume, dtype=wp_dtype)
    wp_g_alpha = _wp_from_torch(grad_alpha, dtype=wp_dtype)
    wp_g_qtot = _wp_from_torch(grad_qtot, dtype=wp_dtype)

    with _pme_scoped_warp_stream(raw_energies.device):
        _ec_backward_launch(
            wp_gE,
            wp_raw,
            wp_chg,
            wp_vol,
            wp_alpha,
            wp_qtot_in,
            wp_g_raw,
            wp_g_chg,
            wp_g_vol,
            wp_g_alpha,
            wp_g_qtot,
            wp_dtype=wp_dtype,
            device=device,
        )
    return grad_raw, grad_charges, grad_volume, grad_alpha, grad_qtot


def _energy_corrections_double_backward_launch(
    h_raw: torch.Tensor,
    h_chg: torch.Tensor,
    h_vol: torch.Tensor,
    h_alpha: torch.Tensor,
    h_qtot: torch.Tensor,
    grad_E: torch.Tensor,
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Single-system 2nd-order launcher — returns 6 grads."""
    from nvalchemiops.interactions.electrostatics.pme_kernels import (
        pme_energy_corrections_double_backward as _ec_dbwd_launch,
    )

    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)

    grad_grad_E = torch.empty_like(grad_E)
    grad_raw = torch.empty_like(raw_energies)
    grad_charges = torch.empty_like(charges, dtype=input_dtype)
    grad_volume = torch.zeros(1, dtype=input_dtype, device=raw_energies.device)
    grad_alpha = torch.zeros(1, dtype=input_dtype, device=raw_energies.device)
    grad_qtot = torch.zeros(1, dtype=input_dtype, device=raw_energies.device)

    wp_h_raw = _wp_from_torch(h_raw.contiguous(), dtype=wp_dtype)
    wp_h_chg = _wp_from_torch(h_chg.contiguous(), dtype=wp_dtype)
    wp_h_vol = _wp_from_torch(h_vol.contiguous(), dtype=wp_dtype)
    wp_h_alpha = _wp_from_torch(h_alpha.contiguous(), dtype=wp_dtype)
    wp_h_qtot = _wp_from_torch(h_qtot.contiguous(), dtype=wp_dtype)
    wp_gE = _wp_from_torch(grad_E.contiguous(), dtype=wp_dtype)
    wp_raw = _wp_from_torch(raw_energies.contiguous(), dtype=wp_dtype)
    wp_chg = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_vol = _wp_from_torch(volume.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_qtot_in = _wp_from_torch(
        total_charge.to(input_dtype).contiguous(), dtype=wp_dtype
    )

    wp_g_gE = _wp_from_torch(grad_grad_E, dtype=wp_dtype)
    wp_g_raw = _wp_from_torch(grad_raw, dtype=wp_dtype)
    wp_g_chg = _wp_from_torch(grad_charges, dtype=wp_dtype)
    wp_g_vol = _wp_from_torch(grad_volume, dtype=wp_dtype)
    wp_g_alpha = _wp_from_torch(grad_alpha, dtype=wp_dtype)
    wp_g_qtot = _wp_from_torch(grad_qtot, dtype=wp_dtype)

    with _pme_scoped_warp_stream(raw_energies.device):
        _ec_dbwd_launch(
            wp_h_raw,
            wp_h_chg,
            wp_h_vol,
            wp_h_alpha,
            wp_h_qtot,
            wp_gE,
            wp_raw,
            wp_chg,
            wp_vol,
            wp_alpha,
            wp_qtot_in,
            wp_g_gE,
            wp_g_raw,
            wp_g_chg,
            wp_g_vol,
            wp_g_alpha,
            wp_g_qtot,
            wp_dtype=wp_dtype,
            device=device,
        )
    return grad_grad_E, grad_raw, grad_charges, grad_volume, grad_alpha, grad_qtot


def _pme_energy_corrections(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> torch.Tensor:
    """Internal: single-system energy corrections via the registered custom op."""
    register_pme_ops()
    return torch.ops.nvalchemiops.pme_energy_corrections(
        raw_energies,
        charges.to(raw_energies.dtype),
        volume.to(raw_energies.dtype),
        alpha.to(raw_energies.dtype),
        total_charge.to(raw_energies.dtype),
    )


def _batch_energy_corrections_forward_launch(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> torch.Tensor:
    from nvalchemiops.interactions.electrostatics.pme_kernels import (
        batch_pme_energy_corrections as _batch_ec_launch,
    )

    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    n = raw_energies.shape[0]
    corrected = torch.zeros(n, dtype=input_dtype, device=raw_energies.device)

    wp_raw = _wp_from_torch(raw_energies.contiguous(), dtype=wp_dtype)
    wp_chg = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_bidx = _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32)
    wp_vol = _wp_from_torch(volumes.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_qtot = _wp_from_torch(total_charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_corrected = _wp_from_torch(corrected, dtype=wp_dtype)

    with _pme_scoped_warp_stream(raw_energies.device):
        _batch_ec_launch(
            wp_raw,
            wp_chg,
            wp_bidx,
            wp_vol,
            wp_alpha,
            wp_qtot,
            wp_corrected,
            wp_dtype=wp_dtype,
            device=device,
        )
    return corrected


def _batch_energy_corrections_backward_launch(
    grad_E: torch.Tensor,
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    from nvalchemiops.interactions.electrostatics.pme_kernels import (
        batch_pme_energy_corrections_backward as _batch_ec_backward_launch,
    )

    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    B = volumes.shape[0]

    grad_raw = torch.empty_like(raw_energies)
    grad_charges = torch.empty_like(charges, dtype=input_dtype)
    grad_volumes = torch.zeros(B, dtype=input_dtype, device=raw_energies.device)
    grad_alpha = torch.zeros(B, dtype=input_dtype, device=raw_energies.device)
    grad_qtots = torch.zeros(B, dtype=input_dtype, device=raw_energies.device)

    wp_gE = _wp_from_torch(grad_E.contiguous(), dtype=wp_dtype)
    wp_raw = _wp_from_torch(raw_energies.contiguous(), dtype=wp_dtype)
    wp_chg = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_bidx = _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32)
    wp_vol = _wp_from_torch(volumes.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_qtot_in = _wp_from_torch(
        total_charges.to(input_dtype).contiguous(), dtype=wp_dtype
    )

    wp_g_raw = _wp_from_torch(grad_raw, dtype=wp_dtype)
    wp_g_chg = _wp_from_torch(grad_charges, dtype=wp_dtype)
    wp_g_vol = _wp_from_torch(grad_volumes, dtype=wp_dtype)
    wp_g_alpha = _wp_from_torch(grad_alpha, dtype=wp_dtype)
    wp_g_qtot = _wp_from_torch(grad_qtots, dtype=wp_dtype)

    with _pme_scoped_warp_stream(raw_energies.device):
        _batch_ec_backward_launch(
            wp_gE,
            wp_raw,
            wp_chg,
            wp_bidx,
            wp_vol,
            wp_alpha,
            wp_qtot_in,
            wp_g_raw,
            wp_g_chg,
            wp_g_vol,
            wp_g_alpha,
            wp_g_qtot,
            wp_dtype=wp_dtype,
            device=device,
        )
    return grad_raw, grad_charges, grad_volumes, grad_alpha, grad_qtots


def _batch_energy_corrections_double_backward_launch(
    h_raw: torch.Tensor,
    h_chg: torch.Tensor,
    h_vol: torch.Tensor,
    h_alpha: torch.Tensor,
    h_qtot: torch.Tensor,
    grad_E: torch.Tensor,
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    from nvalchemiops.interactions.electrostatics.pme_kernels import (
        batch_pme_energy_corrections_double_backward as _batch_ec_dbwd_launch,
    )

    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    B = volumes.shape[0]

    grad_grad_E = torch.empty_like(grad_E)
    grad_raw = torch.empty_like(raw_energies)
    grad_charges = torch.empty_like(charges, dtype=input_dtype)
    grad_volumes = torch.zeros(B, dtype=input_dtype, device=raw_energies.device)
    grad_alpha = torch.zeros(B, dtype=input_dtype, device=raw_energies.device)
    grad_qtots = torch.zeros(B, dtype=input_dtype, device=raw_energies.device)

    wp_h_raw = _wp_from_torch(h_raw.contiguous(), dtype=wp_dtype)
    wp_h_chg = _wp_from_torch(h_chg.contiguous(), dtype=wp_dtype)
    wp_h_vol = _wp_from_torch(h_vol.contiguous(), dtype=wp_dtype)
    wp_h_alpha = _wp_from_torch(h_alpha.contiguous(), dtype=wp_dtype)
    wp_h_qtot = _wp_from_torch(h_qtot.contiguous(), dtype=wp_dtype)
    wp_gE = _wp_from_torch(grad_E.contiguous(), dtype=wp_dtype)
    wp_raw = _wp_from_torch(raw_energies.contiguous(), dtype=wp_dtype)
    wp_chg = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_bidx = _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32)
    wp_vol = _wp_from_torch(volumes.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_qtot_in = _wp_from_torch(
        total_charges.to(input_dtype).contiguous(), dtype=wp_dtype
    )

    wp_g_gE = _wp_from_torch(grad_grad_E, dtype=wp_dtype)
    wp_g_raw = _wp_from_torch(grad_raw, dtype=wp_dtype)
    wp_g_chg = _wp_from_torch(grad_charges, dtype=wp_dtype)
    wp_g_vol = _wp_from_torch(grad_volumes, dtype=wp_dtype)
    wp_g_alpha = _wp_from_torch(grad_alpha, dtype=wp_dtype)
    wp_g_qtot = _wp_from_torch(grad_qtots, dtype=wp_dtype)

    with _pme_scoped_warp_stream(raw_energies.device):
        _batch_ec_dbwd_launch(
            wp_h_raw,
            wp_h_chg,
            wp_h_vol,
            wp_h_alpha,
            wp_h_qtot,
            wp_gE,
            wp_raw,
            wp_chg,
            wp_bidx,
            wp_vol,
            wp_alpha,
            wp_qtot_in,
            wp_g_gE,
            wp_g_raw,
            wp_g_chg,
            wp_g_vol,
            wp_g_alpha,
            wp_g_qtot,
            wp_dtype=wp_dtype,
            device=device,
        )
    return grad_grad_E, grad_raw, grad_charges, grad_volumes, grad_alpha, grad_qtots


def _batch_pme_energy_corrections(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> torch.Tensor:
    """Internal: batched energy corrections via the registered custom op."""
    register_pme_ops()
    return torch.ops.nvalchemiops.pme_energy_corrections_batch(
        raw_energies,
        charges.to(raw_energies.dtype),
        batch_idx,
        volumes.to(raw_energies.dtype),
        alpha.to(raw_energies.dtype),
        total_charges.to(raw_energies.dtype),
    )


def _energy_corrections_charge_grad_forward_launch(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-system forward launch for fused corrected_energies + charge_gradients.

    Pure forward — no autograd plumbing. The charge_gradient output is used
    by ``_InjectChargeGrad`` (which doesn't backprop into it), so we treat
    it as non-differentiable in the wrapping ``Function``.
    """
    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    num_atoms = raw_energies.shape[0]

    corrected_energies = torch.zeros(
        num_atoms, dtype=input_dtype, device=raw_energies.device
    )
    charge_gradients = torch.zeros(
        num_atoms, dtype=input_dtype, device=raw_energies.device
    )

    wp_raw = _wp_from_torch(raw_energies.contiguous(), dtype=wp_dtype)
    wp_charges = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_volume = _wp_from_torch(volume.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_qtot = _wp_from_torch(total_charge.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_corrected = _wp_from_torch(corrected_energies, dtype=wp_dtype)
    wp_charge_grads = _wp_from_torch(charge_gradients, dtype=wp_dtype)

    with _pme_scoped_warp_stream(raw_energies.device):
        _pme_energy_corrections_with_charge_grad_warp(
            wp_raw,
            wp_charges,
            wp_volume,
            wp_alpha,
            wp_qtot,
            wp_corrected,
            wp_charge_grads,
            wp_dtype,
            device=device,
        )
    return corrected_energies, charge_gradients


# ---------------------------------------------------------------------------
# Single-system energy_corrections_with_charge_grad as torch.library.custom_op.
#
# The kernel returns (corrected_energies, charge_gradients) in one pass:
# charge_gradients = analytical ∂E_total/∂q_i. The second output is consumed
# downstream by ``_InjectChargeGrad`` (the dsf.py-style autograd.Function in
# ``_util.py``) which returns None for its grad — so we treat charge_gradients
# as non-differentiable here, exactly mirroring the prior autograd.Function.
#
# The backward of this op delegates to the regular pme_energy_corrections_
# backward op (charge_grad cotangent is ignored), so no new backward kernel
# is needed.


def _pme_energy_corrections_with_charge_grad(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: single-system fused corrections + analytical charge gradient."""
    register_pme_ops()
    return torch.ops.nvalchemiops.pme_energy_corrections_with_charge_grad(
        raw_energies,
        charges.to(raw_energies.dtype),
        volume.to(raw_energies.dtype),
        alpha.to(raw_energies.dtype),
        total_charge.to(raw_energies.dtype),
    )


def _batch_energy_corrections_charge_grad_forward_launch(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched forward launch for fused corrected_energies + charge_gradients."""
    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    num_atoms = raw_energies.shape[0]

    corrected_energies = torch.zeros(
        num_atoms, dtype=input_dtype, device=raw_energies.device
    )
    charge_gradients = torch.zeros(
        num_atoms, dtype=input_dtype, device=raw_energies.device
    )

    wp_raw = _wp_from_torch(raw_energies.contiguous(), dtype=wp_dtype)
    wp_charges = _wp_from_torch(charges.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_bidx = _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32)
    wp_volumes = _wp_from_torch(volumes.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha.to(input_dtype).contiguous(), dtype=wp_dtype)
    wp_qtots = _wp_from_torch(
        total_charges.to(input_dtype).contiguous(), dtype=wp_dtype
    )
    wp_corrected = _wp_from_torch(corrected_energies, dtype=wp_dtype)
    wp_charge_grads = _wp_from_torch(charge_gradients, dtype=wp_dtype)

    with _pme_scoped_warp_stream(raw_energies.device):
        _batch_pme_energy_corrections_with_charge_grad_warp(
            wp_raw,
            wp_charges,
            wp_bidx,
            wp_volumes,
            wp_alpha,
            wp_qtots,
            wp_corrected,
            wp_charge_grads,
            wp_dtype,
            device=device,
        )
    return corrected_energies, charge_gradients


def _batch_pme_energy_corrections_with_charge_grad(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: batched fused corrections + analytical charge gradient."""
    register_pme_ops()
    return torch.ops.nvalchemiops.pme_energy_corrections_with_charge_grad_batch(
        raw_energies,
        charges.to(raw_energies.dtype),
        batch_idx,
        volumes.to(raw_energies.dtype),
        alpha.to(raw_energies.dtype),
        total_charges.to(raw_energies.dtype),
    )


def pme_energy_corrections(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    volume: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply self-energy and background corrections to PME energies.

    Converts raw interpolated potential to energy and subtracts corrections:

    .. math::

        E_i = q_i \\phi_i - E_{\\text{self},i} - E_{\\text{background},i}

    Self-energy correction (removes Gaussian self-interaction):

    .. math::

        E_{\\text{self},i} = \\frac{\\alpha}{\\sqrt{\\pi}} q_i^2

    Background correction (for non-neutral systems):

    .. math::

        E_{\\text{background},i} = \\frac{\\pi}{2\\alpha^2 V} q_i Q_{\\text{total}}

    Parameters
    ----------
    raw_energies : torch.Tensor, shape (N,) or (N_total,)
        Raw potential values :math:`\\phi_i` from mesh interpolation.
    charges : torch.Tensor, shape (N,) or (N_total,)
        Atomic charges.
    cell : torch.Tensor
        Unit cell matrices.
        - Single-system: shape (3, 3) or (1, 3, 3)
        - Batch: shape (B, 3, 3)
    alpha : torch.Tensor
        Ewald splitting parameter.
        - Single-system: shape (1,)
        - Batch: shape (B,)
    batch_idx : torch.Tensor | None, default=None
        System index for each atom. If provided, uses batch kernels.

    Returns
    -------
    corrected_energies : torch.Tensor, shape (N,) or (N_total,)
        Final per-atom reciprocal-space energy with corrections applied.

    Notes
    -----
    - For neutral systems, background correction is zero
    - Matches torchpme's self_contribution and background_correction formulas
    - Supports both float32 and float64 dtypes
    """
    ensure_electrostatics_ops_registered()
    input_dtype = raw_energies.dtype

    if batch_idx is None:
        # Single system - ensure tensors are 1D for kernel indexing
        total_charge = charges.sum().reshape(1)
        if volume is None:
            volume = torch.abs(torch.det(cell)).reshape(1)
        else:
            volume = volume.reshape(1)

        result = _pme_energy_corrections(
            raw_energies,
            charges.to(input_dtype),
            volume.to(input_dtype),
            alpha.to(input_dtype),
            total_charge.to(input_dtype),
        )
    else:
        # Batch
        num_systems = cell.shape[0]
        if volume is None:
            volumes = torch.abs(torch.linalg.det(cell)).to(input_dtype)
        else:
            volumes = volume.to(input_dtype)

        # Compute total charge per system
        total_charges = torch.zeros(
            num_systems, dtype=input_dtype, device=raw_energies.device
        )
        total_charges.scatter_add_(0, batch_idx, charges.to(input_dtype))

        result = _batch_pme_energy_corrections(
            raw_energies,
            charges.to(input_dtype),
            batch_idx,
            volumes,
            alpha.to(input_dtype),
            total_charges,
        )

    return result


def pme_energy_corrections_with_charge_grad(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    volume: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply corrections and compute charge gradients for PME energies.

    Computes both corrected energies and analytical charge gradients:
        E_i = q_i * φ_i - E_self_i - E_background_i
        ∂E/∂q_i = 2*φ_i - 2*(α/√π)*q_i - (π/(α²V))*Q_total

    The factor of 2 on φ_i arises because changing q_i affects both the
    direct energy term (q_i * φ_i) and all other potentials through the
    structure factor (∑_j q_j * ∂φ_j/∂q_i = φ_i).

    Parameters
    ----------
    raw_energies : torch.Tensor, shape (N,) or (N_total,)
        Raw potential values φ_i from mesh interpolation.
    charges : torch.Tensor, shape (N,) or (N_total,)
        Atomic charges.
    cell : torch.Tensor
        Unit cell matrices.
        - Single-system: shape (3, 3) or (1, 3, 3)
        - Batch: shape (B, 3, 3)
    alpha : torch.Tensor
        Ewald splitting parameter.
        - Single-system: shape (1,)
        - Batch: shape (B,)
    batch_idx : torch.Tensor | None, default=None
        System index for each atom. If provided, uses batch kernels.

    Returns
    -------
    corrected_energies : torch.Tensor, shape (N,) or (N_total,)
        Final per-atom reciprocal-space energy with corrections applied.
    charge_gradients : torch.Tensor, shape (N,) or (N_total,)
        Analytical charge gradients ∂E/∂q_i.
    """
    ensure_electrostatics_ops_registered()
    input_dtype = raw_energies.dtype

    if batch_idx is None:
        # Single system
        total_charge = charges.sum().reshape(1)
        if volume is None:
            volume = torch.abs(torch.det(cell)).reshape(1)
        else:
            volume = volume.reshape(1)
        return _pme_energy_corrections_with_charge_grad(
            raw_energies,
            charges.to(input_dtype),
            volume.to(input_dtype),
            alpha.to(input_dtype),
            total_charge.to(input_dtype),
        )
    else:
        # Batch
        num_systems = cell.shape[0]
        if volume is None:
            volumes = torch.abs(torch.linalg.det(cell)).to(input_dtype)
        else:
            volumes = volume.to(input_dtype)

        # Compute total charge per system
        total_charges = torch.zeros(
            num_systems, dtype=input_dtype, device=raw_energies.device
        )
        total_charges.scatter_add_(0, batch_idx, charges.to(input_dtype))

        return _batch_pme_energy_corrections_with_charge_grad(
            raw_energies,
            charges.to(input_dtype),
            batch_idx,
            volumes,
            alpha.to(input_dtype),
            total_charges,
        )


###########################################################################################
########################### Virial Background Correction ##################################
###########################################################################################
# Functional warp-backed op returning ``virial_in - E_bg(s)·I`` with
# analytic backward through charges / cell / alpha.


def _virial_bg_correction_forward_launch(
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    cell: torch.Tensor,
    volume: torch.Tensor,
    use_supplied_volume: bool,
    alpha: torch.Tensor,
    virial_in: torch.Tensor,
) -> torch.Tensor:
    real_dtype = virial_in.dtype
    wp_dtype = wp.float32 if real_dtype == torch.float32 else wp.float64
    device = wp.device_from_torch(virial_in.device)

    virial_out = torch.empty_like(virial_in)
    total_charges = torch.zeros(
        virial_in.shape[0],
        dtype=real_dtype,
        device=virial_in.device,
    )

    wp_charges = _wp_from_torch(charges.contiguous(), dtype=wp_dtype)
    wp_batch_idx = _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32)
    wp_cell = _wp_from_torch(cell.contiguous(), dtype=wp_dtype)
    wp_volume = _wp_from_torch(volume.contiguous(), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha.contiguous(), dtype=wp_dtype)
    wp_total = _wp_from_torch(total_charges, dtype=wp_dtype)
    wp_virial_in = _wp_from_torch(virial_in.contiguous(), dtype=wp_dtype)
    wp_virial_out = _wp_from_torch(virial_out, dtype=wp_dtype)

    with _pme_scoped_warp_stream(virial_in.device):
        _pme_virial_bg_correction_warp(
            charges=wp_charges,
            batch_idx=wp_batch_idx,
            cell=wp_cell,
            volume=wp_volume,
            use_supplied_volume=use_supplied_volume,
            alpha=wp_alpha,
            total_charges=wp_total,
            virial_in=wp_virial_in,
            virial_out=wp_virial_out,
            wp_dtype=wp_dtype,
            device=device,
        )
    return virial_out


def _virial_bg_correction_backward_launch(
    grad_virial: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    cell: torch.Tensor,
    volume: torch.Tensor,
    use_supplied_volume: bool,
    alpha: torch.Tensor,
    virial_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    real_dtype = virial_in.dtype
    wp_dtype = wp.float32 if real_dtype == torch.float32 else wp.float64
    device = wp.device_from_torch(virial_in.device)
    n = charges.shape[0]
    B = virial_in.shape[0]

    total_charges = torch.zeros(B, dtype=real_dtype, device=virial_in.device)
    grad_total_charges = torch.zeros(B, dtype=real_dtype, device=virial_in.device)
    grad_charges = torch.empty(n, dtype=real_dtype, device=virial_in.device)
    grad_alpha = torch.empty(B, dtype=real_dtype, device=virial_in.device)
    grad_cell = torch.empty_like(cell)

    wp_gV = _wp_from_torch(grad_virial.contiguous(), dtype=wp_dtype)
    wp_charges = _wp_from_torch(charges.contiguous(), dtype=wp_dtype)
    wp_batch_idx = _wp_from_torch(batch_idx.contiguous(), dtype=wp.int32)
    wp_cell = _wp_from_torch(cell.contiguous(), dtype=wp_dtype)
    wp_volume = _wp_from_torch(volume.contiguous(), dtype=wp_dtype)
    wp_alpha = _wp_from_torch(alpha.contiguous(), dtype=wp_dtype)
    wp_total = _wp_from_torch(total_charges, dtype=wp_dtype)
    wp_g_total = _wp_from_torch(grad_total_charges, dtype=wp_dtype)
    wp_g_chg = _wp_from_torch(grad_charges, dtype=wp_dtype)
    wp_g_alpha = _wp_from_torch(grad_alpha, dtype=wp_dtype)
    wp_g_cell = _wp_from_torch(grad_cell, dtype=wp_dtype)

    with _pme_scoped_warp_stream(virial_in.device):
        _pme_virial_bg_correction_backward_warp(
            grad_virial=wp_gV,
            charges=wp_charges,
            batch_idx=wp_batch_idx,
            cell=wp_cell,
            volume=wp_volume,
            use_supplied_volume=use_supplied_volume,
            alpha=wp_alpha,
            total_charges=wp_total,
            grad_total_charges=wp_g_total,
            grad_charges=wp_g_chg,
            grad_alpha=wp_g_alpha,
            grad_cell=wp_g_cell,
            wp_dtype=wp_dtype,
            device=device,
        )
    # ``virial_out = virial_in - E_bg·I`` makes the cotangent w.r.t.
    # ``virial_in`` the identity image of ``grad_virial``. Return a clone
    # so the output tensor does not alias the input cotangent — PyTorch's
    # custom_op runtime rejects any input↔output aliasing in backwards.
    return grad_charges, grad_cell, grad_alpha, grad_virial.clone()


def register_pme_ops() -> None:
    """Register PME Torch custom ops once."""
    global _PME_OPS_REGISTERED
    if _PME_OPS_REGISTERED:
        return

    register_warp_op_chain(
        name="nvalchemiops::pme_fused_convolve",
        forward=_pme_convolve_forward,
        forward_fake=_convolve_forward_fake,
        backward=_pme_convolve_backward,
        backward_fake=_convolve_backward_fake,
        backward_return_arity=4,
        diff_input_positions=(0, 5, 6, 1),
        n_forward_inputs=8,
        backward_args=lambda g, f: (
            f[0],
            g[0],
            f[1],
            f[2],
            f[3],
            f[4],
            f[5],
            f[6],
            f[7],
        ),
        double_backward=_pme_convolve_double_backward,
        double_backward_fake=_convolve_double_backward_fake,
        double_backward_return_arity=5,
        second_order_diff_positions=(0, 1, 2, 6, 7),
        n_backward_inputs=9,
        second_order_backward_args=lambda g, f: (
            f[2],
            g[0],
            g[1],
            g[2],
            g[3],
            f[0],
            f[1],
            f[3],
            f[4],
            f[5],
            f[6],
            f[7],
            f[8],
        ),
    )

    register_warp_op_chain(
        name="nvalchemiops::pme_energy_corrections",
        forward=_energy_corrections_forward_launch,
        backward=_energy_corrections_backward_launch,
        double_backward=_energy_corrections_double_backward_launch,
        diff_input_positions=(0, 1, 2, 3, 4),
        n_forward_inputs=5,
        second_order_diff_positions=(0, 1, 2, 3, 4, 5),
        n_backward_inputs=6,
    )

    register_warp_op_chain(
        name="nvalchemiops::pme_energy_corrections_batch",
        forward=_batch_energy_corrections_forward_launch,
        backward=_batch_energy_corrections_backward_launch,
        double_backward=_batch_energy_corrections_double_backward_launch,
        diff_input_positions=(0, 1, 3, 4, 5),
        n_forward_inputs=6,
        second_order_diff_positions=(0, 1, 2, 4, 5, 6),
        n_backward_inputs=7,
        batch_match=True,
    )

    register_warp_op_chain(
        name="nvalchemiops::pme_energy_corrections_with_charge_grad",
        forward=_energy_corrections_charge_grad_forward_launch,
        forward_return_arity=2,
        forward_fake=lambda raw, *_: (torch.empty_like(raw), torch.empty_like(raw)),
    )
    attach_simple_backward(
        "nvalchemiops::pme_energy_corrections_with_charge_grad",
        torch.ops.nvalchemiops.pme_energy_corrections_backward,
        diff_input_positions=(0, 1, 2, 3, 4),
        n_forward_inputs=5,
        propagate_outputs=(0,),
    )

    register_warp_op_chain(
        name="nvalchemiops::pme_energy_corrections_with_charge_grad_batch",
        forward=_batch_energy_corrections_charge_grad_forward_launch,
        forward_return_arity=2,
        forward_fake=lambda raw, *_: (torch.empty_like(raw), torch.empty_like(raw)),
    )
    attach_simple_backward(
        "nvalchemiops::pme_energy_corrections_with_charge_grad_batch",
        torch.ops.nvalchemiops.pme_energy_corrections_batch_backward,
        diff_input_positions=(0, 1, 3, 4, 5),
        n_forward_inputs=6,
        batch_match=True,
        propagate_outputs=(0,),
    )

    register_warp_op_chain(
        name="nvalchemiops::pme_virial_bg_correction",
        forward=_virial_bg_correction_forward_launch,
        backward=_virial_bg_correction_backward_launch,
        diff_input_positions=(0, 2, 5, 6),  # charges, cell, alpha, virial_in
        n_forward_inputs=7,
        forward_fake=lambda charges,
        batch_idx,
        cell,
        volume,
        use_supplied_volume,
        alpha,
        virial_in: (torch.empty_like(virial_in)),
        batch_match=True,
    )
    _PME_OPS_REGISTERED = True


###########################################################################################
########################### Unified PME Reciprocal Space ##################################
###########################################################################################


def _compute_pme_reciprocal_virial(
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
    """Compute PME reciprocal-space virial tensor in k-space.

    Uses the exact spectral pair from the pipeline (mesh_fft_raw before
    deconvolution, and convolved_mesh after Green's function multiplication)
    to compute the per-k energy density directly via Parseval's theorem.

    The virial per k-point is W_ab(k) = E_k * sigma_ab(k) where:
    - E_k = prefactor * weight(k) * Re(mesh_fft_raw(k) * convolved_mesh(k)*)
    - sigma_ab(k) = delta_ab - 2*k_a*k_b/k^2 * (1 + k^2/(4*alpha^2))
    (sign reflects W = -dE/dε convention)

    Parameters
    ----------
    mesh_fft_raw : torch.Tensor
        Raw rfftn output before B-spline deconvolution.
        Shape (nx, ny, nz//2+1) or (B, nx, ny, nz//2+1), complex.
    convolved_mesh : torch.Tensor
        Deconvolved mesh FFT multiplied by Green's function: (mesh_fft/B^2)*G.
        Shape matching mesh_fft_raw.
    k_vectors : torch.Tensor
        k-vectors on the mesh. Shape (..., nx, ny, nz//2+1, 3).
    k_squared : torch.Tensor
        |k|^2. Shape (..., nx, ny, nz//2+1).
    alpha : torch.Tensor
        Ewald splitting parameter.
    mesh_dimensions : tuple
        (nx, ny, nz).
    is_batch : bool
        Whether this is a batched calculation.
    device : torch.device
        Computation device.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    virial : torch.Tensor, shape (B, 3, 3) or (1, 3, 3)
        Per-system virial tensor.
    """
    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions

    # Per-k energy density from exact pipeline spectral pair.
    # Re(mesh_fft_raw * convolved_mesh*) = |mesh_fft_raw|^2 * G / B^2
    #
    # Explicit complex/real dtype mapping is needed because `dtype` is a
    # real-valued dtype (float32 or float64) but the FFT mesh data is complex.
    # PyTorch has no implicit real-to-complex dtype promotion, so we map
    # float32 -> complex64 and float64 -> complex128 explicitly.
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    acc_dtype = dtype  # real accumulation dtype matches input precision
    fft_raw_cast = mesh_fft_raw.to(complex_dtype)
    conv_cast = convolved_mesh.to(complex_dtype)
    energy_density = (fft_raw_cast * conv_cast.conj()).real

    # Weight for rfft symmetry: 2 for interior k_z, 1 for boundary
    weight = torch.full_like(energy_density, 2.0)
    weight[..., 0] = 1.0  # k_z = 0
    if mesh_nz % 2 == 0:
        weight[..., -1] = 1.0  # k_z = nz//2 (Nyquist)

    # Weighted energy density
    weighted_energy = weight * energy_density

    # Virial W = -dE/dε, so sigma_ab = delta_ab - 2*k_a*k_b/k^2 * (1 + k^2/(4*alpha^2))
    k_sq_acc = k_squared.to(acc_dtype)
    alpha_acc = alpha.to(acc_dtype)

    # generate_k_vectors_pme squeezes the batch dim when B=1; restore it so
    # the batched einsum and sum_dims=(1,2,3) operate on the correct axes.
    if is_batch and k_sq_acc.dim() == 3:
        k_sq_acc = k_sq_acc.unsqueeze(0)

    # Handle alpha broadcasting: alpha may be (B,) for batch
    if is_batch and alpha_acc.dim() == 1:
        alpha_view = alpha_acc.view(-1, 1, 1, 1)
    else:
        alpha_view = alpha_acc.view(-1) if alpha_acc.dim() == 0 else alpha_acc

    exp_factor = 0.25 / (alpha_view**2)

    # Avoid division by zero at k=0
    safe_k_sq = k_sq_acc.clamp(min=1e-30)
    k_factor = 2.0 * (1.0 + k_sq_acc * exp_factor) / safe_k_sq

    # Zero out k=0 contribution (no virial from k=0)
    k_mask = k_sq_acc > 1e-10

    # Six per-component weighted reductions instead of an einsum: the
    # einsum's (M=N=3, K=mesh_size) shape hits a slow cuBLAS sgemm corner.
    # virial_ab = sum_k masked_energy * (delta_ab - k_factor * k_a * k_b)
    k_vecs_acc = k_vectors.to(acc_dtype)  # (..., nx, ny, nz//2+1, 3)
    if is_batch and k_vecs_acc.dim() == 4:
        k_vecs_acc = k_vecs_acc.unsqueeze(0)

    masked_energy = weighted_energy * k_mask  # (..., nx, ny, nz//2+1)
    masked_energy_kf = masked_energy * k_factor  # (..., nx, ny, nz//2+1)

    # Sum dimensions depend on batch vs single
    if is_batch:
        sum_dims = (1, 2, 3)
    else:
        sum_dims = (0, 1, 2)

    # Trace term: delta_ab * sum_k masked_energy
    trace_term = masked_energy.sum(dim=sum_dims)  # scalar or (B,)

    # kk term components — six symmetric (a,b) reductions in one expression.
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
        # Assemble symmetric (B, 3, 3) tensor.
        kk_term = torch.stack(
            [
                torch.stack([xx, xy, xz], dim=-1),
                torch.stack([xy, yy, yz], dim=-1),
                torch.stack([xz, yz, zz], dim=-1),
            ],
            dim=-2,
        )
        virial = eye * trace_term[:, None, None] - kk_term  # (B, 3, 3)
    else:
        kk_term = torch.stack(
            [
                torch.stack([xx, xy, xz]),
                torch.stack([xy, yy, yz]),
                torch.stack([xz, yz, zz]),
            ],
        )  # (3, 3)
        virial = (eye * trace_term - kk_term).unsqueeze(0)  # (1, 3, 3)

    return virial.to(dtype)


def _pme_cell_grad_from_virial(
    positions: torch.Tensor,
    dEdR: torch.Tensor,
    cell: torch.Tensor,
    virial: torch.Tensor,
    batch_idx: torch.Tensor | None,
    cell_inv_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert strain virial plus ``dE/dR`` into partial ``dE/dcell``.

    Direct PME virial is ``W = -dE/dstrain`` for simultaneous row-vector
    displacement of positions and cell. The eval fastpath returns partial
    gradients for the actual autograd inputs, so solve
    ``cell.T @ dE/dcell = -W - positions.T @ dE/dR`` per system.
    """
    cell_3d = cell if cell.dim() == 3 else cell.unsqueeze(0)
    num_systems = cell_3d.shape[0]
    pos_term = torch.zeros(
        num_systems,
        3,
        3,
        device=positions.device,
        dtype=torch.float64,
    )
    outer = positions.to(torch.float64).unsqueeze(2) * dEdR.to(torch.float64).unsqueeze(
        1
    )
    if batch_idx is None:
        if outer.numel():
            pos_term[0] = outer.sum(dim=0)
    else:
        pos_term = pos_term.index_add(0, batch_idx.to(torch.long), outer)
    target = -virial.to(torch.float64) - pos_term
    if cell_inv_t is not None:
        inv_t_3d = _normalize_cell_inv_t_cache(cell_inv_t).to(torch.float64)
        return torch.matmul(inv_t_3d, target).to(cell.dtype)
    return torch.linalg.solve(cell_3d.transpose(-1, -2).to(torch.float64), target).to(
        cell.dtype
    )


class _PMEReciprocalCachedFirstGrad(torch.autograd.Function):
    """PME reciprocal energy with detached first-derivative eval caches."""

    @staticmethod
    def forward(
        ctx,
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        k_vectors,
        k_squared,
        volume,
        cell_inv_t,
        moduli_x,
        moduli_y,
        moduli_z,
        mesh_dimensions,
        spline_order,
        need_pos,
        need_charge,
        need_cell,
    ):
        """Compute energy and direct first-derivative states."""
        need_forces = bool(need_pos) or bool(need_cell)
        need_charges = bool(need_charge)
        need_virial = bool(need_cell)

        impl_out = _pme_reciprocal_space_impl(
            positions.detach(),
            charges.detach(),
            cell.detach(),
            alpha.detach(),
            mesh_dimensions,
            spline_order,
            batch_idx.detach() if batch_idx is not None else None,
            compute_forces=need_forces,
            compute_charge_gradients=need_charges,
            compute_virial=need_virial,
            k_vectors=k_vectors.detach() if k_vectors is not None else None,
            k_squared=k_squared.detach() if k_squared is not None else None,
            volume=volume.detach() if volume is not None else None,
            cell_inv_t=cell_inv_t.detach() if cell_inv_t is not None else None,
            moduli_x=moduli_x.detach() if moduli_x is not None else None,
            moduli_y=moduli_y.detach() if moduli_y is not None else None,
            moduli_z=moduli_z.detach() if moduli_z is not None else None,
            return_cell_inv_t=need_virial,
        )
        if need_virial:
            energies, forces, charge_grads, virial, cached_cell_inv_t = impl_out
        else:
            energies, forces, charge_grads, virial = impl_out
            cached_cell_inv_t = None

        cached_dEdR = -forces if need_forces else None
        cached_dEdq = charge_grads if need_charges else None
        cached_dEdcell = None
        if need_virial:
            cached_dEdcell = _pme_cell_grad_from_virial(
                positions.detach(),
                cached_dEdR,
                cell.detach(),
                virial,
                batch_idx,
                cached_cell_inv_t,
            )

        ctx.save_for_backward(
            positions,
            charges,
            cell,
            alpha,
            batch_idx,
            k_vectors,
            k_squared,
            volume,
            cell_inv_t,
            moduli_x,
            moduli_y,
            moduli_z,
            cached_dEdR,
            cached_dEdq,
            cached_dEdcell,
        )
        ctx.mesh_dimensions = mesh_dimensions
        ctx.spline_order = spline_order
        ctx.need_pos = bool(need_pos)
        ctx.need_charge = bool(need_charge)
        ctx.need_cell = bool(need_cell)
        return energies

    @staticmethod
    def backward(ctx, grad_energy):
        """Return cached first gradients or recompute for higher-order fallback."""
        (
            positions,
            charges,
            cell,
            alpha,
            batch_idx,
            k_vectors,
            k_squared,
            volume,
            cell_inv_t,
            moduli_x,
            moduli_y,
            moduli_z,
            cached_dEdR,
            cached_dEdq,
            cached_dEdcell,
        ) = ctx.saved_tensors

        if torch.is_grad_enabled() or not _is_uniform_cotangent(grad_energy):
            with torch.enable_grad():
                recomputed, _forces, _charge_grads, _virial = (
                    _pme_reciprocal_space_impl(
                        positions,
                        charges,
                        cell,
                        alpha,
                        ctx.mesh_dimensions,
                        ctx.spline_order,
                        batch_idx,
                        compute_forces=False,
                        compute_charge_gradients=False,
                        compute_virial=False,
                        k_vectors=k_vectors,
                        k_squared=k_squared,
                        volume=volume,
                        cell_inv_t=cell_inv_t,
                        moduli_x=moduli_x,
                        moduli_y=moduli_y,
                        moduli_z=moduli_z,
                    )
                )
                diff_inputs = []
                diff_names = []
                for name, tensor in (
                    ("positions", positions),
                    ("charges", charges),
                    ("cell", cell),
                    ("alpha", alpha),
                ):
                    if tensor.requires_grad:
                        diff_inputs.append(tensor)
                        diff_names.append(name)
                diff_grads = torch.autograd.grad(
                    recomputed,
                    tuple(diff_inputs),
                    grad_outputs=grad_energy,
                    allow_unused=True,
                    create_graph=torch.is_grad_enabled(),
                )
                grad_map = dict(zip(diff_names, diff_grads, strict=True))
                grad_positions = grad_map.get("positions")
                grad_charges = grad_map.get("charges")
                grad_cell = grad_map.get("cell")
                grad_alpha = grad_map.get("alpha")
            return (
                grad_positions,
                grad_charges,
                grad_cell,
                grad_alpha,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        grad = grad_energy.reshape(-1)
        atom_scale = grad[0]
        if batch_idx is None:
            system_scale = atom_scale
        else:
            system_scale = atom_scale

        grad_positions = cached_dEdR * atom_scale if ctx.need_pos else None
        grad_charges = cached_dEdq * atom_scale if ctx.need_charge else None
        grad_cell = cached_dEdcell * system_scale if ctx.need_cell else None

        return (
            grad_positions,
            grad_charges,
            grad_cell,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _pme_reciprocal_cached_first_grad(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    spline_order: int,
    batch_idx: torch.Tensor | None,
    k_vectors: torch.Tensor | None,
    k_squared: torch.Tensor | None,
    volume: torch.Tensor | None,
    cell_inv_t: torch.Tensor | None,
    moduli_x: torch.Tensor | None,
    moduli_y: torch.Tensor | None,
    moduli_z: torch.Tensor | None,
    *,
    need_pos: bool,
    need_charge: bool,
    need_cell: bool,
) -> torch.Tensor:
    """Run the private first-order cached PME reciprocal energy path."""
    return _PMEReciprocalCachedFirstGrad.apply(
        positions,
        charges,
        cell,
        alpha,
        batch_idx if batch_idx is not None else None,
        k_vectors if k_vectors is not None else None,
        k_squared if k_squared is not None else None,
        volume if volume is not None else None,
        cell_inv_t if cell_inv_t is not None else None,
        moduli_x if moduli_x is not None else None,
        moduli_y if moduli_y is not None else None,
        moduli_z if moduli_z is not None else None,
        mesh_dimensions,
        spline_order,
        need_pos,
        need_charge,
        need_cell,
    )


def _normalize_cell_inv_t_cache(
    cell_inv_t: torch.Tensor | None,
) -> torch.Tensor | None:
    """Normalize optional single-system ``cell_inv_t`` cache to batched shape."""
    if cell_inv_t is not None and cell_inv_t.dim() == 2:
        return cell_inv_t.unsqueeze(0)
    return cell_inv_t


def _pme_reciprocal_space_impl(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    spline_order: int,
    batch_idx: torch.Tensor | None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    k_vectors: torch.Tensor | None = None,
    k_squared: torch.Tensor | None = None,
    volume: torch.Tensor | None = None,
    cell_inv_t: torch.Tensor | None = None,
    moduli_x: torch.Tensor | None = None,
    moduli_y: torch.Tensor | None = None,
    moduli_z: torch.Tensor | None = None,
    hybrid_forces: bool = False,
    cache_forces: bool = False,
    cache_charge_gradients: bool = False,
    cache_virial: bool = False,
    return_cell_inv_t: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
    | tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]
):
    """Internal implementation of PME reciprocal space calculation.

    Uses unified spline functions from nvalchemiops.spline for charge assignment
    and potential interpolation, and Warp kernels for Green's function and corrections.

    Supports both float32 and float64 dtypes - all operations are performed
    in the input dtype without conversion.
    """
    device = positions.device
    input_dtype = positions.dtype
    num_atoms = positions.shape[0]
    is_batch = batch_idx is not None
    fft_dims = (1, 2, 3) if is_batch else (0, 1, 2)
    volume_is_supplied = volume is not None

    if hybrid_forces:
        compute_charge_gradients = True

    alpha = _detach_setup_tensor(alpha)
    k_vectors = _detach_setup_tensor(k_vectors)
    k_squared = _detach_setup_tensor(k_squared)
    volume = _detach_setup_tensor(volume)
    cell_inv_t = _normalize_cell_inv_t_cache(_detach_setup_tensor(cell_inv_t))
    moduli_x = _detach_setup_tensor(moduli_x)
    moduli_y = _detach_setup_tensor(moduli_y)
    moduli_z = _detach_setup_tensor(moduli_z)

    if num_atoms == 0:
        energies = torch.zeros(num_atoms, device=device, dtype=input_dtype)
        forces = (
            torch.zeros(num_atoms, 3, device=device, dtype=input_dtype)
            if compute_forces
            else None
        )
        charge_grads = (
            torch.zeros(num_atoms, device=device, dtype=input_dtype)
            if compute_charge_gradients
            else None
        )
        num_systems = cell.shape[0] if is_batch else 1
        virial = (
            torch.zeros(num_systems, 3, 3, device=device, dtype=input_dtype)
            if compute_virial
            else None
        )
        if return_cell_inv_t:
            if cell_inv_t is None:
                cell_3d = cell if cell.dim() == 3 else cell.unsqueeze(0)
                cell_inv = torch.linalg.inv_ex(cell_3d)[0]
                cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
            return energies, forces, charge_grads, virial, cell_inv_t
        return energies, forces, charge_grads, virial

    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions

    # In hybrid mode, detach positions/charges/cell to sever autograd paths
    # through the spline/FFT chain. Charge gradients are attached via
    # straight-through trick after the forward pass.
    pos_spline = positions.detach() if hybrid_forces else positions
    chg_spline = charges.detach() if hybrid_forces else charges
    cell_spline = cell.detach() if hybrid_forces else cell
    if hybrid_forces and cell_inv_t is not None:
        cell_inv_t = cell_inv_t.detach()

    # Cell inverse + transpose: callers in MD loops can pass these in via the
    # cell_inv_t= kwarg to skip recomputation (typical NVT case). When provided,
    # we still need the un-transposed inverse for `reciprocal_cell`; derive it
    # back from the transpose so the caller only has to pass one tensor.
    if cell_inv_t is None:
        cell_inv = torch.linalg.inv_ex(cell_spline)[0]
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
    else:
        cell_inv = cell_inv_t.transpose(-1, -2)
    reciprocal_cell = TWOPI * cell_inv

    # Step 1: Charge assignment using unified spline_spread API
    mesh_grid = spline_spread(
        pos_spline,
        chg_spline,
        cell_spline,
        mesh_dims=(mesh_nx, mesh_ny, mesh_nz),
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t,
    )

    # Step 3: Generate k-space grid and compute Green's function + structure factor
    # Green's function: G(k) = 2*pi * exp(-k^2/(4*alpha^2)) / (V * k^2)
    # (includes 1/2 pair-counting factor; see pme_kernels.py)
    # Use precomputed k_vectors/k_squared if provided, otherwise generate them
    if k_vectors is None or k_squared is None:
        k_vectors, k_squared = generate_k_vectors_pme(
            cell_spline,
            mesh_dimensions=mesh_dimensions,
            reciprocal_cell=reciprocal_cell,
        )
    if hybrid_forces:
        k_vectors = k_vectors.detach()
        k_squared = k_squared.detach()

    alpha_gsf = alpha.detach() if hybrid_forces else alpha

    # Precomputed 1D B-spline modulus LUTs feed the fused convolve. Caller
    # can supply moduli_x/y/z to skip the fftfreq + sinc^p rebuild every
    # call (depends only on mesh + spline_order).
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
    # Volume: caller can supply via `volume=` kwarg (MD steady-state path);
    # otherwise compute from cell.
    if volume is None:
        cell_for_vol = (
            cell_spline if cell_spline.dim() == 3 else cell_spline.unsqueeze(0)
        )
        volume = torch.abs(torch.linalg.det(cell_for_vol)).to(input_dtype)
    else:
        if volume.dim() == 0:
            volume = volume.reshape(1)
        if hybrid_forces:
            volume = volume.detach()

    # FFT → fused convolve → inverse FFT. Both torch.fft.rfftn/irfftn and
    # the ``pme_fused_convolve`` custom op are fullgraph-traceable, so
    # there's no compile-vs-eager split anymore.
    #
    # cuFFT emits non-contiguous output; under torch.compile we must copy
    # to match the convolve launcher's stride contract, in eager we don't.
    mesh_fft = torch.fft.rfftn(mesh_grid, norm="backward", dim=fft_dims)
    if torch.compiler.is_compiling():
        mesh_fft = mesh_fft.contiguous()
    need_virial_output = compute_virial or cache_virial
    mesh_fft_raw = mesh_fft if need_virial_output else None
    register_pme_ops()
    convolved_mesh = torch.ops.nvalchemiops.pme_fused_convolve(
        mesh_fft,
        k_squared,
        moduli_x,
        moduli_y,
        moduli_z,
        alpha_gsf,
        volume,
        is_batch,
    )
    potential_mesh = torch.fft.irfftn(
        convolved_mesh, norm="forward", s=mesh_dimensions, dim=fft_dims
    ).to(input_dtype)

    # When forces are requested, the fused gather-with-force kernel
    # writes potential + spline-derivative force in one stencil walk.
    if compute_forces:
        raw_energies, gathered_force = spline_gather_with_force(
            pos_spline,
            chg_spline,
            potential_mesh,
            cell_spline,
            spline_order=spline_order,
            batch_idx=batch_idx,
            cell_inv_t=cell_inv_t,
        )
    else:
        raw_energies = spline_gather(
            pos_spline,
            potential_mesh,
            cell_spline,
            spline_order=spline_order,
            batch_idx=batch_idx,
            cell_inv_t=cell_inv_t,
        )
        gathered_force = None

    # Step 7: Apply corrections using Warp kernel
    # Reuse the `volume` computed above so the corrections path skips another
    # ``torch.linalg.det`` (which dispatches getrf/trsm/laswp on the 3x3 cell).
    charge_grads = None
    if compute_charge_gradients:
        reciprocal_energies, charge_grads = pme_energy_corrections_with_charge_grad(
            raw_energies,
            chg_spline,
            cell_spline,
            alpha,
            batch_idx,
            volume=volume,
        )
    else:
        reciprocal_energies = pme_energy_corrections(
            raw_energies,
            chg_spline,
            cell_spline,
            alpha,
            batch_idx,
            volume=volume,
        )
        if cache_charge_gradients:
            with torch.no_grad():
                _, charge_grads = pme_energy_corrections_with_charge_grad(
                    raw_energies.detach(),
                    chg_spline.detach(),
                    cell_spline.detach(),
                    alpha.detach(),
                    batch_idx,
                    volume=volume.detach(),
                )

    # Step 8: Compute virial before forces to allow early release of mesh_fft_raw
    # (virial needs mesh_fft_raw; forces only need convolved_mesh)
    virial = None
    if need_virial_output:
        if compute_virial:
            virial = _compute_pme_reciprocal_virial(
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
        else:
            with torch.no_grad():
                virial = _compute_pme_reciprocal_virial(
                    mesh_fft_raw=mesh_fft_raw.detach(),
                    convolved_mesh=convolved_mesh.detach(),
                    k_vectors=k_vectors.detach(),
                    k_squared=k_squared.detach(),
                    alpha=alpha.detach(),
                    mesh_dimensions=mesh_dimensions,
                    is_batch=is_batch,
                    device=device,
                    dtype=input_dtype,
                )
        del mesh_fft_raw  # Free before force field meshes are allocated

        # Background virial correction for non-neutral systems.
        # E_bg = π Q² / (2 α² V) is subtracted from energy; since
        # dE_bg/dε = -E_bg I (volume derivative), the virial contribution
        # is W_bg = -E_bg I. Single-system fans out via batch_idx=zeros.
        bg_batch_idx = (
            batch_idx
            if is_batch
            else torch.zeros(
                chg_spline.shape[0],
                dtype=torch.int32,
                device=device,
            )
        )
        register_pme_ops()
        virial = torch.ops.nvalchemiops.pme_virial_bg_correction(
            (
                chg_spline.to(input_dtype)
                if compute_virial
                else chg_spline.detach().to(input_dtype)
            ),
            bg_batch_idx,
            (
                cell_spline.to(input_dtype)
                if compute_virial
                else cell_spline.detach().to(input_dtype)
            ),
            volume.to(input_dtype),
            volume_is_supplied,
            (
                alpha.to(input_dtype)
                if compute_virial
                else alpha.detach().to(input_dtype)
            ),
            virial,
        )

    # Step 9: Forces from the fused gather above.
    # gathered_force is -q * ∇Φ in Cartesian coordinates; the 2× scaling
    # accounts for the 1/2 pair-counting factor baked into the Green's
    # function (G = 2π/(V k²) instead of 4π/(V k²)).
    forces = None
    if compute_forces:
        # 2× scaling absorbs the 1/2 pair-counting factor baked into the
        # Green's function (G = 2π/(V k²) instead of 4π/(V k²)).
        forces = 2.0 * gathered_force
    elif cache_forces:
        with torch.no_grad():
            _, cached_gathered_force = spline_gather_with_force(
                positions.detach(),
                charges.detach(),
                potential_mesh.detach(),
                cell.detach(),
                spline_order=spline_order,
                batch_idx=batch_idx,
                cell_inv_t=cell_inv_t.detach(),
            )
            forces = 2.0 * cached_gathered_force

    if hybrid_forces and charges.requires_grad:

        def _fallback(p, q, c):
            fallback_energies, _forces, _charge_grads, _virial = (
                _pme_reciprocal_space_impl(
                    p,
                    q,
                    c,
                    alpha,
                    mesh_dimensions,
                    spline_order,
                    batch_idx,
                    compute_forces=False,
                    compute_charge_gradients=False,
                    compute_virial=False,
                    k_vectors=k_vectors,
                    k_squared=k_squared,
                    volume=volume,
                    cell_inv_t=cell_inv_t,
                    moduli_x=moduli_x,
                    moduli_y=moduli_y,
                    moduli_z=moduli_z,
                    hybrid_forces=False,
                )
            )
            return fallback_energies

        reciprocal_energies = _InjectCachedEvalGradWithFallback.apply(
            reciprocal_energies,
            positions,
            charges,
            cell,
            None,
            charge_grads.detach(),
            None,
            batch_idx,
            _fallback,
        )

    if return_cell_inv_t:
        return reciprocal_energies, forces, charge_grads, virial, cell_inv_t
    return reciprocal_energies, forces, charge_grads, virial


def pme_reciprocal_space(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: float | torch.Tensor,
    mesh_dimensions: tuple[int, int, int] | None = None,
    mesh_spacing: float | None = None,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    k_vectors: torch.Tensor | None = None,
    k_squared: torch.Tensor | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    hybrid_forces: bool = False,
    *,
    cell_inv_t: torch.Tensor | None = None,
    volume: torch.Tensor | None = None,
    moduli_x: torch.Tensor | None = None,
    moduli_y: torch.Tensor | None = None,
    moduli_z: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Compute PME reciprocal-space energy and optionally forces and/or charge gradients.

    Performs the FFT-based reciprocal-space calculation using the Particle Mesh
    Ewald algorithm. This achieves O(N log N) scaling through:

    1. B-spline charge interpolation to mesh (spreading)
    2. FFT of charge mesh to reciprocal space
    3. Convolution with raw Green's function and B-spline deconvolution
    4. Inverse FFT back to real space (potential mesh)
    5. B-spline interpolation of potential to atoms (gathering)
    6. Self-energy and background corrections

    Formula
    -------
    The reciprocal-space energy is computed via the mesh potential:

    .. math::

        \\varphi_{\\text{mesh}}(k) = \\frac{G(k)}{C^2(k)} \\rho_{\\text{mesh}}(k)

    where:

    - :math:`G(k) = (2\\pi/(V k^2)) \\times \\exp(-k^2/(4\\alpha^2))` is the
      volume-normalized PME Green's function used by this implementation
    - :math:`C^2(k)` is the squared B-spline structure factor
    - :math:`\\rho_{\\text{mesh}}(k)` is the FFT of interpolated charges

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates. Supports float32 or float64 dtype.
    charges : torch.Tensor, shape (N,)
        Atomic partial charges in elementary charge units.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrices with lattice vectors as rows. Shape (3, 3) is
        automatically promoted to (1, 3, 3).
    alpha : float or torch.Tensor
        Ewald splitting parameter controlling real/reciprocal space balance.
        - float: Same α for all systems
        - Tensor shape (B,): Per-system α values
    mesh_dimensions : tuple[int, int, int], optional
        Explicit FFT mesh dimensions (nx, ny, nz). Power-of-2 values are
        optimal for FFT performance. Either mesh_dimensions or mesh_spacing
        must be provided.
    mesh_spacing : float, optional
        Target mesh spacing in same units as cell. Mesh dimensions computed as
        ceil(cell_length / mesh_spacing). Typical value: ~1 Å. This setup path
        reads cell lengths into Python integers; pass explicit
        ``mesh_dimensions`` when cell-dependent mesh sizing is not desired.
    spline_order : int, default=4
        B-spline interpolation order. Higher orders are more accurate but slower.
        - 4: Cubic B-splines (good balance, most common)
        - 5-6: Higher accuracy for demanding applications
        - Must be ≥ 3 for smooth interpolation
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index for each atom (0 to B-1). Determines kernel dispatch:
        - None: Single-system optimized kernels
        - Provided: Batched kernels for multiple independent systems
        When provided, atoms must be grouped by system: ``batch_idx`` must be
        contiguous, nondecreasing, and use system IDs ``0..B-1``.
    k_vectors : torch.Tensor, shape (nx, ny, nz//2+1, 3), optional
        Precomputed k-vectors from ``generate_k_vectors_pme``. Providing this
        along with k_squared skips k-vector generation (~15% speedup).
        Can be precomputed once and reused when cell and mesh are unchanged.
        When supplied while ``cell.requires_grad`` is true, the cache is
        assumed to correspond to the current ``cell``.
    k_squared : torch.Tensor, shape (nx, ny, nz//2+1), optional
        Precomputed :math:`|k|^2` values. Must be provided together with k_vectors.
        PME metadata tensors are setup constants and are detached from public
        autograd outputs.
    compute_forces : bool, default=False
        Whether to compute explicit component reciprocal-space forces. This
        direct output is kept for no-autograd MD/inference use; use energy
        autograd for differentiable training.
    compute_charge_gradients : bool, default=False
        Whether to compute explicit component charge gradients ∂E/∂q_i. This
        direct output follows the same no-autograd contract as
        ``compute_forces``.
    compute_virial : bool, default=False
        Whether to compute the component virial tensor
        ``W = -dE/d(displacement)`` for the row-vector displacement recipe.
        Stress = ``-virial / volume``.
    hybrid_forces : bool, default=False
        When True, positions and cell are detached from the autograd graph and
        charge gradients are attached to the energy via a straight-through
        trick.  Forces and virial are forward-only (not differentiable).
        See :func:`ewald_real_space` for details.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom reciprocal-space energy (includes self and background corrections).
    forces : torch.Tensor, shape (N, 3), optional
        Direct reciprocal-space forces. Only returned if compute_forces=True.
    charge_gradients : torch.Tensor, shape (N,), optional
        Direct charge gradients ∂E_recip/∂q_i. Only returned if compute_charge_gradients=True.
    virial : torch.Tensor, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor. Only returned if compute_virial=True. Always last in tuple.

    Note
    ----
    Internal reductions use float64 where needed for numerical stability.
    Returned energies, forces, and virials match the input dtype.
    Energy gradients are part of the public contract only for ``positions``,
    ``charges``, and ``cell``. Caller-supplied reciprocal metadata such as
    ``k_vectors``, ``k_squared``, ``volume``, and ``cell_inv_t`` is treated as
    static setup state that corresponds to the current ``cell``.

    ``torch.compile`` is supported by the public wrapper tests, although custom
    Warp operators and FFTs can still limit compiler fusion for PME workloads.

    Enabled output flags are appended in order: energies, [forces],
    [charge_gradients], [virial]. A single output is returned unwrapped;
    multiple outputs are returned as a tuple.

    Raises
    ------
    ValueError
        If neither mesh_dimensions nor mesh_spacing is provided.

    Examples
    --------
    Energy only with explicit mesh dimensions::

        >>> energies = pme_reciprocal_space(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ... )
        >>> total_recip_energy = energies.sum()

    With forces using mesh spacing::

        >>> energies, forces = pme_reciprocal_space(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_spacing=1.0,
        ...     compute_forces=True,
        ... )

    Precomputed k-vectors for MD loop (fixed cell)::

        >>> from nvalchemiops.torch.interactions.electrostatics import generate_k_vectors_pme
        >>> mesh_dims = (32, 32, 32)
        >>> k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dims)
        >>> for step in range(num_steps):
        ...     energies = pme_reciprocal_space(
        ...         positions, charges, cell,
        ...         alpha=0.3, mesh_dimensions=mesh_dims,
        ...         k_vectors=k_vectors, k_squared=k_squared,
        ...     )

    With charge gradients for ML training::

        >>> energies, charge_grads = pme_reciprocal_space(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ...     compute_charge_gradients=True,
        ... )

    See Also
    --------
    particle_mesh_ewald : Complete PME calculation (real + reciprocal).
    generate_k_vectors_pme : Generate k-vectors for this function.
    """
    component_deprecated_flags = tuple(
        name
        for name, enabled in (
            ("compute_charge_gradients", compute_charge_gradients),
            ("compute_virial", compute_virial),
            ("hybrid_forces", hybrid_forces),
        )
        if enabled
    )
    if component_deprecated_flags and not torch.compiler.is_compiling():
        warnings.warn(
            _component_direct_output_deprecation_msg(
                "pme_reciprocal_space", component_deprecated_flags
            ),
            DeprecationWarning,
            stacklevel=2,
        )

    ensure_electrostatics_ops_registered()
    cell, num_systems = _prepare_cell(cell)
    alpha_tensor = _detach_setup_tensor(
        _prepare_alpha(alpha, num_systems, torch.float64, positions.device)
    )

    # Determine mesh dimensions
    if mesh_dimensions is None:
        if mesh_spacing is None:
            raise ValueError("Either mesh_dimensions or mesh_spacing must be provided")
        cell_lengths = torch.norm(cell[0], dim=1)
        mesh_dimensions = tuple(
            int(torch.ceil(length / mesh_spacing).item()) for length in cell_lengths
        )

    k_vectors = _detach_setup_tensor(k_vectors)
    k_squared = _detach_setup_tensor(k_squared)
    volume = _detach_setup_tensor(volume)
    cell_inv_t = _normalize_cell_inv_t_cache(_detach_setup_tensor(cell_inv_t))
    moduli_x = _detach_setup_tensor(moduli_x)
    moduli_y = _detach_setup_tensor(moduli_y)
    moduli_z = _detach_setup_tensor(moduli_z)

    position_grad = bool(positions.requires_grad)
    charge_grad = bool(charges.requires_grad)
    cell_grad = bool(cell.requires_grad)
    output_grad_requested = compute_forces or compute_charge_gradients or compute_virial
    use_cached_first_grad = (
        not output_grad_requested
        and not hybrid_forces
        and not torch.compiler.is_compiling()
        and not alpha_tensor.requires_grad
        and (position_grad or charge_grad or cell_grad)
    )
    if use_cached_first_grad:
        return _pme_reciprocal_cached_first_grad(
            positions,
            charges,
            cell,
            alpha_tensor,
            mesh_dimensions,
            spline_order,
            batch_idx,
            k_vectors,
            k_squared,
            volume,
            cell_inv_t,
            moduli_x,
            moduli_y,
            moduli_z,
            need_pos=position_grad,
            need_charge=charge_grad,
            need_cell=cell_grad,
        )

    # Deprecated direct-output calls still return a differentiable energy. For
    # ordinary uniform first-order losses, consume the direct derivatives already
    # produced for those outputs instead of traversing the full spline/FFT graph.
    # Weighted losses and create_graph=True fall through to the original graph in
    # _InjectCachedEvalGrad.backward.
    need_cached_pos = position_grad and output_grad_requested and not hybrid_forces
    need_cached_charge = (
        charge_grad
        and not hybrid_forces
        and (output_grad_requested or not (position_grad or cell_grad))
    )
    need_cached_cell = cell_grad and output_grad_requested and not hybrid_forces
    need_force_cache = need_cached_pos or need_cached_cell

    # Energy is the single differentiable output. The eager graph remains present
    # for create_graph / non-uniform-cotangent cases; direct derivative caches are
    # attached below only for uniform first-order eval.
    impl_out = _pme_reciprocal_space_impl(
        positions,
        charges,
        cell,
        alpha_tensor,
        mesh_dimensions,
        spline_order,
        batch_idx,
        compute_forces=compute_forces,
        compute_charge_gradients=compute_charge_gradients,
        compute_virial=compute_virial,
        k_vectors=k_vectors,
        k_squared=k_squared,
        volume=volume,
        cell_inv_t=cell_inv_t,
        moduli_x=moduli_x,
        moduli_y=moduli_y,
        moduli_z=moduli_z,
        hybrid_forces=hybrid_forces,
        cache_forces=need_force_cache and not compute_forces,
        cache_charge_gradients=need_cached_charge and not compute_charge_gradients,
        cache_virial=need_cached_cell and not compute_virial,
        return_cell_inv_t=need_cached_cell,
    )
    if need_cached_cell:
        energies, forces, charge_grads, virial, cached_cell_inv_t = impl_out
    else:
        energies, forces, charge_grads, virial = impl_out
        cached_cell_inv_t = None

    if need_cached_pos or need_cached_charge or need_cached_cell:
        cached_dEdR = -forces.detach() if need_cached_pos else None
        cached_dEdq = charge_grads.detach() if need_cached_charge else None
        cached_dEdcell = None
        if need_cached_cell:
            dEdR_for_cell = -forces.detach()
            cached_dEdcell = _pme_cell_grad_from_virial(
                positions.detach(),
                dEdR_for_cell,
                cell.detach(),
                virial.detach(),
                batch_idx,
                cached_cell_inv_t,
            )
        energies = _InjectCachedEvalGrad.apply(
            energies,
            positions,
            charges,
            cell,
            cached_dEdR,
            cached_dEdq,
            cached_dEdcell,
            batch_idx,
        )

    # Build return tuple based on flags
    match (compute_forces, compute_charge_gradients, compute_virial):
        case (True, True, True):
            return energies, forces, charge_grads, virial
        case (True, True, False):
            return energies, forces, charge_grads
        case (True, False, True):
            return energies, forces, virial
        case (True, False, False):
            return energies, forces
        case (False, True, True):
            return energies, charge_grads, virial
        case (False, True, False):
            return energies, charge_grads
        case (False, False, True):
            return energies, virial
        case _:
            return energies


###########################################################################################
########################### Unified PME API ###############################################
###########################################################################################


def particle_mesh_ewald(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: float | torch.Tensor | None = None,
    mesh_spacing: float | None = None,
    mesh_dimensions: tuple[int, int, int] | None = None,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    k_vectors: torch.Tensor | None = None,
    k_squared: torch.Tensor | None = None,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    neighbor_shifts: torch.Tensor | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    mask_value: int | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    accuracy: float = 1e-6,
    hybrid_forces: bool = False,
    pbc: torch.Tensor | None = None,
    slab_correction: bool = False,
    *,
    cell_inv_t: torch.Tensor | None = None,
    volume: torch.Tensor | None = None,
    moduli_x: torch.Tensor | None = None,
    moduli_y: torch.Tensor | None = None,
    moduli_z: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Complete Particle Mesh Ewald (PME) calculation for long-range electrostatics.

    Computes total Coulomb energy using the PME method, which achieves :math:`O(N \\log N)`
    scaling through FFT-based reciprocal space calculations. Combines:
    1. Real-space contribution (short-range, erfc-damped)
    2. Reciprocal-space contribution (long-range, FFT + B-spline interpolation)
    3. Self-energy and background corrections

    Total Energy Formula:

    .. math::

        E_{\\text{total}} = E_{\\text{real}} + E_{\\text{reciprocal}} - E_{\\text{self}} - E_{\\text{background}}

    where:

    .. math::

        E_{\\text{real}} = \\frac{1}{2} \\sum_{i \\neq j} q_i q_j \\frac{\\text{erfc}(\\alpha r_{ij})}{r_{ij}}
        E_{\\text{reciprocal}} = FFT-based smooth long-range contribution
        E_{\\text{self}} = \\sum_i \\frac{\\alpha}{\\sqrt{\\pi}} q_i^2
        E_{\\text{background}} = \\frac{\\pi}{2\\alpha^2 V} Q_{\\text{total}}^2

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates. Supports float32 or float64 dtype.
    charges : torch.Tensor, shape (N,)
        Atomic partial charges in elementary charge units.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrices with lattice vectors as rows. Shape (3, 3) is
        automatically promoted to (1, 3, 3) for single-system mode.
    alpha : float, torch.Tensor, or None, default=None
        Ewald splitting parameter controlling real/reciprocal space balance.
        - float: Same α for all systems
        - Tensor shape (B,): Per-system α values
        - None: Automatically estimated using Kolafa-Perram formula
        Larger α shifts more computation to reciprocal space.
    mesh_spacing : float, optional
        Target mesh spacing in same units as cell (typically Å). Mesh dimensions
        computed as ceil(cell_length / mesh_spacing). Typical value: 0.8-1.2 Å.
        This setup path reads cell lengths into Python integers; pass explicit
        ``mesh_dimensions`` when cell-dependent mesh sizing is not desired.
    mesh_dimensions : tuple[int, int, int], optional
        Explicit FFT mesh dimensions (nx, ny, nz). Power-of-2 values recommended
        for optimal FFT performance. If None and mesh_spacing is None, computed
        from accuracy parameter.
    spline_order : int, default=4
        B-spline interpolation order. Higher orders are more accurate but slower.
        - 4: Cubic B-splines (standard, good accuracy/speed balance)
        - 5-6: Higher accuracy for demanding applications
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index for each atom (0 to B-1). Determines execution mode:
        - None: Single-system optimized kernels
        - Provided: Batched kernels for multiple independent systems
        When provided, atoms must be grouped by system: ``batch_idx`` must be
        contiguous, nondecreasing, and use system IDs ``0..B-1``.
    k_vectors : torch.Tensor, shape (nx, ny, nz//2+1, 3), optional
        Precomputed k-vectors from ``generate_k_vectors_pme``. Providing this
        along with k_squared skips k-vector generation (~15% speedup).
        Useful for fixed-cell MD simulations (NVT/NVE). When supplied while
        ``cell.requires_grad`` is true, the cache is assumed to correspond to
        the current ``cell``.
    k_squared : torch.Tensor, shape (nx, ny, nz//2+1), optional
        Precomputed :math:`|k|^2` values. Must be provided together with k_vectors.
    cell_inv_t : torch.Tensor, shape (3, 3) or (B, 3, 3), optional
        Precomputed transposed cell inverse :math:`(M^{-1})^T`. When supplied,
        the reciprocal-space path skips the per-call ``torch.linalg.inv`` of
        the cell (which dispatches getrf/trsm/laswp on the 3x3 cell every
        iteration). This is a setup constant for fixed-cell calls and is
        assumed to correspond to the current ``cell`` when supplied while
        ``cell.requires_grad`` is true.
    volume : torch.Tensor, shape (1,) or (B,), optional
        Precomputed cell volume :math:`|\\det(M)|`. When supplied, both the
        Green's-function normalization and the self/background correction
        skip ``torch.linalg.det`` (which also dispatches getrf under the
        hood). Same fixed-cell use-case as ``cell_inv_t``.
    moduli_x, moduli_y, moduli_z : torch.Tensor, optional
        Precomputed 1D B-spline modulus LUTs
        (``sinc(m/N)^spline_order`` per axis) from
        ``compute_bspline_moduli_1d``. When supplied, the reciprocal-space
        path skips the per-call ``fftfreq + sinc^p`` rebuild. The moduli
        only depend on mesh dimension + spline order, so callers can precompute
        them once for repeated calls with the same mesh and spline order.
    neighbor_list : torch.Tensor, shape (2, M), dtype=int32, optional
        Neighbor pairs for real-space in COO format. Row 0 = source indices,
        row 1 = target indices. Mutually exclusive with neighbor_matrix.
    neighbor_ptr : torch.Tensor, shape (N+1,), dtype=int32, optional
        CSR row pointers for neighbor_list. neighbor_ptr[i] gives the starting
        index in neighbor_list for atom i's neighbors. Required with neighbor_list.
    neighbor_shifts : torch.Tensor, shape (M, 3), dtype=int32, optional
        Periodic image shifts for neighbor_list. Required with neighbor_list.
    neighbor_matrix : torch.Tensor, shape (N, max_neighbors), dtype=int32, optional
        Dense neighbor matrix format. Entry [i, k] = j means j is k-th neighbor of i.
        Invalid entries should be set to mask_value.
        Mutually exclusive with neighbor_list.
    neighbor_matrix_shifts : torch.Tensor, shape (N, max_neighbors, 3), dtype=int32, optional
        Periodic image shifts for neighbor_matrix. Required with neighbor_matrix.
    mask_value : int, optional
        Value indicating invalid entries in neighbor_matrix. Defaults to N.
    compute_forces : bool, default=False
        Deprecated direct-output flag. Compute energy and use
        ``torch.autograd.grad`` for differentiable forces.
    compute_charge_gradients : bool, default=False
        Deprecated direct-output flag. Compute energy and use
        ``torch.autograd.grad`` for ``dE/dq_i``.
    compute_virial : bool, default=False
        Deprecated direct-output flag for the virial tensor
        ``W = -dE/d(displacement)``.
        Stress = -virial / volume.
    accuracy : float, default=1e-6
        Target relative accuracy for automatic parameter estimation (α, mesh dims).
        Only used when alpha or mesh_dimensions is None.
        Smaller values increase accuracy but also computational cost.
    hybrid_forces : bool, default=False
        When True, positions and cell are detached from the autograd graph and
        charge gradients are attached to the energy via a straight-through
        trick.  Forces and virial are forward-only (not differentiable).
        See :func:`ewald_real_space` for details.
    pbc : torch.Tensor, shape (3,) or (B, 3), optional
        Per-system periodic boundary conditions for slab correction. Required
        when ``slab_correction=True``. Each row has True for periodic
        directions and False for the non-periodic slab direction. Batched
        slab correction requires explicit shape (B, 3).
    slab_correction : bool, default=False
        Whether to add the two-dimensional Yeh-Berkowitz / Ballenegger slab
        correction to the 3D-periodic PME result. This is only available for
        the full PME interface; use :func:`compute_slab_correction` explicitly
        when manually composing ``ewald_real_space`` and ``pme_reciprocal_space``.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom contribution to total PME energy. Sum gives total energy.
    forces : torch.Tensor, shape (N, 3), optional
        Deprecated direct forces. Only returned if compute_forces=True.
    charge_gradients : torch.Tensor, shape (N,), optional
        Deprecated direct charge gradients ∂E/∂q_i. Only returned if compute_charge_gradients=True.
    virial : torch.Tensor, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor. Only returned if compute_virial=True. Always last in tuple.

    Note
    ----
    Internal reductions use float64 where needed for numerical stability.
    Returned energies, forces, and virials match the input dtype.
    Energy gradients are part of the public contract only for ``positions``,
    ``charges``, and ``cell``. Caller-supplied reciprocal metadata such as
    ``k_vectors``, ``k_squared``, ``volume``, and ``cell_inv_t`` is treated as
    static setup state that corresponds to the current ``cell``.

    Enabled output flags are appended in order: energies, [forces],
    [charge_gradients], [virial]. A single output is returned unwrapped;
    multiple outputs are returned as a tuple.

    Raises
    ------
    ValueError
        If neither neighbor_list nor neighbor_matrix is provided for real-space.
    TypeError
        If alpha has an unsupported type.

    Examples
    --------
    Automatic parameter estimation (recommended for most cases)::

        >>> energies = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=shifts,
        ...     accuracy=1e-6,
        ... )
        >>> total_energy = energies.sum()

    Explicit parameters for reproducibility::

        >>> energies = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ...     spline_order=4,
        ...     neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=shifts,
        ... )
        >>> forces = -torch.autograd.grad(energies.sum(), positions, create_graph=True)[0]

    Using mesh spacing for automatic mesh sizing::

        >>> energies = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_spacing=1.0,  # ~1 Å spacing
        ...     neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=shifts,
        ... )

    Batched systems (multiple independent structures)::

        >>> # positions: concatenated atoms from all systems
        >>> # batch_idx: [0,0,0,0, 1,1,1,1, 2,2,2,2] for 4 atoms × 3 systems
        >>> energies = particle_mesh_ewald(
        ...     positions, charges, cells,  # cells shape (3, 3, 3)
        ...     alpha=torch.tensor([0.3, 0.35, 0.3]),
        ...     batch_idx=batch_idx,
        ...     mesh_dimensions=(32, 32, 32),
        ...     neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=shifts,
        ... )

    Precomputed k-vectors for MD loop (fixed cell)::

        >>> from nvalchemiops.torch.interactions.electrostatics import generate_k_vectors_pme
        >>> mesh_dims = (32, 32, 32)
        >>> k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dims)
        >>> for step in range(num_steps):
        ...     energies = particle_mesh_ewald(
        ...         positions, charges, cell,
        ...         alpha=0.3, mesh_dimensions=mesh_dims,
        ...         k_vectors=k_vectors, k_squared=k_squared,
        ...         neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=shifts,
        ...     )

    With charge gradients for ML training::

        >>> charges.requires_grad_(True)
        >>> energies = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ...     neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=shifts,
        ... )
        >>> charge_grads = torch.autograd.grad(energies.sum(), charges, create_graph=True)[0]

    PME with slab correction::

        >>> pbc_slab = torch.tensor([[True, True, False]], device=positions.device)
        >>> energies, forces = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ...     neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=shifts,
        ...     compute_forces=True,
        ...     pbc=pbc_slab,
        ...     slab_correction=True,
        ... )

    Using PyTorch autograd::

        >>> positions.requires_grad_(True)
        >>> energies = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ...     neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=shifts,
        ... )
        >>> total_energy = energies.sum()
        >>> total_energy.backward()
        >>> autograd_forces = -positions.grad  # Should match explicit forces

    Notes
    -----
    Automatic Parameter Estimation (when alpha is None):
        Uses Kolafa-Perram formula:

    .. math::

        \\begin{aligned}
        \\eta &= \\frac{(V^2 / N)^{1/6}}{\\sqrt{2\\pi}} \\\\
        \\alpha &= \\frac{1}{2\\eta}
        \\end{aligned}

    Mesh dimensions (when mesh_dimensions is None):

    .. math::

        n_x = \\left\\lceil \\frac{2 \\alpha L_x}{3 \\varepsilon^{1/5}} \\right\\rceil

    Autograd Support:
        All inputs (positions, charges, cell) support gradient computation.

    See Also
    --------
    pme_reciprocal_space : Reciprocal-space component only
    ewald_real_space : Real-space component (used internally)
    estimate_pme_parameters : Automatic parameter estimation
    PMEParameters : Container for PME parameters
    """
    if compute_forces or compute_virial or compute_charge_gradients or hybrid_forces:
        if torch.compiler.is_compiling():
            _compiled_direct_output_deprecation_signal("particle_mesh_ewald")
        else:
            warnings.warn(
                _direct_output_deprecation_msg("particle_mesh_ewald"),
                DeprecationWarning,
                stacklevel=2,
            )

    ensure_electrostatics_ops_registered()
    num_atoms = positions.shape[0]

    # Prepare cell
    cell, num_systems = _prepare_cell(cell)

    if slab_correction:
        pbc = _prepare_pbc_for_slab(pbc, num_systems, positions.device)

    # Estimate parameters if not provided
    if alpha is None:
        params = estimate_pme_parameters(positions, cell, batch_idx, accuracy)
        alpha = params.alpha
        if mesh_dimensions is None and mesh_spacing is None:
            mesh_dimensions = tuple(params.mesh_dimensions)  # Unpack the tuple

    # Prepare alpha tensor
    alpha = _prepare_alpha(alpha, num_systems, positions.dtype, positions.device)

    if mask_value is None:
        mask_value = num_atoms

    # Determine mesh dimensions
    if mesh_dimensions is None:
        if mesh_spacing is not None:
            mesh_dimensions = mesh_spacing_to_dimensions(cell, mesh_spacing)
        else:
            # Use accuracy-based estimation
            mesh_dimensions = estimate_pme_mesh_dimensions(cell, alpha, accuracy)

    output_grad_requested = compute_forces or compute_charge_gradients or compute_virial
    differentiable_inputs = (
        positions.requires_grad or charges.requires_grad or cell.requires_grad
    )
    if (
        not output_grad_requested
        and not hybrid_forces
        and not slab_correction
        and not torch.compiler.is_compiling()
        and differentiable_inputs
        and not alpha.requires_grad
    ):
        need_pos = positions.requires_grad
        need_charge = charges.requires_grad
        need_cell = cell.requires_grad
        need_forces = need_pos or need_cell

        cached_cell_inv_t = None

        def _compute_detached_components():
            nonlocal cached_cell_inv_t
            rs_out = ewald_real_space(
                positions=positions.detach(),
                charges=charges.detach(),
                cell=cell.detach(),
                alpha=alpha.detach(),
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                mask_value=mask_value,
                batch_idx=batch_idx,
                compute_forces=need_forces,
                compute_charge_gradients=need_charge,
                compute_virial=need_cell,
            )
            rec_impl_out = _pme_reciprocal_space_impl(
                positions.detach(),
                charges.detach(),
                cell.detach(),
                alpha.detach(),
                mesh_dimensions,
                spline_order,
                batch_idx,
                compute_forces=need_forces,
                compute_charge_gradients=need_charge,
                compute_virial=need_cell,
                k_vectors=k_vectors,
                k_squared=k_squared,
                volume=volume,
                cell_inv_t=cell_inv_t,
                moduli_x=moduli_x,
                moduli_y=moduli_y,
                moduli_z=moduli_z,
                return_cell_inv_t=need_cell,
            )
            if need_cell:
                (
                    rec_energies,
                    rec_forces,
                    rec_charge_grads,
                    rec_virial,
                    cached_cell_inv_t,
                ) = rec_impl_out
            else:
                rec_energies, rec_forces, rec_charge_grads, rec_virial = rec_impl_out
            rec_out = _build_electrostatic_result(
                rec_energies,
                rec_forces,
                rec_charge_grads,
                rec_virial,
                need_forces,
                need_charge,
                need_cell,
            )
            return rs_out, rec_out

        if torch.compiler.is_compiling():
            rs, rec = _compute_detached_components()
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"The component direct-output flag\(s\).*",
                    category=DeprecationWarning,
                )
                rs, rec = _compute_detached_components()

        direct_outputs = _combine_electrostatic_outputs(
            rs,
            rec,
            None,
            need_forces,
            need_charge,
            need_cell,
        )
        energies, forces, charge_grads, virial = _unpack_electrostatic_outputs(
            direct_outputs,
            need_forces,
            need_charge,
            need_cell,
        )
        dEdR = -forces.detach() if forces is not None else None
        cached_dEdcell = None
        if need_cell:
            cached_dEdcell = _pme_cell_grad_from_virial(
                positions.detach(),
                dEdR,
                cell.detach(),
                virial.detach(),
                batch_idx,
                cached_cell_inv_t,
            )

        def _fallback(p, q, c):
            rs_energy = ewald_real_space(
                positions=p,
                charges=q,
                cell=c,
                alpha=alpha,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                mask_value=mask_value,
                batch_idx=batch_idx,
            )
            rec_energy = pme_reciprocal_space(
                positions=p,
                charges=q,
                cell=c,
                alpha=alpha,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                batch_idx=batch_idx,
                k_vectors=k_vectors,
                k_squared=k_squared,
                cell_inv_t=cell_inv_t,
                volume=volume,
                moduli_x=moduli_x,
                moduli_y=moduli_y,
                moduli_z=moduli_z,
            )
            return rs_energy + rec_energy

        return _InjectCachedEvalGradWithFallback.apply(
            energies,
            positions,
            charges,
            cell,
            dEdR if need_pos else None,
            charge_grads.detach() if charge_grads is not None else None,
            cached_dEdcell,
            batch_idx,
            _fallback,
        )

    if hybrid_forces and charges.requires_grad and not slab_correction:
        detached_charges = charges.detach()

        def _compute_hybrid_detached_components():
            rs_out = ewald_real_space(
                positions=positions,
                charges=detached_charges,
                cell=cell,
                alpha=alpha,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                mask_value=mask_value,
                batch_idx=batch_idx,
                compute_forces=compute_forces,
                compute_charge_gradients=True,
                compute_virial=compute_virial,
                hybrid_forces=True,
            )
            rec_out = pme_reciprocal_space(
                positions=positions,
                charges=detached_charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                batch_idx=batch_idx,
                compute_forces=compute_forces,
                compute_charge_gradients=True,
                compute_virial=compute_virial,
                k_vectors=k_vectors,
                k_squared=k_squared,
                cell_inv_t=cell_inv_t,
                volume=volume,
                moduli_x=moduli_x,
                moduli_y=moduli_y,
                moduli_z=moduli_z,
                hybrid_forces=True,
            )
            return rs_out, rec_out

        if torch.compiler.is_compiling():
            rs, rec = _compute_hybrid_detached_components()
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"The component direct-output flag\(s\).*",
                    category=DeprecationWarning,
                )
                rs, rec = _compute_hybrid_detached_components()

        real_energies, real_forces, real_charge_grads, real_virial = (
            _unpack_electrostatic_outputs(rs, compute_forces, True, compute_virial)
        )
        rec_energies, rec_forces, rec_charge_grads, rec_virial = (
            _unpack_electrostatic_outputs(rec, compute_forces, True, compute_virial)
        )

        energies = real_energies + rec_energies
        forces = (
            real_forces + rec_forces
            if compute_forces and real_forces is not None and rec_forces is not None
            else None
        )
        charge_grads = real_charge_grads + rec_charge_grads
        virial = (
            real_virial + rec_virial
            if compute_virial and real_virial is not None and rec_virial is not None
            else None
        )

        def _fallback(p, q, c):
            rs_energy = ewald_real_space(
                positions=p,
                charges=q,
                cell=c,
                alpha=alpha,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                mask_value=mask_value,
                batch_idx=batch_idx,
            )
            rec_energy = pme_reciprocal_space(
                positions=p,
                charges=q,
                cell=c,
                alpha=alpha,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
                batch_idx=batch_idx,
                k_vectors=k_vectors,
                k_squared=k_squared,
                cell_inv_t=cell_inv_t,
                volume=volume,
                moduli_x=moduli_x,
                moduli_y=moduli_y,
                moduli_z=moduli_z,
            )
            return rs_energy + rec_energy

        energies = _InjectCachedEvalGradWithFallback.apply(
            energies,
            positions,
            charges,
            cell,
            None,
            charge_grads.detach(),
            None,
            batch_idx,
            _fallback,
        )

        return _build_electrostatic_result(
            energies,
            forces,
            charge_grads,
            virial,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )

    def _compute_components():
        # Compute real-space contribution
        rs = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=mask_value,
            batch_idx=batch_idx,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
            hybrid_forces=hybrid_forces,
        )

        # Compute reciprocal-space contribution
        rec = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            spline_order=spline_order,
            batch_idx=batch_idx,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
            k_vectors=k_vectors,
            k_squared=k_squared,
            cell_inv_t=cell_inv_t,
            volume=volume,
            moduli_x=moduli_x,
            moduli_y=moduli_y,
            moduli_z=moduli_z,
            hybrid_forces=hybrid_forces,
        )
        return rs, rec

    suppress_component_warnings = (
        compute_charge_gradients or compute_virial or hybrid_forces
    )
    if suppress_component_warnings and not torch.compiler.is_compiling():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The component direct-output flag\(s\).*",
                category=DeprecationWarning,
            )
            rs, rec = _compute_components()
    else:
        rs, rec = _compute_components()

    slab_result: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    if slab_correction:
        if hybrid_forces:
            slab_out = _compute_slab_correction(
                positions.detach(),
                charges.detach(),
                cell.detach(),
                pbc,
                batch_idx=batch_idx,
                compute_forces=compute_forces,
                compute_charge_gradients=True,
                compute_virial=compute_virial,
            )
            slab_energies, slab_forces, slab_charge_grads, slab_virial = (
                _unpack_electrostatic_outputs(
                    slab_out,
                    compute_forces,
                    compute_charge_gradients=True,
                    compute_virial=compute_virial,
                )
            )

            if charges.requires_grad:
                slab_energies = _compute_slab_correction(
                    positions,
                    charges,
                    cell,
                    pbc,
                    batch_idx=batch_idx,
                    compute_forces=False,
                    compute_charge_gradients=False,
                    compute_virial=False,
                )
                slab_energies = _InjectChargeGrad.apply(
                    slab_energies, charges, slab_charge_grads, batch_idx
                )

            slab_result = _build_electrostatic_result(
                slab_energies,
                slab_forces,
                slab_charge_grads,
                slab_virial,
                compute_forces,
                compute_charge_gradients,
                compute_virial,
            )
        else:
            slab_result = _compute_slab_correction(
                positions,
                charges,
                cell,
                pbc,
                batch_idx=batch_idx,
                compute_forces=compute_forces,
                compute_charge_gradients=compute_charge_gradients,
                compute_virial=compute_virial,
            )

    return _combine_electrostatic_outputs(
        rs,
        rec,
        slab_result,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )


__all__ = [
    # Public APIs
    "particle_mesh_ewald",
    "pme_reciprocal_space",
    "pme_energy_corrections",
    "pme_energy_corrections_with_charge_grad",
    "pme_green_structure_factor",
    "compute_bspline_moduli_1d",
]
