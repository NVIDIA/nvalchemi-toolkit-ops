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

The output dtype convention follows ewald.py: energies in float64, forces/virial
match input precision.

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

USAGE EXAMPLES
==============

Automatic parameter estimation::

    >>> from nvalchemiops.torch.interactions.electrostatics import particle_mesh_ewald
    >>> energies, forces = particle_mesh_ewald(
    ...     positions, charges, cell,
    ...     neighbor_list=nl, neighbor_shifts=shifts,
    ...     accuracy=1e-6,  # alpha and mesh estimated automatically
    ... )

Explicit parameters::

    >>> energies, forces = particle_mesh_ewald(
    ...     positions, charges, cell,
    ...     alpha=0.3,
    ...     mesh_dimensions=(32, 32, 32),
    ...     spline_order=4,
    ...     neighbor_list=nl, neighbor_shifts=shifts,
    ... )

Batched systems::

    >>> energies, forces = particle_mesh_ewald(
    ...     positions, charges, cells,  # cells shape (B, 3, 3)
    ...     alpha=torch.tensor([0.3, 0.35]),
    ...     batch_idx=batch_idx,
    ...     mesh_dimensions=(32, 32, 32),
    ...     neighbor_list=nl, neighbor_shifts=shifts,
    ... )

Reciprocal-space only (no real-space)::

    >>> energies = pme_reciprocal_space(
    ...     positions, charges, cell,
    ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
    ... )
REFERENCES

==========

- Essmann et al. (1995). J. Chem. Phys. 103, 8577 (SPME paper)
- Darden et al. (1993). J. Chem. Phys. 98, 10089 (Original PME)
- torchpme: https://github.com/lab-cosmo/torch-pme (Reference implementation)
"""

import math

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics.pme_kernels import (
    _batch_pme_energy_corrections_with_charge_grad_kernel_overload,
    _pme_energy_corrections_with_charge_grad_kernel_overload,
)
from nvalchemiops.interactions.electrostatics.pme_kernels import (
    pme_virial_bg_correction as _pme_virial_bg_correction_warp,
)
from nvalchemiops.interactions.electrostatics.pme_kernels import (
    pme_virial_bg_correction_backward as _pme_virial_bg_correction_backward_warp,
)
from nvalchemiops.torch.interactions.electrostatics._util import (
    _build_electrostatic_result,
    _combine_electrostatic_outputs,
    _InjectChargeGrad,
    _unpack_electrostatic_outputs,
)
from nvalchemiops.torch.interactions.electrostatics._warp_op_helpers import (
    attach_simple_backward,
    register_warp_op_chain,
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
    structure factor consumed by ``_pme_convolve_kernel`` after a 1e-10
    clamp + square. Precomputing the LUT lets the convolve kernel
    replace three sinc transcendentals + an order-dependent power loop
    per (i, j, k) thread with three reads + two multiplies.
    """
    # sinc(x) for x in [-0.5, 0.5] is bounded in [2/pi, 1], so s^spline_order
    # (for orders 2-6) stays well within fp32 range. Stay in the input dtype
    # to avoid an fp32 -> fp64 -> fp32 round-trip every call.
    arg = miller_indices / float(mesh_N)
    s = torch.special.sinc(arg)
    return s**spline_order


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


register_warp_op_chain(
    name="nvalchemiops::pme_fused_convolve",
    forward=_pme_convolve_forward,
    forward_fake=_convolve_forward_fake,
    backward=_pme_convolve_backward,
    backward_fake=_convolve_backward_fake,
    backward_return_arity=4,
    # Backward outputs (grad_mesh_fft, grad_alpha, grad_volume, grad_k_squared)
    # map to forward input positions (0, 5, 6, 1).
    diff_input_positions=(0, 5, 6, 1),
    n_forward_inputs=8,
    # Non-default call ordering for the backward op (mesh_fft, grad_convolved,
    # then the rest of forward inputs) — see comment block above.
    backward_args=lambda g, f: (f[0], g[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]),
)


# Second-order autograd: convolve is LINEAR in mesh_fft, so the Jacobian of
# the first-order backward w.r.t. grad_convolved is the same forward op
# applied to the cotangent. Other partials (∂grad_*/∂{mesh_fft, alpha, volume,
# k_squared}) need a dedicated double-backward kernel if ever required.
attach_simple_backward(
    "nvalchemiops::pme_fused_convolve_backward",
    torch.ops.nvalchemiops.pme_fused_convolve,
    diff_input_positions=(1,),  # only grad_convolved (input pos 1)
    n_forward_inputs=9,
    propagate_outputs=(0,),  # only h_grad_mesh_fft flows
    backward_args=lambda g, f: (g[0], f[2], f[3], f[4], f[5], f[6], f[7], f[8]),
)


_pme_fused_convolve = torch.ops.nvalchemiops.pme_fused_convolve


###########################################################################################
########################### PME Energy Corrections Custom Ops #############################
###########################################################################################


###########################################################################################
###### Explicit Warp-backed backward chain for energy_corrections ##########################
###########################################################################################
#
# Forward kernel: ``_pme_energy_corrections_kernel`` (single) /
#                 ``_batch_pme_energy_corrections_kernel`` (batch).
# Backward kernel: ``_pme_energy_corrections_backward_kernel`` /
#                 ``_batch_pme_energy_corrections_backward_kernel``.
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


# PME energy corrections (single system): registered as a 3-op chain
# (forward / backward / double_backward), all 5 inputs differentiable.
# Default fakes auto-derive output shapes from inputs at diff positions.
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


def _pme_energy_corrections(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> torch.Tensor:
    """Internal: single-system energy corrections via the registered custom op."""
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


# Batched PME energy corrections: same chain as single-system, but batch_idx
# at position 2 (forward) / 3 (backward) is non-differentiable, and the
# per-system alpha/volume/total_charges are length-B vectors so we use
# ``batch_match=True`` to skip 0-d collapse.
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


def _batch_pme_energy_corrections(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> torch.Tensor:
    """Internal: batched energy corrections via the registered custom op."""
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

    kernel = _pme_energy_corrections_with_charge_grad_kernel_overload[wp_dtype]
    with _pme_scoped_warp_stream(raw_energies.device):
        wp.launch(
            kernel,
            dim=num_atoms,
            inputs=[wp_raw, wp_charges, wp_volume, wp_alpha, wp_qtot],
            outputs=[wp_corrected, wp_charge_grads],
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


# Forward returns (corrected, charge_gradients) in one warp kernel pass.
# charge_gradients is precomputed analytical dE/dq — non-differentiable,
# so the backward delegates to the regular pme_energy_corrections_backward
# op via propagate_outputs=(0,) (drops grad_charge_grad cotangent).
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


def _pme_energy_corrections_with_charge_grad(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: single-system fused corrections + analytical charge gradient."""
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

    kernel = _batch_pme_energy_corrections_with_charge_grad_kernel_overload[wp_dtype]
    with _pme_scoped_warp_stream(raw_energies.device):
        wp.launch(
            kernel,
            dim=num_atoms,
            inputs=[wp_raw, wp_charges, wp_bidx, wp_volumes, wp_alpha, wp_qtots],
            outputs=[wp_corrected, wp_charge_grads],
            device=device,
        )
    return corrected_energies, charge_gradients


# Batched variant of the with_charge_grad op. Same delegation pattern.
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


def _batch_pme_energy_corrections_with_charge_grad(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: batched fused corrections + analytical charge gradient."""
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
    wp_alpha = _wp_from_torch(alpha.contiguous(), dtype=wp_dtype)
    wp_total = _wp_from_torch(total_charges, dtype=wp_dtype)
    wp_virial_in = _wp_from_torch(virial_in.contiguous(), dtype=wp_dtype)
    wp_virial_out = _wp_from_torch(virial_out, dtype=wp_dtype)

    with _pme_scoped_warp_stream(virial_in.device):
        _pme_virial_bg_correction_warp(
            charges=wp_charges,
            batch_idx=wp_batch_idx,
            cell=wp_cell,
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


register_warp_op_chain(
    name="nvalchemiops::pme_virial_bg_correction",
    forward=_virial_bg_correction_forward_launch,
    backward=_virial_bg_correction_backward_launch,
    diff_input_positions=(0, 2, 3, 4),  # charges, cell, alpha, virial_in
    n_forward_inputs=5,
    forward_fake=lambda charges, batch_idx, cell, alpha, virial_in: (
        torch.empty_like(virial_in)
    ),
    batch_match=True,
)


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
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
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

    if hybrid_forces:
        compute_charge_gradients = True

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
        return energies, forces, charge_grads, virial

    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions

    # In hybrid mode, detach positions/charges/cell to sever autograd paths
    # through the spline/FFT chain. Charge gradients are attached via
    # straight-through trick after the forward pass.
    pos_spline = positions.detach() if hybrid_forces else positions
    chg_spline = charges.detach() if hybrid_forces else charges
    cell_spline = cell.detach() if hybrid_forces else cell

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

    # FFT → fused convolve → inverse FFT. Both torch.fft.rfftn/irfftn and
    # the ``pme_fused_convolve`` custom op are fullgraph-traceable, so
    # there's no compile-vs-eager split anymore.
    #
    # cuFFT emits non-contiguous output; under torch.compile we must copy
    # to match the convolve launcher's stride contract, in eager we don't.
    mesh_fft = torch.fft.rfftn(mesh_grid, norm="backward", dim=fft_dims)
    if torch.compiler.is_compiling():
        mesh_fft = mesh_fft.contiguous()
    mesh_fft_raw = mesh_fft if compute_virial else None
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

    # Step 8: Compute virial before forces to allow early release of mesh_fft_raw
    # (virial needs mesh_fft_raw; forces only need convolved_mesh)
    virial = None
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
        virial = torch.ops.nvalchemiops.pme_virial_bg_correction(
            chg_spline.to(input_dtype),
            bg_batch_idx,
            cell_spline.to(input_dtype),
            alpha.to(input_dtype),
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

    if hybrid_forces and charges.requires_grad:
        reciprocal_energies = _InjectChargeGrad.apply(
            reciprocal_energies, charges, charge_grads, batch_idx
        )

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
    volume: torch.Tensor | None = None,
    cell_inv_t: torch.Tensor | None = None,
    moduli_x: torch.Tensor | None = None,
    moduli_y: torch.Tensor | None = None,
    moduli_z: torch.Tensor | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    hybrid_forces: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Compute PME reciprocal-space energy and optionally forces and/or charge gradients.

    Performs the FFT-based reciprocal-space calculation using the Particle Mesh
    Ewald algorithm. This achieves O(N log N) scaling through:

    1. B-spline charge interpolation to mesh (spreading)
    2. FFT of charge mesh to reciprocal space
    3. Convolution with Green's function (multiply by G(k))
    4. Inverse FFT back to real space (potential mesh)
    5. B-spline interpolation of potential to atoms (gathering)
    6. Self-energy and background corrections

    Formula
    -------
    The reciprocal-space energy is computed via the mesh potential:

    .. math::

        \\varphi_{\\text{mesh}}(k) = G(k) \\times B^2(k) \\times \\rho_{\\text{mesh}}(k)

    where:

    - :math:`G(k) = (4\\pi/k^2) \\times \\exp(-k^2/(4\\alpha^2))` is the Green's function
    - :math:`B(k)` is the B-spline structure factor (interpolation correction)
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
        ceil(cell_length / mesh_spacing). Typical value: ~1 Å.
    spline_order : int, default=4
        B-spline interpolation order. Higher orders are more accurate but slower.
        - 4: Cubic B-splines (good balance, most common)
        - 5-6: Higher accuracy for demanding applications
        - Must be ≥ 3 for smooth interpolation
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index for each atom (0 to B-1). Determines kernel dispatch:
        - None: Single-system optimized kernels
        - Provided: Batched kernels for multiple independent systems
    k_vectors : torch.Tensor, shape (nx, ny, nz//2+1, 3), optional
        Precomputed k-vectors from ``generate_k_vectors_pme``. Providing this
        along with k_squared skips k-vector generation (~15% speedup).
        Can be precomputed once and reused when cell and mesh are unchanged.
    k_squared : torch.Tensor, shape (nx, ny, nz//2+1), optional
        Precomputed :math:`|k|^2` values. Must be provided together with k_vectors.
    compute_forces : bool, default=False
        Whether to compute explicit reciprocal-space forces.
    compute_charge_gradients : bool, default=False
        Whether to compute analytical charge gradients ∂E/∂q_i. Useful for
        computing charge Hessians in ML potential training.
    compute_virial : bool, default=False
        Whether to compute the virial tensor W = -dE/d(epsilon).
        Stress = virial / volume.
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
        Reciprocal-space forces. Only returned if compute_forces=True.
    charge_gradients : torch.Tensor, shape (N,), optional
        Charge gradients ∂E_recip/∂q_i. Only returned if compute_charge_gradients=True.
    virial : torch.Tensor, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor. Only returned if compute_virial=True. Always last in tuple.

    Note
    ----
    Energies are always float64 for numerical stability during accumulation.
    Forces and virial match the input dtype (float32 or float64).

    The FFT-heavy reciprocal-space block currently runs through a narrow eager
    helper on compiled paths because TorchInductor does not yet lower the
    required complex FFT algebra reliably for PME.

    Return Patterns
    ---------------
    Enabled flags are appended in order: energies, [forces], [charge_gradients], [virial].
    A single output is returned unwrapped; multiple outputs as a tuple.

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
    cell, num_systems = _prepare_cell(cell)
    alpha_tensor = _prepare_alpha(alpha, num_systems, torch.float64, positions.device)

    # Determine mesh dimensions
    if mesh_dimensions is None:
        if mesh_spacing is None:
            raise ValueError("Either mesh_dimensions or mesh_spacing must be provided")
        cell_lengths = torch.norm(cell[0], dim=1)
        mesh_dimensions = tuple(
            int(torch.ceil(length / mesh_spacing).item()) for length in cell_lengths
        )

    energies, forces, charge_grads, virial = _pme_reciprocal_space_impl(
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
    cell_inv_t: torch.Tensor | None = None,
    volume: torch.Tensor | None = None,
    moduli_x: torch.Tensor | None = None,
    moduli_y: torch.Tensor | None = None,
    moduli_z: torch.Tensor | None = None,
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

        E_{\\text{real}} = \\frac{1}{2} \\sum_{i \\neq j} q_i q_j \\frac{\\text{erfc}(\\alpha r_{ij}/\\sqrt{2})}{r_{ij}}
        E_{\\text{reciprocal}} = FFT-based smooth long-range contribution
        E_{\\text{self}} = \\sum_i \\frac{\\alpha}{\\sqrt{2\\pi}} q_i^2
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
    k_vectors : torch.Tensor, shape (nx, ny, nz//2+1, 3), optional
        Precomputed k-vectors from ``generate_k_vectors_pme``. Providing this
        along with k_squared skips k-vector generation (~15% speedup).
        Useful for fixed-cell MD simulations (NVT/NVE).
    k_squared : torch.Tensor, shape (nx, ny, nz//2+1), optional
        Precomputed :math:`|k|^2` values. Must be provided together with k_vectors.
    cell_inv_t : torch.Tensor, shape (3, 3) or (B, 3, 3), optional
        Precomputed transposed cell inverse :math:`(M^{-1})^T`. When supplied,
        the reciprocal-space path skips the per-call ``torch.linalg.inv`` of
        the cell (which dispatches getrf/trsm/laswp on the 3x3 cell every
        iteration). Fixed-cell MD (NVT/NVE) and MLIP training callers should
        compute this once outside the loop and pass it through.
    volume : torch.Tensor, shape (1,) or (B,), optional
        Precomputed cell volume :math:`|\\det(M)|`. When supplied, both the
        Green's-function normalization and the self/background correction
        skip ``torch.linalg.det`` (which also dispatches getrf under the
        hood). Same use-case as ``cell_inv_t``.
    moduli_x, moduli_y, moduli_z : torch.Tensor, optional
        Precomputed 1D B-spline modulus LUTs
        (``sinc(m/N)^spline_order`` per axis) from
        ``compute_bspline_moduli_1d``. When supplied, the reciprocal-space
        path skips the per-call ``fftfreq + sinc^p`` rebuild. The moduli
        only depend on mesh dimension + spline order so fixed-cell MD /
        MLIP training callers should precompute once.
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
        Whether to compute explicit analytical forces.
    compute_charge_gradients : bool, default=False
        Whether to compute analytical charge gradients ∂E/∂q_i. Useful for
        training ML potentials that require second derivatives (charge Hessians).
    compute_virial : bool, default=False
        Whether to compute the virial tensor W = -dE/d(epsilon).
        Stress = virial / volume.
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
        Forces on each atom. Only returned if compute_forces=True.
    charge_gradients : torch.Tensor, shape (N,), optional
        Charge gradients ∂E/∂q_i. Only returned if compute_charge_gradients=True.
    virial : torch.Tensor, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor. Only returned if compute_virial=True. Always last in tuple.

    Note
    ----
    Energies are always float64 for numerical stability during accumulation.
    Forces and virial match the input dtype (float32 or float64).

    Return Patterns
    ---------------
    Enabled flags are appended in order: energies, [forces], [charge_gradients], [virial].
    A single output is returned unwrapped; multiple outputs as a tuple.

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
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ...     neighbor_ptr=nptr, accuracy=1e-6,
        ... )
        >>> total_energy = energies.sum()

    Explicit parameters for reproducibility::

        >>> energies, forces = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ...     spline_order=4, neighbor_list=nl,
        ...     neighbor_shifts=shifts, neighbor_ptr=nptr,
        ...     compute_forces=True,
        ... )

    Using mesh spacing for automatic mesh sizing::

        >>> energies, forces = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_spacing=1.0,  # ~1 Å spacing
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ...     neighbor_ptr=nptr, compute_forces=True,
        ... )

    Batched systems (multiple independent structures)::

        >>> # positions: concatenated atoms from all systems
        >>> # batch_idx: [0,0,0,0, 1,1,1,1, 2,2,2,2] for 4 atoms × 3 systems
        >>> energies, forces = particle_mesh_ewald(
        ...     positions, charges, cells,  # cells shape (3, 3, 3)
        ...     alpha=torch.tensor([0.3, 0.35, 0.3]),
        ...     batch_idx=batch_idx,
        ...     mesh_dimensions=(32, 32, 32),
        ...     neighbor_list=nl,
        ...     neighbor_shifts=shifts, neighbor_ptr=nptr,
        ...     compute_forces=True,
        ... )

    Precomputed k-vectors for MD loop (fixed cell)::

        >>> from nvalchemiops.torch.interactions.electrostatics import generate_k_vectors_pme
        >>> mesh_dims = (32, 32, 32)
        >>> k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dims)
        >>> for step in range(num_steps):
        ...     energies, forces = particle_mesh_ewald(
        ...         positions, charges, cell,
        ...         alpha=0.3, mesh_dimensions=mesh_dims,
        ...         k_vectors=k_vectors, k_squared=k_squared,
        ...         neighbor_list=nl, neighbor_shifts=shifts,
        ...         neighbor_ptr=nptr,
        ...         compute_forces=True,
        ...     )

    With charge gradients for ML training::

        >>> energies, forces, charge_grads = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ...     neighbor_ptr=nptr,
        ...     compute_forces=True, compute_charge_gradients=True,
        ... )
        >>> # Use charge_grads for training on ∂E/∂q

    PME with slab correction::

        >>> pbc_slab = torch.tensor([[True, True, False]], device=positions.device)
        >>> energies, forces = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ...     neighbor_ptr=nptr,
        ...     compute_forces=True,
        ...     pbc=pbc_slab,
        ...     slab_correction=True,
        ... )

    Using PyTorch autograd::

        >>> positions.requires_grad_(True)
        >>> energies = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ...     neighbor_ptr=nptr,
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
]
