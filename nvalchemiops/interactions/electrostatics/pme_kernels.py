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
Unified PME Kernels
===================

This module provides GPU-accelerated Warp kernels for Particle Mesh Ewald (PME)
calculations, specifically for Green's function and energy corrections.
Charge assignment and force interpolation are handled by the spline module.

MATHEMATICAL FORMULATION
========================

PME splits the Coulomb energy into components:

.. math::

    E_{\\text{total}} = E_{\\text{real}} + E_{\\text{reciprocal}} - E_{\\text{self}} - E_{\\text{background}}

This module provides kernels for:

1. Green's Function and Structure Factor Correction:

.. math::

    G(k) = \\frac{2\\pi}{V} \\frac{\\exp(-k^2/(4\\alpha^2))}{k^2}

The B-spline charge assignment introduces aliasing, corrected by:

.. math::

    C(k) = \\left[\\text{sinc}(k_x/N_x) \\cdot \\text{sinc}(k_y/N_y) \\cdot \\text{sinc}(k_z/N_z)\\right]^{-2p}

where p is the spline order.

2. Energy Corrections:

   - Self-energy: :math:`E_{\\text{self}} = \\frac{\\alpha}{\\sqrt{\\pi}} \\sum_i q_i^2`
   - Background (for non-neutral systems): :math:`E_{\\text{background}} = \\frac{\\pi}{2\\alpha^2 V} \\sum_i q_i Q_{\\text{total}}`

DTYPE FLEXIBILITY
=================

All kernels support both float32 and float64 inputs via wp.Any type annotations
and explicit overloads. Use the overload dictionaries (e.g.,
_pme_green_structure_factor_kernel_overload) to select the appropriate kernel
based on input dtype.

KERNEL ORGANIZATION
===================

Green's Function Kernels:
    _pme_green_structure_factor_kernel: Single-system G(k) and C(k)
    _batch_pme_green_structure_factor_kernel: Batched version

Energy Correction Kernels:
    _pme_energy_corrections_kernel: Single-system self + background correction
    _batch_pme_energy_corrections_kernel: Batched version

.. warning
    In contrast to the other electrostatic kernels that offer end-to-end
    ``warp`` launchers, PME requires FFT for the convolution step that is
    currently not available in ``warp``. As a result, bindings must call
    FFT within their own framework in between kernel launches. The sequence
    of calls looks like the following:

    1. Spread charges to mesh: ``spline_spread()``
    2. Forward FFT: ``framework.fft.rfftn(mesh)``
    3. Compute Green's function: ``pme_green_structure_factor()``
    4. Convolution: ``mesh_fft * green_function / structure_factor_sq``
    5. Inverse FFT: ``framework.fft.irfftn(...)``
    6. Gather potential: ``spline_gather()``
    7. Apply corrections: ``pme_energy_corrections()``

REFERENCES
==========

- Essmann et al. (1995). J. Chem. Phys. 103, 8577 (SPME paper)
- Darden et al. (1993). J. Chem. Phys. 98, 10089 (Original PME)
- torchpme: https://github.com/lab-cosmo/torch-pme (Reference implementation)
"""

import math
from typing import Any

import warp as wp

# Mathematical constants
PI = math.pi
TWOPI = 2.0 * PI
FOURPI = 4.0 * PI


###########################################################################################
########################### Helper Functions ##############################################
###########################################################################################


@wp.func
def compute_sinc(x: Any) -> Any:
    """Compute normalized sinc function: :math:`\\sin(\\pi x)/(\\pi x)`.

    Uses Taylor expansion near zero for numerical stability.
    """
    abs_x = wp.abs(x)
    one = type(x)(1.0)
    threshold = type(x)(1e-6)

    if abs_x < threshold:
        return one

    pi_x = type(x)(PI) * x
    return wp.sin(pi_x) / pi_x


@wp.func
def wp_exp_kernel(k_sq: Any, prefactor: Any) -> Any:
    """Compute exp(-prefactor * k_sq) / k_sq."""
    return wp.exp(-prefactor * k_sq) / k_sq


###########################################################################################
########################### Green Function with Structure Factor ##########################
###########################################################################################


@wp.kernel
def _pme_green_structure_factor_kernel(
    k_squared: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft)
    miller_x: wp.array(dtype=Any),  # (Nx,)
    miller_y: wp.array(dtype=Any),  # (Ny,)
    miller_z: wp.array(dtype=Any),  # (Nz_rfft,)
    alpha: wp.array(dtype=Any),  # (1,)
    volume: wp.array(dtype=Any),  # (1,)
    mesh_nx: wp.int32,
    mesh_ny: wp.int32,
    mesh_nz: wp.int32,
    spline_order: wp.int32,
    green_function: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft)
    structure_factor_sq: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft)
):
    """Compute PME Green's function and B-spline structure factor correction.

    Computes two arrays needed for PME reciprocal space:
    1. Green's function: G(k) = (2π/V) * exp(-k²/(4α²)) / k²
    2. Structure factor squared: :math:`|B(k)|^2` for B-spline dealiasing

    The structure factor correction accounts for aliasing from B-spline
    charge spreading: C(k) = [sinc(h/N_x) * sinc(k/N_y) * sinc(l/N_z)]^(2p)

    Launch Grid
    -----------
    dim = [Nx, Ny, Nz_rfft]

    Each thread processes one grid point in the FFT mesh (using rfft symmetry).

    Parameters
    ----------
    k_squared : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        Squared magnitude of k-vectors at each grid point.
    miller_x : wp.array, shape (Nx,), dtype=wp.float32 or wp.float64
        Miller indices in x direction (from fftfreq).
    miller_y : wp.array, shape (Ny,), dtype=wp.float32 or wp.float64
        Miller indices in y direction (from fftfreq).
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32 or wp.float64
        Miller indices in z direction (from rfftfreq).
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Unit cell volume.
    mesh_nx, mesh_ny, mesh_nz : wp.int32
        Full mesh dimensions (Nz is the full size, not rfft size).
    spline_order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended.
    green_function : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: Green's function G(k) at each grid point.
    structure_factor_sq : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: :math:`|B(k)|^2` structure factor squared at each grid point.

    Notes
    -----
    - k=0 (grid point [0,0,0]) is explicitly set to zero (tin-foil boundary conditions).
    - Near-zero k² values are set to zero to avoid division by zero.
    - Structure factor is clamped to avoid division by zero in dealiasing.
    - Uses rfft symmetry: only Nz_rfft = Nz//2 + 1 points in z.
    """
    i, j, k = wp.tid()

    k_sq = k_squared[i, j, k]
    alpha_ = alpha[0]
    volume_ = volume[0]
    mi_x = miller_x[i]
    mi_y = miller_y[j]
    mi_z = miller_z[k]

    # Get dtype-specific constants
    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    four = type(k_sq)(4.0)

    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(TWOPI)

    # Structure factor: sinc(mi_x/Nx) * sinc(mi_y/Ny) * sinc(mi_z/Nz)
    # We compute sf^2 here and fold it into Green's function below, so that
    # the FFT pipeline's per-element complex/real "mesh_fft / structure_factor_sq"
    # step (~18 ms over 10 iters at single_128k) goes away.
    sinc_x = compute_sinc(mi_x / type(mi_x)(mesh_nx))
    sinc_y = compute_sinc(mi_y / type(mi_y)(mesh_ny))
    sinc_z = compute_sinc(mi_z / type(mi_z)(mesh_nz))

    sinc_product = sinc_x * sinc_y * sinc_z

    # Raise to spline_order power. The loop runs up to 5 extra multiplies
    # so we cover spline_order in [1, 6]. The inner `_ < spline_order` guard
    # stops at the correct power for each supported order.
    sf = sinc_product
    for _ in range(1, 6):  # supports spline_order in [1, 6]
        if _ < spline_order:
            sf = sf * sinc_product

    # Clamp to avoid division by zero
    if sf < clamp_threshold:
        sf = clamp_threshold

    sf_sq = sf * sf
    structure_factor_sq[i, j, k] = sf_sq

    # Effective Green's function: G(k) / B^2(k). Folding the deconvolution
    # into G saves one full-mesh complex/real elementwise op per PME call.
    if k_sq < threshold:
        green_function[i, j, k] = zero
    else:
        exp_factor = wp_exp_kernel(k_sq, one / (four * alpha_ * alpha_))
        green_function[i, j, k] = twopi * exp_factor / (volume_ * sf_sq)

    if i == 0 and j == 0 and k == 0:
        green_function[i, j, k] = zero


@wp.kernel
def _batch_pme_green_structure_factor_kernel(
    k_squared: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft)
    miller_x: wp.array(dtype=Any),  # (Nx,)
    miller_y: wp.array(dtype=Any),  # (Ny,)
    miller_z: wp.array(dtype=Any),  # (Nz_rfft,)
    alpha: wp.array(dtype=Any),  # (B,)
    volumes: wp.array(dtype=Any),  # (B,)
    mesh_nx: wp.int32,
    mesh_ny: wp.int32,
    mesh_nz: wp.int32,
    spline_order: wp.int32,
    green_function: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft)
    structure_factor_sq: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft)
):
    """Compute PME Green's function and B-spline structure factor for batched systems.

    Batched version of _pme_green_structure_factor_kernel. Each system can have
    different alpha and volume values, but shares the same mesh dimensions.

    Green's function: G_s(k) = (2π/V_s) * exp(-k²/(4α_s²)) / k²
    Structure factor: :math:`|B(k)|^2` (computed once, shared across systems)

    Launch Grid
    -----------
    dim = [B, Nx, Ny, Nz_rfft]

    Each thread processes one (system, grid_point) pair.

    Parameters
    ----------
    k_squared : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        Per-system squared magnitude of k-vectors at each grid point.
    miller_x : wp.array, shape (Nx,), dtype=wp.float32 or wp.float64
        Miller indices in x direction (shared across systems).
    miller_y : wp.array, shape (Ny,), dtype=wp.float32 or wp.float64
        Miller indices in y direction (shared across systems).
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32 or wp.float64
        Miller indices in z direction (shared across systems).
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system unit cell volume.
    mesh_nx, mesh_ny, mesh_nz : wp.int32
        Full mesh dimensions (Nz is the full size, not rfft size).
    spline_order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended.
    green_function : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system Green's function G_s(k) at each grid point.
    structure_factor_sq : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: :math:`|B(k)|^2` structure factor squared (computed only at batch_idx=0).

    Notes
    -----
    - k=0 (grid point [0,0,0]) is explicitly set to zero for each system.
    - Near-zero k² values are set to zero to avoid division by zero.
    - Structure factor is computed only once (at batch_idx=0) since it depends
      only on mesh dimensions and spline order, not on system parameters.
    - Uses rfft symmetry: only Nz_rfft = Nz//2 + 1 points in z.
    """
    batch_idx, i, j, k = wp.tid()

    k_sq = k_squared[batch_idx, i, j, k]
    system_alpha = alpha[batch_idx]
    system_volume = volumes[batch_idx]
    mi_x = miller_x[i]
    mi_y = miller_y[j]
    mi_z = miller_z[k]

    # Get dtype-specific constants
    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    four = type(k_sq)(4.0)
    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(TWOPI)

    # Structure factor sf^2 folded into the per-system Green's function
    # below. Written once at batch_idx=0 for external consumers.
    sinc_x = compute_sinc(mi_x / type(mi_x)(mesh_nx))
    sinc_y = compute_sinc(mi_y / type(mi_y)(mesh_ny))
    sinc_z = compute_sinc(mi_z / type(mi_z)(mesh_nz))

    sinc_product = sinc_x * sinc_y * sinc_z
    sf = sinc_product
    for _ in range(1, 6):
        if _ < spline_order:
            sf = sf * sinc_product

    if sf < clamp_threshold:
        sf = clamp_threshold

    sf_sq = sf * sf
    if batch_idx == wp.int32(0):
        structure_factor_sq[i, j, k] = sf_sq

    # Effective Green's function: G_s(k) / B^2(k).
    if k_sq < threshold:
        green_function[batch_idx, i, j, k] = zero
    else:
        exp_factor = wp_exp_kernel(k_sq, one / (four * system_alpha * system_alpha))
        green_function[batch_idx, i, j, k] = (
            twopi * exp_factor / (system_volume * sf_sq)
        )

    if i == 0 and j == 0 and k == 0:
        green_function[batch_idx, i, j, k] = zero


###########################################################################################
########################### PME Fused Convolution #########################################
###########################################################################################
#
# Fused Green's factor + B-spline deconvolution + multiply in one warp
# kernel, replacing the two-step Green-then-multiply path. Coulomb-PME
# Green's formula is hardcoded — pluggable via @wp.func is a follow-up.


@wp.kernel
def _pme_convolve_kernel(
    mesh_fft: wp.array3d(dtype=Any),  # input, complex as vec2 (nx, ny, nz_r)
    k_squared: wp.array3d(dtype=Any),
    moduli_x: wp.array(dtype=Any),  # 1D B-spline modulus LUT: sinc(m/N)^order
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    convolved_mesh: wp.array3d(dtype=Any),  # output, complex as vec2
):
    """Fused Green's function compute + structure-factor deconvolution + multiply.

    For each k-point (i, j, k), computes:
        factor = (2π / V) * exp(-k² / (4α²)) / (k² * B²(k))
        convolved_mesh[i,j,k] = mesh_fft[i,j,k] * factor    (complex × real)

    B²(k) is the squared 3D B-spline structure factor. P-C precomputes the
    1D moduli ``b_x[i] = sinc(mi/Nx)^order`` (etc.) on the torch side and
    passes them in as LUTs so the kernel does three reads + two multiplies
    instead of three sinc transcendentals + an order-dependent power loop
    per (i, j, k) thread.

    k=0 is explicitly zeroed (tin-foil boundary).
    """
    i, j, k = wp.tid()

    k_sq = k_squared[i, j, k]
    alpha_ = alpha[0]
    volume_ = volume[0]

    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    four = type(k_sq)(4.0)
    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(TWOPI)

    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    # Effective Green's factor: G(k) / B²(k). Zero at k=0 (tin-foil BC).
    if k_sq < threshold:
        factor = zero
    else:
        exp_factor = wp_exp_kernel(k_sq, one / (four * alpha_ * alpha_))
        factor = twopi * exp_factor / (volume_ * sf_sq)
    if i == 0 and j == 0 and k == 0:
        factor = zero

    # Complex (vec2) × real multiply, written directly to the output mesh.
    c = mesh_fft[i, j, k]
    convolved_mesh[i, j, k] = type(c)(c[0] * factor, c[1] * factor)


@wp.kernel
def _batch_pme_convolve_kernel(
    mesh_fft: wp.array4d(dtype=Any),  # (B, nx, ny, nz_r), complex as vec2
    k_squared: wp.array4d(dtype=Any),
    moduli_x: wp.array(dtype=Any),
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),  # (B,)
    volumes: wp.array(dtype=Any),  # (B,)
    convolved_mesh: wp.array4d(dtype=Any),  # (B, nx, ny, nz_r), complex as vec2
):
    """Batched version of ``_pme_convolve_kernel``. Each thread handles one
    ``(batch_idx, i, j, k)``; alpha and volume are per-system, B-spline
    structure factor is shared across the batch (single LUT per axis)."""
    batch_idx, i, j, k = wp.tid()

    k_sq = k_squared[batch_idx, i, j, k]
    system_alpha = alpha[batch_idx]
    system_volume = volumes[batch_idx]

    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    four = type(k_sq)(4.0)
    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(TWOPI)

    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    if k_sq < threshold:
        factor = zero
    else:
        exp_factor = wp_exp_kernel(k_sq, one / (four * system_alpha * system_alpha))
        factor = twopi * exp_factor / (system_volume * sf_sq)
    if i == 0 and j == 0 and k == 0:
        factor = zero

    c = mesh_fft[batch_idx, i, j, k]
    convolved_mesh[batch_idx, i, j, k] = type(c)(c[0] * factor, c[1] * factor)


###########################################################################################
########################### PME Fused Convolve Backward ###################################
###########################################################################################
#
# Backward kernel reduces grad_alpha and grad_volume via atomic_add over (i,j,k).
# Per-grid contribution: 2·Re(g·conj(m̂)) · {G·k²/(2α³), -G/V}.


@wp.kernel
def _pme_convolve_backward_kernel(
    mesh_fft: wp.array3d(dtype=Any),  # input (vec2): saved from forward
    grad_convolved: wp.array3d(dtype=Any),  # input (vec2): cotangent
    k_squared: wp.array3d(dtype=Any),
    moduli_x: wp.array(dtype=Any),
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    grad_mesh_fft: wp.array3d(dtype=Any),  # output (vec2)
    grad_alpha: wp.array(dtype=Any),  # output scalar (shape (1,))
    grad_volume: wp.array(dtype=Any),  # output scalar (shape (1,))
    grad_k_squared: wp.array3d(dtype=Any),  # output (mesh_nx, mesh_ny, mesh_nz_r)
):
    """Single-system backward for the fused PME convolve.

    Writes ``grad_mesh_fft[i,j,k] = G(k) * grad_convolved[i,j,k]`` and
    ``grad_k_squared[i,j,k] = re_inner * dG/dk²`` elementwise; atomically
    accumulates the scalar gradients ``grad_alpha`` and ``grad_volume``
    per k-point. ``grad_k_squared`` chains back through the
    ``generate_k_vectors_pme`` op to ``cell.grad`` and is required for
    correct cell gradients (k² depends on cell via the reciprocal lattice).
    """
    i, j, k = wp.tid()

    k_sq = k_squared[i, j, k]
    alpha_ = alpha[0]
    volume_ = volume[0]

    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    two = type(k_sq)(2.0)
    four = type(k_sq)(4.0)
    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(TWOPI)

    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    if k_sq < threshold:
        factor = zero
    else:
        exp_factor = wp_exp_kernel(k_sq, one / (four * alpha_ * alpha_))
        factor = twopi * exp_factor / (volume_ * sf_sq)
    if i == 0 and j == 0 and k == 0:
        factor = zero

    # Cotangent and saved mesh_fft.
    g = grad_convolved[i, j, k]
    m = mesh_fft[i, j, k]

    # grad_mesh_fft = G * grad_convolved  (same kernel as forward — G is real).
    grad_mesh_fft[i, j, k] = type(g)(g[0] * factor, g[1] * factor)

    # Real inner product of (grad_conv) and conj(mesh_fft): g.re*m.re + g.im*m.im.
    # NB: no Wirtinger factor of 2 here — torch.fft.rfftn's autograd already
    # accumulates the conjugate-pair contribution into ``grad_convolved`` at
    # rfftn-output indices, so a plain sum-over-rfftn-half gives the correct
    # gradient (verified by FD against `volume.requires_grad_(True)`).
    re_inner = g[0] * m[0] + g[1] * m[1]
    contrib = re_inner * factor

    if factor > zero:
        # dG/dα = G(k) · k² / (2α³)
        d_alpha = contrib * k_sq / (two * alpha_ * alpha_ * alpha_)
        wp.atomic_add(grad_alpha, 0, d_alpha)
        # dG/dV = -G / V
        d_vol = -contrib / volume_
        wp.atomic_add(grad_volume, 0, d_vol)
        # dG/d(k²) = -G · (1/(4α²) + 1/k²)
        grad_k_squared[i, j, k] = -contrib * (
            one / (four * alpha_ * alpha_) + one / k_sq
        )
    else:
        grad_k_squared[i, j, k] = zero


@wp.kernel
def _batch_pme_convolve_backward_kernel(
    mesh_fft: wp.array4d(dtype=Any),  # (B, nx, ny, nz_r)
    grad_convolved: wp.array4d(dtype=Any),  # (B, nx, ny, nz_r)
    k_squared: wp.array4d(dtype=Any),
    moduli_x: wp.array(dtype=Any),
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),  # (B,)
    volumes: wp.array(dtype=Any),  # (B,)
    grad_mesh_fft: wp.array4d(dtype=Any),  # (B, nx, ny, nz_r)
    grad_alpha: wp.array(dtype=Any),  # (B,)
    grad_volume: wp.array(dtype=Any),  # (B,)
    grad_k_squared: wp.array4d(dtype=Any),  # (B, nx, ny, nz_r)
):
    """Batched backward for ``_batch_pme_convolve_kernel``. Per-system
    ``grad_alpha[batch_idx]`` and ``grad_volume[batch_idx]`` are
    atomically accumulated across k-points; ``grad_mesh_fft`` and
    ``grad_k_squared`` are written elementwise."""
    batch_idx, i, j, k = wp.tid()

    k_sq = k_squared[batch_idx, i, j, k]
    system_alpha = alpha[batch_idx]
    system_volume = volumes[batch_idx]

    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    two = type(k_sq)(2.0)
    four = type(k_sq)(4.0)
    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(TWOPI)

    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    if k_sq < threshold:
        factor = zero
    else:
        exp_factor = wp_exp_kernel(k_sq, one / (four * system_alpha * system_alpha))
        factor = twopi * exp_factor / (system_volume * sf_sq)
    if i == 0 and j == 0 and k == 0:
        factor = zero

    g = grad_convolved[batch_idx, i, j, k]
    m = mesh_fft[batch_idx, i, j, k]
    grad_mesh_fft[batch_idx, i, j, k] = type(g)(g[0] * factor, g[1] * factor)

    # See note in single-system kernel: no Wirtinger 2x — rfftn autograd already
    # accumulates conjugate-pair contribution into grad_convolved.
    re_inner = g[0] * m[0] + g[1] * m[1]
    contrib = re_inner * factor
    if factor > zero:
        d_alpha = contrib * k_sq / (two * system_alpha * system_alpha * system_alpha)
        wp.atomic_add(grad_alpha, batch_idx, d_alpha)
        d_vol = -contrib / system_volume
        wp.atomic_add(grad_volume, batch_idx, d_vol)
        grad_k_squared[batch_idx, i, j, k] = -contrib * (
            one / (four * system_alpha * system_alpha) + one / k_sq
        )
    else:
        grad_k_squared[batch_idx, i, j, k] = zero


###########################################################################################
########################### PME Energy Corrections ########################################
###########################################################################################


@wp.kernel
def _pme_energy_corrections_kernel(
    raw_energies: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    total_charge: wp.array(dtype=Any),
    corrected_energies: wp.array(dtype=Any),
):
    """Apply self-energy and background corrections to PME energies.

    Converts raw potential values (φ_i) to corrected per-atom energies by:
    1. Multiplying potential by charge: E_pot = q_i * φ_i
    2. Subtracting self-energy: E_self = (α/√π) * q_i²
    3. Subtracting background: E_bg = (π/(2α²V)) * q_i * Q_total

    Final: E_i = q_i * φ_i - (α/√π) * q_i² - (π/(2α²V)) * q_i * Q_total

    Launch Grid
    -----------
    dim = [num_atoms]

    Each thread processes one atom independently.

    Parameters
    ----------
    raw_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Raw potential values φ_i from mesh interpolation.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Unit cell volume.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    total_charge : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Sum of all charges (Q_total = ∑_i q_i).
    corrected_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Corrected per-atom energies.

    Notes
    -----
    - Self-energy removes spurious interaction of each Gaussian with itself.
    - Background correction accounts for uniform neutralizing background.
    - For neutral systems (Q_total = 0), background correction is zero.
    """
    atom_idx = wp.tid()

    charge = charges[atom_idx]
    raw_energy = raw_energies[atom_idx]
    alpha_ = alpha[0]
    total_charge_ = total_charge[0]
    volume_ = volume[0]

    # Get dtype-specific constants
    pi = type(charge)(PI)
    two = type(charge)(2.0)

    # Convert potential to energy: E = q * phi, where phi = raw_energy
    potential_energy = charge * raw_energy

    # Self-energy correction: -q^2 * alpha / sqrt(pi)
    self_contrib = charge * charge * alpha_ / wp.sqrt(pi)

    # Background correction: -q * pi * Q_tot / (2*alpha^2 * V)
    background_contrib = charge * pi * total_charge_ / (two * alpha_ * alpha_ * volume_)

    # Final corrected energy per atom
    corrected_energies[atom_idx] = potential_energy - self_contrib - background_contrib


@wp.kernel
def _pme_energy_corrections_with_charge_grad_kernel(
    raw_energies: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    total_charge: wp.array(dtype=Any),
    corrected_energies: wp.array(dtype=Any),
    charge_gradients: wp.array(dtype=Any),
):
    """Apply corrections and compute charge gradients for PME energies.

    Computes both corrected energies and analytical charge gradients in a single pass:

    Energy: E_i = q_i * φ_i - (α/√π) * q_i² - (π/(2α²V)) * q_i * Q_total

    Charge gradient: ∂E_total/∂q_i = 2*φ_i - 2*(α/√π)*q_i - (π/(α²V))*Q_total

    The factor of 2 on φ_i arises because changing q_i affects:
    1. The direct term: ∂(q_i * φ_i)/∂q_i = φ_i
    2. All potentials: ∑_j q_j * ∂φ_j/∂q_i = φ_i (since ∂φ_j/∂q_i = φ_i/q_i)

    Total: 2*φ_i

    Launch Grid
    -----------
    dim = [num_atoms]

    Each thread processes one atom independently.

    Parameters
    ----------
    raw_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Raw potential values φ_i from mesh interpolation.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Unit cell volume.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    total_charge : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Sum of all charges (Q_total = ∑_i q_i).
    corrected_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Corrected per-atom energies.
    charge_gradients : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Analytical charge gradients ∂E_total/∂q_i.

    Notes
    -----
    - Charge gradients are useful for second-derivative training in ML potentials.
    - Combines energy and charge gradient computation for efficiency.
    - Self-energy and background corrections are applied to both outputs.
    """
    atom_idx = wp.tid()

    charge = charges[atom_idx]
    raw_energy = raw_energies[atom_idx]  # This is φ_i (the potential)
    alpha_ = alpha[0]
    total_charge_ = total_charge[0]
    volume_ = volume[0]

    # Get dtype-specific constants
    pi = type(charge)(PI)
    two = type(charge)(2.0)

    # === Energy calculation ===
    # Convert potential to energy: E = q * φ
    potential_energy = charge * raw_energy

    # Self-energy correction: -q² * α / √π
    self_contrib = charge * charge * alpha_ / wp.sqrt(pi)

    # Background correction: -q * π * Q_tot / (2α² * V)
    background_contrib = charge * pi * total_charge_ / (two * alpha_ * alpha_ * volume_)

    corrected_energies[atom_idx] = potential_energy - self_contrib - background_contrib

    # === Charge gradient calculation ===
    # ∂E/∂q_i = 2*φ_i - 2*(α/√π)*q_i - (π/(α²V))*Q_total
    # The 2*φ_i factor accounts for both direct contribution and induced potential changes
    self_energy_grad = two * alpha_ * charge / wp.sqrt(pi)
    background_grad = pi * total_charge_ / (alpha_ * alpha_ * volume_)

    charge_gradients[atom_idx] = two * raw_energy - self_energy_grad - background_grad


@wp.kernel
def _batch_pme_energy_corrections_kernel(
    raw_energies: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    volumes: wp.array(dtype=Any),  # (B,)
    alpha: wp.array(dtype=Any),  # (B,)
    total_charges: wp.array(dtype=Any),  # (B,)
    corrected_energies: wp.array(dtype=Any),
):
    """Apply self-energy and background corrections for batched PME.

    Batched version of _pme_energy_corrections_kernel. Each atom looks up its
    system's parameters (volume, alpha, total_charge) via batch_idx.

    Final: E_i = q_i * φ_i - (α_s/√π) * q_i² - (π/(2α_s²V_s)) * q_i * Q_s

    where s = batch_idx[i] is the system index for atom i.

    Launch Grid
    -----------
    dim = [num_atoms_total]

    Each thread processes one atom independently.

    Parameters
    ----------
    raw_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Raw potential values φ_i from mesh interpolation.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system unit cell volume.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    total_charges : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system sum of charges (Q_s = ∑_{i∈s} q_i).
    corrected_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Corrected per-atom energies.

    Notes
    -----
    - Each system can have different alpha, volume, and total charge.
    - Atoms are assigned to systems via batch_idx array.
    """
    atom_idx = wp.tid()

    system_id = batch_idx[atom_idx]
    charge = charges[atom_idx]
    raw_energy = raw_energies[atom_idx]
    volume = volumes[system_id]
    system_alpha = alpha[system_id]
    total_charge = total_charges[system_id]

    # Get dtype-specific constants
    pi = type(charge)(PI)
    two = type(charge)(2.0)

    # Convert potential to energy: E = q * phi, where phi = raw_energy
    potential_energy = charge * raw_energy

    # Self-energy correction: -q^2 * alpha / sqrt(pi)
    self_contrib = charge * charge * system_alpha / wp.sqrt(pi)

    # Background correction: -q * pi * Q_tot / (2*alpha^2 * V)
    background_contrib = (
        charge * pi * total_charge / (two * system_alpha * system_alpha * volume)
    )

    # Final corrected energy per atom
    corrected_energies[atom_idx] = potential_energy - self_contrib - background_contrib


@wp.kernel
def _batch_pme_energy_corrections_with_charge_grad_kernel(
    raw_energies: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    volumes: wp.array(dtype=Any),  # (B,)
    alpha: wp.array(dtype=Any),  # (B,)
    total_charges: wp.array(dtype=Any),  # (B,)
    corrected_energies: wp.array(dtype=Any),
    charge_gradients: wp.array(dtype=Any),
):
    """Apply corrections and compute charge gradients for batched PME.

    Batched version of _pme_energy_corrections_with_charge_grad_kernel.

    Computes both corrected energies and analytical charge gradients:

    Energy: E_i = q_i * φ_i - (α_s/√π) * q_i² - (π/(2α_s²V_s)) * q_i * Q_s

    Charge gradient: ∂E_total/∂q_i = 2*φ_i - 2*(α_s/√π)*q_i - (π/(α_s²V_s))*Q_s

    where s = batch_idx[i] is the system index for atom i.

    Launch Grid
    -----------
    dim = [num_atoms_total]

    Each thread processes one atom independently.

    Parameters
    ----------
    raw_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Raw potential values φ_i from mesh interpolation.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system unit cell volume.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    total_charges : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system sum of charges (Q_s = ∑_{i∈s} q_i).
    corrected_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Corrected per-atom energies.
    charge_gradients : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Analytical charge gradients ∂E_total/∂q_i.

    Notes
    -----
    - Each system can have different alpha, volume, and total charge.
    - Atoms are assigned to systems via batch_idx array.
    - Charge gradients are useful for second-derivative training in ML potentials.
    """
    atom_idx = wp.tid()

    system_id = batch_idx[atom_idx]
    charge = charges[atom_idx]
    raw_energy = raw_energies[atom_idx]  # This is φ_i (the potential)
    volume = volumes[system_id]
    system_alpha = alpha[system_id]
    total_charge = total_charges[system_id]

    # Get dtype-specific constants
    pi = type(charge)(PI)
    two = type(charge)(2.0)

    # === Energy calculation ===
    # Convert potential to energy: E = q * φ
    potential_energy = charge * raw_energy

    # Self-energy correction: -q² * α / √π
    self_contrib = charge * charge * system_alpha / wp.sqrt(pi)

    # Background correction: -q * π * Q_tot / (2α² * V)
    background_contrib = (
        charge * pi * total_charge / (two * system_alpha * system_alpha * volume)
    )

    corrected_energies[atom_idx] = potential_energy - self_contrib - background_contrib

    # === Charge gradient calculation ===
    # ∂E/∂q_i = 2*φ_i - 2*(α/√π)*q_i - (π/(α²V))*Q_total
    # The 2*φ_i factor accounts for both direct contribution and induced potential changes
    self_energy_grad = two * system_alpha * charge / wp.sqrt(pi)
    background_grad = pi * total_charge / (system_alpha * system_alpha * volume)

    charge_gradients[atom_idx] = two * raw_energy - self_energy_grad - background_grad


###########################################################################################
########################### Energy-Corrections Backward Kernels ############################
###########################################################################################
#
# Per-atom energy correction:
#   E_i = q_i · raw_i - (α/√π) · q_i² - (π/(2α²V)) · q_i · Q_total
# Backward computes the partials w.r.t. raw_i, q_i, alpha, V, Q_total;
# the three scalars are reduced via atomic_add per atom.


@wp.kernel
def _pme_energy_corrections_backward_kernel(
    grad_E: wp.array(dtype=Any),  # (N,) cotangent of corrected_energies
    raw_energies: wp.array(dtype=Any),  # (N,) saved forward input
    charges: wp.array(dtype=Any),  # (N,)
    volume: wp.array(dtype=Any),  # (1,)
    alpha: wp.array(dtype=Any),  # (1,)
    total_charge: wp.array(dtype=Any),  # (1,)
    grad_raw: wp.array(dtype=Any),  # (N,) output
    grad_charges: wp.array(dtype=Any),  # (N,) output
    grad_volume: wp.array(dtype=Any),  # (1,) output — atomic accumulation
    grad_alpha: wp.array(dtype=Any),  # (1,) output — atomic accumulation
    grad_total_charge: wp.array(dtype=Any),  # (1,) output — atomic accumulation
):
    """Single-system backward for the per-atom energy corrections kernel.

    All three scalar gradients (volume, alpha, total_charge) must be
    zero-initialized by the caller — they are accumulated atomically across
    atoms.
    """
    i = wp.tid()
    g = grad_E[i]
    q = charges[i]
    r = raw_energies[i]
    a = alpha[0]
    v = volume[0]
    qtot = total_charge[0]

    pi = type(g)(PI)
    two = type(g)(2.0)
    sqrt_pi = wp.sqrt(pi)
    c1 = a / sqrt_pi
    c2 = pi / (two * a * a * v)

    # grad_raw[i] = g · q_i  (elementwise)
    grad_raw[i] = g * q

    # grad_q[i] = g · (r_i - 2·c1·q_i - c2·Q_total)  (elementwise)
    grad_charges[i] = g * (r - two * c1 * q - c2 * qtot)

    # Scalar grads — atomic_add into shape-(1,) arrays.
    # dE_i/dα = -q²/√π + π·q·Q_total/(α³·V)
    d_alpha = g * (-(q * q) / sqrt_pi + pi * q * qtot / (a * a * a * v))
    wp.atomic_add(grad_alpha, 0, d_alpha)

    # dE_i/dV = π·q·Q_total / (2·α²·V²)
    d_volume = g * pi * q * qtot / (two * a * a * v * v)
    wp.atomic_add(grad_volume, 0, d_volume)

    # dE_i/dQ_total = -π·q / (2·α²·V) = -c2 · q
    d_qtot = -g * c2 * q
    wp.atomic_add(grad_total_charge, 0, d_qtot)


@wp.kernel
def _batch_pme_energy_corrections_backward_kernel(
    grad_E: wp.array(dtype=Any),  # (N_total,)
    raw_energies: wp.array(dtype=Any),  # (N_total,)
    charges: wp.array(dtype=Any),  # (N_total,)
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    volumes: wp.array(dtype=Any),  # (B,)
    alpha: wp.array(dtype=Any),  # (B,)
    total_charges: wp.array(dtype=Any),  # (B,)
    grad_raw: wp.array(dtype=Any),  # (N_total,)
    grad_charges: wp.array(dtype=Any),  # (N_total,)
    grad_volumes: wp.array(dtype=Any),  # (B,) — atomic per-system
    grad_alpha: wp.array(dtype=Any),  # (B,) — atomic per-system
    grad_total_charges: wp.array(dtype=Any),  # (B,) — atomic per-system
):
    """Batched backward for the per-atom energy corrections kernel.

    Per-system scalar grads (volume, alpha, total_charge) accumulate
    atomically into length-B arrays keyed by ``batch_idx``.
    """
    i = wp.tid()
    s = batch_idx[i]
    g = grad_E[i]
    q = charges[i]
    r = raw_energies[i]
    a = alpha[s]
    v = volumes[s]
    qtot = total_charges[s]

    pi = type(g)(PI)
    two = type(g)(2.0)
    sqrt_pi = wp.sqrt(pi)
    c1 = a / sqrt_pi
    c2 = pi / (two * a * a * v)

    grad_raw[i] = g * q
    grad_charges[i] = g * (r - two * c1 * q - c2 * qtot)

    d_alpha = g * (-(q * q) / sqrt_pi + pi * q * qtot / (a * a * a * v))
    wp.atomic_add(grad_alpha, s, d_alpha)

    d_volume = g * pi * q * qtot / (two * a * a * v * v)
    wp.atomic_add(grad_volumes, s, d_volume)

    d_qtot = -g * c2 * q
    wp.atomic_add(grad_total_charges, s, d_qtot)


###########################################################################################
########################### Energy-Corrections Double-Backward #############################
###########################################################################################
#
# Forward:  E_i = q_i·r_i - C1·q_i² - C2·q_i·Qtot   (C1=α/√π, C2=π/(2α²V))
# 1st bwd:  partials in q, α, V, Qtot of E_i; 2nd bwd follows from the
# linear + quadratic terms of the forward formula.


@wp.kernel
def _pme_energy_corrections_double_backward_kernel(
    # Upstream cotangents for outputs of the first backward:
    h_raw: wp.array(dtype=Any),  # (N,) cotangent of g_raw
    h_chg: wp.array(dtype=Any),  # (N,) cotangent of g_chg
    h_vol: wp.array(dtype=Any),  # (1,) cotangent of g_V
    h_alpha: wp.array(dtype=Any),  # (1,) cotangent of g_α
    h_qtot: wp.array(dtype=Any),  # (1,) cotangent of g_Qtot
    # Saved inputs from first backward:
    grad_E: wp.array(dtype=Any),  # (N,) — g_i
    raw_energies: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    total_charge: wp.array(dtype=Any),
    # Outputs:
    grad_grad_E: wp.array(dtype=Any),  # (N,)
    grad_raw: wp.array(dtype=Any),  # (N,)
    grad_charges: wp.array(dtype=Any),  # (N,)
    grad_volume: wp.array(dtype=Any),  # (1,) — atomic_add
    grad_alpha: wp.array(dtype=Any),  # (1,) — atomic_add
    grad_total_charge: wp.array(dtype=Any),  # (1,) — atomic_add
):
    """Single-system double-backward for energy corrections.

    All scalar grads must be zero-initialized by the caller.
    """
    i = wp.tid()
    g_i = grad_E[i]
    q = charges[i]
    r = raw_energies[i]
    a = alpha[0]
    v = volume[0]
    qtot = total_charge[0]
    hr = h_raw[i]
    hc = h_chg[i]
    hv = h_vol[0]
    ha = h_alpha[0]
    hq = h_qtot[0]

    pi = type(g_i)(PI)
    two = type(g_i)(2.0)
    three = type(g_i)(3.0)
    sqrt_pi = wp.sqrt(pi)
    c1 = a / sqrt_pi
    c2 = pi / (two * a * a * v)
    A_i = -(q * q) / sqrt_pi + pi * q * qtot / (a * a * a * v)
    B_i = pi * q * qtot / (two * a * a * v * v)
    D_i = -pi * q / (two * a * a * v)

    # ∂L/∂(grad_E[i]) — per atom
    grad_grad_E[i] = (
        hr * q + hc * (r - two * c1 * q - c2 * qtot) + ha * A_i + hv * B_i + hq * D_i
    )

    # ∂L/∂(raw[i]) — per atom (only g_chg depends on r)
    grad_raw[i] = hc * g_i

    # ∂L/∂(q[i]) — per atom; collects all q[i]-dependent contributions
    dq = g_i * (
        hr
        + hc * (-two * c1)
        + ha * (-two * q / sqrt_pi + pi * qtot / (a * a * a * v))
        + hv * (pi * qtot / (two * a * a * v * v))
        + hq * (-pi / (two * a * a * v))
    )
    grad_charges[i] = dq

    # Per-atom contributions to scalar grads (atomic_add).
    g_q = g_i * q  # appears repeatedly in the scalar grad rows.

    # ∂L/∂V
    dV_atom = (
        hc * g_i * qtot * pi / (two * a * a * v * v)
        + ha * (-pi * qtot / (a * a * a * v * v)) * g_q
        + hv * (-pi * qtot / (a * a * v * v * v)) * g_q
        + hq * (pi / (two * a * a * v * v)) * g_q
    )
    wp.atomic_add(grad_volume, 0, dV_atom)

    # ∂L/∂α
    dA_atom = (
        hc * g_i * (-two * q / sqrt_pi + pi * qtot / (a * a * a * v))
        + ha * (-three * pi * qtot / (a * a * a * a * v)) * g_q
        + hv * (-pi * qtot / (a * a * a * v * v)) * g_q
        + hq * (pi / (a * a * a * v)) * g_q
    )
    wp.atomic_add(grad_alpha, 0, dA_atom)

    # ∂L/∂Qtot
    dQ_atom = (
        hc * g_i * (-pi / (two * a * a * v))
        + ha * (pi / (a * a * a * v)) * g_q
        + hv * (pi / (two * a * a * v * v)) * g_q
        # hq term is 0 since ∂D_i/∂Qtot = 0
    )
    wp.atomic_add(grad_total_charge, 0, dQ_atom)


@wp.kernel
def _batch_pme_energy_corrections_double_backward_kernel(
    h_raw: wp.array(dtype=Any),
    h_chg: wp.array(dtype=Any),
    h_vol: wp.array(dtype=Any),
    h_alpha: wp.array(dtype=Any),
    h_qtot: wp.array(dtype=Any),
    grad_E: wp.array(dtype=Any),
    raw_energies: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    volumes: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    total_charges: wp.array(dtype=Any),
    grad_grad_E: wp.array(dtype=Any),
    grad_raw: wp.array(dtype=Any),
    grad_charges: wp.array(dtype=Any),
    grad_volumes: wp.array(dtype=Any),
    grad_alpha: wp.array(dtype=Any),
    grad_total_charges: wp.array(dtype=Any),
):
    """Batched double-backward for energy corrections."""
    i = wp.tid()
    s = batch_idx[i]
    g_i = grad_E[i]
    q = charges[i]
    r = raw_energies[i]
    a = alpha[s]
    v = volumes[s]
    qtot = total_charges[s]
    hr = h_raw[i]
    hc = h_chg[i]
    hv = h_vol[s]
    ha = h_alpha[s]
    hq = h_qtot[s]

    pi = type(g_i)(PI)
    two = type(g_i)(2.0)
    three = type(g_i)(3.0)
    sqrt_pi = wp.sqrt(pi)
    c1 = a / sqrt_pi
    c2 = pi / (two * a * a * v)
    A_i = -(q * q) / sqrt_pi + pi * q * qtot / (a * a * a * v)
    B_i = pi * q * qtot / (two * a * a * v * v)
    D_i = -pi * q / (two * a * a * v)

    grad_grad_E[i] = (
        hr * q + hc * (r - two * c1 * q - c2 * qtot) + ha * A_i + hv * B_i + hq * D_i
    )

    grad_raw[i] = hc * g_i

    dq = g_i * (
        hr
        + hc * (-two * c1)
        + ha * (-two * q / sqrt_pi + pi * qtot / (a * a * a * v))
        + hv * (pi * qtot / (two * a * a * v * v))
        + hq * (-pi / (two * a * a * v))
    )
    grad_charges[i] = dq

    g_q = g_i * q
    dV_atom = (
        hc * g_i * qtot * pi / (two * a * a * v * v)
        + ha * (-pi * qtot / (a * a * a * v * v)) * g_q
        + hv * (-pi * qtot / (a * a * v * v * v)) * g_q
        + hq * (pi / (two * a * a * v * v)) * g_q
    )
    wp.atomic_add(grad_volumes, s, dV_atom)

    dA_atom = (
        hc * g_i * (-two * q / sqrt_pi + pi * qtot / (a * a * a * v))
        + ha * (-three * pi * qtot / (a * a * a * a * v)) * g_q
        + hv * (-pi * qtot / (a * a * a * v * v)) * g_q
        + hq * (pi / (a * a * a * v)) * g_q
    )
    wp.atomic_add(grad_alpha, s, dA_atom)

    dQ_atom = (
        hc * g_i * (-pi / (two * a * a * v))
        + ha * (pi / (a * a * a * v)) * g_q
        + hv * (pi / (two * a * a * v * v)) * g_q
    )
    wp.atomic_add(grad_total_charges, s, dQ_atom)


###########################################################################################
########################### PME Virial Background Correction ##############################
###########################################################################################
#
# Non-neutral PME systems have a background charge term in the energy:
#     E_bg = (π · Q² ) / (2 α² V)
# whose volume derivative gives a diagonal contribution to the virial:
#     W_bg = -d E_bg / dε = -(E_bg) · I    (where ε is the strain tensor)
# We subtract ``E_bg · I`` from the virial diagonal to apply that correction.
#
# Pipeline: pass 1 scatter-adds per-atom q into total_charges[batch_idx];
# pass 2 (per system) computes E_bg = π Q² / (2 α² V) and subtracts from
# the virial diagonal.


@wp.kernel(enable_backward=False)
def _pme_virial_bg_reduce_kernel(
    charges: wp.array(dtype=Any),  # (N,)
    batch_idx: wp.array(dtype=wp.int32),  # (N,) — system index per atom
    total_charges: wp.array(dtype=Any),  # (B,) — IN/OUT, zero-initialized by caller
):
    """Pass 1: scatter-add per-atom charges into ``total_charges[batch_idx]``."""
    atom_idx = wp.tid()
    s = batch_idx[atom_idx]
    wp.atomic_add(total_charges, s, charges[atom_idx])


@wp.kernel(enable_backward=False)
def _pme_virial_bg_apply_kernel(
    total_charges: wp.array(dtype=Any),  # (B,) computed in pass 1
    cell: wp.array3d(dtype=Any),  # (B, 3, 3)
    alpha: wp.array(dtype=Any),  # (B,) — per-system Ewald splitting
    virial_in: wp.array3d(dtype=Any),  # (B, 3, 3) input
    virial_out: wp.array3d(dtype=Any),  # (B, 3, 3) output = virial_in - E_bg·I
):
    """Pass 2: compute V = |det(cell[s])|, E_bg, subtract from virial diagonal."""
    s = wp.tid()

    q = total_charges[s]
    a = alpha[s]
    pi = type(q)(PI)
    two = type(q)(2.0)

    c00 = cell[s, 0, 0]
    c01 = cell[s, 0, 1]
    c02 = cell[s, 0, 2]
    c10 = cell[s, 1, 0]
    c11 = cell[s, 1, 1]
    c12 = cell[s, 1, 2]
    c20 = cell[s, 2, 0]
    c21 = cell[s, 2, 1]
    c22 = cell[s, 2, 2]
    det = (
        c00 * (c11 * c22 - c12 * c21)
        - c01 * (c10 * c22 - c12 * c20)
        + c02 * (c10 * c21 - c11 * c20)
    )
    volume = wp.abs(det)

    e_bg = pi * q * q / (two * a * a * volume)

    virial_out[s, 0, 0] = virial_in[s, 0, 0] - e_bg
    virial_out[s, 0, 1] = virial_in[s, 0, 1]
    virial_out[s, 0, 2] = virial_in[s, 0, 2]
    virial_out[s, 1, 0] = virial_in[s, 1, 0]
    virial_out[s, 1, 1] = virial_in[s, 1, 1] - e_bg
    virial_out[s, 1, 2] = virial_in[s, 1, 2]
    virial_out[s, 2, 0] = virial_in[s, 2, 0]
    virial_out[s, 2, 1] = virial_in[s, 2, 1]
    virial_out[s, 2, 2] = virial_in[s, 2, 2] - e_bg


# Analytic backward kernel — see launcher for the math.
@wp.kernel(enable_backward=False)
def _pme_virial_bg_backward_per_system_kernel(
    grad_virial: wp.array3d(dtype=Any),  # (B, 3, 3) cotangent of virial_out
    total_charges: wp.array(dtype=Any),  # (B,) recomputed from charges
    cell: wp.array3d(dtype=Any),  # (B, 3, 3)
    alpha: wp.array(dtype=Any),  # (B,)
    grad_total_charges: wp.array(dtype=Any),  # (B,) OUT — dL/dQ per system
    grad_alpha: wp.array(dtype=Any),  # (B,) OUT — dL/dα per system
    grad_cell: wp.array3d(dtype=Any),  # (B, 3, 3) OUT — dL/dC
):
    """Per-system: turn the cotangent of virial_out into per-system dL/dQ, dL/dα, dL/dC.

    From ``virial_out[s,i,j] = virial_in[s,i,j] - δ_ij · E_bg(s)`` (where
    ``E_bg = π Q² / (2 α² V)`` and ``V = |det(C)|``):
      dL/dE_bg(s) = -(g[s,0,0] + g[s,1,1] + g[s,2,2])
      dE_bg/dQ    =  π Q / (α² V)
      dE_bg/dα    = -π Q² / (α³ V)
      dE_bg/dV    = -π Q² / (2 α² V²)
      d|det C|/dC = sign(det C) · cofactor(C)   (Jacobi's formula)
    """
    s = wp.tid()

    q = total_charges[s]
    a = alpha[s]
    pi = type(q)(PI)
    two = type(q)(2.0)

    c00 = cell[s, 0, 0]
    c01 = cell[s, 0, 1]
    c02 = cell[s, 0, 2]
    c10 = cell[s, 1, 0]
    c11 = cell[s, 1, 1]
    c12 = cell[s, 1, 2]
    c20 = cell[s, 2, 0]
    c21 = cell[s, 2, 1]
    c22 = cell[s, 2, 2]
    det = (
        c00 * (c11 * c22 - c12 * c21)
        - c01 * (c10 * c22 - c12 * c20)
        + c02 * (c10 * c21 - c11 * c20)
    )
    volume = wp.abs(det)
    sgn = wp.sign(det)

    g_diag_sum = grad_virial[s, 0, 0] + grad_virial[s, 1, 1] + grad_virial[s, 2, 2]
    g_E_bg = -g_diag_sum  # dL/dE_bg

    a2 = a * a
    a3 = a2 * a
    v2 = volume * volume

    dE_dQ = pi * q / (a2 * volume)
    dE_dA = -pi * q * q / (a3 * volume)
    dE_dV = -pi * q * q / (two * a2 * v2)

    grad_total_charges[s] = g_E_bg * dE_dQ
    grad_alpha[s] = g_E_bg * dE_dA

    dV_dC00 = sgn * (c11 * c22 - c12 * c21)
    dV_dC01 = sgn * -(c10 * c22 - c12 * c20)
    dV_dC02 = sgn * (c10 * c21 - c11 * c20)
    dV_dC10 = sgn * -(c01 * c22 - c02 * c21)
    dV_dC11 = sgn * (c00 * c22 - c02 * c20)
    dV_dC12 = sgn * -(c00 * c21 - c01 * c20)
    dV_dC20 = sgn * (c01 * c12 - c02 * c11)
    dV_dC21 = sgn * -(c00 * c12 - c02 * c10)
    dV_dC22 = sgn * (c00 * c11 - c01 * c10)

    gV = g_E_bg * dE_dV
    grad_cell[s, 0, 0] = gV * dV_dC00
    grad_cell[s, 0, 1] = gV * dV_dC01
    grad_cell[s, 0, 2] = gV * dV_dC02
    grad_cell[s, 1, 0] = gV * dV_dC10
    grad_cell[s, 1, 1] = gV * dV_dC11
    grad_cell[s, 1, 2] = gV * dV_dC12
    grad_cell[s, 2, 0] = gV * dV_dC20
    grad_cell[s, 2, 1] = gV * dV_dC21
    grad_cell[s, 2, 2] = gV * dV_dC22


@wp.kernel(enable_backward=False)
def _pme_virial_bg_backward_per_atom_kernel(
    batch_idx: wp.array(dtype=wp.int32),  # (N,)
    grad_total_charges: wp.array(dtype=Any),  # (B,) per-system dL/dQ
    grad_charges: wp.array(dtype=Any),  # (N,) OUT — dL/dq_j = dL/dQ(s(j))
):
    """Per-atom: dL/dq_j = dL/dQ(s(j))."""
    j = wp.tid()
    s = batch_idx[j]
    grad_charges[j] = grad_total_charges[s]


###########################################################################################
########################### Kernel Overloads for Dtype Flexibility ########################
###########################################################################################

# Type lists for creating overloads
_T = [wp.float32, wp.float64]
# Complex-as-vec2 type per dtype (rfftn output is complex64 for float32 input,
# complex128 for float64 input). We pass these to Warp via torch.view_as_real.
_C = {wp.float32: wp.vec2f, wp.float64: wp.vec2d}

# Single-system kernel overloads
_pme_green_structure_factor_kernel_overload = {}
_pme_convolve_kernel_overload = {}
_pme_convolve_backward_kernel_overload = {}
_pme_energy_corrections_kernel_overload = {}
_pme_energy_corrections_backward_kernel_overload = {}
_pme_energy_corrections_double_backward_kernel_overload = {}
_pme_energy_corrections_with_charge_grad_kernel_overload = {}
_pme_virial_bg_reduce_kernel_overload = {}
_pme_virial_bg_apply_kernel_overload = {}
_pme_virial_bg_backward_per_system_kernel_overload = {}
_pme_virial_bg_backward_per_atom_kernel_overload = {}

# Batch kernel overloads
_batch_pme_green_structure_factor_kernel_overload = {}
_batch_pme_convolve_kernel_overload = {}
_batch_pme_convolve_backward_kernel_overload = {}
_batch_pme_energy_corrections_kernel_overload = {}
_batch_pme_energy_corrections_backward_kernel_overload = {}
_batch_pme_energy_corrections_double_backward_kernel_overload = {}
_batch_pme_energy_corrections_with_charge_grad_kernel_overload = {}

for t in _T:
    # Green's function kernel overloads
    _pme_green_structure_factor_kernel_overload[t] = wp.overload(
        _pme_green_structure_factor_kernel,
        [
            wp.array3d(dtype=t),  # k_squared
            wp.array(dtype=t),  # miller_x
            wp.array(dtype=t),  # miller_y
            wp.array(dtype=t),  # miller_z
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # volume
            wp.int32,  # mesh_nx
            wp.int32,  # mesh_ny
            wp.int32,  # mesh_nz
            wp.int32,  # spline_order
            wp.array3d(dtype=t),  # green_function
            wp.array3d(dtype=t),  # structure_factor_sq
        ],
    )

    _batch_pme_green_structure_factor_kernel_overload[t] = wp.overload(
        _batch_pme_green_structure_factor_kernel,
        [
            wp.array4d(dtype=t),  # k_squared
            wp.array(dtype=t),  # miller_x
            wp.array(dtype=t),  # miller_y
            wp.array(dtype=t),  # miller_z
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # volumes
            wp.int32,  # mesh_nx
            wp.int32,  # mesh_ny
            wp.int32,  # mesh_nz
            wp.int32,  # spline_order
            wp.array4d(dtype=t),  # green_function
            wp.array3d(dtype=t),  # structure_factor_sq
        ],
    )

    # Fused convolution kernel (uses precomputed 1D B-spline moduli).
    _pme_convolve_kernel_overload[t] = wp.overload(
        _pme_convolve_kernel,
        [
            wp.array3d(dtype=_C[t]),  # mesh_fft (complex as vec2)
            wp.array3d(dtype=t),  # k_squared
            wp.array(dtype=t),  # moduli_x
            wp.array(dtype=t),  # moduli_y
            wp.array(dtype=t),  # moduli_z
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # volume
            wp.array3d(dtype=_C[t]),  # convolved_mesh (complex as vec2)
        ],
    )

    _batch_pme_convolve_kernel_overload[t] = wp.overload(
        _batch_pme_convolve_kernel,
        [
            wp.array4d(dtype=_C[t]),  # mesh_fft
            wp.array4d(dtype=t),  # k_squared
            wp.array(dtype=t),  # moduli_x
            wp.array(dtype=t),  # moduli_y
            wp.array(dtype=t),  # moduli_z
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # volumes
            wp.array4d(dtype=_C[t]),  # convolved_mesh
        ],
    )

    # Fused convolve backward (uses precomputed 1D B-spline moduli).
    _pme_convolve_backward_kernel_overload[t] = wp.overload(
        _pme_convolve_backward_kernel,
        [
            wp.array3d(dtype=_C[t]),  # mesh_fft
            wp.array3d(dtype=_C[t]),  # grad_convolved
            wp.array3d(dtype=t),  # k_squared
            wp.array(dtype=t),  # moduli_x
            wp.array(dtype=t),  # moduli_y
            wp.array(dtype=t),  # moduli_z
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # volume
            wp.array3d(dtype=_C[t]),  # grad_mesh_fft
            wp.array(dtype=t),  # grad_alpha (1,)
            wp.array(dtype=t),  # grad_volume (1,)
            wp.array3d(dtype=t),  # grad_k_squared
        ],
    )
    _batch_pme_convolve_backward_kernel_overload[t] = wp.overload(
        _batch_pme_convolve_backward_kernel,
        [
            wp.array4d(dtype=_C[t]),  # mesh_fft
            wp.array4d(dtype=_C[t]),  # grad_convolved
            wp.array4d(dtype=t),  # k_squared
            wp.array(dtype=t),  # moduli_x
            wp.array(dtype=t),  # moduli_y
            wp.array(dtype=t),  # moduli_z
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # volumes
            wp.array4d(dtype=_C[t]),  # grad_mesh_fft
            wp.array(dtype=t),  # grad_alpha (B,)
            wp.array(dtype=t),  # grad_volumes (B,)
            wp.array4d(dtype=t),  # grad_k_squared (B, nx, ny, nz_r)
        ],
    )

    # Energy corrections kernel overloads
    _pme_energy_corrections_kernel_overload[t] = wp.overload(
        _pme_energy_corrections_kernel,
        [
            wp.array(dtype=t),  # raw_energies
            wp.array(dtype=t),  # charges
            wp.array(dtype=t),  # volume
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # total_charge
            wp.array(dtype=t),  # corrected_energies
        ],
    )

    _batch_pme_energy_corrections_kernel_overload[t] = wp.overload(
        _batch_pme_energy_corrections_kernel,
        [
            wp.array(dtype=t),  # raw_energies
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # volumes
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # total_charges
            wp.array(dtype=t),  # corrected_energies
        ],
    )

    # Energy corrections backward kernel overloads
    _pme_energy_corrections_backward_kernel_overload[t] = wp.overload(
        _pme_energy_corrections_backward_kernel,
        [
            wp.array(dtype=t),  # grad_E
            wp.array(dtype=t),  # raw_energies
            wp.array(dtype=t),  # charges
            wp.array(dtype=t),  # volume
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # total_charge
            wp.array(dtype=t),  # grad_raw
            wp.array(dtype=t),  # grad_charges
            wp.array(dtype=t),  # grad_volume
            wp.array(dtype=t),  # grad_alpha
            wp.array(dtype=t),  # grad_total_charge
        ],
    )
    _batch_pme_energy_corrections_backward_kernel_overload[t] = wp.overload(
        _batch_pme_energy_corrections_backward_kernel,
        [
            wp.array(dtype=t),  # grad_E
            wp.array(dtype=t),  # raw_energies
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # volumes
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # total_charges
            wp.array(dtype=t),  # grad_raw
            wp.array(dtype=t),  # grad_charges
            wp.array(dtype=t),  # grad_volumes (B,)
            wp.array(dtype=t),  # grad_alpha (B,)
            wp.array(dtype=t),  # grad_total_charges (B,)
        ],
    )

    # Energy corrections DOUBLE-backward kernel overloads
    _pme_energy_corrections_double_backward_kernel_overload[t] = wp.overload(
        _pme_energy_corrections_double_backward_kernel,
        [
            wp.array(dtype=t),  # h_raw
            wp.array(dtype=t),  # h_chg
            wp.array(dtype=t),  # h_vol
            wp.array(dtype=t),  # h_alpha
            wp.array(dtype=t),  # h_qtot
            wp.array(dtype=t),  # grad_E (saved)
            wp.array(dtype=t),  # raw_energies (saved)
            wp.array(dtype=t),  # charges (saved)
            wp.array(dtype=t),  # volume
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # total_charge
            wp.array(dtype=t),  # grad_grad_E
            wp.array(dtype=t),  # grad_raw
            wp.array(dtype=t),  # grad_charges
            wp.array(dtype=t),  # grad_volume
            wp.array(dtype=t),  # grad_alpha
            wp.array(dtype=t),  # grad_total_charge
        ],
    )
    _batch_pme_energy_corrections_double_backward_kernel_overload[t] = wp.overload(
        _batch_pme_energy_corrections_double_backward_kernel,
        [
            wp.array(dtype=t),  # h_raw
            wp.array(dtype=t),  # h_chg
            wp.array(dtype=t),  # h_vol  (B,)
            wp.array(dtype=t),  # h_alpha (B,)
            wp.array(dtype=t),  # h_qtot (B,)
            wp.array(dtype=t),  # grad_E
            wp.array(dtype=t),  # raw_energies
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # volumes
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # total_charges
            wp.array(dtype=t),  # grad_grad_E
            wp.array(dtype=t),  # grad_raw
            wp.array(dtype=t),  # grad_charges
            wp.array(dtype=t),  # grad_volumes (B,)
            wp.array(dtype=t),  # grad_alpha (B,)
            wp.array(dtype=t),  # grad_total_charges (B,)
        ],
    )

    # Energy corrections with charge gradient kernel overloads
    _pme_energy_corrections_with_charge_grad_kernel_overload[t] = wp.overload(
        _pme_energy_corrections_with_charge_grad_kernel,
        [
            wp.array(dtype=t),  # raw_energies
            wp.array(dtype=t),  # charges
            wp.array(dtype=t),  # volume
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # total_charge
            wp.array(dtype=t),  # corrected_energies
            wp.array(dtype=t),  # charge_gradients
        ],
    )

    _batch_pme_energy_corrections_with_charge_grad_kernel_overload[t] = wp.overload(
        _batch_pme_energy_corrections_with_charge_grad_kernel,
        [
            wp.array(dtype=t),  # raw_energies
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # volumes
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # total_charges
            wp.array(dtype=t),  # corrected_energies
            wp.array(dtype=t),  # charge_gradients
        ],
    )

    _pme_virial_bg_reduce_kernel_overload[t] = wp.overload(
        _pme_virial_bg_reduce_kernel,
        [
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # total_charges
        ],
    )
    _pme_virial_bg_apply_kernel_overload[t] = wp.overload(
        _pme_virial_bg_apply_kernel,
        [
            wp.array(dtype=t),  # total_charges
            wp.array3d(dtype=t),  # cell
            wp.array(dtype=t),  # alpha
            wp.array3d(dtype=t),  # virial_in
            wp.array3d(dtype=t),  # virial_out
        ],
    )
    _pme_virial_bg_backward_per_system_kernel_overload[t] = wp.overload(
        _pme_virial_bg_backward_per_system_kernel,
        [
            wp.array3d(dtype=t),  # grad_virial
            wp.array(dtype=t),  # total_charges
            wp.array3d(dtype=t),  # cell
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # grad_total_charges
            wp.array(dtype=t),  # grad_alpha
            wp.array3d(dtype=t),  # grad_cell
        ],
    )
    _pme_virial_bg_backward_per_atom_kernel_overload[t] = wp.overload(
        _pme_virial_bg_backward_per_atom_kernel,
        [
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # grad_total_charges
            wp.array(dtype=t),  # grad_charges
        ],
    )


###########################################################################################
########################### Warp Launcher Functions (wp_*) ################################
###########################################################################################


def pme_green_structure_factor(
    k_squared: wp.array,
    miller_x: wp.array,
    miller_y: wp.array,
    miller_z: wp.array,
    alpha: wp.array,
    volume: wp.array,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    green_function: wp.array,
    structure_factor_sq: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Compute PME Green's function and B-spline structure factor correction.

    Framework-agnostic launcher for single-system Green's function computation.

    Note: FFT Operations Offloaded to Framework
    -------------------------------------------
    This kernel computes the Green's function multipliers for PME.
    The complete PME reciprocal-space workflow requires FFT operations
    that are not available in Warp and must be performed by the calling
    framework. The typical workflow is:

    1. Spread charges to mesh: spline_spread()
    2. Forward FFT: framework.fft.rfftn(mesh)      <-- Framework-specific
    3. Compute Green's function: pme_green_structure_factor()
    4. Convolution: mesh_fft * green_function / structure_factor_sq
    5. Inverse FFT: framework.fft.irfftn(...)     <-- Framework-specific
    6. Gather potential: spline_gather()
    7. Apply corrections: pme_energy_corrections()

    Parameters
    ----------
    k_squared : wp.array, shape (Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        Squared magnitude of k-vectors at each grid point.
    miller_x : wp.array, shape (Nx,), dtype=wp.float32 or wp.float64
        Miller indices in x direction (from fftfreq).
    miller_y : wp.array, shape (Ny,), dtype=wp.float32 or wp.float64
        Miller indices in y direction (from fftfreq).
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32 or wp.float64
        Miller indices in z direction (from rfftfreq).
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Unit cell volume.
    mesh_nx, mesh_ny, mesh_nz : int
        Full mesh dimensions (Nz is the full size, not rfft size).
    spline_order : int
        B-spline order (1-4). Order 4 (cubic) recommended.
    green_function : wp.array, shape (Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: Green's function G(k) at each grid point.
    structure_factor_sq : wp.array, shape (Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: :math:`|B(k)|^2` structure factor squared at each grid point.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    nvalchemiops.torch.interactions.electrostatics.pme : Complete PyTorch implementation
    """
    nx, ny, nz_rfft = k_squared.shape[0], k_squared.shape[1], k_squared.shape[2]

    kernel = _pme_green_structure_factor_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(nx, ny, nz_rfft),
        inputs=[
            k_squared,
            miller_x,
            miller_y,
            miller_z,
            alpha,
            volume,
            wp.int32(mesh_nx),
            wp.int32(mesh_ny),
            wp.int32(mesh_nz),
            wp.int32(spline_order),
        ],
        outputs=[green_function, structure_factor_sq],
        device=device,
    )


def batch_pme_green_structure_factor(
    k_squared: wp.array,
    miller_x: wp.array,
    miller_y: wp.array,
    miller_z: wp.array,
    alpha: wp.array,
    volumes: wp.array,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    green_function: wp.array,
    structure_factor_sq: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Compute PME Green's function and B-spline structure factor for batched systems.

    Framework-agnostic launcher for batched Green's function computation.
    Each system can have different alpha and volume values, but shares
    the same mesh dimensions.

    Parameters
    ----------
    k_squared : wp.array, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        Per-system squared magnitude of k-vectors at each grid point.
    miller_x : wp.array, shape (Nx,), dtype=wp.float32 or wp.float64
        Miller indices in x direction (shared across systems).
    miller_y : wp.array, shape (Ny,), dtype=wp.float32 or wp.float64
        Miller indices in y direction (shared across systems).
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32 or wp.float64
        Miller indices in z direction (shared across systems).
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system unit cell volume.
    mesh_nx, mesh_ny, mesh_nz : int
        Full mesh dimensions (Nz is the full size, not rfft size).
    spline_order : int
        B-spline order (1-4). Order 4 (cubic) recommended.
    green_function : wp.array, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system Green's function G_s(k) at each grid point.
    structure_factor_sq : wp.array, shape (Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: :math:`|B(k)|^2` structure factor squared (computed only at batch_idx=0).
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    nvalchemiops.torch.interactions.electrostatics.pme : Complete PyTorch implementation
    """
    num_systems = k_squared.shape[0]
    nx, ny, nz_rfft = k_squared.shape[1], k_squared.shape[2], k_squared.shape[3]

    kernel = _batch_pme_green_structure_factor_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_systems, nx, ny, nz_rfft),
        inputs=[
            k_squared,
            miller_x,
            miller_y,
            miller_z,
            alpha,
            volumes,
            wp.int32(mesh_nx),
            wp.int32(mesh_ny),
            wp.int32(mesh_nz),
            wp.int32(spline_order),
        ],
        outputs=[green_function, structure_factor_sq],
        device=device,
    )


def pme_convolve(
    mesh_fft: wp.array,
    k_squared: wp.array,
    moduli_x: wp.array,
    moduli_y: wp.array,
    moduli_z: wp.array,
    alpha: wp.array,
    volume: wp.array,
    convolved_mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Fused per-k-point Green's compute + B-spline deconvolution + multiply.

    Single-system. ``moduli_x/y/z`` are precomputed 1D B-spline modulus LUTs
    (``sinc(m/N)^spline_order`` per miller index, one per axis); the kernel
    reads three values + multiplies + squares them per (i, j, k) thread,
    replacing the inline sinc-and-power computation used pre-P-C.

    Parameters
    ----------
    mesh_fft : wp.array3d, shape (nx, ny, nz_rfft), dtype=vec2f/vec2d
        Input mesh after forward rFFT, complex represented as (real, imag).
    convolved_mesh : wp.array3d, same shape/dtype as ``mesh_fft``
        OUTPUT. May alias ``mesh_fft`` for in-place.
    """
    nx, ny, nz_rfft = mesh_fft.shape[0], mesh_fft.shape[1], mesh_fft.shape[2]
    kernel = _pme_convolve_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(nx, ny, nz_rfft),
        inputs=[
            mesh_fft,
            k_squared,
            moduli_x,
            moduli_y,
            moduli_z,
            alpha,
            volume,
        ],
        outputs=[convolved_mesh],
        device=device,
    )


def batch_pme_convolve(
    mesh_fft: wp.array,
    k_squared: wp.array,
    moduli_x: wp.array,
    moduli_y: wp.array,
    moduli_z: wp.array,
    alpha: wp.array,
    volumes: wp.array,
    convolved_mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Batched version of ``pme_convolve``. Mesh shapes are (B, nx, ny, nz_r)."""
    num_systems = mesh_fft.shape[0]
    nx, ny, nz_rfft = mesh_fft.shape[1], mesh_fft.shape[2], mesh_fft.shape[3]
    kernel = _batch_pme_convolve_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_systems, nx, ny, nz_rfft),
        inputs=[
            mesh_fft,
            k_squared,
            moduli_x,
            moduli_y,
            moduli_z,
            alpha,
            volumes,
        ],
        outputs=[convolved_mesh],
        device=device,
    )


def pme_convolve_backward(
    mesh_fft: wp.array,
    grad_convolved: wp.array,
    k_squared: wp.array,
    moduli_x: wp.array,
    moduli_y: wp.array,
    moduli_z: wp.array,
    alpha: wp.array,
    volume: wp.array,
    grad_mesh_fft: wp.array,
    grad_alpha: wp.array,
    grad_volume: wp.array,
    grad_k_squared: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Single-system backward for ``pme_convolve``. See kernel docstring for math.

    ``grad_alpha`` and ``grad_volume`` must be zero-initialized 1-element arrays
    (the kernel atomically accumulates into them across all k-points).
    ``grad_k_squared`` is written elementwise (no zero-init required).
    """
    nx, ny, nz_rfft = mesh_fft.shape[0], mesh_fft.shape[1], mesh_fft.shape[2]
    kernel = _pme_convolve_backward_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(nx, ny, nz_rfft),
        inputs=[
            mesh_fft,
            grad_convolved,
            k_squared,
            moduli_x,
            moduli_y,
            moduli_z,
            alpha,
            volume,
        ],
        outputs=[grad_mesh_fft, grad_alpha, grad_volume, grad_k_squared],
        device=device,
    )


def batch_pme_convolve_backward(
    mesh_fft: wp.array,
    grad_convolved: wp.array,
    k_squared: wp.array,
    moduli_x: wp.array,
    moduli_y: wp.array,
    moduli_z: wp.array,
    alpha: wp.array,
    volumes: wp.array,
    grad_mesh_fft: wp.array,
    grad_alpha: wp.array,
    grad_volumes: wp.array,
    grad_k_squared: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Batched backward for ``batch_pme_convolve``. ``grad_alpha`` and
    ``grad_volumes`` are length-B arrays zero-initialized by the caller.
    ``grad_k_squared`` is written elementwise (no zero-init required)."""
    num_systems = mesh_fft.shape[0]
    nx, ny, nz_rfft = mesh_fft.shape[1], mesh_fft.shape[2], mesh_fft.shape[3]
    kernel = _batch_pme_convolve_backward_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_systems, nx, ny, nz_rfft),
        inputs=[
            mesh_fft,
            grad_convolved,
            k_squared,
            moduli_x,
            moduli_y,
            moduli_z,
            alpha,
            volumes,
        ],
        outputs=[grad_mesh_fft, grad_alpha, grad_volumes, grad_k_squared],
        device=device,
    )


def pme_energy_corrections(
    raw_energies: wp.array,
    charges: wp.array,
    volume: wp.array,
    alpha: wp.array,
    total_charge: wp.array,
    corrected_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Apply self-energy and background corrections to PME energies.

    Framework-agnostic launcher for single-system energy corrections.

    Converts raw potential values (φ_i) to corrected per-atom energies by:
    1. Multiplying potential by charge: E_pot = q_i * φ_i
    2. Subtracting self-energy: E_self = (α/√π) * q_i²
    3. Subtracting background: E_bg = (π/(2α²V)) * q_i * Q_total

    Final: E_i = q_i * φ_i - (α/√π) * q_i² - (π/(2α²V)) * q_i * Q_total

    Parameters
    ----------
    raw_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Raw potential values φ_i from mesh interpolation.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Unit cell volume.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    total_charge : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Sum of all charges (Q_total = ∑_i q_i).
    corrected_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Corrected per-atom energies.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = raw_energies.shape[0]

    kernel = _pme_energy_corrections_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[raw_energies, charges, volume, alpha, total_charge],
        outputs=[corrected_energies],
        device=device,
    )


def batch_pme_energy_corrections(
    raw_energies: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    volumes: wp.array,
    alpha: wp.array,
    total_charges: wp.array,
    corrected_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Apply self-energy and background corrections for batched PME.

    Framework-agnostic launcher for batched energy corrections.
    Each atom looks up its system's parameters via batch_idx.

    Parameters
    ----------
    raw_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Raw potential values φ_i from mesh interpolation.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system unit cell volume.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    total_charges : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system sum of charges (Q_s = ∑_{i∈s} q_i).
    corrected_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Corrected per-atom energies.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = raw_energies.shape[0]

    kernel = _batch_pme_energy_corrections_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[raw_energies, charges, batch_idx, volumes, alpha, total_charges],
        outputs=[corrected_energies],
        device=device,
    )


def pme_energy_corrections_backward(
    grad_E: wp.array,
    raw_energies: wp.array,
    charges: wp.array,
    volume: wp.array,
    alpha: wp.array,
    total_charge: wp.array,
    grad_raw: wp.array,
    grad_charges: wp.array,
    grad_volume: wp.array,
    grad_alpha: wp.array,
    grad_total_charge: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Single-system launcher for ``_pme_energy_corrections_backward_kernel``.

    ``grad_volume``, ``grad_alpha``, and ``grad_total_charge`` must be
    zero-initialized 1-element arrays.
    """
    kernel = _pme_energy_corrections_backward_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=raw_energies.shape[0],
        inputs=[grad_E, raw_energies, charges, volume, alpha, total_charge],
        outputs=[
            grad_raw,
            grad_charges,
            grad_volume,
            grad_alpha,
            grad_total_charge,
        ],
        device=device,
    )


def batch_pme_energy_corrections_backward(
    grad_E: wp.array,
    raw_energies: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    volumes: wp.array,
    alpha: wp.array,
    total_charges: wp.array,
    grad_raw: wp.array,
    grad_charges: wp.array,
    grad_volumes: wp.array,
    grad_alpha: wp.array,
    grad_total_charges: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Batched launcher for ``_batch_pme_energy_corrections_backward_kernel``.

    Per-system grads must be zero-initialized length-B arrays.
    """
    kernel = _batch_pme_energy_corrections_backward_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=raw_energies.shape[0],
        inputs=[
            grad_E,
            raw_energies,
            charges,
            batch_idx,
            volumes,
            alpha,
            total_charges,
        ],
        outputs=[
            grad_raw,
            grad_charges,
            grad_volumes,
            grad_alpha,
            grad_total_charges,
        ],
        device=device,
    )


def pme_energy_corrections_double_backward(
    h_raw: wp.array,
    h_chg: wp.array,
    h_vol: wp.array,
    h_alpha: wp.array,
    h_qtot: wp.array,
    grad_E: wp.array,
    raw_energies: wp.array,
    charges: wp.array,
    volume: wp.array,
    alpha: wp.array,
    total_charge: wp.array,
    grad_grad_E: wp.array,
    grad_raw: wp.array,
    grad_charges: wp.array,
    grad_volume: wp.array,
    grad_alpha: wp.array,
    grad_total_charge: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Single-system launcher for ``_pme_energy_corrections_double_backward_kernel``.

    ``grad_volume`` / ``grad_alpha`` / ``grad_total_charge`` must be
    zero-initialized 1-element arrays.
    """
    kernel = _pme_energy_corrections_double_backward_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=raw_energies.shape[0],
        inputs=[
            h_raw,
            h_chg,
            h_vol,
            h_alpha,
            h_qtot,
            grad_E,
            raw_energies,
            charges,
            volume,
            alpha,
            total_charge,
        ],
        outputs=[
            grad_grad_E,
            grad_raw,
            grad_charges,
            grad_volume,
            grad_alpha,
            grad_total_charge,
        ],
        device=device,
    )


def batch_pme_energy_corrections_double_backward(
    h_raw: wp.array,
    h_chg: wp.array,
    h_vol: wp.array,
    h_alpha: wp.array,
    h_qtot: wp.array,
    grad_E: wp.array,
    raw_energies: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    volumes: wp.array,
    alpha: wp.array,
    total_charges: wp.array,
    grad_grad_E: wp.array,
    grad_raw: wp.array,
    grad_charges: wp.array,
    grad_volumes: wp.array,
    grad_alpha: wp.array,
    grad_total_charges: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Batched launcher for ``_batch_pme_energy_corrections_double_backward_kernel``."""
    kernel = _batch_pme_energy_corrections_double_backward_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=raw_energies.shape[0],
        inputs=[
            h_raw,
            h_chg,
            h_vol,
            h_alpha,
            h_qtot,
            grad_E,
            raw_energies,
            charges,
            batch_idx,
            volumes,
            alpha,
            total_charges,
        ],
        outputs=[
            grad_grad_E,
            grad_raw,
            grad_charges,
            grad_volumes,
            grad_alpha,
            grad_total_charges,
        ],
        device=device,
    )


def pme_energy_corrections_with_charge_grad(
    raw_energies: wp.array,
    charges: wp.array,
    volume: wp.array,
    alpha: wp.array,
    total_charge: wp.array,
    corrected_energies: wp.array,
    charge_gradients: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Apply corrections and compute charge gradients for PME energies.

    Framework-agnostic launcher for single-system energy corrections
    with analytical charge gradient computation.

    Computes both corrected energies and analytical charge gradients:
    - Energy: E_i = q_i * φ_i - (α/√π) * q_i² - (π/(2α²V)) * q_i * Q_total
    - Charge gradient: ∂E_total/∂q_i = 2*φ_i - 2*(α/√π)*q_i - (π/(α²V))*Q_total

    Parameters
    ----------
    raw_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Raw potential values φ_i from mesh interpolation.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Unit cell volume.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    total_charge : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Sum of all charges (Q_total = ∑_i q_i).
    corrected_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Corrected per-atom energies.
    charge_gradients : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Analytical charge gradients ∂E_total/∂q_i.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = raw_energies.shape[0]

    kernel = _pme_energy_corrections_with_charge_grad_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[raw_energies, charges, volume, alpha, total_charge],
        outputs=[corrected_energies, charge_gradients],
        device=device,
    )


def batch_pme_energy_corrections_with_charge_grad(
    raw_energies: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    volumes: wp.array,
    alpha: wp.array,
    total_charges: wp.array,
    corrected_energies: wp.array,
    charge_gradients: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Apply corrections and compute charge gradients for batched PME.

    Framework-agnostic launcher for batched energy corrections
    with analytical charge gradient computation.

    Parameters
    ----------
    raw_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Raw potential values φ_i from mesh interpolation.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system unit cell volume.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    total_charges : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system sum of charges (Q_s = ∑_{i∈s} q_i).
    corrected_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Corrected per-atom energies.
    charge_gradients : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Analytical charge gradients ∂E_total/∂q_i.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = raw_energies.shape[0]

    kernel = _batch_pme_energy_corrections_with_charge_grad_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[raw_energies, charges, batch_idx, volumes, alpha, total_charges],
        outputs=[corrected_energies, charge_gradients],
        device=device,
    )


def pme_virial_bg_correction(
    charges: wp.array,
    batch_idx: wp.array,
    cell: wp.array,
    alpha: wp.array,
    total_charges: wp.array,
    virial_in: wp.array,
    virial_out: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Apply non-neutral background virial correction.

    Two-pass: pass 1 reduces per-atom ``charges`` into per-system
    ``total_charges`` (zero-initialized by the caller) via atomic_add;
    pass 2 computes ``V = |det(cell[s])|``, ``E_bg = π Q² / (2 α² V)``,
    subtracts ``E_bg`` from the three diagonal entries of ``virial_in``,
    and writes the result to ``virial_out``. Single-system uses
    ``batch_idx`` filled with zeros.

    Shapes:
      charges       (N,)
      batch_idx     (N,) int32
      cell          (B, 3, 3)
      alpha         (B,)
      total_charges (B,)  — zero-initialized by caller; written in pass 1
      virial_in     (B, 3, 3)
      virial_out    (B, 3, 3) — written in pass 2 (may alias virial_in)
    """
    num_atoms = charges.shape[0]
    num_systems = total_charges.shape[0]
    wp.launch(
        _pme_virial_bg_reduce_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[charges, batch_idx, total_charges],
        device=device,
    )
    wp.launch(
        _pme_virial_bg_apply_kernel_overload[wp_dtype],
        dim=num_systems,
        inputs=[total_charges, cell, alpha, virial_in, virial_out],
        device=device,
    )


def pme_virial_bg_correction_backward(
    grad_virial: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    cell: wp.array,
    alpha: wp.array,
    total_charges: wp.array,
    grad_total_charges: wp.array,
    grad_charges: wp.array,
    grad_alpha: wp.array,
    grad_cell: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Analytic backward for ``pme_virial_bg_correction``.

    Three passes:
      1) reduce ``charges`` into ``total_charges`` (Q per system)
      2) per-system: turn cotangent ``grad_virial`` into ``grad_total_charges``,
         ``grad_alpha``, ``grad_cell`` via dE_bg/dQ, dE_bg/dα, and Jacobi's
         formula for d|det C|/dC
      3) per-atom: scatter ``grad_total_charges[s(j)]`` to ``grad_charges[j]``

    All output buffers (``total_charges``, ``grad_*``) are zero-initialized
    by the caller.
    """
    num_atoms = charges.shape[0]
    num_systems = total_charges.shape[0]
    wp.launch(
        _pme_virial_bg_reduce_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[charges, batch_idx, total_charges],
        device=device,
    )
    wp.launch(
        _pme_virial_bg_backward_per_system_kernel_overload[wp_dtype],
        dim=num_systems,
        inputs=[
            grad_virial,
            total_charges,
            cell,
            alpha,
            grad_total_charges,
            grad_alpha,
            grad_cell,
        ],
        device=device,
    )
    wp.launch(
        _pme_virial_bg_backward_per_atom_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[batch_idx, grad_total_charges, grad_charges],
        device=device,
    )


###########################################################################################
########################### Module Exports #################################################
###########################################################################################

__all__ = [
    # Kernel overloads
    "_pme_green_structure_factor_kernel_overload",
    "_batch_pme_green_structure_factor_kernel_overload",
    "_pme_energy_corrections_kernel_overload",
    "_batch_pme_energy_corrections_kernel_overload",
    "_pme_energy_corrections_backward_kernel_overload",
    "_batch_pme_energy_corrections_backward_kernel_overload",
    "_pme_energy_corrections_double_backward_kernel_overload",
    "_batch_pme_energy_corrections_double_backward_kernel_overload",
    "_pme_energy_corrections_with_charge_grad_kernel_overload",
    "_batch_pme_energy_corrections_with_charge_grad_kernel_overload",
    "_pme_convolve_kernel_overload",
    "_batch_pme_convolve_kernel_overload",
    "_pme_convolve_backward_kernel_overload",
    "_batch_pme_convolve_backward_kernel_overload",
    # Warp launchers
    "pme_green_structure_factor",
    "batch_pme_green_structure_factor",
    "pme_convolve",
    "batch_pme_convolve",
    "pme_convolve_backward",
    "batch_pme_convolve_backward",
    "pme_energy_corrections",
    "batch_pme_energy_corrections",
    "pme_energy_corrections_backward",
    "batch_pme_energy_corrections_backward",
    "pme_energy_corrections_double_backward",
    "batch_pme_energy_corrections_double_backward",
    "pme_energy_corrections_with_charge_grad",
    "batch_pme_energy_corrections_with_charge_grad",
    "pme_virial_bg_correction",
    "pme_virial_bg_correction_backward",
]
