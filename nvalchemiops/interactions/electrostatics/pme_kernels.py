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

This module provides GPU-accelerated Warp launchers for Particle Mesh Ewald
(PME) calculations. Green structure-factor and virial-background kernels live
here; convolve and correction launchers route through ``pme_factory.py``.
Charge assignment and force interpolation are handled by the spline module.

MATHEMATICAL FORMULATION
========================

PME splits the Coulomb energy into components:

.. math::

    E_{\\text{total}} = E_{\\text{real}} + E_{\\text{reciprocal}} - E_{\\text{self}} - E_{\\text{background}}

This module provides low-level support for:

1. Green's Function and Structure Factor Correction:

.. math::

    G(k) = \\frac{2\\pi}{V} \\frac{\\exp(-k^2/(4\\alpha^2))}{k^2}

The B-spline charge assignment introduces aliasing, corrected by:

.. math::

    C(k) = \\left[\\text{sinc}(k_x/N_x) \\cdot \\text{sinc}(k_y/N_y) \\cdot \\text{sinc}(k_z/N_z)\\right]^{-2p}

where p is the spline order.

2. Factory-backed energy corrections:

   - Self-energy: :math:`E_{\\text{self}} = \\frac{\\alpha}{\\sqrt{\\pi}} \\sum_i q_i^2`
   - Background (for non-neutral systems): :math:`E_{\\text{background}} = \\frac{\\pi}{2\\alpha^2 V} \\sum_i q_i Q_{\\text{total}}`

DTYPE FLEXIBILITY
=================

The hand-written Green structure-factor and virial-background kernels support
both float32 and float64 inputs via explicit overloads. Factory-backed convolve
and correction launchers select typed kernels through ``get_pme_kernel``.

KERNEL ORGANIZATION
===================

Green's Function Kernels:
    _pme_green_structure_factor_kernel: Single-system G(k) and C(k)
    _batch_pme_green_structure_factor_kernel: Batched version

Factory-Backed Correction Launchers:
    pme_energy_corrections: Single-system self + background correction
    batch_pme_energy_corrections: Batched self + background correction

Internal Factory-Backed Convolve Helpers:
    pme_convolve: Single-system PME reciprocal convolution
    batch_pme_convolve: Batched PME reciprocal convolution

.. warning
    In contrast to the other electrostatic kernels that offer end-to-end
    ``warp`` launchers, PME requires FFT for the convolution step that is
    currently not available in ``warp``. As a result, bindings must call
    FFT within their own framework in between kernel launches. The sequence
    of calls looks like the following:

    1. Spread charges to mesh: ``spline_spread()``
    2. Forward FFT: ``framework.fft.rfftn(mesh)``
    3. Legacy helper: ``pme_green_structure_factor()`` returns raw ``G(k)``
       and ``C^2(k)``
    4. Convolution: ``mesh_fft * green_function / structure_factor_sq``
    5. Inverse FFT: ``framework.fft.irfftn(...)``
    6. Gather potential: ``spline_gather()``
    7. Apply corrections: ``pme_energy_corrections()``

    The Torch/JAX PME paths use internal factory-backed convolve helpers that
    compute the effective folded multiplier ``G(k) / C^2(k)`` inside the fused
    convolve kernel.

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


###########################################################################################
########################### Helper Functions ##############################################
###########################################################################################


@wp.func
def compute_sinc(x: Any) -> Any:
    """Compute normalized sinc function: :math:`\\sin(\\pi x)/(\\pi x)`.

    Uses Taylor expansion near zero for numerical stability.

    Parameters
    ----------
    x : Any
        Scalar argument; dtype determined by the calling context (wp.float32 or wp.float64).

    Returns
    -------
    Any
        :math:`\\sin(\\pi x)/(\\pi x)`, or 1.0 when ``|x| < 1e-6``.
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
    """Compute exp(-prefactor * k_sq) / k_sq.

    Parameters
    ----------
    k_sq : Any
        Squared wave-vector magnitude; dtype determined by calling context.
    prefactor : Any
        Scalar prefactor in the exponent; same dtype as ``k_sq``.

    Returns
    -------
    Any
        :math:`\\exp(-\\text{prefactor} \\cdot k^2) / k^2`.
    """
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
    r"""Compute PME Green's function and B-spline structure factor correction.

    Computes two arrays needed for PME reciprocal space:
    1. Green's function: :math:`G(k) = (2\pi/V) \cdot \exp(-k^2/(4\alpha^2)) / k^2`
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
        B-spline order (1-6). Order 4 (cubic) recommended.
    green_function : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: Green's function G(k) at each grid point.
    structure_factor_sq : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: :math:`|B(k)|^2` structure factor squared at each grid point.

    Notes
    -----
    - k=0 (grid point [0,0,0]) is explicitly set to zero (tin-foil boundary conditions).
    - Near-zero :math:`k^2` values are set to zero to avoid division by zero.
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

    # Structure factor: sinc(mi_x/Nx) * sinc(mi_y/Ny) * sinc(mi_z/Nz).
    # This helper returns raw G(k) plus C^2(k). The folded G/C^2 path lives in
    # the factory-backed fused convolve helpers.
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

    # Raw volume-normalized Green's function. External callers that use this
    # helper apply the B-spline deconvolution with ``structure_factor_sq``.
    if k_sq < threshold:
        green_function[i, j, k] = zero
    else:
        exp_factor = wp_exp_kernel(k_sq, one / (four * alpha_ * alpha_))
        green_function[i, j, k] = twopi * exp_factor / volume_

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
    r"""Compute PME Green's function and B-spline structure factor for batched systems.

    Batched version of _pme_green_structure_factor_kernel. Each system can have
    different alpha and volume values, but shares the same mesh dimensions.

    Green's function: :math:`G_s(k) = (2\pi/V_s) \cdot \exp(-k^2/(4\alpha_s^2)) / k^2`
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
        B-spline order (1-6). Order 4 (cubic) recommended.
    green_function : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system Green's function G_s(k) at each grid point.
    structure_factor_sq : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: :math:`|B(k)|^2` structure factor squared (computed only at batch_idx=0).

    Notes
    -----
    - k=0 (grid point [0,0,0]) is explicitly set to zero for each system.
    - Near-zero :math:`k^2` values are set to zero to avoid division by zero.
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

    # Structure factor C^2(k). Written once at batch_idx=0 because it depends
    # only on mesh dimensions and spline order.
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

    # Raw volume-normalized Green's function; fused convolve owns folded G/C^2.
    if k_sq < threshold:
        green_function[batch_idx, i, j, k] = zero
    else:
        exp_factor = wp_exp_kernel(k_sq, one / (four * system_alpha * system_alpha))
        green_function[batch_idx, i, j, k] = twopi * exp_factor / system_volume

    if i == 0 and j == 0 and k == 0:
        green_function[batch_idx, i, j, k] = zero


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
    volume: wp.array(dtype=Any),  # (B,) caller-supplied or dummy
    use_supplied_volume: wp.int32,
    alpha: wp.array(dtype=Any),  # (B,) — per-system Ewald splitting
    virial_in: wp.array3d(dtype=Any),  # (B, 3, 3) input
    virial_out: wp.array3d(dtype=Any),  # (B, 3, 3) output = virial_in - E_bg·I
):
    """Pass 2: compute E_bg and subtract it from the virial diagonal."""
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
    cell_volume = wp.abs(det)
    volume_value = cell_volume
    if use_supplied_volume != 0:
        volume_value = volume[s]

    e_bg = pi * q * q / (two * a * a * volume_value)

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
    volume: wp.array(dtype=Any),  # (B,) caller-supplied or dummy
    use_supplied_volume: wp.int32,
    alpha: wp.array(dtype=Any),  # (B,)
    grad_total_charges: wp.array(dtype=Any),  # (B,) OUT — dL/dQ per system
    grad_alpha: wp.array(dtype=Any),  # (B,) OUT — dL/dα per system
    grad_cell: wp.array3d(dtype=Any),  # (B, 3, 3) OUT — dL/dC
):
    r"""Per-system: turn the cotangent of virial_out into per-system dL/dQ, dL/dalpha, dL/dC.

    From ``virial_out[s,i,j] = virial_in[s,i,j] - delta_ij * E_bg(s)`` (where
    :math:`E_{bg} = \pi Q^2 / (2 \alpha^2 V)` and ``V = |det(C)|``):

    .. math::

        dL/dE_{bg}(s) = -(g[s,0,0] + g[s,1,1] + g[s,2,2])

        dE_{bg}/dQ = \pi Q / (\alpha^2 V)

        dE_{bg}/d\alpha = -\pi Q^2 / (\alpha^3 V)

        dE_{bg}/dV = -\pi Q^2 / (2 \alpha^2 V^2)

        d|\det C|/dC = \text{sign}(\det C) \cdot \text{cofactor}(C) \quad \text{(Jacobi's formula)}

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
    cell_volume = wp.abs(det)
    volume_value = cell_volume
    if use_supplied_volume != 0:
        volume_value = volume[s]
    sgn = wp.sign(det)

    g_diag_sum = grad_virial[s, 0, 0] + grad_virial[s, 1, 1] + grad_virial[s, 2, 2]
    g_E_bg = -g_diag_sum  # dL/dE_bg

    a2 = a * a
    a3 = a2 * a
    v2 = volume_value * volume_value

    dE_dQ = pi * q / (a2 * volume_value)
    dE_dA = -pi * q * q / (a3 * volume_value)
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
    if use_supplied_volume != 0:
        grad_cell[s, 0, 0] = type(q)(0.0)
        grad_cell[s, 0, 1] = type(q)(0.0)
        grad_cell[s, 0, 2] = type(q)(0.0)
        grad_cell[s, 1, 0] = type(q)(0.0)
        grad_cell[s, 1, 1] = type(q)(0.0)
        grad_cell[s, 1, 2] = type(q)(0.0)
        grad_cell[s, 2, 0] = type(q)(0.0)
        grad_cell[s, 2, 1] = type(q)(0.0)
        grad_cell[s, 2, 2] = type(q)(0.0)
    else:
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
# Single-system kernel overloads
_pme_green_structure_factor_kernel_overload = {}
_pme_virial_bg_reduce_kernel_overload = {}
_pme_virial_bg_apply_kernel_overload = {}
_pme_virial_bg_backward_per_system_kernel_overload = {}
_pme_virial_bg_backward_per_atom_kernel_overload = {}

# Batch kernel overloads
_batch_pme_green_structure_factor_kernel_overload = {}

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
            wp.array(dtype=t),  # volume
            wp.int32,  # use_supplied_volume
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
            wp.array(dtype=t),  # volume
            wp.int32,  # use_supplied_volume
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


def _get_pme_factory_kernel(
    wp_dtype: type,
    *,
    component: str,
    batched: bool = False,
    order: str = "forward",
    charge_grad: bool = False,
) -> wp.Kernel:
    """Return a PME factory kernel without creating a module import cycle."""
    from nvalchemiops.interactions.electrostatics.pme_factory import get_pme_kernel

    return get_pme_kernel(
        wp_dtype,
        component=component,
        batched=batched,
        order=order,
        charge_grad=charge_grad,
    )


def _get_pme_factory_sentinels(wp_dtype: type, device: str) -> dict[str, wp.array]:
    """Return PME factory sentinel arrays without creating a module import cycle."""
    from nvalchemiops.interactions.electrostatics.pme_factory import (
        alloc_pme_sentinels,
    )

    return alloc_pme_sentinels(wp_dtype, device)


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
    This helper computes raw Green's function multipliers and B-spline
    deconvolution factors for PME. The internal factory-backed convolve helper
    computes the effective folded multiplier ``G(k) / C^2(k)`` internally.
    The complete PME reciprocal-space workflow requires FFT operations
    that are not available in Warp and must be performed by the calling
    framework. The typical workflow is:

    1. Spread charges to mesh: spline_spread()
    2. Forward FFT: framework.fft.rfftn(mesh)      <-- Framework-specific
    3. Compute Green's function and structure factor: pme_green_structure_factor()
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
        B-spline order (1-6). Order 4 (cubic) recommended.
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
        B-spline order (1-6). Order 4 (cubic) recommended.
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
    replacing repeated inline sinc-and-power work in each convolve launch.

    Parameters
    ----------
    mesh_fft : wp.array, shape (nx, ny, nz_rfft), dtype=vec2f or vec2d
        Input mesh after forward rFFT, complex values as (real, imag) pairs.
    k_squared : wp.array, shape (nx, ny, nz_rfft), dtype=wp.float32 or wp.float64
        Squared magnitude of k-vectors at each grid point.
    moduli_x : wp.array, shape (nx,), dtype=wp.float32 or wp.float64
        Precomputed 1D B-spline modulus LUT along x: ``sinc(m/Nx)^spline_order``.
    moduli_y : wp.array, shape (ny,), dtype=wp.float32 or wp.float64
        Precomputed 1D B-spline modulus LUT along y: ``sinc(m/Ny)^spline_order``.
    moduli_z : wp.array, shape (nz_rfft,), dtype=wp.float32 or wp.float64
        Precomputed 1D B-spline modulus LUT along z: ``sinc(m/Nz)^spline_order``.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Unit cell volume.
    convolved_mesh : wp.array, shape (nx, ny, nz_rfft), dtype=vec2f or vec2d
        OUTPUT: Convolved mesh ``mesh_fft * G(k) / C^2(k)``. May alias ``mesh_fft``
        for in-place operation.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str or None, optional
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.batch_pme_convolve` : Batched variant.
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.pme_convolve_backward` : Backward pass.
    """
    nx, ny, nz_rfft = mesh_fft.shape[0], mesh_fft.shape[1], mesh_fft.shape[2]
    kernel = _get_pme_factory_kernel(wp_dtype, component="pme_convolve")
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
    """Batched version of ``pme_convolve``; fused Green's + B-spline deconvolution for B systems.

    Parameters
    ----------
    mesh_fft : wp.array, shape (B, nx, ny, nz_rfft), dtype=vec2f or vec2d
        Per-system input mesh after forward rFFT, complex values as (real, imag) pairs.
    k_squared : wp.array, shape (B, nx, ny, nz_rfft), dtype=wp.float32 or wp.float64
        Per-system squared magnitude of k-vectors at each grid point.
    moduli_x : wp.array, shape (nx,), dtype=wp.float32 or wp.float64
        Precomputed 1D B-spline modulus LUT along x (shared across systems).
    moduli_y : wp.array, shape (ny,), dtype=wp.float32 or wp.float64
        Precomputed 1D B-spline modulus LUT along y (shared across systems).
    moduli_z : wp.array, shape (nz_rfft,), dtype=wp.float32 or wp.float64
        Precomputed 1D B-spline modulus LUT along z (shared across systems).
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system unit cell volume.
    convolved_mesh : wp.array, shape (B, nx, ny, nz_rfft), dtype=vec2f or vec2d
        OUTPUT: Per-system convolved mesh ``mesh_fft * G_s(k) / C^2(k)``.
        May alias ``mesh_fft`` for in-place operation.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str or None, optional
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.pme_convolve` : Single-system variant.
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.batch_pme_convolve_backward` : Backward pass.
    """
    num_systems = mesh_fft.shape[0]
    nx, ny, nz_rfft = mesh_fft.shape[1], mesh_fft.shape[2], mesh_fft.shape[3]
    kernel = _get_pme_factory_kernel(wp_dtype, component="pme_convolve", batched=True)
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
    """Single-system backward for ``pme_convolve``.

    Parameters
    ----------
    mesh_fft : wp.array, shape (nx, ny, nz_rfft), dtype=vec2f or vec2d
        Forward mesh values saved from the forward pass.
    grad_convolved : wp.array, shape (nx, ny, nz_rfft), dtype=vec2f or vec2d
        Cotangent of the convolved mesh output.
    k_squared : wp.array, shape (nx, ny, nz_rfft), dtype=wp.float32 or wp.float64
        Squared k-vector magnitudes from the forward pass.
    moduli_x : wp.array, shape (nx,), dtype=wp.float32 or wp.float64
        B-spline modulus LUT along x from the forward pass.
    moduli_y : wp.array, shape (ny,), dtype=wp.float32 or wp.float64
        B-spline modulus LUT along y from the forward pass.
    moduli_z : wp.array, shape (nz_rfft,), dtype=wp.float32 or wp.float64
        B-spline modulus LUT along z from the forward pass.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter from the forward pass.
    volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Unit cell volume from the forward pass.
    grad_mesh_fft : wp.array, shape (nx, ny, nz_rfft), dtype=vec2f or vec2d
        OUTPUT: Gradient w.r.t. ``mesh_fft``.
    grad_alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        OUTPUT: Gradient w.r.t. ``alpha``. Must be zero-initialized; kernel
        accumulates atomically across k-points.
    grad_volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        OUTPUT: Gradient w.r.t. ``volume``. Must be zero-initialized; kernel
        accumulates atomically across k-points.
    grad_k_squared : wp.array, shape (nx, ny, nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: Gradient w.r.t. ``k_squared``. Written elementwise; no zero-init required.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str or None, optional
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.pme_convolve` : Corresponding forward pass.
    """
    nx, ny, nz_rfft = mesh_fft.shape[0], mesh_fft.shape[1], mesh_fft.shape[2]
    kernel = _get_pme_factory_kernel(
        wp_dtype, component="pme_convolve", order="backward"
    )
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
    """Batched backward for ``batch_pme_convolve``.

    Parameters
    ----------
    mesh_fft : wp.array, shape (B, nx, ny, nz_rfft), dtype=vec2f or vec2d
        Per-system forward mesh values saved from the forward pass.
    grad_convolved : wp.array, shape (B, nx, ny, nz_rfft), dtype=vec2f or vec2d
        Cotangent of the convolved mesh output.
    k_squared : wp.array, shape (B, nx, ny, nz_rfft), dtype=wp.float32 or wp.float64
        Per-system squared k-vector magnitudes from the forward pass.
    moduli_x : wp.array, shape (nx,), dtype=wp.float32 or wp.float64
        B-spline modulus LUT along x from the forward pass.
    moduli_y : wp.array, shape (ny,), dtype=wp.float32 or wp.float64
        B-spline modulus LUT along y from the forward pass.
    moduli_z : wp.array, shape (nz_rfft,), dtype=wp.float32 or wp.float64
        B-spline modulus LUT along z from the forward pass.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter from the forward pass.
    volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system unit cell volumes from the forward pass.
    grad_mesh_fft : wp.array, shape (B, nx, ny, nz_rfft), dtype=vec2f or vec2d
        OUTPUT: Gradient w.r.t. ``mesh_fft``.
    grad_alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system gradient w.r.t. ``alpha``. Must be zero-initialized; kernel
        accumulates atomically across k-points.
    grad_volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system gradient w.r.t. ``volumes``. Must be zero-initialized; kernel
        accumulates atomically across k-points.
    grad_k_squared : wp.array, shape (B, nx, ny, nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: Gradient w.r.t. ``k_squared``. Written elementwise; no zero-init required.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str or None, optional
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.batch_pme_convolve` : Corresponding forward pass.
    """
    num_systems = mesh_fft.shape[0]
    nx, ny, nz_rfft = mesh_fft.shape[1], mesh_fft.shape[2], mesh_fft.shape[3]
    kernel = _get_pme_factory_kernel(
        wp_dtype, component="pme_convolve", batched=True, order="backward"
    )
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


def pme_convolve_double_backward(
    h_grad_mesh: wp.array,
    h_alpha: wp.array,
    h_volume: wp.array,
    h_grad_ksq: wp.array,
    mesh_fft: wp.array,
    grad_convolved: wp.array,
    k_squared: wp.array,
    moduli_x: wp.array,
    moduli_y: wp.array,
    moduli_z: wp.array,
    alpha: wp.array,
    volume: wp.array,
    grad_mesh_out: wp.array,
    grad_grad_convolved: wp.array,
    grad_k_squared_out: wp.array,
    grad_alpha_out: wp.array,
    grad_volume_out: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Single-system double-backward for ``pme_convolve``.

    Emits position-relevant second-order terms and cell/stress second-order terms.

    Parameters
    ----------
    h_grad_mesh : wp.array, shape (nx, ny, nz_rfft), dtype=vec2f or vec2d
        Incoming cotangent for the ``grad_mesh_fft`` output of the backward pass.
    h_alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_alpha`` output of the backward pass.
    h_volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_volume`` output of the backward pass.
    h_grad_ksq : wp.array, shape (nx, ny, nz_rfft), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_k_squared`` output of the backward pass.
    mesh_fft : wp.array, shape (nx, ny, nz_rfft), dtype=vec2f or vec2d
        Forward mesh values saved from the original forward pass.
    grad_convolved : wp.array, shape (nx, ny, nz_rfft), dtype=vec2f or vec2d
        Cotangent of the convolved mesh from the backward pass.
    k_squared : wp.array, shape (nx, ny, nz_rfft), dtype=wp.float32 or wp.float64
        Squared k-vector magnitudes from the forward pass.
    moduli_x : wp.array, shape (nx,), dtype=wp.float32 or wp.float64
        B-spline modulus LUT along x.
    moduli_y : wp.array, shape (ny,), dtype=wp.float32 or wp.float64
        B-spline modulus LUT along y.
    moduli_z : wp.array, shape (nz_rfft,), dtype=wp.float32 or wp.float64
        B-spline modulus LUT along z.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter from the forward pass.
    volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Unit cell volume from the forward pass.
    grad_mesh_out : wp.array, shape (nx, ny, nz_rfft), dtype=vec2f or vec2d
        OUTPUT: Second-order gradient w.r.t. ``mesh_fft`` (dL/dmesh_fft).
    grad_grad_convolved : wp.array, shape (nx, ny, nz_rfft), dtype=vec2f or vec2d
        OUTPUT: Second-order gradient w.r.t. ``grad_convolved`` (dL/dgrad_convolved).
    grad_k_squared_out : wp.array, shape (nx, ny, nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: Per-k cell/stress second-order gradient (dL/ds per k-point).
        Must be zero-initialized before launch.
    grad_alpha_out : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        OUTPUT: Second-order gradient w.r.t. ``alpha``. Must be zero-initialized;
        accumulated atomically over k-points.
    grad_volume_out : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        OUTPUT: Second-order gradient w.r.t. ``volume``. Must be zero-initialized;
        accumulated atomically over k-points.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str or None, optional
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.pme_convolve_backward` : First-order backward pass.
    """
    nx, ny, nz_rfft = mesh_fft.shape[0], mesh_fft.shape[1], mesh_fft.shape[2]
    kernel = _get_pme_factory_kernel(
        wp_dtype, component="pme_convolve", order="double_backward"
    )
    wp.launch(
        kernel,
        dim=(nx, ny, nz_rfft),
        inputs=[
            h_grad_mesh,
            h_alpha,
            h_volume,
            h_grad_ksq,
            mesh_fft,
            grad_convolved,
            k_squared,
            moduli_x,
            moduli_y,
            moduli_z,
            alpha,
            volume,
        ],
        outputs=[
            grad_mesh_out,
            grad_grad_convolved,
            grad_k_squared_out,
            grad_alpha_out,
            grad_volume_out,
        ],
        device=device,
    )


def batch_pme_convolve_double_backward(
    h_grad_mesh: wp.array,
    h_alpha: wp.array,
    h_volume: wp.array,
    h_grad_ksq: wp.array,
    mesh_fft: wp.array,
    grad_convolved: wp.array,
    k_squared: wp.array,
    moduli_x: wp.array,
    moduli_y: wp.array,
    moduli_z: wp.array,
    alpha: wp.array,
    volume: wp.array,
    grad_mesh_out: wp.array,
    grad_grad_convolved: wp.array,
    grad_k_squared_out: wp.array,
    grad_alpha_out: wp.array,
    grad_volume_out: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Batched double-backward for ``batch_pme_convolve``.

    Parameters
    ----------
    h_grad_mesh : wp.array, shape (B, nx, ny, nz_rfft), dtype=vec2f or vec2d
        Incoming cotangent for the ``grad_mesh_fft`` output of the backward pass.
    h_alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the per-system ``grad_alpha`` output of the backward pass.
    h_volume : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the per-system ``grad_volume`` output of the backward pass.
    h_grad_ksq : wp.array, shape (B, nx, ny, nz_rfft), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_k_squared`` output of the backward pass.
    mesh_fft : wp.array, shape (B, nx, ny, nz_rfft), dtype=vec2f or vec2d
        Per-system forward mesh values saved from the original forward pass.
    grad_convolved : wp.array, shape (B, nx, ny, nz_rfft), dtype=vec2f or vec2d
        Per-system cotangent of the convolved mesh from the backward pass.
    k_squared : wp.array, shape (B, nx, ny, nz_rfft), dtype=wp.float32 or wp.float64
        Per-system squared k-vector magnitudes from the forward pass.
    moduli_x : wp.array, shape (nx,), dtype=wp.float32 or wp.float64
        B-spline modulus LUT along x (shared across systems).
    moduli_y : wp.array, shape (ny,), dtype=wp.float32 or wp.float64
        B-spline modulus LUT along y (shared across systems).
    moduli_z : wp.array, shape (nz_rfft,), dtype=wp.float32 or wp.float64
        B-spline modulus LUT along z (shared across systems).
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter from the forward pass.
    volume : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system unit cell volumes from the forward pass.
    grad_mesh_out : wp.array, shape (B, nx, ny, nz_rfft), dtype=vec2f or vec2d
        OUTPUT: Second-order gradient w.r.t. ``mesh_fft``.
    grad_grad_convolved : wp.array, shape (B, nx, ny, nz_rfft), dtype=vec2f or vec2d
        OUTPUT: Second-order gradient w.r.t. ``grad_convolved``.
    grad_k_squared_out : wp.array, shape (B, nx, ny, nz_rfft), dtype=wp.float32 or wp.float64
        OUTPUT: Per-k cell/stress second-order gradient. Must be zero-initialized.
    grad_alpha_out : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system second-order gradient w.r.t. ``alpha``. Must be
        zero-initialized; accumulated atomically over k-points.
    grad_volume_out : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system second-order gradient w.r.t. ``volume``. Must be
        zero-initialized; accumulated atomically over k-points.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str or None, optional
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.batch_pme_convolve_backward` : First-order backward pass.
    """
    num_systems = mesh_fft.shape[0]
    nx, ny, nz_rfft = mesh_fft.shape[1], mesh_fft.shape[2], mesh_fft.shape[3]
    kernel = _get_pme_factory_kernel(
        wp_dtype, component="pme_convolve", batched=True, order="double_backward"
    )
    wp.launch(
        kernel,
        dim=(num_systems, nx, ny, nz_rfft),
        inputs=[
            h_grad_mesh,
            h_alpha,
            h_volume,
            h_grad_ksq,
            mesh_fft,
            grad_convolved,
            k_squared,
            moduli_x,
            moduli_y,
            moduli_z,
            alpha,
            volume,
        ],
        outputs=[
            grad_mesh_out,
            grad_grad_convolved,
            grad_k_squared_out,
            grad_alpha_out,
            grad_volume_out,
        ],
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
    r"""Apply self-energy and background corrections to PME energies.

    Framework-agnostic launcher for single-system energy corrections.

    Converts raw potential values :math:`\phi_i` to corrected per-atom energies by:

    1. Multiplying potential by charge: :math:`E_{pot} = q_i \cdot \phi_i`
    2. Subtracting self-energy: :math:`E_{self} = (\alpha/\sqrt{\pi}) \cdot q_i^2`
    3. Subtracting background: :math:`E_{bg} = (\pi/(2\alpha^2 V)) \cdot q_i \cdot Q_{total}`

    Final:

    .. math::

        E_i = q_i \cdot \phi_i - \frac{\alpha}{\sqrt{\pi}} q_i^2 - \frac{\pi}{2\alpha^2 V} q_i Q_{total}


    Parameters
    ----------
    raw_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Raw potential values :math:`\phi_i` from mesh interpolation.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Unit cell volume.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    total_charge : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Sum of all charges (:math:`Q_{total} = \sum_i q_i`).
    corrected_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Corrected per-atom energies.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = raw_energies.shape[0]
    launch_device = device if device is not None else str(raw_energies.device)
    sentinels = _get_pme_factory_sentinels(wp_dtype, launch_device)

    kernel = _get_pme_factory_kernel(wp_dtype, component="pme_corrections")
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[
            raw_energies,
            charges,
            sentinels["batch_idx"],
            volume,
            alpha,
            total_charge,
        ],
        outputs=[corrected_energies, sentinels["atoms"]],
        device=launch_device,
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
    r"""Apply self-energy and background corrections for batched PME.

    Framework-agnostic launcher for batched energy corrections.
    Each atom looks up its system's parameters via batch_idx.

    Parameters
    ----------
    raw_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Raw potential values :math:`\phi_i` from mesh interpolation.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system unit cell volume.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    total_charges : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system sum of charges (:math:`Q_s = \sum_{i \in s} q_i`).
    corrected_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Corrected per-atom energies.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = raw_energies.shape[0]
    launch_device = device if device is not None else str(raw_energies.device)
    sentinels = _get_pme_factory_sentinels(wp_dtype, launch_device)

    kernel = _get_pme_factory_kernel(
        wp_dtype, component="pme_corrections", batched=True
    )
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[raw_energies, charges, batch_idx, volumes, alpha, total_charges],
        outputs=[corrected_energies, sentinels["atoms"]],
        device=launch_device,
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
    """Single-system launcher for factory-backed PME correction backward.

    Parameters
    ----------
    grad_E : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Cotangent of the corrected per-atom energies.
    raw_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Raw potential values :math:`\\phi_i` saved from the forward pass.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges saved from the forward pass.
    volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Unit cell volume saved from the forward pass.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter saved from the forward pass.
    total_charge : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Sum of all charges saved from the forward pass.
    grad_raw : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Gradient w.r.t. ``raw_energies``.
    grad_charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Gradient w.r.t. ``charges``.
    grad_volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        OUTPUT: Gradient w.r.t. ``volume``. Must be zero-initialized.
    grad_alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        OUTPUT: Gradient w.r.t. ``alpha``. Must be zero-initialized.
    grad_total_charge : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        OUTPUT: Gradient w.r.t. ``total_charge``. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str or None, optional
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.pme_energy_corrections` : Corresponding forward pass.
    """
    launch_device = device if device is not None else str(raw_energies.device)
    sentinels = _get_pme_factory_sentinels(wp_dtype, launch_device)

    kernel = _get_pme_factory_kernel(
        wp_dtype, component="pme_corrections", order="backward"
    )
    wp.launch(
        kernel,
        dim=raw_energies.shape[0],
        inputs=[
            grad_E,
            raw_energies,
            charges,
            sentinels["batch_idx"],
            volume,
            alpha,
            total_charge,
        ],
        outputs=[
            grad_raw,
            grad_charges,
            grad_volume,
            grad_alpha,
            grad_total_charge,
        ],
        device=launch_device,
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
    """Batched launcher for factory-backed PME correction backward.

    Parameters
    ----------
    grad_E : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Cotangent of the corrected per-atom energies.
    raw_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Raw potential values :math:`\\phi_i` saved from the forward pass.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated, saved from the forward pass.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system unit cell volumes saved from the forward pass.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter saved from the forward pass.
    total_charges : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system sum of charges saved from the forward pass.
    grad_raw : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Gradient w.r.t. ``raw_energies``.
    grad_charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Gradient w.r.t. ``charges``.
    grad_volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system gradient w.r.t. ``volumes``. Must be zero-initialized.
    grad_alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system gradient w.r.t. ``alpha``. Must be zero-initialized.
    grad_total_charges : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system gradient w.r.t. ``total_charges``. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str or None, optional
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.batch_pme_energy_corrections` : Corresponding forward pass.
    """
    launch_device = device if device is not None else str(raw_energies.device)

    kernel = _get_pme_factory_kernel(
        wp_dtype, component="pme_corrections", batched=True, order="backward"
    )
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
        device=launch_device,
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
    """Single-system launcher for factory-backed PME correction double-backward.

    Parameters
    ----------
    h_raw : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_raw`` output of the backward pass.
    h_chg : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_charges`` output of the backward pass.
    h_vol : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_volume`` output of the backward pass.
    h_alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_alpha`` output of the backward pass.
    h_qtot : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_total_charge`` output of the backward pass.
    grad_E : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Cotangent of the corrected per-atom energies from the backward pass.
    raw_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Raw potential values :math:`\\phi_i` saved from the original forward pass.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges saved from the original forward pass.
    volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Unit cell volume saved from the original forward pass.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter saved from the original forward pass.
    total_charge : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Sum of all charges saved from the original forward pass.
    grad_grad_E : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Second-order gradient w.r.t. ``grad_E``.
    grad_raw : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Second-order gradient w.r.t. ``raw_energies``.
    grad_charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Second-order gradient w.r.t. ``charges``.
    grad_volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        OUTPUT: Second-order gradient w.r.t. ``volume``. Must be zero-initialized.
    grad_alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        OUTPUT: Second-order gradient w.r.t. ``alpha``. Must be zero-initialized.
    grad_total_charge : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        OUTPUT: Second-order gradient w.r.t. ``total_charge``. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str or None, optional
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.pme_energy_corrections_backward` : First-order backward pass.
    """
    launch_device = device if device is not None else str(raw_energies.device)
    sentinels = _get_pme_factory_sentinels(wp_dtype, launch_device)

    kernel = _get_pme_factory_kernel(
        wp_dtype, component="pme_corrections", order="double_backward"
    )
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
            sentinels["batch_idx"],
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
        device=launch_device,
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
    """Batched launcher for factory-backed PME correction double-backward.

    Parameters
    ----------
    h_raw : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_raw`` output of the backward pass.
    h_chg : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_charges`` output of the backward pass.
    h_vol : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_volumes`` output of the backward pass.
    h_alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_alpha`` output of the backward pass.
    h_qtot : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Incoming cotangent for the ``grad_total_charges`` output of the backward pass.
    grad_E : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Cotangent of the corrected per-atom energies from the backward pass.
    raw_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Raw potential values :math:`\\phi_i` saved from the original forward pass.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated, saved from the original forward pass.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system unit cell volumes saved from the original forward pass.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter saved from the original forward pass.
    total_charges : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system sum of charges saved from the original forward pass.
    grad_grad_E : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Second-order gradient w.r.t. ``grad_E``.
    grad_raw : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Second-order gradient w.r.t. ``raw_energies``.
    grad_charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Second-order gradient w.r.t. ``charges``.
    grad_volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system second-order gradient w.r.t. ``volumes``. Must be zero-initialized.
    grad_alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system second-order gradient w.r.t. ``alpha``. Must be zero-initialized.
    grad_total_charges : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system second-order gradient w.r.t. ``total_charges``. Must be zero-initialized.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str or None, optional
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.batch_pme_energy_corrections_backward` : First-order backward pass.
    """
    launch_device = device if device is not None else str(raw_energies.device)

    kernel = _get_pme_factory_kernel(
        wp_dtype, component="pme_corrections", batched=True, order="double_backward"
    )
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
        device=launch_device,
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
    r"""Apply corrections and compute charge gradients for PME energies.

    Framework-agnostic launcher for single-system energy corrections
    with analytical charge gradient computation.

    Computes both corrected energies and analytical charge gradients:

    - Energy: :math:`E_i = q_i \phi_i - (\alpha/\sqrt{\pi}) q_i^2 - (\pi/(2\alpha^2 V)) q_i Q_{total}`
    - Charge gradient: :math:`\partial E_{total}/\partial q_i = 2\phi_i - 2(\alpha/\sqrt{\pi})q_i - (\pi/(\alpha^2 V))Q_{total}`

    Parameters
    ----------
    raw_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Raw potential values :math:`\phi_i` from mesh interpolation.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    volume : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Unit cell volume.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    total_charge : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Sum of all charges (:math:`Q_{total} = \sum_i q_i`).
    corrected_energies : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Corrected per-atom energies.
    charge_gradients : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Analytical charge gradients :math:`\partial E_{total}/\partial q_i`.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = raw_energies.shape[0]
    launch_device = device if device is not None else str(raw_energies.device)
    sentinels = _get_pme_factory_sentinels(wp_dtype, launch_device)

    kernel = _get_pme_factory_kernel(
        wp_dtype, component="pme_corrections", charge_grad=True
    )
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[
            raw_energies,
            charges,
            sentinels["batch_idx"],
            volume,
            alpha,
            total_charge,
        ],
        outputs=[corrected_energies, charge_gradients],
        device=launch_device,
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
    r"""Apply corrections and compute charge gradients for batched PME.

    Framework-agnostic launcher for batched energy corrections
    with analytical charge gradient computation.

    Parameters
    ----------
    raw_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Raw potential values :math:`\phi_i` from mesh interpolation.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    volumes : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system unit cell volume.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    total_charges : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system sum of charges (:math:`Q_s = \sum_{i \in s} q_i`).
    corrected_energies : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Corrected per-atom energies.
    charge_gradients : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        OUTPUT: Analytical charge gradients :math:`\partial E_{total}/\partial q_i`.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    num_atoms = raw_energies.shape[0]
    launch_device = device if device is not None else str(raw_energies.device)

    kernel = _get_pme_factory_kernel(
        wp_dtype, component="pme_corrections", batched=True, charge_grad=True
    )
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[raw_energies, charges, batch_idx, volumes, alpha, total_charges],
        outputs=[corrected_energies, charge_gradients],
        device=launch_device,
    )


def pme_virial_bg_correction(
    charges: wp.array,
    batch_idx: wp.array,
    cell: wp.array,
    volume: wp.array,
    use_supplied_volume: bool,
    alpha: wp.array,
    total_charges: wp.array,
    virial_in: wp.array,
    virial_out: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Apply non-neutral background virial correction.

    Two-pass launcher: pass 1 scatter-adds per-atom ``charges`` into per-system
    ``total_charges`` via atomic_add; pass 2 computes
    :math:`E_{bg} = \pi Q^2 / (2 \alpha^2 V)` and subtracts it from the three
    diagonal entries of ``virial_in``, writing the result to ``virial_out``.
    For a single system, fill ``batch_idx`` with zeros.

    Parameters
    ----------
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Per-atom charges.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell : wp.array, shape (B, 3, 3), dtype=wp.float32 or wp.float64
        Per-system unit cell matrix; volume computed as ``|det(cell[s])|``
        when ``use_supplied_volume`` is False.
    volume : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Caller-supplied per-system volume. Used only when ``use_supplied_volume``
        is True; otherwise a dummy array is acceptable.
    use_supplied_volume : bool
        If True, use values from ``volume`` rather than computing from ``cell``.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    total_charges : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system total charge accumulator. Must be zero-initialized by the caller;
        written in pass 1 via atomic_add.
    virial_in : wp.array, shape (B, 3, 3), dtype=wp.float32 or wp.float64
        Input virial tensor per system.
    virial_out : wp.array, shape (B, 3, 3), dtype=wp.float32 or wp.float64
        OUTPUT: ``virial_in`` with :math:`E_{bg}` subtracted from diagonal entries.
        May alias ``virial_in`` for in-place operation.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str or None, optional
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.pme_virial_bg_correction_backward` : Backward pass.
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
        inputs=[
            total_charges,
            cell,
            volume,
            int(use_supplied_volume),
            alpha,
            virial_in,
            virial_out,
        ],
        device=device,
    )


def pme_virial_bg_correction_backward(
    grad_virial: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    cell: wp.array,
    volume: wp.array,
    use_supplied_volume: bool,
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

    Three-pass launcher:

    1. Reduce per-atom ``charges`` into ``total_charges`` (Q per system).
    2. Per-system: turn cotangent ``grad_virial`` into ``grad_total_charges``,
       ``grad_alpha``, and ``grad_cell`` via :math:`dE_{bg}/dQ`, :math:`dE_{bg}/d\\alpha`,
       and Jacobi's formula for :math:`d|\\det C|/dC`.
    3. Per-atom: scatter ``grad_total_charges[s(j)]`` to ``grad_charges[j]``.

    Parameters
    ----------
    grad_virial : wp.array, shape (B, 3, 3), dtype=wp.float32 or wp.float64
        Cotangent of the ``virial_out`` output.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Per-atom charges (same as forward pass).
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cell : wp.array, shape (B, 3, 3), dtype=wp.float32 or wp.float64
        Per-system unit cell matrix from the forward pass.
    volume : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Caller-supplied per-system volume. Used only when ``use_supplied_volume`` is True.
    use_supplied_volume : bool
        If True, use values from ``volume`` rather than computing from ``cell``.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter from the forward pass.
    total_charges : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system total charge accumulator. Must be zero-initialized; written in pass 1.
    grad_total_charges : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system gradient w.r.t. total charge. Must be zero-initialized.
    grad_charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-atom gradient w.r.t. ``charges``. Must be zero-initialized.
    grad_alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system gradient w.r.t. ``alpha``. Must be zero-initialized.
    grad_cell : wp.array, shape (B, 3, 3), dtype=wp.float32 or wp.float64
        OUTPUT: Per-system gradient w.r.t. ``cell``. Must be zero-initialized.
        Zero when ``use_supplied_volume`` is True.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str or None, optional
        Warp device string. If None, inferred from arrays.

    See Also
    --------
    :func:`nvalchemiops.interactions.electrostatics.pme_kernels.pme_virial_bg_correction` : Corresponding forward pass.
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
            volume,
            int(use_supplied_volume),
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
    # Warp launchers
    "pme_green_structure_factor",
    "batch_pme_green_structure_factor",
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
