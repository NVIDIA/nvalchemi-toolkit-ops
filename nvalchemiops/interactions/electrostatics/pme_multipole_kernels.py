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
Multipole Particle-Mesh Ewald (PME) Warp kernels.

Framework-agnostic Warp launchers that extend the monopole PME
infrastructure (``pme_kernels.py``, ``nvalchemiops/math/spline.py``) to
multipole sources (charges, dipoles, and quadrupoles).

The reciprocal-space pipeline mirrors monopole PME — spread → FFT →
Green's function multiply → inverse FFT → gather — with the same FFT
plumbing (``torch.fft.rfftn`` / ``irfftn``) and aliasing correction. The
dipole branch enters at the spread step (the density gains an extra
:math:`\boldsymbol{\mu}_i \cdot \nabla B_p(r_\text{grid} - r_i)` term)
and exits at the gather step (forces on dipoles need
:math:`\nabla^2\phi` in addition to :math:`\nabla\phi`). Each Warp
kernel is paired with one ``torch.autograd.Function`` in
``nvalchemiops/torch/interactions/electrostatics/pme_multipole.py``.
"""

from __future__ import annotations

import math
from typing import Any

import warp as wp

from nvalchemiops.interactions.electrostatics.pme_kernels import (
    compute_sinc,
)
from nvalchemiops.math.spline import (
    bspline_derivative,
    bspline_fourth_derivative,
    bspline_grid_offset,
    bspline_second_derivative,
    bspline_third_derivative,
    bspline_weight,
    bspline_weight_hessian_3d,
    compute_fractional_coords,
    wrap_grid_index,
)
from nvalchemiops.warp_dispatch import register_overloads

_TWOPI = 2.0 * math.pi


@wp.kernel(enable_backward=False)
def _bspline_moduli_1d_kernel(
    miller: wp.array(dtype=wp.float64),  # (n_axis,) integer Miller indices (float)
    inv_n: wp.float64,  # 1 / mesh dim along this axis
    order: wp.int32,  # B-spline order p
    moduli: wp.array(dtype=wp.float64),  # (n_axis,) OUTPUT b[i] = sinc(m_i/N)^p
):
    r"""Fill a 1-D B-spline modulus LUT ``b[i] = sinc(m_i / N)^order``.

    One thread per axis index. Reuses the monopole PME ``compute_sinc``
    ``@wp.func`` (``pme_kernels.py``); precomputing the per-axis 1-D LUT lets the
    multipole Green's kernel read 3 entries + 2 multiplies instead of
    recomputing ``sinc`` at every grid point (the MZ-1 LUT optimization, now
    Warp-native — replaces the previous ``torch.sinc`` host compute).
    """
    i = wp.tid()
    s = compute_sinc(miller[i] * inv_n)
    val = wp.float64(1.0)
    for _ in range(order):
        val = val * s
    moduli[i] = val


def bspline_moduli_1d_launch(
    miller: wp.array,
    inv_n: float,
    order: int,
    moduli: wp.array,
    device: str | None = None,
) -> None:
    """Launcher for :func:`_bspline_moduli_1d_kernel` (one thread per axis index)."""
    if device is None:
        device = str(miller.device)
    wp.launch(
        _bspline_moduli_1d_kernel,
        dim=miller.shape[0],
        inputs=[miller, wp.float64(inv_n), wp.int32(order), moduli],
        device=device,
    )


# =============================================================================
# Single-system spread
# =============================================================================


def multipole_pme_spread_launch(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Single-system Multipole-PME spread launcher (framework-agnostic).

    Spreads each atom's charge + dipole onto the B-spline mesh:
    :math:`\rho[g] \mathrel{+}= q_i B_p(g - r_i) + \boldsymbol{\mu}_i \cdot
    \nabla_{\text{cart}} B_p(g - r_i)`. Caller is responsible for
    allocating the mesh pre-zeroed at the desired dtype. Order is passed
    as a Python int and forwarded to the per-order specialized kernel.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=vec3f/vec3d
        Cartesian atom positions.
    charges : wp.array, shape (N,), dtype=wp.float32/float64
        Per-atom monopole charges.
    dipoles : wp.array, shape (N,), dtype=vec3f/vec3d
        Per-atom Cartesian dipole vectors.
    cell_inv_t : wp.array, shape (1,), dtype=mat33f/mat33d
        Transpose of the inverse cell matrix (fractional → Cartesian map).
    order : int
        B-spline order (one of ``_PER_ORDER_SUPPORTED`` = ``(3, 4, 5, 6)``).
    mesh : wp.array3d, shape (Nx, Ny, Nz), dtype=wp.float32/float64
        OUTPUT. Charge-density mesh, pre-zeroed; accumulated via atomics.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom; each emits ``order**3`` atomic
    adds into ``mesh``.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    per_order = _maybe_per_order_spread_kernel(order, wp_dtype)
    if per_order is None:
        raise NotImplementedError(
            f"multipole_pme_spread_launch: no per-order kernel registered for "
            f"(order={order}, dtype={wp_dtype}); supported orders are "
            f"{_PER_ORDER_SUPPORTED}."
        )
    wp.launch(
        per_order,
        dim=(num_atoms,),
        inputs=[positions, charges, dipoles, cell_inv_t],
        outputs=[mesh],
        device=device,
    )


# =============================================================================
# Single-system spread backward
# =============================================================================


def multipole_pme_spread_backward_launch(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    cell_inv_t: wp.array,
    order: int,
    grad_mesh: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_dipoles: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Single-system Multipole-PME spread-backward launcher.

    Backward of :func:`multipole_pme_spread_launch`: given the upstream
    gradient ``grad_mesh`` (:math:`\partial L / \partial \rho`),
    accumulates gradients w.r.t. positions, charges, and dipoles by
    contracting against the same B-spline stencil weights / derivatives.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=vec3f/vec3d
        Cartesian atom positions.
    charges : wp.array, shape (N,), dtype=wp.float32/float64
        Per-atom monopole charges.
    dipoles : wp.array, shape (N,), dtype=vec3f/vec3d
        Per-atom Cartesian dipole vectors.
    cell_inv_t : wp.array, shape (1,), dtype=mat33f/mat33d
        Transpose of the inverse cell matrix.
    order : int
        B-spline order (one of ``(3, 4, 5, 6)``).
    grad_mesh : wp.array3d, shape (Nx, Ny, Nz), dtype=wp.float32/float64
        Upstream gradient w.r.t. the spread mesh, :math:`\partial L /
        \partial \rho`.
    grad_positions : wp.array, shape (N,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Gradient w.r.t. positions.
    grad_charges : wp.array, shape (N,), dtype=wp.float32/float64
        OUTPUT, pre-zeroed. Gradient w.r.t. charges.
    grad_dipoles : wp.array, shape (N,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Gradient w.r.t. dipoles.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    per_order = _maybe_per_order_spread_backward_kernel(order, wp_dtype)
    if per_order is None:
        raise NotImplementedError(
            f"multipole_pme_spread_backward_launch: no per-order kernel registered "
            f"for (order={order}, dtype={wp_dtype}); supported orders are "
            f"{_PER_ORDER_SUPPORTED}."
        )
    wp.launch(
        per_order,
        dim=(num_atoms,),
        inputs=[positions, charges, dipoles, cell_inv_t, grad_mesh],
        outputs=[grad_positions, grad_charges, grad_dipoles],
        device=device,
    )


# =============================================================================
# Batched spread
# =============================================================================


def batch_multipole_pme_spread_launch(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Batched Multipole-PME spread launcher (B systems, ragged atoms).

    Batched analog of :func:`multipole_pme_spread_launch`: each atom is
    routed to its system's mesh slice via ``batch_idx``. Caller pre-zeros
    the ``(B, Nx, Ny, Nz)`` mesh.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=vec3f/vec3d
        Concatenated Cartesian atom positions across all systems.
    charges : wp.array, shape (N_total,), dtype=wp.float32/float64
        Per-atom monopole charges.
    dipoles : wp.array, shape (N_total,), dtype=vec3f/vec3d
        Per-atom Cartesian dipole vectors.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        Per-atom system index into the leading mesh / ``cell_inv_t`` axis.
    cell_inv_t : wp.array, shape (B,), dtype=mat33f/mat33d
        Per-system transpose of the inverse cell matrix.
    order : int
        B-spline order (one of ``(3, 4, 5, 6)``).
    mesh : wp.array, shape (B, Nx, Ny, Nz), dtype=wp.float32/float64
        OUTPUT. Per-system charge-density mesh, pre-zeroed.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Launch Grid
    -----------
    ``dim = (N_total,)`` — one thread per atom across the batch.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    per_order = _maybe_per_order_batch_spread_kernel(order, wp_dtype)
    if per_order is None:
        raise NotImplementedError(
            f"batch_multipole_pme_spread_launch: no per-order kernel registered "
            f"for (order={order}, dtype={wp_dtype}); supported orders are "
            f"{_PER_ORDER_SUPPORTED}."
        )
    wp.launch(
        per_order,
        dim=(num_atoms,),
        inputs=[positions, charges, dipoles, batch_idx, cell_inv_t],
        outputs=[mesh],
        device=device,
    )


# =============================================================================
# Batched spread backward
# =============================================================================


def batch_multipole_pme_spread_backward_launch(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    grad_mesh: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_dipoles: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Batched Multipole-PME spread-backward launcher.

    Batched analog of :func:`multipole_pme_spread_backward_launch`.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=vec3f/vec3d
        Concatenated Cartesian atom positions.
    charges : wp.array, shape (N_total,), dtype=wp.float32/float64
        Per-atom monopole charges.
    dipoles : wp.array, shape (N_total,), dtype=vec3f/vec3d
        Per-atom Cartesian dipole vectors.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        Per-atom system index into the leading mesh axis.
    cell_inv_t : wp.array, shape (B,), dtype=mat33f/mat33d
        Per-system transpose of the inverse cell matrix.
    order : int
        B-spline order (one of ``(3, 4, 5, 6)``).
    grad_mesh : wp.array, shape (B, Nx, Ny, Nz), dtype=wp.float32/float64
        Upstream gradient w.r.t. the per-system spread mesh.
    grad_positions : wp.array, shape (N_total,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Gradient w.r.t. positions.
    grad_charges : wp.array, shape (N_total,), dtype=wp.float32/float64
        OUTPUT, pre-zeroed. Gradient w.r.t. charges.
    grad_dipoles : wp.array, shape (N_total,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Gradient w.r.t. dipoles.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Launch Grid
    -----------
    ``dim = (N_total,)`` — one thread per atom across the batch.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    per_order = _maybe_per_order_batch_spread_backward_kernel(order, wp_dtype)
    if per_order is None:
        raise NotImplementedError(
            f"batch_multipole_pme_spread_backward_launch: no per-order kernel "
            f"registered for (order={order}, dtype={wp_dtype}); supported orders "
            f"are {_PER_ORDER_SUPPORTED}."
        )
    wp.launch(
        per_order,
        dim=(num_atoms,),
        inputs=[positions, charges, dipoles, batch_idx, cell_inv_t, grad_mesh],
        outputs=[grad_positions, grad_charges, grad_dipoles],
        device=device,
    )


# =============================================================================
# Single-system Green's function + structure factor
# =============================================================================


@wp.kernel(enable_backward=False)
def _pme_multipole_green_structure_factor_kernel(
    k_squared: wp.array3d(dtype=Any),
    miller_x: wp.array(dtype=Any),
    miller_y: wp.array(dtype=Any),
    miller_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    mesh_nx: wp.int32,
    mesh_ny: wp.int32,
    mesh_nz: wp.int32,
    spline_order: wp.int32,
    green_function: wp.array3d(dtype=Any),
    structure_factor_sq: wp.array3d(dtype=Any),
):
    r"""Multipole PME Green's function + B-spline structure factor.

    Extension of the monopole PME Green's function with one extra factor
    :math:`e^{-\sigma^2 k^2}` to account for the GTO smearing of **both**
    the source and receiver multipoles in the pair sum (each carries a
    GTO Fourier transform :math:`e^{-\sigma^2 k^2/2}`; the pair-sum
    convolution produces :math:`e^{-\sigma^2 k^2}`):

    .. math::

        \tilde{G}(k) = \frac{2\pi \, e^{-k^2/(4\alpha^2)} \, e^{-\sigma^2 k^2}}{V \, k^2}

    At :math:`\sigma = 0` this collapses to the monopole Green's function
    bit-for-bit. The structure factor :math:`|C(k)|^2` is unchanged from
    monopole PME: B-spline aliasing depends on the spline order and grid
    geometry but not on the source distribution.

    Launch Grid
    -----------
    ``dim = [Nx, Ny, Nz_rfft]`` — one thread per k-grid point in the
    rfft half-space.

    Parameters
    ----------
    k_squared : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Squared magnitude of the k-vector at each rfft grid point.
    miller_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        Miller indices along x (from ``fftfreq``).
    miller_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        Miller indices along y (from ``fftfreq``).
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        Miller indices along z (from ``rfftfreq``).
    alpha : wp.array, shape (1,), dtype=wp.float32/float64
        Ewald splitting parameter :math:`\alpha`.
    sigma : wp.array, shape (1,), dtype=wp.float32/float64
        GTO density-basis Gaussian width :math:`\sigma`. Pass zeros for a
        pure-monopole call.
    volume : wp.array, shape (1,), dtype=wp.float32/float64
        Unit cell volume :math:`V`.
    mesh_nx : wp.int32
        Full mesh size along x (used for sinc dealiasing).
    mesh_ny : wp.int32
        Full mesh size along y.
    mesh_nz : wp.int32
        Full mesh size along z (full size, not the rfft size).
    spline_order : wp.int32
        B-spline order :math:`p` (1-6) used in charge assignment.
    green_function : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. Green's function :math:`\tilde{G}(k)` per grid point.
    structure_factor_sq : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. B-spline structure factor squared :math:`|C(k)|^2`.

    Notes
    -----
    - k=0 (grid point [0, 0, 0]) is set to zero (tin-foil boundary).
    - Near-zero :math:`k^2` (threshold 1e-10) sets the Green's function
      to zero to avoid division blow-up.
    - Structure factor clamped below 1e-10 to avoid division-by-zero
      downstream when the caller uses :math:`C^2` as a denominator.
    - ``rfft`` symmetry: only ``Nz_rfft = Nz // 2 + 1`` points in z.
    """
    i, j, k = wp.tid()

    k_sq = k_squared[i, j, k]
    alpha_ = alpha[0]
    sigma_ = sigma[0]
    volume_ = volume[0]
    mi_x = miller_x[i]
    mi_y = miller_y[j]
    mi_z = miller_z[k]

    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    four = type(k_sq)(4.0)

    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(_TWOPI)

    # GTO factor is σ² (not σ²/2): each side of the pair-sum convolution
    # contributes exp(-σ²k²/2), and the product gives σ².
    if k_sq < threshold:
        green_function[i, j, k] = zero
    else:
        combined_prefactor = one / (four * alpha_ * alpha_) + sigma_ * sigma_
        exp_factor = wp.exp(-combined_prefactor * k_sq) / k_sq
        green_function[i, j, k] = twopi * exp_factor / volume_

    if i == 0 and j == 0 and k == 0:
        green_function[i, j, k] = zero

    sinc_x = compute_sinc(mi_x / type(mi_x)(mesh_nx))
    sinc_y = compute_sinc(mi_y / type(mi_y)(mesh_ny))
    sinc_z = compute_sinc(mi_z / type(mi_z)(mesh_nz))
    sinc_product = sinc_x * sinc_y * sinc_z

    sf = sinc_product
    for _ in range(1, 6):  # Max supported order = 6
        if _ < spline_order:
            sf = sf * sinc_product

    if sf < clamp_threshold:
        sf = clamp_threshold
    structure_factor_sq[i, j, k] = sf * sf


def _pme_multipole_green_sig(v, t):
    """Signature builder for the multipole Green's function kernel.

    All scalar arrays share the input dtype ``t``. ``v`` is accepted for
    signature-builder uniformity with the rest of the Phase-3 kernels
    but not used by this kernel (no vec3 inputs).
    """
    del v
    return [
        wp.array3d(dtype=t),  # k_squared
        wp.array(dtype=t),  # miller_x
        wp.array(dtype=t),  # miller_y
        wp.array(dtype=t),  # miller_z
        wp.array(dtype=t),  # alpha
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # volume
        wp.int32,  # mesh_nx
        wp.int32,  # mesh_ny
        wp.int32,  # mesh_nz
        wp.int32,  # spline_order
        wp.array3d(dtype=t),  # green_function
        wp.array3d(dtype=t),  # structure_factor_sq
    ]


_pme_multipole_green_overloads = register_overloads(
    _pme_multipole_green_structure_factor_kernel, _pme_multipole_green_sig
)


def multipole_pme_green_structure_factor_launch(
    k_squared: wp.array,
    miller_x: wp.array,
    miller_y: wp.array,
    miller_z: wp.array,
    alpha: wp.array,
    sigma: wp.array,
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
    r"""Single-system launcher for the multipole Green's + structure-factor kernel.

    Thin wrapper around :func:`_pme_multipole_green_structure_factor_kernel`.
    Outputs are pre-allocated by the caller at the same dtype as
    ``k_squared``. Kernel overloads are registered for fp32 and fp64.

    Parameters
    ----------
    k_squared : wp.array, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Squared magnitude of the k-vector at each rfft grid point.
    miller_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        Miller indices along x.
    miller_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        Miller indices along y.
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        Miller indices along z (rfft half-space).
    alpha : wp.array, shape (1,), dtype=wp.float32/float64
        Ewald splitting parameter :math:`\alpha`.
    sigma : wp.array, shape (1,), dtype=wp.float32/float64
        GTO density-basis Gaussian width :math:`\sigma`.
    volume : wp.array, shape (1,), dtype=wp.float32/float64
        Unit cell volume :math:`V`.
    mesh_nx : int
        Full mesh size along x.
    mesh_ny : int
        Full mesh size along y.
    mesh_nz : int
        Full mesh size along z.
    spline_order : int
        B-spline order :math:`p`.
    green_function : wp.array, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. Green's function :math:`\tilde{G}(k)`.
    structure_factor_sq : wp.array, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. B-spline structure factor squared :math:`|C(k)|^2`.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``k_squared.device``.

    Launch Grid
    -----------
    ``dim = (Nx, Ny, Nz_rfft)`` — one thread per rfft k-grid point.
    """
    if device is None:
        device = str(k_squared.device)
    nx, ny, nz_rfft = k_squared.shape
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_multipole_green_overloads[vec_dtype],
        dim=(nx, ny, nz_rfft),
        inputs=[
            k_squared,
            miller_x,
            miller_y,
            miller_z,
            alpha,
            sigma,
            volume,
            wp.int32(mesh_nx),
            wp.int32(mesh_ny),
            wp.int32(mesh_nz),
            wp.int32(spline_order),
        ],
        outputs=[green_function, structure_factor_sq],
        device=device,
    )


# =============================================================================
# Batched Green's function + structure factor
# =============================================================================


@wp.kernel(enable_backward=False)
def _batch_pme_multipole_green_structure_factor_kernel(
    k_squared: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft)
    miller_x: wp.array(dtype=Any),  # (Nx,)
    miller_y: wp.array(dtype=Any),  # (Ny,)
    miller_z: wp.array(dtype=Any),  # (Nz_rfft,)
    alpha: wp.array(dtype=Any),  # (B,)
    sigma: wp.array(dtype=Any),  # (B,)
    volume: wp.array(dtype=Any),  # (B,)
    mesh_nx: wp.int32,
    mesh_ny: wp.int32,
    mesh_nz: wp.int32,
    spline_order: wp.int32,
    green_function: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft)
    structure_factor_sq: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft)
):
    r"""Batched multipole PME Green's function + B-spline structure factor.

    Per-system extension of the single-system Green's function kernel:

    .. math::

        \tilde{G}_b(k) = \frac{2\pi \, e^{-k^2/(4\alpha_b^2)} \, e^{-\sigma_b^2 k^2}}{V_b \, k^2}

    ``k_squared`` and ``green_function`` carry a leading batch axis;
    ``alpha``, ``sigma``, ``volume`` are per-system ``(B,)`` arrays.
    ``structure_factor_sq`` is shared across the batch
    (``(Nx, Ny, Nz_rfft)``) because B-spline aliasing depends only on
    mesh geometry; it is computed once at ``batch_idx == 0``.

    Launch Grid
    -----------
    ``dim = (B, Nx, Ny, Nz_rfft)`` — one thread per (system, k-grid point).

    Parameters
    ----------
    k_squared : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Per-system squared k-vector magnitude.
    miller_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        Miller indices along x (shared across systems).
    miller_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        Miller indices along y (shared across systems).
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        Miller indices along z (shared across systems).
    alpha : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system Ewald splitting parameter.
    sigma : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system GTO Gaussian width.
    volume : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system unit cell volume.
    mesh_nx : wp.int32
        Full mesh size along x.
    mesh_ny : wp.int32
        Full mesh size along y.
    mesh_nz : wp.int32
        Full mesh size along z.
    spline_order : wp.int32
        B-spline order :math:`p`.
    green_function : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. Per-system Green's function.
    structure_factor_sq : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. Shared structure factor squared (written only at
        ``batch_idx == 0``).
    """
    batch_idx, i, j, k = wp.tid()

    k_sq = k_squared[batch_idx, i, j, k]
    alpha_ = alpha[batch_idx]
    sigma_ = sigma[batch_idx]
    volume_ = volume[batch_idx]
    mi_x = miller_x[i]
    mi_y = miller_y[j]
    mi_z = miller_z[k]

    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    four = type(k_sq)(4.0)

    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(_TWOPI)

    if k_sq < threshold:
        green_function[batch_idx, i, j, k] = zero
    else:
        combined_prefactor = one / (four * alpha_ * alpha_) + sigma_ * sigma_
        exp_factor = wp.exp(-combined_prefactor * k_sq) / k_sq
        green_function[batch_idx, i, j, k] = twopi * exp_factor / volume_

    if i == 0 and j == 0 and k == 0:
        green_function[batch_idx, i, j, k] = zero

    # Structure factor depends only on mesh geometry; emit once at batch_idx=0.
    if batch_idx == wp.int32(0):
        sinc_x = compute_sinc(mi_x / type(mi_x)(mesh_nx))
        sinc_y = compute_sinc(mi_y / type(mi_y)(mesh_ny))
        sinc_z = compute_sinc(mi_z / type(mi_z)(mesh_nz))
        sinc_product = sinc_x * sinc_y * sinc_z

        sf = sinc_product
        for _ in range(1, 6):  # Max supported order = 6
            if _ < spline_order:
                sf = sf * sinc_product

        if sf < clamp_threshold:
            sf = clamp_threshold
        structure_factor_sq[i, j, k] = sf * sf


def _batch_pme_multipole_green_sig(v, t):
    """Signature builder for the batched multipole Green's-function kernel."""
    del v
    return [
        wp.array4d(dtype=t),  # k_squared
        wp.array(dtype=t),  # miller_x
        wp.array(dtype=t),  # miller_y
        wp.array(dtype=t),  # miller_z
        wp.array(dtype=t),  # alpha
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # volume
        wp.int32,  # mesh_nx
        wp.int32,  # mesh_ny
        wp.int32,  # mesh_nz
        wp.int32,  # spline_order
        wp.array4d(dtype=t),  # green_function
        wp.array3d(dtype=t),  # structure_factor_sq
    ]


_batch_pme_multipole_green_overloads = register_overloads(
    _batch_pme_multipole_green_structure_factor_kernel,
    _batch_pme_multipole_green_sig,
)


def batch_multipole_pme_green_structure_factor_launch(
    k_squared: wp.array,
    miller_x: wp.array,
    miller_y: wp.array,
    miller_z: wp.array,
    alpha: wp.array,
    sigma: wp.array,
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
    r"""Batched launcher for the multipole Green's + structure-factor kernel.

    Thin wrapper around
    :func:`_batch_pme_multipole_green_structure_factor_kernel`. Caller
    pre-allocates the ``(B, Nx, Ny, Nz_rfft)`` ``green_function`` and the
    shared ``(Nx, Ny, Nz_rfft)`` ``structure_factor_sq``.

    Parameters
    ----------
    k_squared : wp.array, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Per-system squared k-vector magnitude.
    miller_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        Miller indices along x (shared across systems).
    miller_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        Miller indices along y (shared across systems).
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        Miller indices along z (shared across systems).
    alpha : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system Ewald splitting parameter.
    sigma : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system GTO Gaussian width.
    volume : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system unit cell volume.
    mesh_nx : int
        Full mesh size along x.
    mesh_ny : int
        Full mesh size along y.
    mesh_nz : int
        Full mesh size along z.
    spline_order : int
        B-spline order :math:`p`.
    green_function : wp.array, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. Per-system Green's function.
    structure_factor_sq : wp.array, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. Shared structure factor squared.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``k_squared.device``.

    Launch Grid
    -----------
    ``dim = (B, Nx, Ny, Nz_rfft)`` — one thread per (system, k-grid point).
    """
    if device is None:
        device = str(k_squared.device)
    B, nx, ny, nz_rfft = k_squared.shape
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _batch_pme_multipole_green_overloads[vec_dtype],
        dim=(B, nx, ny, nz_rfft),
        inputs=[
            k_squared,
            miller_x,
            miller_y,
            miller_z,
            alpha,
            sigma,
            volume,
            wp.int32(mesh_nx),
            wp.int32(mesh_ny),
            wp.int32(mesh_nz),
            wp.int32(spline_order),
        ],
        outputs=[green_function, structure_factor_sq],
        device=device,
    )


# =============================================================================
# Single-system Hessian gather
# =============================================================================


@wp.kernel(enable_backward=False)
def _pme_multipole_gather_hessian_kernel(
    positions: wp.array(dtype=Any),
    cell_inv_t: wp.array(dtype=Any),
    order: wp.int32,
    mesh: wp.array3d(dtype=Any),
    hessian_diag: wp.array(dtype=Any),
    hessian_off: wp.array(dtype=Any),
):
    r"""Gather the Cartesian Hessian of ``φ`` at atom positions.

    For each atom, gathers

    .. math::

        H(r_i) = \sum_g \nabla^2_{\!\text{cart}} B_p(r_i, g) \cdot \phi_\text{grid}[g]

    where the Cartesian Hessian is obtained from the fractional-mesh
    Hessian (``bspline_weight_hessian_3d``) via the chain rule:

    .. math::

        H_\text{cart} = \text{transpose}(M)\, H_\text{frac}\, M
        \quad\text{with}\ M = \text{cell\_inv\_t}.

    Outputs the symmetric Hessian as two ``(N,)`` ``vec3`` arrays:

    * ``hessian_diag[i] = (H_xx, H_yy, H_zz)``
    * ``hessian_off[i]  = (H_xy, H_xz, H_yz)``

    The torch wrapper assembles the full ``(N, 3, 3)`` symmetric matrix
    from these. Two ``vec3`` atomic adds per stencil point — half the
    cost of a 9-component ``mat33`` write.

    Launch
    ------
    ``wp.launch(dim=(num_atoms, order**3), ...)`` — same grid shape
    as the spread / gather kernels.

    Parameters
    ----------
    positions, cell_inv_t, order
        Same convention as the spread kernel.
    mesh : wp.array3d
        Potential grid ``φ(g)`` — the inverse FFT of ``ρ̃ · G̃`` in
        the PME pipeline.
    hessian_diag, hessian_off : wp.array(dtype=vec3)
        OUTPUTS, pre-zeroed. Per-atom diagonal and off-diagonal of
        the Cartesian Hessian of ``φ``.
    """
    atom_idx, point_idx = wp.tid()

    mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
    position = positions[atom_idx]

    base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
    offset = bspline_grid_offset(point_idx, order, theta)

    diag_frac, off_frac = bspline_weight_hessian_3d(theta, offset, order, mesh_dims)

    gx = wrap_grid_index(base_grid[0] + offset[0], mesh_dims[0])
    gy = wrap_grid_index(base_grid[1] + offset[1], mesh_dims[1])
    gz = wrap_grid_index(base_grid[2] + offset[2], mesh_dims[2])
    mesh_val = mesh[gx, gy, gz]

    # Transform fractional Hessian → Cartesian: H_cart = M^T H_frac M.
    M = cell_inv_t[0]

    h_xx = diag_frac[0]
    h_yy = diag_frac[1]
    h_zz = diag_frac[2]
    h_xy = off_frac[0]
    h_xz = off_frac[1]
    h_yz = off_frac[2]

    # T = H_frac · M.
    t00 = h_xx * M[0, 0] + h_xy * M[1, 0] + h_xz * M[2, 0]
    t01 = h_xx * M[0, 1] + h_xy * M[1, 1] + h_xz * M[2, 1]
    t02 = h_xx * M[0, 2] + h_xy * M[1, 2] + h_xz * M[2, 2]
    t10 = h_xy * M[0, 0] + h_yy * M[1, 0] + h_yz * M[2, 0]
    t11 = h_xy * M[0, 1] + h_yy * M[1, 1] + h_yz * M[2, 1]
    t12 = h_xy * M[0, 2] + h_yy * M[1, 2] + h_yz * M[2, 2]
    t20 = h_xz * M[0, 0] + h_yz * M[1, 0] + h_zz * M[2, 0]
    t21 = h_xz * M[0, 1] + h_yz * M[1, 1] + h_zz * M[2, 1]
    t22 = h_xz * M[0, 2] + h_yz * M[1, 2] + h_zz * M[2, 2]

    # H_cart = M^T · T — only the 6 unique entries.
    h_xx_cart = M[0, 0] * t00 + M[1, 0] * t10 + M[2, 0] * t20
    h_yy_cart = M[0, 1] * t01 + M[1, 1] * t11 + M[2, 1] * t21
    h_zz_cart = M[0, 2] * t02 + M[1, 2] * t12 + M[2, 2] * t22
    h_xy_cart = M[0, 0] * t01 + M[1, 0] * t11 + M[2, 0] * t21
    h_xz_cart = M[0, 0] * t02 + M[1, 0] * t12 + M[2, 0] * t22
    h_yz_cart = M[0, 1] * t02 + M[1, 1] * t12 + M[2, 1] * t22

    wp.atomic_add(
        hessian_diag,
        atom_idx,
        type(diag_frac)(
            h_xx_cart * mesh_val,
            h_yy_cart * mesh_val,
            h_zz_cart * mesh_val,
        ),
    )
    wp.atomic_add(
        hessian_off,
        atom_idx,
        type(off_frac)(
            h_xy_cart * mesh_val,
            h_xz_cart * mesh_val,
            h_yz_cart * mesh_val,
        ),
    )


def _pme_multipole_gather_hessian_sig(v, t):
    """Signature builder for :func:`_pme_multipole_gather_hessian_kernel`."""
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=m),  # cell_inv_t
        wp.int32,  # order
        wp.array3d(dtype=t),  # mesh
        wp.array(dtype=v),  # hessian_diag
        wp.array(dtype=v),  # hessian_off
    ]


_pme_multipole_gather_hessian_overloads = register_overloads(
    _pme_multipole_gather_hessian_kernel, _pme_multipole_gather_hessian_sig
)


def multipole_pme_gather_hessian_launch(
    positions: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    hessian_diag: wp.array,
    hessian_off: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_pme_multipole_gather_hessian_kernel`.

    Gathers the per-atom Cartesian Hessian of the potential ``φ`` (needed
    for quadrupole forces). Uses a per-order specialized kernel when one
    is registered, else the generic ``order**3``-grid kernel.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=vec3f/vec3d
        Cartesian atom positions.
    cell_inv_t : wp.array, shape (1,), dtype=mat33f/mat33d
        Transpose of the inverse cell matrix.
    order : int
        B-spline order.
    mesh : wp.array, shape (Nx, Ny, Nz), dtype=wp.float32/float64
        Potential grid ``φ(g)`` (inverse FFT of ``ρ̃ · G̃``).
    hessian_diag : wp.array, shape (N,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Diagonal ``(H_xx, H_yy, H_zz)`` per atom.
    hessian_off : wp.array, shape (N,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Off-diagonal ``(H_xy, H_xz, H_yz)`` per atom.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Launch Grid
    -----------
    ``dim = (N,)`` (per-order kernel) or ``dim = (N, order**3)`` (generic).
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    # Per-order kernel falls back to the generic kernel for orders
    # outside the {3, 4, 5, 6} registered tuples.
    per_order = _maybe_per_order_gather_hessian_kernel(order, wp_dtype)
    if per_order is not None:
        wp.launch(
            per_order,
            dim=(num_atoms,),
            inputs=[
                positions,
                cell_inv_t,
                mesh,
            ],
            outputs=[hessian_diag, hessian_off],
            device=device,
        )
        return

    wp.launch(
        _pme_multipole_gather_hessian_overloads[vec_dtype],
        dim=(num_atoms, order**3),
        inputs=[
            positions,
            cell_inv_t,
            wp.int32(order),
            mesh,
        ],
        outputs=[hessian_diag, hessian_off],
        device=device,
    )


# =============================================================================
# k_squared compute
# =============================================================================


@wp.kernel(enable_backward=False)
def _pme_k_squared_kernel(
    miller_x: wp.array(dtype=Any),  # (Nx,)
    miller_y: wp.array(dtype=Any),  # (Ny,)
    miller_z: wp.array(dtype=Any),  # (Nz_rfft,)
    cell_inv_T: wp.array(dtype=Any),  # (1,) of mat33 — transpose(inv(cell))
    k_squared: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft) OUTPUT
):
    r"""Compute :math:`k^2[i, j, k] = |2\pi \cdot \text{cell\_inv\_T} \cdot m|^2`.

    Per (i, j, k) thread, reads three scalars from the rank-1 Miller
    index arrays + the 3x3 ``cell_inv_T`` matrix, does the 3x3 matvec
    inline, and writes one scalar.

    Launch Grid
    -----------
    ``dim = (Nx, Ny, Nz_rfft)`` — one thread per rfft k-grid cell.

    Parameters
    ----------
    miller_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        Miller indices along x.
    miller_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        Miller indices along y.
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        Miller indices along z (rfft half-space).
    cell_inv_T : wp.array, shape (1,), dtype=mat33f/mat33d
        Transpose of the inverse cell matrix :math:`M`.
    k_squared : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. :math:`k^2 = |2\pi M m|^2` at each grid point.
    """
    i, j, k = wp.tid()

    mx = miller_x[i]
    my = miller_y[j]
    mz = miller_z[k]
    M = cell_inv_T[0]

    # k_vec = (cell_inv^T) · m.
    kx = M[0, 0] * mx + M[0, 1] * my + M[0, 2] * mz
    ky = M[1, 0] * mx + M[1, 1] * my + M[1, 2] * mz
    kz = M[2, 0] * mx + M[2, 1] * my + M[2, 2] * mz

    twopi = type(mx)(_TWOPI)
    twopi_sq = twopi * twopi
    k_squared[i, j, k] = twopi_sq * (kx * kx + ky * ky + kz * kz)


def _pme_k_squared_sig(v, t):
    """Signature builder for :func:`_pme_k_squared_kernel`."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=t),  # miller_x
        wp.array(dtype=t),  # miller_y
        wp.array(dtype=t),  # miller_z
        wp.array(dtype=mat),  # cell_inv_T
        wp.array(dtype=t, ndim=3),  # k_squared (output)
    ]


_pme_k_squared_overloads = register_overloads(_pme_k_squared_kernel, _pme_k_squared_sig)


def pme_k_squared_launch(
    miller_x: wp.array,
    miller_y: wp.array,
    miller_z: wp.array,
    cell_inv_T: wp.array,
    k_squared: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Single-system launcher for :func:`_pme_k_squared_kernel`.

    Parameters
    ----------
    miller_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        Miller indices along x.
    miller_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        Miller indices along y.
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        Miller indices along z (rfft half-space).
    cell_inv_T : wp.array, shape (1,), dtype=mat33f/mat33d
        Transpose of the inverse cell matrix.
    k_squared : wp.array, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. :math:`k^2` grid.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``miller_x.device``.

    Launch Grid
    -----------
    ``dim = (Nx, Ny, Nz_rfft)`` — one thread per rfft k-grid cell.
    """
    if device is None:
        device = str(miller_x.device)
    nx, ny, nz_rfft = k_squared.shape
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_k_squared_overloads[vec_dtype],
        dim=(nx, ny, nz_rfft),
        inputs=[
            miller_x,
            miller_y,
            miller_z,
            cell_inv_T,
        ],
        outputs=[k_squared],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _pme_k_squared_backward_kernel(
    miller_x: wp.array(dtype=Any),  # (Nx,)
    miller_y: wp.array(dtype=Any),  # (Ny,)
    miller_z: wp.array(dtype=Any),  # (Nz_rfft,)
    cell_inv_T: wp.array(dtype=Any),  # (1,) of mat33
    grad_k_squared: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft) input
    grad_cell_inv_T: wp.array2d(dtype=Any),  # (3, 3) output (atomic-accumulated)
):
    r"""Backward for :func:`_pme_k_squared_kernel`.

    Forward: :math:`k^2(i,j,k) = (2\pi)^2 |M m|^2` where
    :math:`M = \text{cell\_inv\_T}` and :math:`m = (m_x[i], m_y[j], m_z[k])`.
    With :math:`k_c` the c-th component of :math:`M m`:

    .. math::

        \partial L / \partial M[c, d] = 8\pi^2 \, k_c \, m_d
                                        \cdot \partial L / \partial k^2

    Accumulated atomically over the rfft grid into the (3, 3)
    ``grad_cell_inv_T`` output.

    Launch Grid
    -----------
    ``dim = (Nx, Ny, Nz_rfft)`` — one thread per rfft k-grid cell;
    each atomically accumulates a rank-1 contribution into the shared
    ``(3, 3)`` gradient.

    Parameters
    ----------
    miller_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        Miller indices along x.
    miller_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        Miller indices along y.
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        Miller indices along z.
    cell_inv_T : wp.array, shape (1,), dtype=mat33f/mat33d
        Transpose of the inverse cell matrix :math:`M`.
    grad_k_squared : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Upstream gradient :math:`\partial L / \partial k^2`.
    grad_cell_inv_T : wp.array2d, shape (3, 3), dtype=wp.float32/float64
        OUTPUT, pre-zeroed. Gradient w.r.t. ``cell_inv_T`` (atomic).
    """
    i, j, k = wp.tid()

    mx = miller_x[i]
    my = miller_y[j]
    mz = miller_z[k]
    M = cell_inv_T[0]

    kx = M[0, 0] * mx + M[0, 1] * my + M[0, 2] * mz
    ky = M[1, 0] * mx + M[1, 1] * my + M[1, 2] * mz
    kz = M[2, 0] * mx + M[2, 1] * my + M[2, 2] * mz

    twopi = type(mx)(_TWOPI)
    # 8π² = 2 · (2π)² (factor of 2 from differentiating |k|²).
    eightpi_sq = type(mx)(8.0) * twopi * twopi / type(mx)(4.0)
    g = grad_k_squared[i, j, k] * eightpi_sq

    wp.atomic_add(grad_cell_inv_T, 0, 0, g * kx * mx)
    wp.atomic_add(grad_cell_inv_T, 0, 1, g * kx * my)
    wp.atomic_add(grad_cell_inv_T, 0, 2, g * kx * mz)
    wp.atomic_add(grad_cell_inv_T, 1, 0, g * ky * mx)
    wp.atomic_add(grad_cell_inv_T, 1, 1, g * ky * my)
    wp.atomic_add(grad_cell_inv_T, 1, 2, g * ky * mz)
    wp.atomic_add(grad_cell_inv_T, 2, 0, g * kz * mx)
    wp.atomic_add(grad_cell_inv_T, 2, 1, g * kz * my)
    wp.atomic_add(grad_cell_inv_T, 2, 2, g * kz * mz)


def _pme_k_squared_backward_sig(v, t):
    """Signature builder for :func:`_pme_k_squared_backward_kernel`."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=t),  # miller_x
        wp.array(dtype=t),  # miller_y
        wp.array(dtype=t),  # miller_z
        wp.array(dtype=mat),  # cell_inv_T
        wp.array(dtype=t, ndim=3),  # grad_k_squared
        wp.array(dtype=t, ndim=2),  # grad_cell_inv_T (out)
    ]


_pme_k_squared_backward_overloads = register_overloads(
    _pme_k_squared_backward_kernel, _pme_k_squared_backward_sig
)


def pme_k_squared_backward_launch(
    miller_x: wp.array,
    miller_y: wp.array,
    miller_z: wp.array,
    cell_inv_T: wp.array,
    grad_k_squared: wp.array,
    grad_cell_inv_T: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for the single-system k_squared backward kernel.

    ``grad_cell_inv_T`` must be zero-initialized — the kernel
    ``atomic_add``s into it.

    Parameters
    ----------
    miller_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        Miller indices along x.
    miller_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        Miller indices along y.
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        Miller indices along z.
    cell_inv_T : wp.array, shape (1,), dtype=mat33f/mat33d
        Transpose of the inverse cell matrix.
    grad_k_squared : wp.array, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Upstream gradient :math:`\partial L / \partial k^2`.
    grad_cell_inv_T : wp.array, shape (3, 3), dtype=wp.float32/float64
        OUTPUT, pre-zeroed. Gradient w.r.t. ``cell_inv_T``.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``miller_x.device``.

    Launch Grid
    -----------
    ``dim = (Nx, Ny, Nz_rfft)`` — one thread per rfft k-grid cell.
    """
    if device is None:
        device = str(miller_x.device)
    nx, ny, nz_rfft = grad_k_squared.shape
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_k_squared_backward_overloads[vec_dtype],
        dim=(nx, ny, nz_rfft),
        inputs=[
            miller_x,
            miller_y,
            miller_z,
            cell_inv_T,
            grad_k_squared,
        ],
        outputs=[grad_cell_inv_T],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _batch_pme_k_squared_kernel(
    miller_x: wp.array(dtype=Any),
    miller_y: wp.array(dtype=Any),
    miller_z: wp.array(dtype=Any),
    cell_inv_T: wp.array(dtype=Any),  # (B,) of mat33
    k_squared: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft) OUTPUT
):
    r"""Batched companion of :func:`_pme_k_squared_kernel` — per-system
    ``cell_inv_T``, shared Miller indices.

    Launch Grid
    -----------
    ``dim = (B, Nx, Ny, Nz_rfft)`` — one thread per (system, k-grid cell).

    Parameters
    ----------
    miller_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        Miller indices along x (shared across systems).
    miller_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        Miller indices along y (shared across systems).
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        Miller indices along z (shared across systems).
    cell_inv_T : wp.array, shape (B,), dtype=mat33f/mat33d
        Per-system transpose of the inverse cell matrix.
    k_squared : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. Per-system :math:`k^2` grid.
    """
    b, i, j, k = wp.tid()

    mx = miller_x[i]
    my = miller_y[j]
    mz = miller_z[k]
    M = cell_inv_T[b]

    kx = M[0, 0] * mx + M[0, 1] * my + M[0, 2] * mz
    ky = M[1, 0] * mx + M[1, 1] * my + M[1, 2] * mz
    kz = M[2, 0] * mx + M[2, 1] * my + M[2, 2] * mz

    twopi = type(mx)(_TWOPI)
    twopi_sq = twopi * twopi
    k_squared[b, i, j, k] = twopi_sq * (kx * kx + ky * ky + kz * kz)


def _batch_pme_k_squared_sig(v, t):
    """Signature builder for :func:`_batch_pme_k_squared_kernel`."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=t),  # miller_x
        wp.array(dtype=t),  # miller_y
        wp.array(dtype=t),  # miller_z
        wp.array(dtype=mat),  # cell_inv_T (B,)
        wp.array(dtype=t, ndim=4),  # k_squared (output)
    ]


_batch_pme_k_squared_overloads = register_overloads(
    _batch_pme_k_squared_kernel, _batch_pme_k_squared_sig
)


def batch_pme_k_squared_launch(
    miller_x: wp.array,
    miller_y: wp.array,
    miller_z: wp.array,
    cell_inv_T: wp.array,
    k_squared: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Batched launcher for :func:`_batch_pme_k_squared_kernel`.

    Parameters
    ----------
    miller_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        Miller indices along x (shared across systems).
    miller_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        Miller indices along y (shared across systems).
    miller_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        Miller indices along z (shared across systems).
    cell_inv_T : wp.array, shape (B,), dtype=mat33f/mat33d
        Per-system transpose of the inverse cell matrix.
    k_squared : wp.array, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. Per-system :math:`k^2` grid.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``miller_x.device``.

    Launch Grid
    -----------
    ``dim = (B, Nx, Ny, Nz_rfft)`` — one thread per (system, k-grid cell).
    """
    if device is None:
        device = str(miller_x.device)
    B, nx, ny, nz_rfft = k_squared.shape
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _batch_pme_k_squared_overloads[vec_dtype],
        dim=(B, nx, ny, nz_rfft),
        inputs=[
            miller_x,
            miller_y,
            miller_z,
            cell_inv_T,
        ],
        outputs=[k_squared],
        device=device,
    )


# =============================================================================
# Fused PME convolve kernel
# =============================================================================
#
# The multipole-specific ``exp(-σ²k²)`` factor (vs monopole PME) collapses to
# the monopole convolve at σ → 0.


@wp.kernel(enable_backward=False)
def _pme_multipole_convolve_kernel(
    mesh_fft: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft), complex as vec2
    k_squared: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft)
    moduli_x: wp.array(dtype=Any),  # 1D LUT: sinc(mi/Nx)^spline_order
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),  # (1,)
    sigma: wp.array(dtype=Any),  # (1,)
    volume: wp.array(dtype=Any),  # (1,)
    convolved_mesh: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft), complex as vec2 OUTPUT
):
    r"""Fused Green's function + B-spline deconvolution + complex multiply.

    For each k-grid point ``(i, j, k)``:

    .. math::

        \tilde{G}(k) = \frac{2\pi \, e^{-k^2 [1/(4\alpha^2) + \sigma^2]}}{V \, k^2}

        \mathrm{factor}(i, j, k) = \frac{\tilde{G}(k)}{(b_x[i] \, b_y[j] \, b_z[k])^2}

        \mathrm{convolved\_mesh}[i, j, k] = \mathrm{mesh\_fft}[i, j, k] \cdot \mathrm{factor}

    where ``b_x[i] = sinc(m_x / N_x)^{spline\_order}`` is the precomputed
    1D B-spline modulus (computed once per call in fp32 in the torch
    wrapper, cast to working dtype).

    ``k = 0`` (the origin) is explicitly zeroed (tin-foil boundary).

    Launch Grid
    -----------
    ``dim = (Nx, Ny, Nz_rfft)`` — one thread per rfft cell.

    Parameters
    ----------
    mesh_fft : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        Forward-FFT of the spread density; complex stored as ``vec2``
        ``(real, imag)``.
    k_squared : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Squared k-vector magnitude per grid point.
    moduli_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        Precomputed 1D B-spline modulus ``b_x[i] = sinc(m_x/N_x)^p``.
    moduli_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        1D B-spline modulus along y.
    moduli_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        1D B-spline modulus along z.
    alpha : wp.array, shape (1,), dtype=wp.float32/float64
        Ewald splitting parameter :math:`\alpha`.
    sigma : wp.array, shape (1,), dtype=wp.float32/float64
        GTO density Gaussian width :math:`\sigma`.
    volume : wp.array, shape (1,), dtype=wp.float32/float64
        Unit cell volume :math:`V`.
    convolved_mesh : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        OUTPUT. ``mesh_fft`` scaled by the effective Green's factor.
    """
    i, j, k = wp.tid()

    k_sq = k_squared[i, j, k]
    alpha_ = alpha[0]
    sigma_ = sigma[0]
    volume_ = volume[0]

    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    four = type(k_sq)(4.0)
    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(_TWOPI)

    # B-spline structure factor (rank-1 product of 1D moduli).
    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    # Effective Green's factor G̃(k) / |C(k)|² with the multipole-
    # specific exp(-σ²k²) factor baked in.
    if k_sq < threshold:
        factor = zero
    else:
        combined_prefactor = one / (four * alpha_ * alpha_) + sigma_ * sigma_
        exp_factor = wp.exp(-combined_prefactor * k_sq) / k_sq
        factor = twopi * exp_factor / (volume_ * sf_sq)
    if i == 0 and j == 0 and k == 0:
        factor = zero

    # Complex × real multiply; complex stored as vec2 (real, imag).
    c = mesh_fft[i, j, k]
    convolved_mesh[i, j, k] = type(c)(c[0] * factor, c[1] * factor)


def _pme_multipole_convolve_sig(v, t):
    """Signature builder for the fused convolve kernel.

    ``v`` is the vec3 dtype (unused here; ``register_overloads``
    convention). ``t`` is the scalar working dtype (fp32 / fp64).
    Complex mesh entries are stored as vec2 of the matching dtype.
    """
    vec2 = wp.vec2d if t == wp.float64 else wp.vec2f
    return [
        wp.array(dtype=vec2, ndim=3),  # mesh_fft
        wp.array(dtype=t, ndim=3),  # k_squared
        wp.array(dtype=t),  # moduli_x
        wp.array(dtype=t),  # moduli_y
        wp.array(dtype=t),  # moduli_z
        wp.array(dtype=t),  # alpha
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # volume
        wp.array(dtype=vec2, ndim=3),  # convolved_mesh (output)
    ]


_pme_multipole_convolve_overloads = register_overloads(
    _pme_multipole_convolve_kernel, _pme_multipole_convolve_sig
)


def multipole_pme_convolve_launch(
    mesh_fft: wp.array,
    k_squared: wp.array,
    moduli_x: wp.array,
    moduli_y: wp.array,
    moduli_z: wp.array,
    alpha: wp.array,
    sigma: wp.array,
    volume: wp.array,
    convolved_mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Single-system launcher for the fused multipole convolve kernel.

    All scalar arrays share the working dtype ``wp_dtype``; complex mesh
    entries are vec2 of the matching dtype.

    Parameters
    ----------
    mesh_fft : wp.array, shape (Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        Forward-FFT of the spread density (complex as ``vec2``).
    k_squared : wp.array, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Squared k-vector magnitude.
    moduli_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        1D B-spline modulus along x.
    moduli_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        1D B-spline modulus along y.
    moduli_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        1D B-spline modulus along z.
    alpha : wp.array, shape (1,), dtype=wp.float32/float64
        Ewald splitting parameter.
    sigma : wp.array, shape (1,), dtype=wp.float32/float64
        GTO Gaussian width.
    volume : wp.array, shape (1,), dtype=wp.float32/float64
        Unit cell volume.
    convolved_mesh : wp.array, shape (Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        OUTPUT. Convolved mesh.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``mesh_fft.device``.

    Launch Grid
    -----------
    ``dim = (Nx, Ny, Nz_rfft)`` — one thread per rfft cell.
    """
    if device is None:
        device = str(mesh_fft.device)
    nx, ny, nz_rfft = mesh_fft.shape
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_multipole_convolve_overloads[vec_dtype],
        dim=(nx, ny, nz_rfft),
        inputs=[
            mesh_fft,
            k_squared,
            moduli_x,
            moduli_y,
            moduli_z,
            alpha,
            sigma,
            volume,
        ],
        outputs=[convolved_mesh],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _pme_multipole_convolve_backward_kernel(
    mesh_fft: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft), vec2
    k_squared: wp.array3d(dtype=Any),
    moduli_x: wp.array(dtype=Any),
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    grad_convolved: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft), vec2
    grad_mesh_fft: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft), vec2 (out)
    grad_k_squared: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft) (out)
    grad_volume: wp.array(dtype=Any),  # (1,) (out, atomic)
):
    r"""Backward kernel for the fused PME convolve.

    Forward: :math:`\text{convolved} = \text{mesh\_fft} \cdot \text{factor}`
    where :math:`\text{factor} = 2\pi e^{-(1/(4\alpha^2) + \sigma^2) k^2}
    / (V k^2 \text{sf}^2)`. Treating ``alpha, sigma, moduli`` as fixed:

    - :math:`\partial L / \partial \text{mesh\_fft} =
      \text{grad\_convolved} \cdot \text{factor}`.
    - :math:`\partial L / \partial V = -\sum_g \mathrm{Re}(
      \text{grad\_convolved} \cdot \overline{\text{convolved}}) / V`
      (atomic add into a scalar (1,) buffer).
    - :math:`\partial L / \partial k^2(g) = \mathrm{Re}(
      \text{grad\_convolved} \cdot \overline{\text{mesh\_fft}} \cdot
      \text{factor}) \cdot (-(1/(4\alpha^2)+\sigma^2) - 1/k^2)`.

    All k=0 contributions are zeroed (the forward sets ``factor = 0``
    at k=0).

    Launch Grid
    -----------
    ``dim = (Nx, Ny, Nz_rfft)`` — one thread per rfft cell.

    Parameters
    ----------
    mesh_fft : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        Forward-FFT of the spread density (complex as ``vec2``).
    k_squared : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Squared k-vector magnitude.
    moduli_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        1D B-spline modulus along x.
    moduli_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        1D B-spline modulus along y.
    moduli_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        1D B-spline modulus along z.
    alpha : wp.array, shape (1,), dtype=wp.float32/float64
        Ewald splitting parameter.
    sigma : wp.array, shape (1,), dtype=wp.float32/float64
        GTO Gaussian width.
    volume : wp.array, shape (1,), dtype=wp.float32/float64
        Unit cell volume.
    grad_convolved : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        Upstream gradient w.r.t. the convolved mesh.
    grad_mesh_fft : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        OUTPUT. Gradient w.r.t. ``mesh_fft``.
    grad_k_squared : wp.array3d, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. Gradient w.r.t. ``k_squared``.
    grad_volume : wp.array, shape (1,), dtype=wp.float32/float64
        OUTPUT, pre-zeroed. Gradient w.r.t. volume (atomic accumulator).
    """
    i, j, k = wp.tid()

    k_sq = k_squared[i, j, k]
    alpha_ = alpha[0]
    sigma_ = sigma[0]
    volume_ = volume[0]

    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    four = type(k_sq)(4.0)
    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(_TWOPI)

    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    # c = combined_prefactor = 1/(4α²) + σ².
    if k_sq < threshold:
        factor = zero
        combined_prefactor = zero
    else:
        combined_prefactor = one / (four * alpha_ * alpha_) + sigma_ * sigma_
        exp_factor = wp.exp(-combined_prefactor * k_sq) / k_sq
        factor = twopi * exp_factor / (volume_ * sf_sq)
    if i == 0 and j == 0 and k == 0:
        factor = zero

    # ∂L/∂factor = Re(grad_convolved · conj(mesh_fft)).
    gc = grad_convolved[i, j, k]
    mf = mesh_fft[i, j, k]
    dL_dfactor = gc[0] * mf[0] + gc[1] * mf[1]

    grad_mesh_fft[i, j, k] = type(gc)(gc[0] * factor, gc[1] * factor)

    if k_sq < threshold:
        grad_k_squared[i, j, k] = zero
    else:
        d_factor_d_ksq = factor * (-combined_prefactor - one / k_sq)
        grad_k_squared[i, j, k] = dL_dfactor * d_factor_d_ksq

    # ∂factor/∂V = -factor/V.
    if k_sq < threshold:
        pass
    else:
        wp.atomic_add(grad_volume, 0, -dL_dfactor * factor / volume_)


def _pme_multipole_convolve_backward_sig(v, t):
    """Signature builder for the convolve backward kernel."""
    vec2 = wp.vec2d if t == wp.float64 else wp.vec2f
    return [
        wp.array(dtype=vec2, ndim=3),  # mesh_fft
        wp.array(dtype=t, ndim=3),  # k_squared
        wp.array(dtype=t),  # moduli_x
        wp.array(dtype=t),  # moduli_y
        wp.array(dtype=t),  # moduli_z
        wp.array(dtype=t),  # alpha
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # volume
        wp.array(dtype=vec2, ndim=3),  # grad_convolved
        wp.array(dtype=vec2, ndim=3),  # grad_mesh_fft (out)
        wp.array(dtype=t, ndim=3),  # grad_k_squared (out)
        wp.array(dtype=t),  # grad_volume (out, atomic)
    ]


_pme_multipole_convolve_backward_overloads = register_overloads(
    _pme_multipole_convolve_backward_kernel, _pme_multipole_convolve_backward_sig
)


def multipole_pme_convolve_backward_launch(
    mesh_fft: wp.array,
    k_squared: wp.array,
    moduli_x: wp.array,
    moduli_y: wp.array,
    moduli_z: wp.array,
    alpha: wp.array,
    sigma: wp.array,
    volume: wp.array,
    grad_convolved: wp.array,
    grad_mesh_fft: wp.array,
    grad_k_squared: wp.array,
    grad_volume: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for the single-system PME convolve backward kernel.

    ``grad_volume`` is a ``(1,)`` array — must be zero-initialized.

    Parameters
    ----------
    mesh_fft : wp.array, shape (Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        Forward-FFT of the spread density (complex as ``vec2``).
    k_squared : wp.array, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Squared k-vector magnitude.
    moduli_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        1D B-spline modulus along x.
    moduli_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        1D B-spline modulus along y.
    moduli_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        1D B-spline modulus along z.
    alpha : wp.array, shape (1,), dtype=wp.float32/float64
        Ewald splitting parameter.
    sigma : wp.array, shape (1,), dtype=wp.float32/float64
        GTO Gaussian width.
    volume : wp.array, shape (1,), dtype=wp.float32/float64
        Unit cell volume.
    grad_convolved : wp.array, shape (Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        Upstream gradient w.r.t. the convolved mesh.
    grad_mesh_fft : wp.array, shape (Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        OUTPUT. Gradient w.r.t. ``mesh_fft``.
    grad_k_squared : wp.array, shape (Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. Gradient w.r.t. ``k_squared``.
    grad_volume : wp.array, shape (1,), dtype=wp.float32/float64
        OUTPUT, pre-zeroed. Gradient w.r.t. volume.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``mesh_fft.device``.

    Launch Grid
    -----------
    ``dim = (Nx, Ny, Nz_rfft)`` — one thread per rfft cell.
    """
    if device is None:
        device = str(mesh_fft.device)
    nx, ny, nz_rfft = mesh_fft.shape
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_multipole_convolve_backward_overloads[vec_dtype],
        dim=(nx, ny, nz_rfft),
        inputs=[
            mesh_fft,
            k_squared,
            moduli_x,
            moduli_y,
            moduli_z,
            alpha,
            sigma,
            volume,
            grad_convolved,
        ],
        outputs=[grad_mesh_fft, grad_k_squared, grad_volume],
        device=device,
    )


@wp.kernel
def _pme_multipole_convolve_double_backward_kernel(
    mesh_fft: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft), vec2
    k_squared: wp.array3d(dtype=Any),
    moduli_x: wp.array(dtype=Any),
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    grad_convolved: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft), vec2 — gc
    gg_mesh_fft: wp.array3d(dtype=Any),  # cotangent on grad_mesh_fft (vec2)
    gg_k_squared: wp.array3d(dtype=Any),  # cotangent on grad_k_squared (scalar)
    gg_volume: wp.array(dtype=Any),  # (1,) cotangent on grad_volume
    d_grad_convolved: wp.array3d(dtype=Any),  # (out) ∂/∂grad_convolved (vec2)
    d_mesh_fft: wp.array3d(dtype=Any),  # (out) ∂/∂mesh_fft (vec2)
    d_k_squared: wp.array3d(dtype=Any),  # (out) ∂/∂k_squared (scalar)
    d_volume: wp.array(dtype=Any),  # (1,) (out, atomic) ∂/∂volume
):
    r"""Double-backward (stress-loss) of the fused PME convolve.

    Second-order of the backward
    :math:`(gc, mf, k^2, V) \mapsto (g_{mf}, g_{k^2}, g_V)`. With
    ``factor`` and ``β = -(1/4α² + σ²) - 1/k²`` as in the backward, the real
    dots ``P = gg_mf·gc``, ``D = gc·mf``, the scalar
    ``s = gg_ksq·β - gg_vol/V`` and ``W = P + D·s``:

    - ``∂/∂grad_convolved = factor·gg_mf + factor·s·mf``
    - ``∂/∂mesh_fft = factor·s·gc``
    - ``∂/∂k² = factor·(β·W + gg_ksq·D/k⁴)``
    - ``∂/∂V = Σ_g (factor/V)·(-W + D·gg_vol/V)`` (atomic).

    The ``mesh_fft`` cotangent path (``gg_mesh_fft``) is the force-loss term;
    the ``gg_k_squared``/``gg_volume`` paths are the cell-coupled stress-loss
    terms that ``create_graph`` through ``dE/dcell`` requires.
    """
    i, j, k = wp.tid()

    k_sq = k_squared[i, j, k]
    alpha_ = alpha[0]
    sigma_ = sigma[0]
    volume_ = volume[0]

    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    four = type(k_sq)(4.0)
    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(_TWOPI)

    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    gc = grad_convolved[i, j, k]
    mf = mesh_fft[i, j, k]
    gmf = gg_mesh_fft[i, j, k]
    gksq = gg_k_squared[i, j, k]
    gvol = gg_volume[0]

    zero_c = type(gc)(zero, zero)
    if k_sq < threshold:
        d_grad_convolved[i, j, k] = zero_c
        d_mesh_fft[i, j, k] = zero_c
        d_k_squared[i, j, k] = zero
        return
    if i == 0 and j == 0 and k == 0:
        d_grad_convolved[i, j, k] = zero_c
        d_mesh_fft[i, j, k] = zero_c
        d_k_squared[i, j, k] = zero
        return

    combined_prefactor = one / (four * alpha_ * alpha_) + sigma_ * sigma_
    factor = twopi * (wp.exp(-combined_prefactor * k_sq) / k_sq) / (volume_ * sf_sq)
    beta = -combined_prefactor - one / k_sq

    p_dot = gmf[0] * gc[0] + gmf[1] * gc[1]  # gg_mf · gc
    d_dot = gc[0] * mf[0] + gc[1] * mf[1]  # gc · mf  (= dL_dfactor)
    s = gksq * beta - gvol / volume_
    w = p_dot + d_dot * s

    # ∂/∂grad_convolved = factor·gg_mf + factor·s·mf.
    d_grad_convolved[i, j, k] = type(gc)(
        factor * gmf[0] + factor * s * mf[0],
        factor * gmf[1] + factor * s * mf[1],
    )
    # ∂/∂mesh_fft = factor·s·gc.
    d_mesh_fft[i, j, k] = type(gc)(factor * s * gc[0], factor * s * gc[1])
    # ∂/∂k² = factor·(β·W + gg_ksq·D/k⁴).
    d_k_squared[i, j, k] = factor * (beta * w + gksq * d_dot / (k_sq * k_sq))
    # ∂/∂V = (factor/V)·(-W + D·gg_vol/V), summed.
    wp.atomic_add(d_volume, 0, (factor / volume_) * (-w + d_dot * gvol / volume_))


def _pme_multipole_convolve_double_backward_sig(v, t):
    """Signature builder for the convolve double-backward kernel."""
    vec2 = wp.vec2d if t == wp.float64 else wp.vec2f
    return [
        wp.array(dtype=vec2, ndim=3),  # mesh_fft
        wp.array(dtype=t, ndim=3),  # k_squared
        wp.array(dtype=t),  # moduli_x
        wp.array(dtype=t),  # moduli_y
        wp.array(dtype=t),  # moduli_z
        wp.array(dtype=t),  # alpha
        wp.array(dtype=t),  # sigma
        wp.array(dtype=t),  # volume
        wp.array(dtype=vec2, ndim=3),  # grad_convolved
        wp.array(dtype=vec2, ndim=3),  # gg_mesh_fft
        wp.array(dtype=t, ndim=3),  # gg_k_squared
        wp.array(dtype=t),  # gg_volume
        wp.array(dtype=vec2, ndim=3),  # d_grad_convolved (out)
        wp.array(dtype=vec2, ndim=3),  # d_mesh_fft (out)
        wp.array(dtype=t, ndim=3),  # d_k_squared (out)
        wp.array(dtype=t),  # d_volume (out, atomic)
    ]


_pme_multipole_convolve_double_backward_overloads = register_overloads(
    _pme_multipole_convolve_double_backward_kernel,
    _pme_multipole_convolve_double_backward_sig,
)


def multipole_pme_convolve_double_backward_launch(
    mesh_fft,
    k_squared,
    moduli_x,
    moduli_y,
    moduli_z,
    alpha,
    sigma,
    volume,
    grad_convolved,
    gg_mesh_fft,
    gg_k_squared,
    gg_volume,
    d_grad_convolved,
    d_mesh_fft,
    d_k_squared,
    d_volume,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launch :func:`_pme_multipole_convolve_double_backward_kernel`.

    ``d_volume`` is a ``(1,)`` array — must be zero-initialized.
    """
    if device is None:
        device = str(mesh_fft.device)
    nx, ny, nz_rfft = mesh_fft.shape
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_multipole_convolve_double_backward_overloads[vec_dtype],
        dim=(nx, ny, nz_rfft),
        inputs=[
            mesh_fft,
            k_squared,
            moduli_x,
            moduli_y,
            moduli_z,
            alpha,
            sigma,
            volume,
            grad_convolved,
            gg_mesh_fft,
            gg_k_squared,
            gg_volume,
        ],
        outputs=[d_grad_convolved, d_mesh_fft, d_k_squared, d_volume],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _batch_pme_multipole_convolve_kernel(
    mesh_fft: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft), complex as vec2
    k_squared: wp.array4d(dtype=Any),
    moduli_x: wp.array(dtype=Any),
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),  # (B,)
    sigma: wp.array(dtype=Any),  # (B,)
    volume: wp.array(dtype=Any),  # (B,)
    convolved_mesh: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft) OUTPUT
):
    r"""Batched fused convolve. Identical math to single-system; per-system
    ``alpha``, ``sigma``, ``volume``; moduli shared across batch.

    Launch Grid
    -----------
    ``dim = (B, Nx, Ny, Nz_rfft)`` — one thread per (system, rfft cell).

    Parameters
    ----------
    mesh_fft : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        Per-system forward-FFT of the spread density (complex as ``vec2``).
    k_squared : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Per-system squared k-vector magnitude.
    moduli_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        1D B-spline modulus along x (shared).
    moduli_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        1D B-spline modulus along y (shared).
    moduli_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        1D B-spline modulus along z (shared).
    alpha : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system Ewald splitting parameter.
    sigma : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system GTO Gaussian width.
    volume : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system unit cell volume.
    convolved_mesh : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        OUTPUT. Per-system convolved mesh.
    """
    b, i, j, k = wp.tid()

    k_sq = k_squared[b, i, j, k]
    alpha_ = alpha[b]
    sigma_ = sigma[b]
    volume_ = volume[b]

    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    four = type(k_sq)(4.0)
    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(_TWOPI)

    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    if k_sq < threshold:
        factor = zero
    else:
        combined_prefactor = one / (four * alpha_ * alpha_) + sigma_ * sigma_
        exp_factor = wp.exp(-combined_prefactor * k_sq) / k_sq
        factor = twopi * exp_factor / (volume_ * sf_sq)
    if i == 0 and j == 0 and k == 0:
        factor = zero

    c = mesh_fft[b, i, j, k]
    convolved_mesh[b, i, j, k] = type(c)(c[0] * factor, c[1] * factor)


def _batch_pme_multipole_convolve_sig(v, t):
    vec2 = wp.vec2d if t == wp.float64 else wp.vec2f
    return [
        wp.array(dtype=vec2, ndim=4),  # mesh_fft (B, Nx, Ny, Nz_rfft)
        wp.array(dtype=t, ndim=4),  # k_squared (B, Nx, Ny, Nz_rfft)
        wp.array(dtype=t),  # moduli_x (Nx,)
        wp.array(dtype=t),  # moduli_y (Ny,)
        wp.array(dtype=t),  # moduli_z (Nz_rfft,)
        wp.array(dtype=t),  # alpha (B,)
        wp.array(dtype=t),  # sigma (B,)
        wp.array(dtype=t),  # volume (B,)
        wp.array(dtype=vec2, ndim=4),  # convolved_mesh (output)
    ]


_batch_pme_multipole_convolve_overloads = register_overloads(
    _batch_pme_multipole_convolve_kernel, _batch_pme_multipole_convolve_sig
)


def batch_multipole_pme_convolve_launch(
    mesh_fft: wp.array,
    k_squared: wp.array,
    moduli_x: wp.array,
    moduli_y: wp.array,
    moduli_z: wp.array,
    alpha: wp.array,
    sigma: wp.array,
    volume: wp.array,
    convolved_mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Batched launcher for the fused multipole convolve kernel.

    Parameters
    ----------
    mesh_fft : wp.array, shape (B, Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        Per-system forward-FFT of the spread density (complex as ``vec2``).
    k_squared : wp.array, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Per-system squared k-vector magnitude.
    moduli_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        1D B-spline modulus along x (shared).
    moduli_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        1D B-spline modulus along y (shared).
    moduli_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        1D B-spline modulus along z (shared).
    alpha : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system Ewald splitting parameter.
    sigma : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system GTO Gaussian width.
    volume : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system unit cell volume.
    convolved_mesh : wp.array, shape (B, Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        OUTPUT. Per-system convolved mesh.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``mesh_fft.device``.

    Launch Grid
    -----------
    ``dim = (B, Nx, Ny, Nz_rfft)`` — one thread per (system, rfft cell).
    """
    if device is None:
        device = str(mesh_fft.device)
    B, nx, ny, nz_rfft = mesh_fft.shape
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _batch_pme_multipole_convolve_overloads[vec_dtype],
        dim=(B, nx, ny, nz_rfft),
        inputs=[
            mesh_fft,
            k_squared,
            moduli_x,
            moduli_y,
            moduli_z,
            alpha,
            sigma,
            volume,
        ],
        outputs=[convolved_mesh],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _batch_pme_multipole_convolve_backward_kernel(
    mesh_fft: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft), vec2
    k_squared: wp.array4d(dtype=Any),
    moduli_x: wp.array(dtype=Any),
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),  # (B,)
    sigma: wp.array(dtype=Any),  # (B,)
    volume: wp.array(dtype=Any),  # (B,)
    grad_convolved: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft), vec2
    grad_mesh_fft: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft), vec2 (out)
    grad_k_squared: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft) (out)
    grad_volume: wp.array(dtype=Any),  # (B,) (out, atomic)
):
    r"""Batched convolve backward. Per-system analog of
    :func:`_pme_multipole_convolve_backward_kernel`; ``grad_volume`` is a
    per-system ``(B,)`` atomic accumulator.

    Launch Grid
    -----------
    ``dim = (B, Nx, Ny, Nz_rfft)`` — one thread per (system, rfft cell).

    Parameters
    ----------
    mesh_fft : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        Per-system forward-FFT of the spread density (complex as ``vec2``).
    k_squared : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Per-system squared k-vector magnitude.
    moduli_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        1D B-spline modulus along x (shared).
    moduli_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        1D B-spline modulus along y (shared).
    moduli_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        1D B-spline modulus along z (shared).
    alpha : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system Ewald splitting parameter.
    sigma : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system GTO Gaussian width.
    volume : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system unit cell volume.
    grad_convolved : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        Upstream gradient w.r.t. the convolved mesh.
    grad_mesh_fft : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        OUTPUT. Gradient w.r.t. ``mesh_fft``.
    grad_k_squared : wp.array4d, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. Gradient w.r.t. ``k_squared``.
    grad_volume : wp.array, shape (B,), dtype=wp.float32/float64
        OUTPUT, pre-zeroed. Per-system gradient w.r.t. volume (atomic).
    """
    b, i, j, k = wp.tid()

    k_sq = k_squared[b, i, j, k]
    alpha_ = alpha[b]
    sigma_ = sigma[b]
    volume_ = volume[b]

    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    four = type(k_sq)(4.0)
    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(_TWOPI)

    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    if k_sq < threshold:
        factor = zero
        combined_prefactor = zero
    else:
        combined_prefactor = one / (four * alpha_ * alpha_) + sigma_ * sigma_
        exp_factor = wp.exp(-combined_prefactor * k_sq) / k_sq
        factor = twopi * exp_factor / (volume_ * sf_sq)
    if i == 0 and j == 0 and k == 0:
        factor = zero

    gc = grad_convolved[b, i, j, k]
    mf = mesh_fft[b, i, j, k]
    dL_dfactor = gc[0] * mf[0] + gc[1] * mf[1]

    grad_mesh_fft[b, i, j, k] = type(gc)(gc[0] * factor, gc[1] * factor)

    if k_sq < threshold:
        grad_k_squared[b, i, j, k] = zero
    else:
        d_factor_d_ksq = factor * (-combined_prefactor - one / k_sq)
        grad_k_squared[b, i, j, k] = dL_dfactor * d_factor_d_ksq

    if k_sq < threshold:
        pass
    else:
        wp.atomic_add(grad_volume, b, -dL_dfactor * factor / volume_)


def _batch_pme_multipole_convolve_backward_sig(v, t):
    vec2 = wp.vec2d if t == wp.float64 else wp.vec2f
    return [
        wp.array(dtype=vec2, ndim=4),  # mesh_fft
        wp.array(dtype=t, ndim=4),  # k_squared
        wp.array(dtype=t),  # moduli_x
        wp.array(dtype=t),  # moduli_y
        wp.array(dtype=t),  # moduli_z
        wp.array(dtype=t),  # alpha (B,)
        wp.array(dtype=t),  # sigma (B,)
        wp.array(dtype=t),  # volume (B,)
        wp.array(dtype=vec2, ndim=4),  # grad_convolved
        wp.array(dtype=vec2, ndim=4),  # grad_mesh_fft (out)
        wp.array(dtype=t, ndim=4),  # grad_k_squared (out)
        wp.array(dtype=t),  # grad_volume (B,) (out, atomic)
    ]


_batch_pme_multipole_convolve_backward_overloads = register_overloads(
    _batch_pme_multipole_convolve_backward_kernel,
    _batch_pme_multipole_convolve_backward_sig,
)


def batch_multipole_pme_convolve_backward_launch(
    mesh_fft: wp.array,
    k_squared: wp.array,
    moduli_x: wp.array,
    moduli_y: wp.array,
    moduli_z: wp.array,
    alpha: wp.array,
    sigma: wp.array,
    volume: wp.array,
    grad_convolved: wp.array,
    grad_mesh_fft: wp.array,
    grad_k_squared: wp.array,
    grad_volume: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Batched launcher for the PME convolve backward kernel.

    ``grad_volume`` is a ``(B,)`` array — must be zero-initialized.

    Parameters
    ----------
    mesh_fft : wp.array, shape (B, Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        Per-system forward-FFT of the spread density (complex as ``vec2``).
    k_squared : wp.array, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        Per-system squared k-vector magnitude.
    moduli_x : wp.array, shape (Nx,), dtype=wp.float32/float64
        1D B-spline modulus along x (shared).
    moduli_y : wp.array, shape (Ny,), dtype=wp.float32/float64
        1D B-spline modulus along y (shared).
    moduli_z : wp.array, shape (Nz_rfft,), dtype=wp.float32/float64
        1D B-spline modulus along z (shared).
    alpha : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system Ewald splitting parameter.
    sigma : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system GTO Gaussian width.
    volume : wp.array, shape (B,), dtype=wp.float32/float64
        Per-system unit cell volume.
    grad_convolved : wp.array, shape (B, Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        Upstream gradient w.r.t. the convolved mesh.
    grad_mesh_fft : wp.array, shape (B, Nx, Ny, Nz_rfft), dtype=vec2f/vec2d
        OUTPUT. Gradient w.r.t. ``mesh_fft``.
    grad_k_squared : wp.array, shape (B, Nx, Ny, Nz_rfft), dtype=wp.float32/float64
        OUTPUT. Gradient w.r.t. ``k_squared``.
    grad_volume : wp.array, shape (B,), dtype=wp.float32/float64
        OUTPUT, pre-zeroed. Per-system gradient w.r.t. volume.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``mesh_fft.device``.

    Launch Grid
    -----------
    ``dim = (B, Nx, Ny, Nz_rfft)`` — one thread per (system, rfft cell).
    """
    if device is None:
        device = str(mesh_fft.device)
    B, nx, ny, nz_rfft = mesh_fft.shape
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _batch_pme_multipole_convolve_backward_overloads[vec_dtype],
        dim=(B, nx, ny, nz_rfft),
        inputs=[
            mesh_fft,
            k_squared,
            moduli_x,
            moduli_y,
            moduli_z,
            alpha,
            sigma,
            volume,
            grad_convolved,
        ],
        outputs=[grad_mesh_fft, grad_k_squared, grad_volume],
        device=device,
    )


@wp.kernel
def _batch_pme_multipole_convolve_double_backward_kernel(
    mesh_fft: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft), vec2
    k_squared: wp.array4d(dtype=Any),
    moduli_x: wp.array(dtype=Any),
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),  # (B,)
    sigma: wp.array(dtype=Any),  # (B,)
    volume: wp.array(dtype=Any),  # (B,)
    grad_convolved: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft), vec2 — gc
    gg_mesh_fft: wp.array4d(dtype=Any),  # cotangent on grad_mesh_fft (vec2)
    gg_k_squared: wp.array4d(dtype=Any),  # cotangent on grad_k_squared (scalar)
    gg_volume: wp.array(dtype=Any),  # (B,) cotangent on grad_volume
    d_grad_convolved: wp.array4d(dtype=Any),  # (out) ∂/∂grad_convolved (vec2)
    d_mesh_fft: wp.array4d(dtype=Any),  # (out) ∂/∂mesh_fft (vec2)
    d_k_squared: wp.array4d(dtype=Any),  # (out) ∂/∂k_squared (scalar)
    d_volume: wp.array(dtype=Any),  # (B,) (out, atomic) ∂/∂volume
):
    r"""Batched analog of :func:`_pme_multipole_convolve_double_backward_kernel`
    (per-system ``alpha``/``sigma``/``volume`` and per-system ``grad_volume``)."""
    b, i, j, k = wp.tid()

    k_sq = k_squared[b, i, j, k]
    alpha_ = alpha[b]
    sigma_ = sigma[b]
    volume_ = volume[b]

    zero = type(k_sq)(0.0)
    one = type(k_sq)(1.0)
    four = type(k_sq)(4.0)
    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    twopi = type(k_sq)(_TWOPI)

    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    gc = grad_convolved[b, i, j, k]
    mf = mesh_fft[b, i, j, k]
    gmf = gg_mesh_fft[b, i, j, k]
    gksq = gg_k_squared[b, i, j, k]
    gvol = gg_volume[b]

    zero_c = type(gc)(zero, zero)
    if k_sq < threshold:
        d_grad_convolved[b, i, j, k] = zero_c
        d_mesh_fft[b, i, j, k] = zero_c
        d_k_squared[b, i, j, k] = zero
        return
    if i == 0 and j == 0 and k == 0:
        d_grad_convolved[b, i, j, k] = zero_c
        d_mesh_fft[b, i, j, k] = zero_c
        d_k_squared[b, i, j, k] = zero
        return

    combined_prefactor = one / (four * alpha_ * alpha_) + sigma_ * sigma_
    factor = twopi * (wp.exp(-combined_prefactor * k_sq) / k_sq) / (volume_ * sf_sq)
    beta = -combined_prefactor - one / k_sq

    p_dot = gmf[0] * gc[0] + gmf[1] * gc[1]
    d_dot = gc[0] * mf[0] + gc[1] * mf[1]
    s = gksq * beta - gvol / volume_
    w = p_dot + d_dot * s

    d_grad_convolved[b, i, j, k] = type(gc)(
        factor * gmf[0] + factor * s * mf[0],
        factor * gmf[1] + factor * s * mf[1],
    )
    d_mesh_fft[b, i, j, k] = type(gc)(factor * s * gc[0], factor * s * gc[1])
    d_k_squared[b, i, j, k] = factor * (beta * w + gksq * d_dot / (k_sq * k_sq))
    wp.atomic_add(d_volume, b, (factor / volume_) * (-w + d_dot * gvol / volume_))


def _batch_pme_multipole_convolve_double_backward_sig(v, t):
    """Signature builder for the batched convolve double-backward kernel."""
    vec2 = wp.vec2d if t == wp.float64 else wp.vec2f
    return [
        wp.array(dtype=vec2, ndim=4),  # mesh_fft
        wp.array(dtype=t, ndim=4),  # k_squared
        wp.array(dtype=t),  # moduli_x
        wp.array(dtype=t),  # moduli_y
        wp.array(dtype=t),  # moduli_z
        wp.array(dtype=t),  # alpha (B,)
        wp.array(dtype=t),  # sigma (B,)
        wp.array(dtype=t),  # volume (B,)
        wp.array(dtype=vec2, ndim=4),  # grad_convolved
        wp.array(dtype=vec2, ndim=4),  # gg_mesh_fft
        wp.array(dtype=t, ndim=4),  # gg_k_squared
        wp.array(dtype=t),  # gg_volume (B,)
        wp.array(dtype=vec2, ndim=4),  # d_grad_convolved (out)
        wp.array(dtype=vec2, ndim=4),  # d_mesh_fft (out)
        wp.array(dtype=t, ndim=4),  # d_k_squared (out)
        wp.array(dtype=t),  # d_volume (B,) (out, atomic)
    ]


_batch_pme_multipole_convolve_double_backward_overloads = register_overloads(
    _batch_pme_multipole_convolve_double_backward_kernel,
    _batch_pme_multipole_convolve_double_backward_sig,
)


def batch_multipole_pme_convolve_double_backward_launch(
    mesh_fft,
    k_squared,
    moduli_x,
    moduli_y,
    moduli_z,
    alpha,
    sigma,
    volume,
    grad_convolved,
    gg_mesh_fft,
    gg_k_squared,
    gg_volume,
    d_grad_convolved,
    d_mesh_fft,
    d_k_squared,
    d_volume,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Batched launcher for :func:`_batch_pme_multipole_convolve_double_backward_kernel`.

    ``d_volume`` is a ``(B,)`` array — must be zero-initialized."""
    if device is None:
        device = str(mesh_fft.device)
    B, nx, ny, nz_rfft = mesh_fft.shape
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _batch_pme_multipole_convolve_double_backward_overloads[vec_dtype],
        dim=(B, nx, ny, nz_rfft),
        inputs=[
            mesh_fft,
            k_squared,
            moduli_x,
            moduli_y,
            moduli_z,
            alpha,
            sigma,
            volume,
            grad_convolved,
            gg_mesh_fft,
            gg_k_squared,
            gg_volume,
        ],
        outputs=[d_grad_convolved, d_mesh_fft, d_k_squared, d_volume],
        device=device,
    )


# =============================================================================
# Per-order specialized multipole spread kernel
# =============================================================================
#
# Per-cell math:
#     contrib = q_i · B + μ_i · ∇_cart B
# where ∇_cart B = transpose(cell_inv_t) · ∇_frac B, and ∇_frac B is the
# per-axis 1D derivative scaled by mesh_dims.

_PER_ORDER_VEC = {
    (3, wp.float32): wp.types.vector(length=3, dtype=wp.float32),
    (3, wp.float64): wp.types.vector(length=3, dtype=wp.float64),
    (4, wp.float32): wp.types.vector(length=4, dtype=wp.float32),
    (4, wp.float64): wp.types.vector(length=4, dtype=wp.float64),
    (5, wp.float32): wp.types.vector(length=5, dtype=wp.float32),
    (5, wp.float64): wp.types.vector(length=5, dtype=wp.float64),
    (6, wp.float32): wp.types.vector(length=6, dtype=wp.float32),
    (6, wp.float64): wp.types.vector(length=6, dtype=wp.float64),
}

_PER_ORDER_SUPPORTED = (3, 4, 5, 6)


def _pme_multipole_per_order_module(kind: str, order: int, scalar_dtype) -> wp.Module:
    """Named Warp module per (kind, order, dtype) tuple.

    Warp NVRTC-compiles only the modules actually launched, so importing
    this file stays metadata-only.
    """
    dtype_tag = "fp32" if scalar_dtype is wp.float32 else "fp64"
    mod = wp.get_module(
        f"nvalchemiops.interactions.electrostatics.pme_multipole_per_order."
        f"{kind}_order{order}_{dtype_tag}"
    )
    # Backward is registered via a separate autograd-Function pair, so the
    # kernel itself needs no Warp adjoint — skipping it cuts ~70% of
    # generated code on these heavily-unrolled kernels.
    mod.options["enable_backward"] = False
    return mod


def _make_pme_multipole_spread_kernel(
    ORDER: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    """Factory for the per-(order, dtype) multipole spread kernel.

    1 thread per atom. 1D B-spline weights ``wx/y/z[k]`` and their
    derivatives ``dwx/y/z[k]`` (scaled by ``mesh_dims[axis]`` for the
    fractional gradient) live in register-resident length-``ORDER``
    vectors. The order^3 stencil walk is fully unrolled at codegen time
    because ``ORDER`` is a Python int literal in scope.

    Per stencil cell ``(i, j, k)``:

    .. math::

        \\mathrm{contrib} = q_i \\, B_{ijk}
                        + \\boldsymbol{\\mu}_i \\cdot
                          \\bigl(\\mathrm{cell\\_inv\\_t}^\\top \\cdot
                                 \\nabla^{\\text{frac}}\\!B_{ijk}\\bigr)

    where :math:`B_{ijk} = w_x[i]\\,w_y[j]\\,w_z[k]` and
    :math:`\\nabla^{\\text{frac}}\\!B_{ijk}` collects the three rank-1
    derivative products (e.g. :math:`dw_x[i]\\,w_y[j]\\,w_z[k]`).

    Atomics into ``mesh`` are unavoidable — multiple atoms can spread to
    the same grid cell — but each thread emits exactly ``ORDER**3`` of
    them per launch.
    """
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(module=_pme_multipole_per_order_module("spread", ORDER, scalar_dtype))
    def kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        cell_inv_t: wp.array(dtype=Any),
        mesh: wp.array3d(dtype=Any),
    ):
        atom_idx = wp.tid()
        mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
        position = positions[atom_idx]
        charge = charges[atom_idx]
        dipole = dipoles[atom_idx]

        base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)

        t0 = theta[0]
        half_order = type(t0)(HALF_ORDER_PY)
        half_n_minus_2 = type(t0)(HALF_N_MINUS_2_PY)

        offset_start_x = wp.int32(wp.floor(theta[0] - half_n_minus_2))
        offset_start_y = wp.int32(wp.floor(theta[1] - half_n_minus_2))
        offset_start_z = wp.int32(wp.floor(theta[2] - half_n_minus_2))

        # 1D B-spline weights + derivatives in registers. ``dw{axis}[k]``
        # includes the mesh-dim scale, giving a fractional-mesh gradient.
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

        # Per-atom constant; hoisted out of the stencil walk.
        cell_inv_T = wp.transpose(cell_inv_t[0])

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wxy = wxi * wy[j]
                dwxy_x = dwxi * wy[j]
                dwxy_y = wxi * dwy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    wzk = wz[k]
                    dwzk = dwz[k]
                    weight = wxy * wzk
                    grad_frac_vec = vec_pos_dtype(
                        dwxy_x * wzk,
                        dwxy_y * wzk,
                        wxy * dwzk,
                    )
                    grad_cart = cell_inv_T * grad_frac_vec
                    contrib = charge * weight + (
                        dipole[0] * grad_cart[0]
                        + dipole[1] * grad_cart[1]
                        + dipole[2] * grad_cart[2]
                    )
                    wp.atomic_add(mesh, gx, gy, gz, contrib)

    return kernel


# Per-order spread kernel dispatch dict: {scalar_dtype: {order: kernel_overload}}.
_PER_ORDER_SPREAD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_pme_multipole_spread_kernel(_order, _scalar, _vec, _mat)
        # Concrete-typed overload so launch can resolve the kernel without
        # inspecting the Any-typed (string) annotations.
        _PER_ORDER_SPREAD_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),  # positions
                wp.array(dtype=_scalar),  # charges
                wp.array(dtype=_vec),  # dipoles
                wp.array(dtype=_mat),  # cell_inv_t
                wp.array3d(dtype=_scalar),  # mesh
            ],
        )


def _maybe_per_order_spread_kernel(order: int, wp_dtype):
    """Return the per-order specialized spread kernel overload for this
    ``(order, dtype)`` if registered, else ``None`` (caller falls back
    to the generic kernel).
    """
    return _PER_ORDER_SPREAD_KERNELS.get(wp_dtype, {}).get(order)


def _make_batch_pme_multipole_spread_kernel(
    ORDER: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    """Factory for the per-(order, dtype) *batched* multipole spread kernel.

    Same atomic-write pattern and per-cell math as
    :func:`_make_pme_multipole_spread_kernel`, with two changes:

    1. Each atom's grid slice is selected by ``batch_idx[atom_idx]``.
    2. ``cell_inv_t`` is shape ``(B, 3, 3)`` (per-system 3x3).

    The batched and single-system kernels are kept structurally identical
    so the two paths agree bit-exactly: any difference in multiplication
    order shows up as float64 round-off noise.
    """
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_module("batch_spread", ORDER, scalar_dtype)
    )
    def kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        batch_idx: wp.array(dtype=wp.int32),
        cell_inv_t: wp.array(dtype=Any),
        mesh: wp.array(dtype=Any, ndim=4),
    ):
        atom_idx = wp.tid()
        sys_idx = batch_idx[atom_idx]
        mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
        position = positions[atom_idx]
        charge = charges[atom_idx]
        dipole = dipoles[atom_idx]

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

        cell_inv_T = wp.transpose(cell_inv_t[sys_idx])

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wxy = wxi * wy[j]
                dwxy_x = dwxi * wy[j]
                dwxy_y = wxi * dwy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    wzk = wz[k]
                    dwzk = dwz[k]
                    weight = wxy * wzk
                    grad_frac_vec = vec_pos_dtype(
                        dwxy_x * wzk,
                        dwxy_y * wzk,
                        wxy * dwzk,
                    )
                    grad_cart = cell_inv_T * grad_frac_vec
                    contrib = charge * weight + (
                        dipole[0] * grad_cart[0]
                        + dipole[1] * grad_cart[1]
                        + dipole[2] * grad_cart[2]
                    )
                    wp.atomic_add(mesh, sys_idx, gx, gy, gz, contrib)

    return kernel


_PER_ORDER_BATCH_SPREAD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_batch_pme_multipole_spread_kernel(_order, _scalar, _vec, _mat)
        _PER_ORDER_BATCH_SPREAD_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),  # positions
                wp.array(dtype=_scalar),  # charges
                wp.array(dtype=_vec),  # dipoles
                wp.array(dtype=wp.int32),  # batch_idx
                wp.array(dtype=_mat),  # cell_inv_t (B, 3, 3)
                wp.array(dtype=_scalar, ndim=4),  # mesh (B, nx, ny, nz)
            ],
        )


def _maybe_per_order_batch_spread_kernel(order: int, wp_dtype):
    """Return the per-order specialized batched spread kernel overload
    for this ``(order, dtype)`` if registered, else ``None`` (caller
    falls back to the generic batched kernel).
    """
    return _PER_ORDER_BATCH_SPREAD_KERNELS.get(wp_dtype, {}).get(order)


def _make_pme_multipole_gather_hessian_kernel(
    ORDER: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    r"""Factory for the per-(order, dtype) Cartesian-Hessian gather kernel.

    1 thread per atom. The order³ stencil walk is fully unrolled at
    codegen time and the fractional-frame Hessian-times-mesh-value is
    accumulated in six scalar registers (xx, yy, zz, xy, xz, yz). The
    ``M^\top H_\text{frac} M`` chain-rule transform is hoisted out of
    the stencil and applied **once per atom** at the end — this is the
    real win over the per-stencil-cell generic kernel, which transforms
    each cell separately, and lets the kernel write its two output
    ``vec3`` slots non-atomically (each thread owns its atom).

    Per stencil cell ``(i, j, k)``:

    .. math::

        H^\text{frac}_{\alpha\beta}(i, j, k) =
            \partial_\alpha\partial_\beta B_{ijk}, \quad
        \mathrm{acc}_{\alpha\beta}\;{+}{=}\; H^\text{frac}_{\alpha\beta} \cdot \phi(g)

    with the diagonal entries using one 2nd-derivative and two weights,
    and the off-diagonals using two 1st-derivatives and one weight (cf.
    :func:`bspline_weight_hessian_3d`). The mesh-dim scale factors are
    folded into the 1D ``dw{axis}[k]`` and ``d2w{axis}[k]`` register
    arrays so the fractional Hessian comes out already scaled.

    Unlike the per-order spread, there's no requirement that the
    multiplication order match a sibling kernel bit-exactly (gather
    Hessian tests only assert symmetry + finite-difference parity), so
    the eval order here is chosen for fewest ops per cell.
    """
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_module("gather_hessian", ORDER, scalar_dtype)
    )
    def kernel(
        positions: wp.array(dtype=Any),
        cell_inv_t: wp.array(dtype=Any),
        mesh: wp.array3d(dtype=Any),
        hessian_diag: wp.array(dtype=Any),
        hessian_off: wp.array(dtype=Any),
    ):
        atom_idx = wp.tid()
        mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
        position = positions[atom_idx]

        base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)

        t0 = theta[0]
        half_order = type(t0)(HALF_ORDER_PY)
        half_n_minus_2 = type(t0)(HALF_N_MINUS_2_PY)
        zero = type(t0)(0.0)

        offset_start_x = wp.int32(wp.floor(theta[0] - half_n_minus_2))
        offset_start_y = wp.int32(wp.floor(theta[1] - half_n_minus_2))
        offset_start_z = wp.int32(wp.floor(theta[2] - half_n_minus_2))

        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        dwx = vec_ord()
        dwy = vec_ord()
        dwz = vec_ord()
        d2wx = vec_ord()
        d2wy = vec_ord()
        d2wz = vec_ord()
        md_x = type(t0)(mesh_dims[0])
        md_y = type(t0)(mesh_dims[1])
        md_z = type(t0)(mesh_dims[2])
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            dwx[k] = bspline_derivative(u_x, ORDER) * md_x
            dwy[k] = bspline_derivative(u_y, ORDER) * md_y
            dwz[k] = bspline_derivative(u_z, ORDER) * md_z
            d2wx[k] = bspline_second_derivative(u_x, ORDER) * md_x * md_x
            d2wy[k] = bspline_second_derivative(u_y, ORDER) * md_y * md_y
            d2wz[k] = bspline_second_derivative(u_z, ORDER) * md_z * md_z

        # Register accumulators for the fractional-frame Hessian times
        # the mesh value, summed over the order^3 stencil. The
        # cell_inv_t-driven Cartesian transform is hoisted out and
        # applied once at the end.
        acc_xx = zero
        acc_yy = zero
        acc_zz = zero
        acc_xy = zero
        acc_xz = zero
        acc_yz = zero

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            d2wxi = d2wx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wxy = wxi * wy[j]
                d2wxy_xx = d2wxi * wy[j]
                d2wxy_yy = wxi * d2wy[j]
                dwxy_xy = dwxi * dwy[j]
                dwxy_x = dwxi * wy[j]
                dwxy_y = wxi * dwy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    mesh_val = mesh[gx, gy, gz]
                    wzk = wz[k]
                    dwzk = dwz[k]
                    d2wzk = d2wz[k]
                    acc_xx = acc_xx + d2wxy_xx * wzk * mesh_val
                    acc_yy = acc_yy + d2wxy_yy * wzk * mesh_val
                    acc_zz = acc_zz + wxy * d2wzk * mesh_val
                    acc_xy = acc_xy + dwxy_xy * wzk * mesh_val
                    acc_xz = acc_xz + dwxy_x * dwzk * mesh_val
                    acc_yz = acc_yz + dwxy_y * dwzk * mesh_val

        # H_cart = M^T · H_frac · M (M = cell_inv_t per atom). Apply the
        # symmetric transform once using the six unique fractional
        # entries — same element-by-element formulation as the generic
        # kernel, just lifted outside the stencil loop.
        M = cell_inv_t[0]
        t00 = acc_xx * M[0, 0] + acc_xy * M[1, 0] + acc_xz * M[2, 0]
        t01 = acc_xx * M[0, 1] + acc_xy * M[1, 1] + acc_xz * M[2, 1]
        t02 = acc_xx * M[0, 2] + acc_xy * M[1, 2] + acc_xz * M[2, 2]
        t10 = acc_xy * M[0, 0] + acc_yy * M[1, 0] + acc_yz * M[2, 0]
        t11 = acc_xy * M[0, 1] + acc_yy * M[1, 1] + acc_yz * M[2, 1]
        t12 = acc_xy * M[0, 2] + acc_yy * M[1, 2] + acc_yz * M[2, 2]
        t20 = acc_xz * M[0, 0] + acc_yz * M[1, 0] + acc_zz * M[2, 0]
        t21 = acc_xz * M[0, 1] + acc_yz * M[1, 1] + acc_zz * M[2, 1]
        t22 = acc_xz * M[0, 2] + acc_yz * M[1, 2] + acc_zz * M[2, 2]

        h_xx_cart = M[0, 0] * t00 + M[1, 0] * t10 + M[2, 0] * t20
        h_yy_cart = M[0, 1] * t01 + M[1, 1] * t11 + M[2, 1] * t21
        h_zz_cart = M[0, 2] * t02 + M[1, 2] * t12 + M[2, 2] * t22
        h_xy_cart = M[0, 0] * t01 + M[1, 0] * t11 + M[2, 0] * t21
        h_xz_cart = M[0, 0] * t02 + M[1, 0] * t12 + M[2, 0] * t22
        h_yz_cart = M[0, 1] * t02 + M[1, 1] * t12 + M[2, 1] * t22

        # One thread per atom owns its output slot — no atomic needed.
        hessian_diag[atom_idx] = vec_pos_dtype(h_xx_cart, h_yy_cart, h_zz_cart)
        hessian_off[atom_idx] = vec_pos_dtype(h_xy_cart, h_xz_cart, h_yz_cart)

    return kernel


_PER_ORDER_GATHER_HESSIAN_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_pme_multipole_gather_hessian_kernel(_order, _scalar, _vec, _mat)
        _PER_ORDER_GATHER_HESSIAN_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),  # positions
                wp.array(dtype=_mat),  # cell_inv_t
                wp.array3d(dtype=_scalar),  # mesh
                wp.array(dtype=_vec),  # hessian_diag
                wp.array(dtype=_vec),  # hessian_off
            ],
        )


def _maybe_per_order_gather_hessian_kernel(order: int, wp_dtype):
    """Return the per-order specialized gather-Hessian kernel overload
    for this ``(order, dtype)`` if registered, else ``None`` (caller
    falls back to the generic kernel).
    """
    return _PER_ORDER_GATHER_HESSIAN_KERNELS.get(wp_dtype, {}).get(order)


def _make_pme_multipole_spread_backward_kernel(
    ORDER: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    r"""Factory for the per-(order, dtype) multipole spread-backward kernel.

    Same analytical backward as :func:`_pme_multipole_spread_backward_kernel`
    but with 1 thread per atom, the order^3 stencil walk fully unrolled,
    and the per-atom accumulators living in registers.

    Per stencil cell ``(i, j, k)``, the kernel updates ten scalar
    register accumulators:

    * ``acc_q``                     — ``Σ_g grad_mesh[g] · B(g)``
    * ``acc_field_frac_{x,y,z}``    — ``Σ_g grad_mesh[g] · ∇_frac B(g)``
    * ``acc_H_{xx,yy,zz,xy,xz,yz}`` — ``Σ_g grad_mesh[g] · ∂²_frac B(g)``

    After the stencil walk, the per-atom output is assembled in one
    pass by transforming the fractional accumulators with the per-atom
    constant ``cell_inv_t`` matrix (avoiding the per-cell
    ``transpose(cell_inv_t) · grad_frac`` and ``M^T · H_frac · M``
    cuBLAS-dispatch shapes the generic kernel paid for every stencil
    point). Outputs are written non-atomically — each thread owns its
    ``atom_idx`` slot.

    Math identities used:

    * ``grad_μ_cart  = M^T · (Σ_g grad_mesh · ∇_frac B)``
    * ``grad_pos     = q · grad_μ_cart + M^T · ((Σ_g grad_mesh · H_frac) · M · μ)``

    See :func:`_pme_multipole_spread_backward_kernel` for the per-cell
    formulation and the sign-convention notes.
    """
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_module("spread_backward", ORDER, scalar_dtype)
    )
    def kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        cell_inv_t: wp.array(dtype=Any),
        grad_mesh: wp.array3d(dtype=Any),
        grad_positions: wp.array(dtype=Any),
        grad_charges: wp.array(dtype=Any),
        grad_dipoles: wp.array(dtype=Any),
    ):
        atom_idx = wp.tid()
        mesh_dims = wp.vec3i(grad_mesh.shape[0], grad_mesh.shape[1], grad_mesh.shape[2])
        position = positions[atom_idx]
        charge = charges[atom_idx]
        dipole = dipoles[atom_idx]

        base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)

        t0 = theta[0]
        half_order = type(t0)(HALF_ORDER_PY)
        half_n_minus_2 = type(t0)(HALF_N_MINUS_2_PY)
        zero = type(t0)(0.0)

        offset_start_x = wp.int32(wp.floor(theta[0] - half_n_minus_2))
        offset_start_y = wp.int32(wp.floor(theta[1] - half_n_minus_2))
        offset_start_z = wp.int32(wp.floor(theta[2] - half_n_minus_2))

        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        dwx = vec_ord()
        dwy = vec_ord()
        dwz = vec_ord()
        d2wx = vec_ord()
        d2wy = vec_ord()
        d2wz = vec_ord()
        md_x = type(t0)(mesh_dims[0])
        md_y = type(t0)(mesh_dims[1])
        md_z = type(t0)(mesh_dims[2])
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            dwx[k] = bspline_derivative(u_x, ORDER) * md_x
            dwy[k] = bspline_derivative(u_y, ORDER) * md_y
            dwz[k] = bspline_derivative(u_z, ORDER) * md_z
            d2wx[k] = bspline_second_derivative(u_x, ORDER) * md_x * md_x
            d2wy[k] = bspline_second_derivative(u_y, ORDER) * md_y * md_y
            d2wz[k] = bspline_second_derivative(u_z, ORDER) * md_z * md_z

        acc_q = zero
        acc_fx = zero
        acc_fy = zero
        acc_fz = zero
        acc_hxx = zero
        acc_hyy = zero
        acc_hzz = zero
        acc_hxy = zero
        acc_hxz = zero
        acc_hyz = zero

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            d2wxi = d2wx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wxy = wxi * wy[j]
                dwxy_x = dwxi * wy[j]
                dwxy_y = wxi * dwy[j]
                d2wxy_xx = d2wxi * wy[j]
                d2wxy_yy = wxi * d2wy[j]
                dwxy_xy = dwxi * dwy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    mesh_val = grad_mesh[gx, gy, gz]
                    wzk = wz[k]
                    dwzk = dwz[k]
                    d2wzk = d2wz[k]
                    weight = wxy * wzk
                    acc_q = acc_q + mesh_val * weight
                    acc_fx = acc_fx + mesh_val * dwxy_x * wzk
                    acc_fy = acc_fy + mesh_val * dwxy_y * wzk
                    acc_fz = acc_fz + mesh_val * wxy * dwzk
                    acc_hxx = acc_hxx + mesh_val * d2wxy_xx * wzk
                    acc_hyy = acc_hyy + mesh_val * d2wxy_yy * wzk
                    acc_hzz = acc_hzz + mesh_val * wxy * d2wzk
                    acc_hxy = acc_hxy + mesh_val * dwxy_xy * wzk
                    acc_hxz = acc_hxz + mesh_val * dwxy_x * dwzk
                    acc_hyz = acc_hyz + mesh_val * dwxy_y * dwzk

        # grad_μ_cart = M^T · (field_frac)
        M = cell_inv_t[0]
        gmu_x = M[0, 0] * acc_fx + M[1, 0] * acc_fy + M[2, 0] * acc_fz
        gmu_y = M[0, 1] * acc_fx + M[1, 1] * acc_fy + M[2, 1] * acc_fz
        gmu_z = M[0, 2] * acc_fx + M[1, 2] * acc_fy + M[2, 2] * acc_fz

        # mu_frac = M · μ ; Hμ_frac = H_frac · mu_frac ; Hμ_cart = M^T · Hμ_frac
        mfx = M[0, 0] * dipole[0] + M[0, 1] * dipole[1] + M[0, 2] * dipole[2]
        mfy = M[1, 0] * dipole[0] + M[1, 1] * dipole[1] + M[1, 2] * dipole[2]
        mfz = M[2, 0] * dipole[0] + M[2, 1] * dipole[1] + M[2, 2] * dipole[2]

        hmu_x = acc_hxx * mfx + acc_hxy * mfy + acc_hxz * mfz
        hmu_y = acc_hxy * mfx + acc_hyy * mfy + acc_hyz * mfz
        hmu_z = acc_hxz * mfx + acc_hyz * mfy + acc_hzz * mfz

        hmuc_x = M[0, 0] * hmu_x + M[1, 0] * hmu_y + M[2, 0] * hmu_z
        hmuc_y = M[0, 1] * hmu_x + M[1, 1] * hmu_y + M[2, 1] * hmu_z
        hmuc_z = M[0, 2] * hmu_x + M[1, 2] * hmu_y + M[2, 2] * hmu_z

        grad_charges[atom_idx] = acc_q
        grad_dipoles[atom_idx] = vec_pos_dtype(gmu_x, gmu_y, gmu_z)
        grad_positions[atom_idx] = vec_pos_dtype(
            charge * gmu_x + hmuc_x,
            charge * gmu_y + hmuc_y,
            charge * gmu_z + hmuc_z,
        )

    return kernel


_PER_ORDER_SPREAD_BACKWARD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_pme_multipole_spread_backward_kernel(_order, _scalar, _vec, _mat)
        _PER_ORDER_SPREAD_BACKWARD_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),  # positions
                wp.array(dtype=_scalar),  # charges
                wp.array(dtype=_vec),  # dipoles
                wp.array(dtype=_mat),  # cell_inv_t
                wp.array3d(dtype=_scalar),  # grad_mesh
                wp.array(dtype=_vec),  # grad_positions (out)
                wp.array(dtype=_scalar),  # grad_charges (out)
                wp.array(dtype=_vec),  # grad_dipoles (out)
            ],
        )


def _maybe_per_order_spread_backward_kernel(order: int, wp_dtype):
    """Return the per-order specialized spread-backward kernel overload
    for this ``(order, dtype)`` if registered, else ``None`` (caller
    falls back to the generic kernel).
    """
    return _PER_ORDER_SPREAD_BACKWARD_KERNELS.get(wp_dtype, {}).get(order)


def _make_batch_pme_multipole_spread_backward_kernel(
    ORDER: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    """Factory for the batched per-(order, dtype) spread-backward kernel.

    Same accumulator structure as
    :func:`_make_pme_multipole_spread_backward_kernel`, with the per-atom
    grid slice selected by ``batch_idx[atom_idx]`` and ``cell_inv_t``
    indexed per system.
    """
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_module(
            "batch_spread_backward", ORDER, scalar_dtype
        )
    )
    def kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        batch_idx: wp.array(dtype=wp.int32),
        cell_inv_t: wp.array(dtype=Any),
        grad_mesh: wp.array(dtype=Any, ndim=4),
        grad_positions: wp.array(dtype=Any),
        grad_charges: wp.array(dtype=Any),
        grad_dipoles: wp.array(dtype=Any),
    ):
        atom_idx = wp.tid()
        sys_idx = batch_idx[atom_idx]
        mesh_dims = wp.vec3i(grad_mesh.shape[1], grad_mesh.shape[2], grad_mesh.shape[3])
        position = positions[atom_idx]
        charge = charges[atom_idx]
        dipole = dipoles[atom_idx]

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

        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        dwx = vec_ord()
        dwy = vec_ord()
        dwz = vec_ord()
        d2wx = vec_ord()
        d2wy = vec_ord()
        d2wz = vec_ord()
        md_x = type(t0)(mesh_dims[0])
        md_y = type(t0)(mesh_dims[1])
        md_z = type(t0)(mesh_dims[2])
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            dwx[k] = bspline_derivative(u_x, ORDER) * md_x
            dwy[k] = bspline_derivative(u_y, ORDER) * md_y
            dwz[k] = bspline_derivative(u_z, ORDER) * md_z
            d2wx[k] = bspline_second_derivative(u_x, ORDER) * md_x * md_x
            d2wy[k] = bspline_second_derivative(u_y, ORDER) * md_y * md_y
            d2wz[k] = bspline_second_derivative(u_z, ORDER) * md_z * md_z

        acc_q = zero
        acc_fx = zero
        acc_fy = zero
        acc_fz = zero
        acc_hxx = zero
        acc_hyy = zero
        acc_hzz = zero
        acc_hxy = zero
        acc_hxz = zero
        acc_hyz = zero

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            d2wxi = d2wx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wxy = wxi * wy[j]
                dwxy_x = dwxi * wy[j]
                dwxy_y = wxi * dwy[j]
                d2wxy_xx = d2wxi * wy[j]
                d2wxy_yy = wxi * d2wy[j]
                dwxy_xy = dwxi * dwy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    mesh_val = grad_mesh[sys_idx, gx, gy, gz]
                    wzk = wz[k]
                    dwzk = dwz[k]
                    d2wzk = d2wz[k]
                    weight = wxy * wzk
                    acc_q = acc_q + mesh_val * weight
                    acc_fx = acc_fx + mesh_val * dwxy_x * wzk
                    acc_fy = acc_fy + mesh_val * dwxy_y * wzk
                    acc_fz = acc_fz + mesh_val * wxy * dwzk
                    acc_hxx = acc_hxx + mesh_val * d2wxy_xx * wzk
                    acc_hyy = acc_hyy + mesh_val * d2wxy_yy * wzk
                    acc_hzz = acc_hzz + mesh_val * wxy * d2wzk
                    acc_hxy = acc_hxy + mesh_val * dwxy_xy * wzk
                    acc_hxz = acc_hxz + mesh_val * dwxy_x * dwzk
                    acc_hyz = acc_hyz + mesh_val * dwxy_y * dwzk

        M = cell_inv_t[sys_idx]
        gmu_x = M[0, 0] * acc_fx + M[1, 0] * acc_fy + M[2, 0] * acc_fz
        gmu_y = M[0, 1] * acc_fx + M[1, 1] * acc_fy + M[2, 1] * acc_fz
        gmu_z = M[0, 2] * acc_fx + M[1, 2] * acc_fy + M[2, 2] * acc_fz

        mfx = M[0, 0] * dipole[0] + M[0, 1] * dipole[1] + M[0, 2] * dipole[2]
        mfy = M[1, 0] * dipole[0] + M[1, 1] * dipole[1] + M[1, 2] * dipole[2]
        mfz = M[2, 0] * dipole[0] + M[2, 1] * dipole[1] + M[2, 2] * dipole[2]

        hmu_x = acc_hxx * mfx + acc_hxy * mfy + acc_hxz * mfz
        hmu_y = acc_hxy * mfx + acc_hyy * mfy + acc_hyz * mfz
        hmu_z = acc_hxz * mfx + acc_hyz * mfy + acc_hzz * mfz

        hmuc_x = M[0, 0] * hmu_x + M[1, 0] * hmu_y + M[2, 0] * hmu_z
        hmuc_y = M[0, 1] * hmu_x + M[1, 1] * hmu_y + M[2, 1] * hmu_z
        hmuc_z = M[0, 2] * hmu_x + M[1, 2] * hmu_y + M[2, 2] * hmu_z

        grad_charges[atom_idx] = acc_q
        grad_dipoles[atom_idx] = vec_pos_dtype(gmu_x, gmu_y, gmu_z)
        grad_positions[atom_idx] = vec_pos_dtype(
            charge * gmu_x + hmuc_x,
            charge * gmu_y + hmuc_y,
            charge * gmu_z + hmuc_z,
        )

    return kernel


_PER_ORDER_BATCH_SPREAD_BACKWARD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_batch_pme_multipole_spread_backward_kernel(
            _order, _scalar, _vec, _mat
        )
        _PER_ORDER_BATCH_SPREAD_BACKWARD_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),  # positions
                wp.array(dtype=_scalar),  # charges
                wp.array(dtype=_vec),  # dipoles
                wp.array(dtype=wp.int32),  # batch_idx
                wp.array(dtype=_mat),  # cell_inv_t (B, 3, 3)
                wp.array(dtype=_scalar, ndim=4),  # grad_mesh (B, nx, ny, nz)
                wp.array(dtype=_vec),  # grad_positions (out)
                wp.array(dtype=_scalar),  # grad_charges (out)
                wp.array(dtype=_vec),  # grad_dipoles (out)
            ],
        )


def _maybe_per_order_batch_spread_backward_kernel(order: int, wp_dtype):
    """Return the per-order batched spread-backward kernel overload for
    this ``(order, dtype)`` if registered, else ``None``.
    """
    return _PER_ORDER_BATCH_SPREAD_BACKWARD_KERNELS.get(wp_dtype, {}).get(order)


def _make_pme_multipole_gather_gradient_kernel(
    ORDER: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    r"""Factory for the per-(order, dtype) gather-gradient kernel.

    Computes the "force" form ``F_i = -q_i · Σ_g φ(g) · ∇_cart B(r_i, g)``
    consumed by the multipole PME gather-potential backward and the
    gather-field forward (which negates the result). Same math as
    ``_bspline_gather_gradient_kernel`` in ``math.spline``, but with the
    standard MZ-2 per-order shape:

    * 1 thread per atom, order^3 stencil fully unrolled.
    * Three scalar fractional-field accumulators in registers.
    * ``M^T`` transform applied once outside the stencil walk.
    * Non-atomic per-atom output (each thread owns its slot).

    Lives next to the multipole-specific spread/gather kernels rather
    than in ``math.spline`` to keep this branch self-contained against
    the parallel monopole spline rewrite.
    """
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_module("gather_gradient", ORDER, scalar_dtype)
    )
    def kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        cell_inv_t: wp.array(dtype=Any),
        mesh: wp.array3d(dtype=Any),
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

        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        dwx = vec_ord()
        dwy = vec_ord()
        dwz = vec_ord()
        md_x = type(t0)(mesh_dims[0])
        md_y = type(t0)(mesh_dims[1])
        md_z = type(t0)(mesh_dims[2])
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            dwx[k] = bspline_derivative(u_x, ORDER) * md_x
            dwy[k] = bspline_derivative(u_y, ORDER) * md_y
            dwz[k] = bspline_derivative(u_z, ORDER) * md_z

        acc_fx = zero
        acc_fy = zero
        acc_fz = zero

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wxy = wxi * wy[j]
                dwxy_x = dwxi * wy[j]
                dwxy_y = wxi * dwy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    mesh_val = mesh[gx, gy, gz]
                    wzk = wz[k]
                    dwzk = dwz[k]
                    acc_fx = acc_fx + mesh_val * dwxy_x * wzk
                    acc_fy = acc_fy + mesh_val * dwxy_y * wzk
                    acc_fz = acc_fz + mesh_val * wxy * dwzk

        # F_cart = -charge · M^T · field_frac. Hoist the ``-charge``
        # multiply and the M^T transform outside the stencil walk.
        M = cell_inv_t[0]
        neg_q = -charge
        fc_x = neg_q * (M[0, 0] * acc_fx + M[1, 0] * acc_fy + M[2, 0] * acc_fz)
        fc_y = neg_q * (M[0, 1] * acc_fx + M[1, 1] * acc_fy + M[2, 1] * acc_fz)
        fc_z = neg_q * (M[0, 2] * acc_fx + M[1, 2] * acc_fy + M[2, 2] * acc_fz)

        forces[atom_idx] = vec_pos_dtype(fc_x, fc_y, fc_z)

    return kernel


_PER_ORDER_GATHER_GRADIENT_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_pme_multipole_gather_gradient_kernel(_order, _scalar, _vec, _mat)
        _PER_ORDER_GATHER_GRADIENT_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),  # positions
                wp.array(dtype=_scalar),  # charges
                wp.array(dtype=_mat),  # cell_inv_t
                wp.array3d(dtype=_scalar),  # mesh
                wp.array(dtype=_vec),  # forces (out)
            ],
        )


def _maybe_per_order_gather_gradient_kernel(order: int, wp_dtype):
    """Return the per-order gather-gradient kernel overload, or ``None``."""
    return _PER_ORDER_GATHER_GRADIENT_KERNELS.get(wp_dtype, {}).get(order)


def multipole_pme_gather_gradient_launch(
    positions: wp.array,
    charges: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    forces: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> bool:
    r"""Per-order specialized gather-gradient (force) launcher.

    Gathers per-atom forces :math:`-q_i \nabla \phi(r_i)` from the
    potential mesh via the per-order specialized kernel.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=vec3f/vec3d
        Cartesian atom positions.
    charges : wp.array, shape (N,), dtype=wp.float32/float64
        Per-atom monopole charges.
    cell_inv_t : wp.array, shape (1,), dtype=mat33f/mat33d
        Transpose of the inverse cell matrix.
    order : int
        B-spline order.
    mesh : wp.array, shape (Nx, Ny, Nz), dtype=wp.float32/float64
        Potential grid ``φ(g)``.
    forces : wp.array, shape (N,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Per-atom force contribution.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Returns
    -------
    bool
        ``True`` when the per-order kernel ran, ``False`` when no overload
        exists for this ``(order, dtype)`` (caller falls back to the
        generic ``spline_gather_gradient``).

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    per_order = _maybe_per_order_gather_gradient_kernel(order, wp_dtype)
    if per_order is None:
        return False
    if device is None:
        device = str(positions.device)
    num_atoms = positions.shape[0]
    wp.launch(
        per_order,
        dim=(num_atoms,),
        inputs=[
            positions,
            charges,
            cell_inv_t,
            mesh,
        ],
        outputs=[forces],
        device=device,
    )
    return True


def _make_batch_pme_multipole_gather_gradient_kernel(
    ORDER: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    """Batched companion of :func:`_make_pme_multipole_gather_gradient_kernel`."""
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_module(
            "batch_gather_gradient", ORDER, scalar_dtype
        )
    )
    def kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        batch_idx: wp.array(dtype=wp.int32),
        cell_inv_t: wp.array(dtype=Any),
        mesh: wp.array(dtype=Any, ndim=4),
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

        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        dwx = vec_ord()
        dwy = vec_ord()
        dwz = vec_ord()
        md_x = type(t0)(mesh_dims[0])
        md_y = type(t0)(mesh_dims[1])
        md_z = type(t0)(mesh_dims[2])
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            dwx[k] = bspline_derivative(u_x, ORDER) * md_x
            dwy[k] = bspline_derivative(u_y, ORDER) * md_y
            dwz[k] = bspline_derivative(u_z, ORDER) * md_z

        acc_fx = zero
        acc_fy = zero
        acc_fz = zero

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wxy = wxi * wy[j]
                dwxy_x = dwxi * wy[j]
                dwxy_y = wxi * dwy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    mesh_val = mesh[sys_idx, gx, gy, gz]
                    wzk = wz[k]
                    dwzk = dwz[k]
                    acc_fx = acc_fx + mesh_val * dwxy_x * wzk
                    acc_fy = acc_fy + mesh_val * dwxy_y * wzk
                    acc_fz = acc_fz + mesh_val * wxy * dwzk

        M = cell_inv_t[sys_idx]
        neg_q = -charge
        fc_x = neg_q * (M[0, 0] * acc_fx + M[1, 0] * acc_fy + M[2, 0] * acc_fz)
        fc_y = neg_q * (M[0, 1] * acc_fx + M[1, 1] * acc_fy + M[2, 1] * acc_fz)
        fc_z = neg_q * (M[0, 2] * acc_fx + M[1, 2] * acc_fy + M[2, 2] * acc_fz)

        forces[atom_idx] = vec_pos_dtype(fc_x, fc_y, fc_z)

    return kernel


_PER_ORDER_BATCH_GATHER_GRADIENT_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_batch_pme_multipole_gather_gradient_kernel(
            _order, _scalar, _vec, _mat
        )
        _PER_ORDER_BATCH_GATHER_GRADIENT_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),  # positions
                wp.array(dtype=_scalar),  # charges
                wp.array(dtype=wp.int32),  # batch_idx
                wp.array(dtype=_mat),  # cell_inv_t (B, 3, 3)
                wp.array(dtype=_scalar, ndim=4),  # mesh (B, nx, ny, nz)
                wp.array(dtype=_vec),  # forces (out)
            ],
        )


def _maybe_per_order_batch_gather_gradient_kernel(order: int, wp_dtype):
    """Return the per-order batched gather-gradient kernel overload, or ``None``."""
    return _PER_ORDER_BATCH_GATHER_GRADIENT_KERNELS.get(wp_dtype, {}).get(order)


def batch_multipole_pme_gather_gradient_launch(
    positions: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    mesh: wp.array,
    forces: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> bool:
    r"""Per-order batched gather-gradient (force) launcher.

    Batched analog of :func:`multipole_pme_gather_gradient_launch`.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=vec3f/vec3d
        Concatenated Cartesian atom positions.
    charges : wp.array, shape (N_total,), dtype=wp.float32/float64
        Per-atom monopole charges.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        Per-atom system index into the leading mesh axis.
    cell_inv_t : wp.array, shape (B,), dtype=mat33f/mat33d
        Per-system transpose of the inverse cell matrix.
    order : int
        B-spline order.
    mesh : wp.array, shape (B, Nx, Ny, Nz), dtype=wp.float32/float64
        Per-system potential grid.
    forces : wp.array, shape (N_total,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Per-atom force contribution.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Returns
    -------
    bool
        ``True`` if the per-order kernel ran, ``False`` on cache miss.

    Launch Grid
    -----------
    ``dim = (N_total,)`` — one thread per atom across the batch.
    """
    per_order = _maybe_per_order_batch_gather_gradient_kernel(order, wp_dtype)
    if per_order is None:
        return False
    if device is None:
        device = str(positions.device)
    num_atoms = positions.shape[0]
    wp.launch(
        per_order,
        dim=(num_atoms,),
        inputs=[
            positions,
            charges,
            batch_idx,
            cell_inv_t,
            mesh,
        ],
        outputs=[forces],
        device=device,
    )
    return True


# =============================================================================
# Unified per-(ORDER, LMAX, dtype) factory
# =============================================================================


def _pme_multipole_per_order_lmax_module(
    kind: str, order: int, lmax: int, scalar_dtype
) -> wp.Module:
    """Named Warp module per ``(kind, order, lmax, dtype)`` tuple."""
    dtype_tag = "fp32" if scalar_dtype is wp.float32 else "fp64"
    mod = wp.get_module(
        f"nvalchemiops.interactions.electrostatics.pme_multipole_per_order."
        f"{kind}_order{order}_l{lmax}_{dtype_tag}"
    )
    return mod


# ---- Unified spread factory ----


def _make_pme_multipole_spread_unified_kernel(
    ORDER: int,
    LMAX: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    r"""Factory for the unified per-(ORDER, LMAX, dtype) spread kernel.

    1 thread per atom. Always emits the charge contribution. Codegen-time
    ``if LMAX >= N`` gates the dipole (LMAX≥1) and quadrupole (LMAX≥2)
    branches: NVRTC sees a flat kernel with only the active channels.

    Per stencil cell ``(i, j, k)``:

    .. math::

        \mathrm{contrib} = q_i\, B
                          + \boldsymbol{\mu}_i \cdot \nabla_{\text{cart}} B
                          + \tfrac{1}{2}\, Q_i^{\alpha\beta}
                            (\nabla \nabla_{\text{cart}})_{\alpha\beta} B

    Common subexpressions ``wx[i]·wy[j]``, ``dwx[i]·wy[j]``, etc. are
    factored out of the innermost loop. The cell-transform matrices
    ``M^T`` and ``M Q M^T`` (latter for ``LMAX ≥ 2``) are hoisted to
    per-atom constants outside the stencil walk.
    """
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_lmax_module("spread", ORDER, LMAX, scalar_dtype)
    )
    def kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        quadrupoles: wp.array(dtype=Any),
        cell_inv_t: wp.array(dtype=Any),
        mesh: wp.array3d(dtype=Any),
    ):
        atom_idx = wp.tid()
        mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
        position = positions[atom_idx]
        charge = charges[atom_idx]

        base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)

        t0 = theta[0]
        half_order = type(t0)(HALF_ORDER_PY)
        half_n_minus_2 = type(t0)(HALF_N_MINUS_2_PY)

        offset_start_x = wp.int32(wp.floor(theta[0] - half_n_minus_2))
        offset_start_y = wp.int32(wp.floor(theta[1] - half_n_minus_2))
        offset_start_z = wp.int32(wp.floor(theta[2] - half_n_minus_2))

        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        if LMAX >= 1:
            dwx = vec_ord()
            dwy = vec_ord()
            dwz = vec_ord()
        if LMAX >= 2:
            d2wx = vec_ord()
            d2wy = vec_ord()
            d2wz = vec_ord()

        md_x = type(t0)(mesh_dims[0])
        md_y = type(t0)(mesh_dims[1])
        md_z = type(t0)(mesh_dims[2])
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            if LMAX >= 1:
                dwx[k] = bspline_derivative(u_x, ORDER) * md_x
                dwy[k] = bspline_derivative(u_y, ORDER) * md_y
                dwz[k] = bspline_derivative(u_z, ORDER) * md_z
            if LMAX >= 2:
                d2wx[k] = bspline_second_derivative(u_x, ORDER) * md_x * md_x
                d2wy[k] = bspline_second_derivative(u_y, ORDER) * md_y * md_y
                d2wz[k] = bspline_second_derivative(u_z, ORDER) * md_z * md_z

        # Hoist per-atom-constant matrix products.
        if LMAX >= 1:
            cell_inv_T = wp.transpose(cell_inv_t[0])
            dipole = dipoles[atom_idx]
        if LMAX >= 2:
            Q = quadrupoles[atom_idx]
            M = cell_inv_t[0]
            # Compute Q_eff = M · Q · M^T (six unique symmetric entries).
            MQ_00 = M[0, 0] * Q[0, 0] + M[0, 1] * Q[1, 0] + M[0, 2] * Q[2, 0]
            MQ_01 = M[0, 0] * Q[0, 1] + M[0, 1] * Q[1, 1] + M[0, 2] * Q[2, 1]
            MQ_02 = M[0, 0] * Q[0, 2] + M[0, 1] * Q[1, 2] + M[0, 2] * Q[2, 2]
            MQ_10 = M[1, 0] * Q[0, 0] + M[1, 1] * Q[1, 0] + M[1, 2] * Q[2, 0]
            MQ_11 = M[1, 0] * Q[0, 1] + M[1, 1] * Q[1, 1] + M[1, 2] * Q[2, 1]
            MQ_12 = M[1, 0] * Q[0, 2] + M[1, 1] * Q[1, 2] + M[1, 2] * Q[2, 2]
            MQ_20 = M[2, 0] * Q[0, 0] + M[2, 1] * Q[1, 0] + M[2, 2] * Q[2, 0]
            MQ_21 = M[2, 0] * Q[0, 1] + M[2, 1] * Q[1, 1] + M[2, 2] * Q[2, 1]
            MQ_22 = M[2, 0] * Q[0, 2] + M[2, 1] * Q[1, 2] + M[2, 2] * Q[2, 2]
            Qe_00 = MQ_00 * M[0, 0] + MQ_01 * M[0, 1] + MQ_02 * M[0, 2]
            Qe_11 = MQ_10 * M[1, 0] + MQ_11 * M[1, 1] + MQ_12 * M[1, 2]
            Qe_22 = MQ_20 * M[2, 0] + MQ_21 * M[2, 1] + MQ_22 * M[2, 2]
            Qe_01 = MQ_00 * M[1, 0] + MQ_01 * M[1, 1] + MQ_02 * M[1, 2]
            Qe_02 = MQ_00 * M[2, 0] + MQ_01 * M[2, 1] + MQ_02 * M[2, 2]
            Qe_12 = MQ_10 * M[2, 0] + MQ_11 * M[2, 1] + MQ_12 * M[2, 2]

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            if LMAX >= 1:
                dwxi = dwx[i]
            if LMAX >= 2:
                d2wxi = d2wx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wxy = wxi * wy[j]
                if LMAX >= 1:
                    dwxy_x = dwxi * wy[j]
                    dwxy_y = wxi * dwy[j]
                if LMAX >= 2:
                    d2wxy_xx = d2wxi * wy[j]
                    d2wxy_yy = wxi * d2wy[j]
                    dwxy_xy = dwxi * dwy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    wzk = wz[k]
                    weight = wxy * wzk
                    contrib = charge * weight

                    if LMAX >= 1:
                        dwzk = dwz[k]
                        grad_frac_vec = vec_pos_dtype(
                            dwxy_x * wzk,
                            dwxy_y * wzk,
                            wxy * dwzk,
                        )
                        grad_cart = cell_inv_T * grad_frac_vec
                        contrib = contrib + (
                            dipole[0] * grad_cart[0]
                            + dipole[1] * grad_cart[1]
                            + dipole[2] * grad_cart[2]
                        )

                    if LMAX >= 2:
                        d2wzk = d2wz[k]
                        # H_frac symmetric entries:
                        H_xx = d2wxy_xx * wzk
                        H_yy = d2wxy_yy * wzk
                        H_zz = wxy * d2wzk
                        H_xy = dwxy_xy * wzk
                        H_xz = dwxy_x * dwzk
                        H_yz = dwxy_y * dwzk
                        # (1/2) Q_eff : H_frac (off-diagonal × 2 for symmetry).
                        half = type(weight)(0.5)
                        two = type(weight)(2.0)
                        contrib = contrib + half * (
                            Qe_00 * H_xx
                            + Qe_11 * H_yy
                            + Qe_22 * H_zz
                            + two * (Qe_01 * H_xy + Qe_02 * H_xz + Qe_12 * H_yz)
                        )

                    wp.atomic_add(mesh, gx, gy, gz, contrib)

    return kernel


_PER_ORDER_LMAX_SPREAD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _lmax in (0, 1, 2):
        for _scalar, _vec, _mat in (
            (wp.float32, wp.vec3f, wp.mat33f),
            (wp.float64, wp.vec3d, wp.mat33d),
        ):
            _k = _make_pme_multipole_spread_unified_kernel(
                _order, _lmax, _scalar, _vec, _mat
            )
            _PER_ORDER_LMAX_SPREAD_KERNELS[_scalar][(_order, _lmax)] = wp.overload(
                _k,
                [
                    wp.array(dtype=_vec),  # positions
                    wp.array(dtype=_scalar),  # charges
                    wp.array(dtype=_vec),  # dipoles
                    wp.array(dtype=_mat),  # quadrupoles
                    wp.array(dtype=_mat),  # cell_inv_t
                    wp.array3d(dtype=_scalar),  # mesh
                ],
            )


def _maybe_per_order_lmax_spread_kernel(order: int, lmax: int, wp_dtype):
    """Return the per-(ORDER, LMAX, dtype) unified spread kernel, or None."""
    return _PER_ORDER_LMAX_SPREAD_KERNELS.get(wp_dtype, {}).get((order, lmax))


def multipole_pme_spread_unified_launch(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    quadrupoles: wp.array,
    cell_inv_t: wp.array,
    order: int,
    lmax: int,
    mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> bool:
    r"""Launch the unified ``(ORDER, LMAX)`` spread kernel.

    Spreads charges (LMAX≥0), dipoles (LMAX≥1), and quadrupoles (LMAX≥2)
    onto a single density mesh. The kernel signature is fixed across LMAX;
    unused moment arrays may be zero-sized dummies.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=vec3f/vec3d
        Cartesian atom positions.
    charges : wp.array, shape (N,), dtype=wp.float32/float64
        Per-atom monopole charges.
    dipoles : wp.array, shape (N,), dtype=vec3f/vec3d
        Per-atom Cartesian dipoles (used when ``lmax >= 1``).
    quadrupoles : wp.array, shape (N,), dtype=mat33f/mat33d
        Per-atom Cartesian quadrupole tensors (used when ``lmax >= 2``).
    cell_inv_t : wp.array, shape (1,), dtype=mat33f/mat33d
        Transpose of the inverse cell matrix.
    order : int
        B-spline order (one of ``(3, 4, 5, 6)``).
    lmax : int
        Maximum multipole order to spread (0, 1, or 2).
    mesh : wp.array, shape (Nx, Ny, Nz), dtype=wp.float32/float64
        OUTPUT. Charge-density mesh, pre-zeroed.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Returns
    -------
    bool
        ``True`` if a per-order overload exists and was launched, ``False``
        if no overload is registered for this ``(order, lmax, dtype)``
        (caller falls back).

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    per_order = _maybe_per_order_lmax_spread_kernel(order, lmax, wp_dtype)
    if per_order is None:
        return False
    if device is None:
        device = str(positions.device)
    num_atoms = positions.shape[0]
    wp.launch(
        per_order,
        dim=(num_atoms,),
        inputs=[positions, charges, dipoles, quadrupoles, cell_inv_t],
        outputs=[mesh],
        device=device,
    )
    return True


# ---- Unified spread-backward factory ----


def _make_pme_multipole_spread_backward_unified_kernel(
    ORDER: int,
    LMAX: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    r"""Factory for the unified per-(ORDER, LMAX, dtype) spread-backward kernel.

    1 thread per atom. Same codegen-time ``if LMAX >= N`` gating as the
    forward factory: outputs per-atom gradients of ``charges``, ``dipoles``
    (LMAX≥1), ``quadrupoles`` (LMAX≥2), and ``positions``.

    Math:

    .. math::

        \partial L/\partial q_i &= \sum_g \partial L/\partial \rho(g)\, B(r_i, g),  \\
        \partial L/\partial \mu_i &= \sum_g \partial L/\partial \rho(g)\, \nabla_\text{cart} B,  \\
        \partial L/\partial Q_i^{\alpha\beta} &= \tfrac{1}{2} \sum_g \partial L/\partial \rho(g)\,
                                                 (\nabla\nabla_\text{cart})_{\alpha\beta} B,  \\
        \partial L/\partial r_i^\gamma &= \sum_g \partial L/\partial \rho(g)\,
            \Bigl[ q_i\, \partial_\gamma B
                 + \mu_i^\alpha \partial_\alpha \partial_\gamma B
                 + \tfrac{1}{2} Q_i^{\alpha\beta} \partial_\alpha\partial_\beta\partial_\gamma B \Bigr].

    Register-resident accumulators:

    - LMAX = 0: ``acc_q`` (1 scalar) + ``acc_f_frac`` (3 scalars) for
      position gradient via field gather.
    - LMAX = 1: + ``acc_H_frac`` (6 sym entries) for the
      ``μ·∇∇B`` position gradient.
    - LMAX = 2: + rank-3 fractional moments (10 sym entries) for the
      ``Q:∇³B`` position gradient.

    All Cartesian transforms (``M^T``, ``M Q M^T``, etc.) are hoisted
    out of the stencil walk.
    """
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_lmax_module(
            "spread_backward", ORDER, LMAX, scalar_dtype
        )
    )
    def kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        quadrupoles: wp.array(dtype=Any),
        cell_inv_t: wp.array(dtype=Any),
        grad_mesh: wp.array3d(dtype=Any),
        grad_positions: wp.array(dtype=Any),
        grad_charges: wp.array(dtype=Any),
        grad_dipoles: wp.array(dtype=Any),
        grad_quadrupoles: wp.array(dtype=Any),
        grad_cell_inv_t: wp.array2d(dtype=Any),
    ):
        atom_idx = wp.tid()
        mesh_dims = wp.vec3i(grad_mesh.shape[0], grad_mesh.shape[1], grad_mesh.shape[2])
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

        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        dwx = vec_ord()
        dwy = vec_ord()
        dwz = vec_ord()
        if LMAX >= 1:
            d2wx = vec_ord()
            d2wy = vec_ord()
            d2wz = vec_ord()
        if LMAX >= 2:
            d3wx = vec_ord()
            d3wy = vec_ord()
            d3wz = vec_ord()
        md_x = type(t0)(mesh_dims[0])
        md_y = type(t0)(mesh_dims[1])
        md_z = type(t0)(mesh_dims[2])
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            dwx[k] = bspline_derivative(u_x, ORDER) * md_x
            dwy[k] = bspline_derivative(u_y, ORDER) * md_y
            dwz[k] = bspline_derivative(u_z, ORDER) * md_z
            if LMAX >= 1:
                d2wx[k] = bspline_second_derivative(u_x, ORDER) * md_x * md_x
                d2wy[k] = bspline_second_derivative(u_y, ORDER) * md_y * md_y
                d2wz[k] = bspline_second_derivative(u_z, ORDER) * md_z * md_z
            if LMAX >= 2:
                d3wx[k] = bspline_third_derivative(u_x, ORDER) * md_x * md_x * md_x
                d3wy[k] = bspline_third_derivative(u_y, ORDER) * md_y * md_y * md_y
                d3wz[k] = bspline_third_derivative(u_z, ORDER) * md_z * md_z * md_z

        # Register accumulators.
        acc_q = zero
        acc_fx = zero
        acc_fy = zero
        acc_fz = zero
        if LMAX >= 1:
            acc_hxx = zero
            acc_hyy = zero
            acc_hzz = zero
            acc_hxy = zero
            acc_hxz = zero
            acc_hyz = zero
        if LMAX >= 2:
            # Rank-3 symmetric fractional moments (10 unique entries):
            # xxx, yyy, zzz, xxy, xxz, yyx, yyz, zzx, zzy, xyz.
            acc_3_xxx = zero
            acc_3_yyy = zero
            acc_3_zzz = zero
            acc_3_xxy = zero
            acc_3_xxz = zero
            acc_3_yyx = zero
            acc_3_yyz = zero
            acc_3_zzx = zero
            acc_3_zzy = zero
            acc_3_xyz = zero

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            if LMAX >= 1:
                d2wxi = d2wx[i]
            if LMAX >= 2:
                d3wxi = d3wx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wxy = wxi * wy[j]
                dwxy_x = dwxi * wy[j]
                dwxy_y = wxi * dwy[j]
                if LMAX >= 1:
                    d2wxy_xx = d2wxi * wy[j]
                    d2wxy_yy = wxi * d2wy[j]
                    dwxy_xy = dwxi * dwy[j]
                if LMAX >= 2:
                    d3wxy_xxx = d3wxi * wy[j]
                    d3wxy_yyy = wxi * d3wy[j]
                    d2wxy_xx_y = d2wxi * dwy[j]
                    d2wxy_x_yy = dwxi * d2wy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    mesh_val = grad_mesh[gx, gy, gz]
                    wzk = wz[k]
                    dwzk = dwz[k]
                    weight = wxy * wzk

                    acc_q = acc_q + mesh_val * weight
                    acc_fx = acc_fx + mesh_val * dwxy_x * wzk
                    acc_fy = acc_fy + mesh_val * dwxy_y * wzk
                    acc_fz = acc_fz + mesh_val * wxy * dwzk

                    if LMAX >= 1:
                        d2wzk = d2wz[k]
                        acc_hxx = acc_hxx + mesh_val * d2wxy_xx * wzk
                        acc_hyy = acc_hyy + mesh_val * d2wxy_yy * wzk
                        acc_hzz = acc_hzz + mesh_val * wxy * d2wzk
                        acc_hxy = acc_hxy + mesh_val * dwxy_xy * wzk
                        acc_hxz = acc_hxz + mesh_val * dwxy_x * dwzk
                        acc_hyz = acc_hyz + mesh_val * dwxy_y * dwzk

                    if LMAX >= 2:
                        d3wzk = d3wz[k]
                        # Rank-3 fractional moments:
                        # xxx: d3wx · wy · wz
                        acc_3_xxx = acc_3_xxx + mesh_val * d3wxy_xxx * wzk
                        # yyy: wx · d3wy · wz
                        acc_3_yyy = acc_3_yyy + mesh_val * d3wxy_yyy * wzk
                        # zzz: wx · wy · d3wz
                        acc_3_zzz = acc_3_zzz + mesh_val * wxy * d3wzk
                        # xxy: d2wx · dwy · wz
                        acc_3_xxy = acc_3_xxy + mesh_val * d2wxy_xx_y * wzk
                        # xxz: d2wx · wy · dwz
                        acc_3_xxz = acc_3_xxz + mesh_val * d2wxy_xx * dwzk
                        # yyx (= xyy): dwx · d2wy · wz
                        acc_3_yyx = acc_3_yyx + mesh_val * d2wxy_x_yy * wzk
                        # yyz: wx · d2wy · dwz
                        acc_3_yyz = acc_3_yyz + mesh_val * d2wxy_yy * dwzk
                        # zzx (= xzz): dwx · wy · d2wz
                        acc_3_zzx = acc_3_zzx + mesh_val * dwxy_x * d2wzk
                        # zzy (= yzz): wx · dwy · d2wz
                        acc_3_zzy = acc_3_zzy + mesh_val * dwxy_y * d2wzk
                        # xyz: dwx · dwy · dwz
                        acc_3_xyz = acc_3_xyz + mesh_val * dwxy_xy * dwzk

        # Build outputs.
        # ∂L/∂q_i
        grad_charges[atom_idx] = acc_q

        # ∂L/∂μ_cart = M^T · field_frac (where field_frac = acc_f).
        # ∂L/∂position from charge: q * ∂L/∂μ_cart (linear chain).
        M = cell_inv_t[0]
        gmu_cart_x = M[0, 0] * acc_fx + M[1, 0] * acc_fy + M[2, 0] * acc_fz
        gmu_cart_y = M[0, 1] * acc_fx + M[1, 1] * acc_fy + M[2, 1] * acc_fz
        gmu_cart_z = M[0, 2] * acc_fx + M[1, 2] * acc_fy + M[2, 2] * acc_fz

        # Position gradient pieces:
        # From q channel: charge * grad_mu_cart
        gpos_x = charge * gmu_cart_x
        gpos_y = charge * gmu_cart_y
        gpos_z = charge * gmu_cart_z

        if LMAX >= 1:
            grad_dipoles[atom_idx] = vec_pos_dtype(gmu_cart_x, gmu_cart_y, gmu_cart_z)

            # From μ channel: M^T · (H_frac · M · μ)
            dipole = dipoles[atom_idx]
            mfx = M[0, 0] * dipole[0] + M[0, 1] * dipole[1] + M[0, 2] * dipole[2]
            mfy = M[1, 0] * dipole[0] + M[1, 1] * dipole[1] + M[1, 2] * dipole[2]
            mfz = M[2, 0] * dipole[0] + M[2, 1] * dipole[1] + M[2, 2] * dipole[2]
            hmu_x = acc_hxx * mfx + acc_hxy * mfy + acc_hxz * mfz
            hmu_y = acc_hxy * mfx + acc_hyy * mfy + acc_hyz * mfz
            hmu_z = acc_hxz * mfx + acc_hyz * mfy + acc_hzz * mfz
            gpos_x = gpos_x + (M[0, 0] * hmu_x + M[1, 0] * hmu_y + M[2, 0] * hmu_z)
            gpos_y = gpos_y + (M[0, 1] * hmu_x + M[1, 1] * hmu_y + M[2, 1] * hmu_z)
            gpos_z = gpos_z + (M[0, 2] * hmu_x + M[1, 2] * hmu_y + M[2, 2] * hmu_z)

        if LMAX >= 2:
            # ∂L/∂Q_i^{αβ} = (1/2) · (M^T H_frac M)_αβ
            # = (1/2) · grad_Q_cart_αβ ; build the 6 unique entries.
            # H_cart = M^T · H_frac · M (using the gather_hessian pattern).
            h_xx = acc_hxx
            h_yy = acc_hyy
            h_zz = acc_hzz
            h_xy = acc_hxy
            h_xz = acc_hxz
            h_yz = acc_hyz
            t00 = h_xx * M[0, 0] + h_xy * M[1, 0] + h_xz * M[2, 0]
            t01 = h_xx * M[0, 1] + h_xy * M[1, 1] + h_xz * M[2, 1]
            t02 = h_xx * M[0, 2] + h_xy * M[1, 2] + h_xz * M[2, 2]
            t10 = h_xy * M[0, 0] + h_yy * M[1, 0] + h_yz * M[2, 0]
            t11 = h_xy * M[0, 1] + h_yy * M[1, 1] + h_yz * M[2, 1]
            t12 = h_xy * M[0, 2] + h_yy * M[1, 2] + h_yz * M[2, 2]
            t20 = h_xz * M[0, 0] + h_yz * M[1, 0] + h_zz * M[2, 0]
            t21 = h_xz * M[0, 1] + h_yz * M[1, 1] + h_zz * M[2, 1]
            t22 = h_xz * M[0, 2] + h_yz * M[1, 2] + h_zz * M[2, 2]
            half = type(charge)(0.5)
            hxx_c = half * (M[0, 0] * t00 + M[1, 0] * t10 + M[2, 0] * t20)
            hyy_c = half * (M[0, 1] * t01 + M[1, 1] * t11 + M[2, 1] * t21)
            hzz_c = half * (M[0, 2] * t02 + M[1, 2] * t12 + M[2, 2] * t22)
            hxy_c = half * (M[0, 0] * t01 + M[1, 0] * t11 + M[2, 0] * t21)
            hxz_c = half * (M[0, 0] * t02 + M[1, 0] * t12 + M[2, 0] * t22)
            hyz_c = half * (M[0, 1] * t02 + M[1, 1] * t12 + M[2, 1] * t22)
            # ``grad_quadrupoles[atom_idx]`` is mat33 (symmetric).
            # Fill diagonal + symmetric off-diagonals.
            grad_quadrupoles[atom_idx] = mat33_dtype(
                hxx_c,
                hxy_c,
                hxz_c,
                hxy_c,
                hyy_c,
                hyz_c,
                hxz_c,
                hyz_c,
                hzz_c,
            )

            # From Q channel: position gradient piece.
            # F_α (Q part) = (1/2) Σ_g grad_rho(g) · Q^{βγ} · ∂_α∂_β∂_γ B
            # ∂³_cart B[α,β,γ] = M[i,α] M[j,β] M[k,γ] · ∂³_frac B[i,j,k]
            # So per-atom contraction:
            #   acc_T3[i,j,k] (frac) → contract with Q (cart) via M
            # Equivalently: (1/2) Σ_ijk M[i,α] (M Q M^T)_ij · ... — getting
            # messy. Cleanest: form the rank-3 symmetric Cartesian tensor
            # T3_cart_αβγ = M[i,α] M[j,β] M[k,γ] · acc_T3_ijk
            # via three sequential matrix-tensor contractions, then
            # contract with Q^{αβ} on (α,β) pair.
            #
            # The 10 unique entries of acc_3 store the symmetric rank-3:
            #   acc_3[i,j,k] for sorted (i,j,k) triples.
            #
            # We unroll: form T3_cart[α,β,γ] = M[i,α] M[j,β] M[k,γ] · acc_3[i,j,k]
            # for all (α,β,γ), then F_γ_Q = (1/2) Q^{αβ} T3_cart[α,β,γ].
            Q = quadrupoles[atom_idx]
            # First contraction: T31[α,j,k] = M[i,α] · acc_3[i,j,k]
            # Expand by sorting indices. Symmetric so we have 10 entries.
            # Pad to full 3x3x3 array for clarity (Warp inlines this).
            # acc_3 entries: xxx, yyy, zzz, xxy, xxz, yyx, yyz, zzx, zzy, xyz
            # Build the full symmetric tensor acc_3_full[i,j,k] = acc_3
            # indexed by sorted (i, j, k).
            # Compute T3 contraction: F_γ_Q_part = (1/2) Q^{αβ} M[i,α] M[j,β] M[k,γ] acc_3[i,j,k]
            # = (1/2) (M^T Q M)_ij · M[k,γ] · acc_3[i,j,k] ?
            # Wait — that's NOT right because Q is in CART indices, M maps cart→frac.
            # Let me redo: ∂_α B = M^T_α_i · ∂_i_frac B → ∂_α B = M[i,α] ∂_i_frac B
            # So ∂_α∂_β∂_γ B_cart = M[i,α] M[j,β] M[k,γ] · ∂_i∂_j∂_k B_frac
            # Then (1/2) Σ_αβ Q^{αβ} ∂_α∂_β∂_γ B_cart
            # = (1/2) Σ_αβ Q^{αβ} M[i,α] M[j,β] M[k,γ] · acc_3[i,j,k]
            # = (1/2) (M Q M^T)_ij · M[k,γ] · acc_3[i,j,k]
            # (since Σ_αβ Q^{αβ} M[i,α] M[j,β] = (M Q M^T)_ij)
            # We already computed M Q M^T = Qe in the forward kernel; here
            # we need it again. Recompute.
            MQ_00 = M[0, 0] * Q[0, 0] + M[0, 1] * Q[1, 0] + M[0, 2] * Q[2, 0]
            MQ_01 = M[0, 0] * Q[0, 1] + M[0, 1] * Q[1, 1] + M[0, 2] * Q[2, 1]
            MQ_02 = M[0, 0] * Q[0, 2] + M[0, 1] * Q[1, 2] + M[0, 2] * Q[2, 2]
            MQ_10 = M[1, 0] * Q[0, 0] + M[1, 1] * Q[1, 0] + M[1, 2] * Q[2, 0]
            MQ_11 = M[1, 0] * Q[0, 1] + M[1, 1] * Q[1, 1] + M[1, 2] * Q[2, 1]
            MQ_12 = M[1, 0] * Q[0, 2] + M[1, 1] * Q[1, 2] + M[1, 2] * Q[2, 2]
            MQ_20 = M[2, 0] * Q[0, 0] + M[2, 1] * Q[1, 0] + M[2, 2] * Q[2, 0]
            MQ_21 = M[2, 0] * Q[0, 1] + M[2, 1] * Q[1, 1] + M[2, 2] * Q[2, 1]
            MQ_22 = M[2, 0] * Q[0, 2] + M[2, 1] * Q[1, 2] + M[2, 2] * Q[2, 2]
            Qe_00 = MQ_00 * M[0, 0] + MQ_01 * M[0, 1] + MQ_02 * M[0, 2]
            Qe_11 = MQ_10 * M[1, 0] + MQ_11 * M[1, 1] + MQ_12 * M[1, 2]
            Qe_22 = MQ_20 * M[2, 0] + MQ_21 * M[2, 1] + MQ_22 * M[2, 2]
            Qe_01 = MQ_00 * M[1, 0] + MQ_01 * M[1, 1] + MQ_02 * M[1, 2]
            Qe_02 = MQ_00 * M[2, 0] + MQ_01 * M[2, 1] + MQ_02 * M[2, 2]
            Qe_12 = MQ_10 * M[2, 0] + MQ_11 * M[2, 1] + MQ_12 * M[2, 2]
            # Now: F_γ_Q_frac = (1/2) Σ_ij Qe_ij · acc_3[i,j,k_γ_index]
            # where k indexes the FRACTIONAL Cartesian dim. After this
            # contraction, F_k_frac[k] = (1/2) Σ_ij Qe_ij · acc_3[i,j,k]
            # for k = 0,1,2 (x,y,z fractional). Then transform to cart:
            #   F_γ_Q_cart = M[k,γ] · F_k_frac.
            # acc_3 symmetric: ij → use Qe_ij + Qe_ji = 2 Qe_ij for i\!=j.
            # acc_3[i,j,k] = acc_3 indexed sorted(i,j,k).
            #
            # F_frac[k=x] (third index = 0): Σ_ij Qe_ij · acc_3[i,j,0]
            #   acc_3[0,0,0]=xxx, acc_3[0,1,0]=xxy/2-wait — for sorted convention:
            #
            # Better: directly enumerate all 27 (i,j,k) ordered triples,
            # mapped to the 10 sorted entries via permutation count.
            # acc_3[i,j,k] = acc_3_sorted_unique[π(i,j,k)] for any permutation
            # of (i,j,k) → sorted (a,b,c).
            #
            # The mapping (sorted triple → unique entry):
            #   (0,0,0)=xxx, (1,1,1)=yyy, (2,2,2)=zzz,
            #   (0,0,1)=xxy, (0,0,2)=xxz, (0,1,1)=yyx,
            #   (1,1,2)=yyz, (0,2,2)=zzx, (1,2,2)=zzy,
            #   (0,1,2)=xyz.
            #
            # For each (i,j,k) ordered, sort → look up entry.
            # F_frac[k_dim] = Σ_{i,j} Qe[i,j] · acc_3[i,j,k_dim]
            # (note: Qe is symmetric.)
            # Two off-diag contributions Qe[i,j] + Qe[j,i] = 2 Qe[i,j].
            # We use only Qe upper-triangular and multiply by 2 for i\!=j.
            #
            # Final F_γ_Q (cart) = M[k,γ] · F_k_frac.

            two = type(charge)(2.0)

            # F_frac[0] = Qe_00·acc[0,0,0] + Qe_11·acc[1,1,0] + Qe_22·acc[2,2,0]
            #           + 2·Qe_01·acc[0,1,0] + 2·Qe_02·acc[0,2,0] + 2·Qe_12·acc[1,2,0]
            # acc[1,1,0] = acc sorted (0,1,1) = acc_3_yyx
            # acc[2,2,0] = acc sorted (0,2,2) = acc_3_zzx
            # acc[0,1,0] = acc sorted (0,0,1) = acc_3_xxy
            # acc[0,2,0] = acc sorted (0,0,2) = acc_3_xxz
            # acc[1,2,0] = acc sorted (0,1,2) = acc_3_xyz
            F_frac_0 = (
                Qe_00 * acc_3_xxx
                + Qe_11 * acc_3_yyx
                + Qe_22 * acc_3_zzx
                + two * (Qe_01 * acc_3_xxy + Qe_02 * acc_3_xxz + Qe_12 * acc_3_xyz)
            )
            # F_frac[1]:
            # acc[0,0,1] = sorted (0,0,1) = xxy
            # acc[1,1,1] = yyy
            # acc[2,2,1] = sorted (1,2,2) = zzy
            # acc[0,1,1] = sorted (0,1,1) = yyx
            # acc[0,2,1] = sorted (0,1,2) = xyz
            # acc[1,2,1] = sorted (1,1,2) = yyz
            F_frac_1 = (
                Qe_00 * acc_3_xxy
                + Qe_11 * acc_3_yyy
                + Qe_22 * acc_3_zzy
                + two * (Qe_01 * acc_3_yyx + Qe_02 * acc_3_xyz + Qe_12 * acc_3_yyz)
            )
            # F_frac[2]:
            # acc[0,0,2] = xxz
            # acc[1,1,2] = yyz
            # acc[2,2,2] = zzz
            # acc[0,1,2] = xyz
            # acc[0,2,2] = sorted (0,2,2) = zzx
            # acc[1,2,2] = sorted (1,2,2) = zzy
            F_frac_2 = (
                Qe_00 * acc_3_xxz
                + Qe_11 * acc_3_yyz
                + Qe_22 * acc_3_zzz
                + two * (Qe_01 * acc_3_xyz + Qe_02 * acc_3_zzx + Qe_12 * acc_3_zzy)
            )
            # F_cart_γ = (1/2) · M[γ_frac, γ_cart] · F_frac[γ_frac]
            # = (1/2) · (M[0,γ_cart]·F_frac_0 + M[1,γ_cart]·F_frac_1 + M[2,γ_cart]·F_frac_2)
            gpos_x = gpos_x + half * (
                M[0, 0] * F_frac_0 + M[1, 0] * F_frac_1 + M[2, 0] * F_frac_2
            )
            gpos_y = gpos_y + half * (
                M[0, 1] * F_frac_0 + M[1, 1] * F_frac_1 + M[2, 1] * F_frac_2
            )
            gpos_z = gpos_z + half * (
                M[0, 2] * F_frac_0 + M[1, 2] * F_frac_1 + M[2, 2] * F_frac_2
            )

        grad_positions[atom_idx] = vec_pos_dtype(gpos_x, gpos_y, gpos_z)

        # =====================================================
        # Cell gradient: ∂L/∂M[c, d] where M = cell_inv_t.
        # =====================================================
        #
        # Per atom, ``mesh(g)`` depends on M through three paths:
        #   (a) ``theta = M @ r`` (all channels — charge B, dipole ∂B,
        #       Q ∂²B all evaluated at ``theta``).
        #   (b) ``μ_frac = M @ μ_cart`` (dipole channel only).
        #   (c) ``Qe = M Q M^T`` (Q channel only).
        #
        # Derivation: contributions are
        #
        #   (a) charge: ``q · acc_f_c · r_d``
        #              dipole-via-theta: ``Σ_a (M μ)_a · acc_h_{ac} · r_d``
        #                              = ``hmu_c · r_d``  (already computed)
        #              Q-via-theta: ``(1/2) F_frac_c · r_d``  (already
        #              computed for grad_positions)
        #   (b) dipole-via-μfrac: ``μ_cart_d · acc_f_c``
        #   (c) Q-via-Qe: ``(acc_h @ M Q)_{cd}``
        #
        # Combine (a) into ``grad_theta_c · r_d`` where
        # ``grad_theta_c = q · acc_f_c + hmu_c + (1/2) F_frac_c`` —
        # which is exactly the *frac-frame* position-gradient. The cart
        # ``grad_positions`` already-computed equals ``M^T @ grad_theta``.
        #
        # 9 ``atomic_add``s per atom — negligible compared to the
        # ORDER^3 stencil walk above.
        r_d_x = position[0]
        r_d_y = position[1]
        r_d_z = position[2]

        # ``grad_theta[c] = q acc_f_c + hmu_c (LMAX≥1) + (1/2) F_frac_c (LMAX≥2)``
        gtheta_x = charge * acc_fx
        gtheta_y = charge * acc_fy
        gtheta_z = charge * acc_fz
        if LMAX >= 1:
            gtheta_x = gtheta_x + hmu_x
            gtheta_y = gtheta_y + hmu_y
            gtheta_z = gtheta_z + hmu_z
        if LMAX >= 2:
            half_cell = type(charge)(0.5)
            gtheta_x = gtheta_x + half_cell * F_frac_0
            gtheta_y = gtheta_y + half_cell * F_frac_1
            gtheta_z = gtheta_z + half_cell * F_frac_2

        # Path (a): ``grad_theta_c · r_d``.
        wp.atomic_add(grad_cell_inv_t, 0, 0, gtheta_x * r_d_x)
        wp.atomic_add(grad_cell_inv_t, 0, 1, gtheta_x * r_d_y)
        wp.atomic_add(grad_cell_inv_t, 0, 2, gtheta_x * r_d_z)
        wp.atomic_add(grad_cell_inv_t, 1, 0, gtheta_y * r_d_x)
        wp.atomic_add(grad_cell_inv_t, 1, 1, gtheta_y * r_d_y)
        wp.atomic_add(grad_cell_inv_t, 1, 2, gtheta_y * r_d_z)
        wp.atomic_add(grad_cell_inv_t, 2, 0, gtheta_z * r_d_x)
        wp.atomic_add(grad_cell_inv_t, 2, 1, gtheta_z * r_d_y)
        wp.atomic_add(grad_cell_inv_t, 2, 2, gtheta_z * r_d_z)

        if LMAX >= 1:
            # Path (b): ``μ_cart_d · acc_f_c``.
            mu_d_x = dipole[0]
            mu_d_y = dipole[1]
            mu_d_z = dipole[2]
            wp.atomic_add(grad_cell_inv_t, 0, 0, mu_d_x * acc_fx)
            wp.atomic_add(grad_cell_inv_t, 0, 1, mu_d_y * acc_fx)
            wp.atomic_add(grad_cell_inv_t, 0, 2, mu_d_z * acc_fx)
            wp.atomic_add(grad_cell_inv_t, 1, 0, mu_d_x * acc_fy)
            wp.atomic_add(grad_cell_inv_t, 1, 1, mu_d_y * acc_fy)
            wp.atomic_add(grad_cell_inv_t, 1, 2, mu_d_z * acc_fy)
            wp.atomic_add(grad_cell_inv_t, 2, 0, mu_d_x * acc_fz)
            wp.atomic_add(grad_cell_inv_t, 2, 1, mu_d_y * acc_fz)
            wp.atomic_add(grad_cell_inv_t, 2, 2, mu_d_z * acc_fz)

        if LMAX >= 2:
            # Path (c): ``(acc_h @ M Q)[c, d]``.
            # ``acc_h`` is symmetric: rows (acc_hxx, acc_hxy, acc_hxz),
            #                        (acc_hxy, acc_hyy, acc_hyz),
            #                        (acc_hxz, acc_hyz, acc_hzz).
            # Recompute ``MQ = M @ Q`` here — Warp's local-scope rules
            # mean the earlier ``MQ_*`` definitions inside the
            # position-gradient ``if LMAX >= 2:`` block aren't reliably
            # visible across distinct ``if`` branches.
            Q_cell = quadrupoles[atom_idx]
            MQc_00 = (
                M[0, 0] * Q_cell[0, 0] + M[0, 1] * Q_cell[1, 0] + M[0, 2] * Q_cell[2, 0]
            )
            MQc_01 = (
                M[0, 0] * Q_cell[0, 1] + M[0, 1] * Q_cell[1, 1] + M[0, 2] * Q_cell[2, 1]
            )
            MQc_02 = (
                M[0, 0] * Q_cell[0, 2] + M[0, 1] * Q_cell[1, 2] + M[0, 2] * Q_cell[2, 2]
            )
            MQc_10 = (
                M[1, 0] * Q_cell[0, 0] + M[1, 1] * Q_cell[1, 0] + M[1, 2] * Q_cell[2, 0]
            )
            MQc_11 = (
                M[1, 0] * Q_cell[0, 1] + M[1, 1] * Q_cell[1, 1] + M[1, 2] * Q_cell[2, 1]
            )
            MQc_12 = (
                M[1, 0] * Q_cell[0, 2] + M[1, 1] * Q_cell[1, 2] + M[1, 2] * Q_cell[2, 2]
            )
            MQc_20 = (
                M[2, 0] * Q_cell[0, 0] + M[2, 1] * Q_cell[1, 0] + M[2, 2] * Q_cell[2, 0]
            )
            MQc_21 = (
                M[2, 0] * Q_cell[0, 1] + M[2, 1] * Q_cell[1, 1] + M[2, 2] * Q_cell[2, 1]
            )
            MQc_22 = (
                M[2, 0] * Q_cell[0, 2] + M[2, 1] * Q_cell[1, 2] + M[2, 2] * Q_cell[2, 2]
            )
            ahmq_00 = acc_hxx * MQc_00 + acc_hxy * MQc_10 + acc_hxz * MQc_20
            ahmq_01 = acc_hxx * MQc_01 + acc_hxy * MQc_11 + acc_hxz * MQc_21
            ahmq_02 = acc_hxx * MQc_02 + acc_hxy * MQc_12 + acc_hxz * MQc_22
            ahmq_10 = acc_hxy * MQc_00 + acc_hyy * MQc_10 + acc_hyz * MQc_20
            ahmq_11 = acc_hxy * MQc_01 + acc_hyy * MQc_11 + acc_hyz * MQc_21
            ahmq_12 = acc_hxy * MQc_02 + acc_hyy * MQc_12 + acc_hyz * MQc_22
            ahmq_20 = acc_hxz * MQc_00 + acc_hyz * MQc_10 + acc_hzz * MQc_20
            ahmq_21 = acc_hxz * MQc_01 + acc_hyz * MQc_11 + acc_hzz * MQc_21
            ahmq_22 = acc_hxz * MQc_02 + acc_hyz * MQc_12 + acc_hzz * MQc_22
            wp.atomic_add(grad_cell_inv_t, 0, 0, ahmq_00)
            wp.atomic_add(grad_cell_inv_t, 0, 1, ahmq_01)
            wp.atomic_add(grad_cell_inv_t, 0, 2, ahmq_02)
            wp.atomic_add(grad_cell_inv_t, 1, 0, ahmq_10)
            wp.atomic_add(grad_cell_inv_t, 1, 1, ahmq_11)
            wp.atomic_add(grad_cell_inv_t, 1, 2, ahmq_12)
            wp.atomic_add(grad_cell_inv_t, 2, 0, ahmq_20)
            wp.atomic_add(grad_cell_inv_t, 2, 1, ahmq_21)
            wp.atomic_add(grad_cell_inv_t, 2, 2, ahmq_22)

    return kernel


_PER_ORDER_LMAX_SPREAD_BACKWARD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _lmax in (0, 1, 2):
        for _scalar, _vec, _mat in (
            (wp.float32, wp.vec3f, wp.mat33f),
            (wp.float64, wp.vec3d, wp.mat33d),
        ):
            _k = _make_pme_multipole_spread_backward_unified_kernel(
                _order, _lmax, _scalar, _vec, _mat
            )
            _PER_ORDER_LMAX_SPREAD_BACKWARD_KERNELS[_scalar][(_order, _lmax)] = (
                wp.overload(
                    _k,
                    [
                        wp.array(dtype=_vec),  # positions
                        wp.array(dtype=_scalar),  # charges
                        wp.array(dtype=_vec),  # dipoles
                        wp.array(dtype=_mat),  # quadrupoles
                        wp.array(dtype=_mat),  # cell_inv_t
                        wp.array3d(dtype=_scalar),  # grad_mesh
                        wp.array(dtype=_vec),  # grad_positions (out)
                        wp.array(dtype=_scalar),  # grad_charges (out)
                        wp.array(dtype=_vec),  # grad_dipoles (out)
                        wp.array(dtype=_mat),  # grad_quadrupoles (out)
                        wp.array2d(dtype=_scalar),  # grad_cell_inv_t (3, 3) (out)
                    ],
                )
            )


def _maybe_per_order_lmax_spread_backward_kernel(order: int, lmax: int, wp_dtype):
    """Return the per-(ORDER, LMAX, dtype) unified spread-backward kernel."""
    return _PER_ORDER_LMAX_SPREAD_BACKWARD_KERNELS.get(wp_dtype, {}).get((order, lmax))


def multipole_pme_spread_backward_unified_launch(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    quadrupoles: wp.array,
    cell_inv_t: wp.array,
    order: int,
    lmax: int,
    grad_mesh: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_dipoles: wp.array,
    grad_quadrupoles: wp.array,
    grad_cell_inv_t: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> bool:
    r"""Launch the unified ``(ORDER, LMAX)`` spread-backward kernel.

    Backward of :func:`multipole_pme_spread_unified_launch`.
    ``grad_cell_inv_t`` is a ``(3, 3)`` 2D Warp array that receives the
    atomic-accumulated ``∂L/∂M`` (M = ``cell_inv_t[0]``). The kernel
    expects the buffer to be zero-initialized — the launcher does not
    clear it for the caller.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=vec3f/vec3d
        Cartesian atom positions.
    charges : wp.array, shape (N,), dtype=wp.float32/float64
        Per-atom monopole charges.
    dipoles : wp.array, shape (N,), dtype=vec3f/vec3d
        Per-atom Cartesian dipoles (used when ``lmax >= 1``).
    quadrupoles : wp.array, shape (N,), dtype=mat33f/mat33d
        Per-atom Cartesian quadrupoles (used when ``lmax >= 2``).
    cell_inv_t : wp.array, shape (1,), dtype=mat33f/mat33d
        Transpose of the inverse cell matrix.
    order : int
        B-spline order (one of ``(3, 4, 5, 6)``).
    lmax : int
        Maximum multipole order (0, 1, or 2).
    grad_mesh : wp.array, shape (Nx, Ny, Nz), dtype=wp.float32/float64
        Upstream gradient w.r.t. the spread mesh.
    grad_positions : wp.array, shape (N,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Gradient w.r.t. positions.
    grad_charges : wp.array, shape (N,), dtype=wp.float32/float64
        OUTPUT, pre-zeroed. Gradient w.r.t. charges.
    grad_dipoles : wp.array, shape (N,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Gradient w.r.t. dipoles.
    grad_quadrupoles : wp.array, shape (N,), dtype=mat33f/mat33d
        OUTPUT, pre-zeroed. Gradient w.r.t. quadrupoles.
    grad_cell_inv_t : wp.array, shape (3, 3), dtype=wp.float32/float64
        OUTPUT, pre-zeroed. Gradient w.r.t. ``cell_inv_t`` (atomic).
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Returns
    -------
    bool
        ``True`` if the per-order overload ran, ``False`` on cache miss.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    per_order = _maybe_per_order_lmax_spread_backward_kernel(order, lmax, wp_dtype)
    if per_order is None:
        return False
    if device is None:
        device = str(positions.device)
    num_atoms = positions.shape[0]
    wp.launch(
        per_order,
        dim=(num_atoms,),
        inputs=[
            positions,
            charges,
            dipoles,
            quadrupoles,
            cell_inv_t,
            grad_mesh,
        ],
        outputs=[
            grad_positions,
            grad_charges,
            grad_dipoles,
            grad_quadrupoles,
            grad_cell_inv_t,
        ],
        device=device,
    )
    return True


# =============================================================================
# Batched unified spread (forward + backward) — l_max 0/1/2
# =============================================================================
#
# Three batch substitutions:
#   * ``sys_idx = batch_idx[atom_idx]`` selects the per-system grid slice;
#   * ``cell_inv_t[sys_idx]`` (per-system 3x3) replaces ``cell_inv_t[0]``;
#   * the mesh / grad_mesh / grad_cell_inv_t arrays gain a leading batch axis.


def _make_batch_pme_multipole_spread_unified_kernel(
    ORDER: int,
    LMAX: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    r"""Batched per-(ORDER, LMAX, dtype) unified spread kernel.

    Batched analog of :func:`_make_pme_multipole_spread_unified_kernel`.
    1 thread per atom; ``batch_idx[atom_idx]`` selects the system.
    """
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_lmax_module(
            "batch_spread", ORDER, LMAX, scalar_dtype
        )
    )
    def kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        quadrupoles: wp.array(dtype=Any),
        batch_idx: wp.array(dtype=wp.int32),
        cell_inv_t: wp.array(dtype=Any),
        mesh: wp.array(dtype=Any, ndim=4),
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

        offset_start_x = wp.int32(wp.floor(theta[0] - half_n_minus_2))
        offset_start_y = wp.int32(wp.floor(theta[1] - half_n_minus_2))
        offset_start_z = wp.int32(wp.floor(theta[2] - half_n_minus_2))

        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        if LMAX >= 1:
            dwx = vec_ord()
            dwy = vec_ord()
            dwz = vec_ord()
        if LMAX >= 2:
            d2wx = vec_ord()
            d2wy = vec_ord()
            d2wz = vec_ord()

        md_x = type(t0)(mesh_dims[0])
        md_y = type(t0)(mesh_dims[1])
        md_z = type(t0)(mesh_dims[2])
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            if LMAX >= 1:
                dwx[k] = bspline_derivative(u_x, ORDER) * md_x
                dwy[k] = bspline_derivative(u_y, ORDER) * md_y
                dwz[k] = bspline_derivative(u_z, ORDER) * md_z
            if LMAX >= 2:
                d2wx[k] = bspline_second_derivative(u_x, ORDER) * md_x * md_x
                d2wy[k] = bspline_second_derivative(u_y, ORDER) * md_y * md_y
                d2wz[k] = bspline_second_derivative(u_z, ORDER) * md_z * md_z

        # Hoist per-atom-constant matrix products.
        if LMAX >= 1:
            cell_inv_T = wp.transpose(cell_inv_t[sys_idx])
            dipole = dipoles[atom_idx]
        if LMAX >= 2:
            Q = quadrupoles[atom_idx]
            M = cell_inv_t[sys_idx]
            MQ_00 = M[0, 0] * Q[0, 0] + M[0, 1] * Q[1, 0] + M[0, 2] * Q[2, 0]
            MQ_01 = M[0, 0] * Q[0, 1] + M[0, 1] * Q[1, 1] + M[0, 2] * Q[2, 1]
            MQ_02 = M[0, 0] * Q[0, 2] + M[0, 1] * Q[1, 2] + M[0, 2] * Q[2, 2]
            MQ_10 = M[1, 0] * Q[0, 0] + M[1, 1] * Q[1, 0] + M[1, 2] * Q[2, 0]
            MQ_11 = M[1, 0] * Q[0, 1] + M[1, 1] * Q[1, 1] + M[1, 2] * Q[2, 1]
            MQ_12 = M[1, 0] * Q[0, 2] + M[1, 1] * Q[1, 2] + M[1, 2] * Q[2, 2]
            MQ_20 = M[2, 0] * Q[0, 0] + M[2, 1] * Q[1, 0] + M[2, 2] * Q[2, 0]
            MQ_21 = M[2, 0] * Q[0, 1] + M[2, 1] * Q[1, 1] + M[2, 2] * Q[2, 1]
            MQ_22 = M[2, 0] * Q[0, 2] + M[2, 1] * Q[1, 2] + M[2, 2] * Q[2, 2]
            Qe_00 = MQ_00 * M[0, 0] + MQ_01 * M[0, 1] + MQ_02 * M[0, 2]
            Qe_11 = MQ_10 * M[1, 0] + MQ_11 * M[1, 1] + MQ_12 * M[1, 2]
            Qe_22 = MQ_20 * M[2, 0] + MQ_21 * M[2, 1] + MQ_22 * M[2, 2]
            Qe_01 = MQ_00 * M[1, 0] + MQ_01 * M[1, 1] + MQ_02 * M[1, 2]
            Qe_02 = MQ_00 * M[2, 0] + MQ_01 * M[2, 1] + MQ_02 * M[2, 2]
            Qe_12 = MQ_10 * M[2, 0] + MQ_11 * M[2, 1] + MQ_12 * M[2, 2]

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            if LMAX >= 1:
                dwxi = dwx[i]
            if LMAX >= 2:
                d2wxi = d2wx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wxy = wxi * wy[j]
                if LMAX >= 1:
                    dwxy_x = dwxi * wy[j]
                    dwxy_y = wxi * dwy[j]
                if LMAX >= 2:
                    d2wxy_xx = d2wxi * wy[j]
                    d2wxy_yy = wxi * d2wy[j]
                    dwxy_xy = dwxi * dwy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    wzk = wz[k]
                    weight = wxy * wzk
                    contrib = charge * weight

                    if LMAX >= 1:
                        dwzk = dwz[k]
                        grad_frac_vec = vec_pos_dtype(
                            dwxy_x * wzk,
                            dwxy_y * wzk,
                            wxy * dwzk,
                        )
                        grad_cart = cell_inv_T * grad_frac_vec
                        contrib = contrib + (
                            dipole[0] * grad_cart[0]
                            + dipole[1] * grad_cart[1]
                            + dipole[2] * grad_cart[2]
                        )

                    if LMAX >= 2:
                        d2wzk = d2wz[k]
                        H_xx = d2wxy_xx * wzk
                        H_yy = d2wxy_yy * wzk
                        H_zz = wxy * d2wzk
                        H_xy = dwxy_xy * wzk
                        H_xz = dwxy_x * dwzk
                        H_yz = dwxy_y * dwzk
                        half = type(weight)(0.5)
                        two = type(weight)(2.0)
                        contrib = contrib + half * (
                            Qe_00 * H_xx
                            + Qe_11 * H_yy
                            + Qe_22 * H_zz
                            + two * (Qe_01 * H_xy + Qe_02 * H_xz + Qe_12 * H_yz)
                        )

                    wp.atomic_add(mesh, sys_idx, gx, gy, gz, contrib)

    return kernel


_PER_ORDER_LMAX_BATCH_SPREAD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _lmax in (0, 1, 2):
        for _scalar, _vec, _mat in (
            (wp.float32, wp.vec3f, wp.mat33f),
            (wp.float64, wp.vec3d, wp.mat33d),
        ):
            _k = _make_batch_pme_multipole_spread_unified_kernel(
                _order, _lmax, _scalar, _vec, _mat
            )
            _PER_ORDER_LMAX_BATCH_SPREAD_KERNELS[_scalar][(_order, _lmax)] = (
                wp.overload(
                    _k,
                    [
                        wp.array(dtype=_vec),  # positions
                        wp.array(dtype=_scalar),  # charges
                        wp.array(dtype=_vec),  # dipoles
                        wp.array(dtype=_mat),  # quadrupoles
                        wp.array(dtype=wp.int32),  # batch_idx
                        wp.array(dtype=_mat),  # cell_inv_t (B, 3, 3)
                        wp.array(dtype=_scalar, ndim=4),  # mesh (B, nx, ny, nz)
                    ],
                )
            )


def _maybe_per_order_lmax_batch_spread_kernel(order: int, lmax: int, wp_dtype):
    """Return the per-(ORDER, LMAX, dtype) batched unified spread kernel."""
    return _PER_ORDER_LMAX_BATCH_SPREAD_KERNELS.get(wp_dtype, {}).get((order, lmax))


def batch_multipole_pme_spread_unified_launch(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    quadrupoles: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    lmax: int,
    mesh: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> bool:
    r"""Launch the batched unified ``(ORDER, LMAX)`` spread kernel.

    Batched analog of :func:`multipole_pme_spread_unified_launch`. Caller
    pre-zeros the ``(B, Nx, Ny, Nz)`` mesh.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=vec3f/vec3d
        Concatenated Cartesian atom positions.
    charges : wp.array, shape (N_total,), dtype=wp.float32/float64
        Per-atom monopole charges.
    dipoles : wp.array, shape (N_total,), dtype=vec3f/vec3d
        Per-atom Cartesian dipoles (used when ``lmax >= 1``).
    quadrupoles : wp.array, shape (N_total,), dtype=mat33f/mat33d
        Per-atom Cartesian quadrupoles (used when ``lmax >= 2``).
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        Per-atom system index into the leading mesh axis.
    cell_inv_t : wp.array, shape (B,), dtype=mat33f/mat33d
        Per-system transpose of the inverse cell matrix.
    order : int
        B-spline order (one of ``(3, 4, 5, 6)``).
    lmax : int
        Maximum multipole order (0, 1, or 2).
    mesh : wp.array, shape (B, Nx, Ny, Nz), dtype=wp.float32/float64
        OUTPUT. Per-system charge-density mesh, pre-zeroed.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Returns
    -------
    bool
        ``True`` if the per-order overload ran, ``False`` on cache miss.

    Launch Grid
    -----------
    ``dim = (N_total,)`` — one thread per atom across the batch.
    """
    per_order = _maybe_per_order_lmax_batch_spread_kernel(order, lmax, wp_dtype)
    if per_order is None:
        return False
    if device is None:
        device = str(positions.device)
    num_atoms = positions.shape[0]
    wp.launch(
        per_order,
        dim=(num_atoms,),
        inputs=[positions, charges, dipoles, quadrupoles, batch_idx, cell_inv_t],
        outputs=[mesh],
        device=device,
    )
    return True


def _make_batch_pme_multipole_spread_backward_unified_kernel(
    ORDER: int,
    LMAX: int,
    scalar_dtype,
    vec_pos_dtype,
    mat33_dtype,
):
    r"""Batched per-(ORDER, LMAX, dtype) unified spread-backward kernel.

    Batched analog of
    :func:`_make_pme_multipole_spread_backward_unified_kernel`. Outputs
    per-atom gradients of charges, dipoles (LMAX>=1), quadrupoles (LMAX>=2),
    positions, and per-system ``grad_cell_inv_t`` (B, 3, 3).
    """
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_lmax_module(
            "batch_spread_backward", ORDER, LMAX, scalar_dtype
        )
    )
    def kernel(
        positions: wp.array(dtype=Any),
        charges: wp.array(dtype=Any),
        dipoles: wp.array(dtype=Any),
        quadrupoles: wp.array(dtype=Any),
        batch_idx: wp.array(dtype=wp.int32),
        cell_inv_t: wp.array(dtype=Any),
        grad_mesh: wp.array(dtype=Any, ndim=4),
        grad_positions: wp.array(dtype=Any),
        grad_charges: wp.array(dtype=Any),
        grad_dipoles: wp.array(dtype=Any),
        grad_quadrupoles: wp.array(dtype=Any),
        grad_cell_inv_t: wp.array(dtype=Any, ndim=3),
    ):
        atom_idx = wp.tid()
        sys_idx = batch_idx[atom_idx]
        mesh_dims = wp.vec3i(grad_mesh.shape[1], grad_mesh.shape[2], grad_mesh.shape[3])
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

        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        dwx = vec_ord()
        dwy = vec_ord()
        dwz = vec_ord()
        if LMAX >= 1:
            d2wx = vec_ord()
            d2wy = vec_ord()
            d2wz = vec_ord()
        if LMAX >= 2:
            d3wx = vec_ord()
            d3wy = vec_ord()
            d3wz = vec_ord()
        md_x = type(t0)(mesh_dims[0])
        md_y = type(t0)(mesh_dims[1])
        md_z = type(t0)(mesh_dims[2])
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            dwx[k] = bspline_derivative(u_x, ORDER) * md_x
            dwy[k] = bspline_derivative(u_y, ORDER) * md_y
            dwz[k] = bspline_derivative(u_z, ORDER) * md_z
            if LMAX >= 1:
                d2wx[k] = bspline_second_derivative(u_x, ORDER) * md_x * md_x
                d2wy[k] = bspline_second_derivative(u_y, ORDER) * md_y * md_y
                d2wz[k] = bspline_second_derivative(u_z, ORDER) * md_z * md_z
            if LMAX >= 2:
                d3wx[k] = bspline_third_derivative(u_x, ORDER) * md_x * md_x * md_x
                d3wy[k] = bspline_third_derivative(u_y, ORDER) * md_y * md_y * md_y
                d3wz[k] = bspline_third_derivative(u_z, ORDER) * md_z * md_z * md_z

        acc_q = zero
        acc_fx = zero
        acc_fy = zero
        acc_fz = zero
        if LMAX >= 1:
            acc_hxx = zero
            acc_hyy = zero
            acc_hzz = zero
            acc_hxy = zero
            acc_hxz = zero
            acc_hyz = zero
        if LMAX >= 2:
            acc_3_xxx = zero
            acc_3_yyy = zero
            acc_3_zzz = zero
            acc_3_xxy = zero
            acc_3_xxz = zero
            acc_3_yyx = zero
            acc_3_yyz = zero
            acc_3_zzx = zero
            acc_3_zzy = zero
            acc_3_xyz = zero

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            if LMAX >= 1:
                d2wxi = d2wx[i]
            if LMAX >= 2:
                d3wxi = d3wx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wxy = wxi * wy[j]
                dwxy_x = dwxi * wy[j]
                dwxy_y = wxi * dwy[j]
                if LMAX >= 1:
                    d2wxy_xx = d2wxi * wy[j]
                    d2wxy_yy = wxi * d2wy[j]
                    dwxy_xy = dwxi * dwy[j]
                if LMAX >= 2:
                    d3wxy_xxx = d3wxi * wy[j]
                    d3wxy_yyy = wxi * d3wy[j]
                    d2wxy_xx_y = d2wxi * dwy[j]
                    d2wxy_x_yy = dwxi * d2wy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    mesh_val = grad_mesh[sys_idx, gx, gy, gz]
                    wzk = wz[k]
                    dwzk = dwz[k]
                    weight = wxy * wzk

                    acc_q = acc_q + mesh_val * weight
                    acc_fx = acc_fx + mesh_val * dwxy_x * wzk
                    acc_fy = acc_fy + mesh_val * dwxy_y * wzk
                    acc_fz = acc_fz + mesh_val * wxy * dwzk

                    if LMAX >= 1:
                        d2wzk = d2wz[k]
                        acc_hxx = acc_hxx + mesh_val * d2wxy_xx * wzk
                        acc_hyy = acc_hyy + mesh_val * d2wxy_yy * wzk
                        acc_hzz = acc_hzz + mesh_val * wxy * d2wzk
                        acc_hxy = acc_hxy + mesh_val * dwxy_xy * wzk
                        acc_hxz = acc_hxz + mesh_val * dwxy_x * dwzk
                        acc_hyz = acc_hyz + mesh_val * dwxy_y * dwzk

                    if LMAX >= 2:
                        d3wzk = d3wz[k]
                        acc_3_xxx = acc_3_xxx + mesh_val * d3wxy_xxx * wzk
                        acc_3_yyy = acc_3_yyy + mesh_val * d3wxy_yyy * wzk
                        acc_3_zzz = acc_3_zzz + mesh_val * wxy * d3wzk
                        acc_3_xxy = acc_3_xxy + mesh_val * d2wxy_xx_y * wzk
                        acc_3_xxz = acc_3_xxz + mesh_val * d2wxy_xx * dwzk
                        acc_3_yyx = acc_3_yyx + mesh_val * d2wxy_x_yy * wzk
                        acc_3_yyz = acc_3_yyz + mesh_val * d2wxy_yy * dwzk
                        acc_3_zzx = acc_3_zzx + mesh_val * dwxy_x * d2wzk
                        acc_3_zzy = acc_3_zzy + mesh_val * dwxy_y * d2wzk
                        acc_3_xyz = acc_3_xyz + mesh_val * dwxy_xy * dwzk

        grad_charges[atom_idx] = acc_q

        M = cell_inv_t[sys_idx]
        gmu_cart_x = M[0, 0] * acc_fx + M[1, 0] * acc_fy + M[2, 0] * acc_fz
        gmu_cart_y = M[0, 1] * acc_fx + M[1, 1] * acc_fy + M[2, 1] * acc_fz
        gmu_cart_z = M[0, 2] * acc_fx + M[1, 2] * acc_fy + M[2, 2] * acc_fz

        gpos_x = charge * gmu_cart_x
        gpos_y = charge * gmu_cart_y
        gpos_z = charge * gmu_cart_z

        if LMAX >= 1:
            grad_dipoles[atom_idx] = vec_pos_dtype(gmu_cart_x, gmu_cart_y, gmu_cart_z)
            dipole = dipoles[atom_idx]
            mfx = M[0, 0] * dipole[0] + M[0, 1] * dipole[1] + M[0, 2] * dipole[2]
            mfy = M[1, 0] * dipole[0] + M[1, 1] * dipole[1] + M[1, 2] * dipole[2]
            mfz = M[2, 0] * dipole[0] + M[2, 1] * dipole[1] + M[2, 2] * dipole[2]
            hmu_x = acc_hxx * mfx + acc_hxy * mfy + acc_hxz * mfz
            hmu_y = acc_hxy * mfx + acc_hyy * mfy + acc_hyz * mfz
            hmu_z = acc_hxz * mfx + acc_hyz * mfy + acc_hzz * mfz
            gpos_x = gpos_x + (M[0, 0] * hmu_x + M[1, 0] * hmu_y + M[2, 0] * hmu_z)
            gpos_y = gpos_y + (M[0, 1] * hmu_x + M[1, 1] * hmu_y + M[2, 1] * hmu_z)
            gpos_z = gpos_z + (M[0, 2] * hmu_x + M[1, 2] * hmu_y + M[2, 2] * hmu_z)

        if LMAX >= 2:
            h_xx = acc_hxx
            h_yy = acc_hyy
            h_zz = acc_hzz
            h_xy = acc_hxy
            h_xz = acc_hxz
            h_yz = acc_hyz
            t00 = h_xx * M[0, 0] + h_xy * M[1, 0] + h_xz * M[2, 0]
            t01 = h_xx * M[0, 1] + h_xy * M[1, 1] + h_xz * M[2, 1]
            t02 = h_xx * M[0, 2] + h_xy * M[1, 2] + h_xz * M[2, 2]
            t10 = h_xy * M[0, 0] + h_yy * M[1, 0] + h_yz * M[2, 0]
            t11 = h_xy * M[0, 1] + h_yy * M[1, 1] + h_yz * M[2, 1]
            t12 = h_xy * M[0, 2] + h_yy * M[1, 2] + h_yz * M[2, 2]
            t20 = h_xz * M[0, 0] + h_yz * M[1, 0] + h_zz * M[2, 0]
            t21 = h_xz * M[0, 1] + h_yz * M[1, 1] + h_zz * M[2, 1]
            t22 = h_xz * M[0, 2] + h_yz * M[1, 2] + h_zz * M[2, 2]
            half = type(charge)(0.5)
            hxx_c = half * (M[0, 0] * t00 + M[1, 0] * t10 + M[2, 0] * t20)
            hyy_c = half * (M[0, 1] * t01 + M[1, 1] * t11 + M[2, 1] * t21)
            hzz_c = half * (M[0, 2] * t02 + M[1, 2] * t12 + M[2, 2] * t22)
            hxy_c = half * (M[0, 0] * t01 + M[1, 0] * t11 + M[2, 0] * t21)
            hxz_c = half * (M[0, 0] * t02 + M[1, 0] * t12 + M[2, 0] * t22)
            hyz_c = half * (M[0, 1] * t02 + M[1, 1] * t12 + M[2, 1] * t22)
            grad_quadrupoles[atom_idx] = mat33_dtype(
                hxx_c,
                hxy_c,
                hxz_c,
                hxy_c,
                hyy_c,
                hyz_c,
                hxz_c,
                hyz_c,
                hzz_c,
            )

            Q = quadrupoles[atom_idx]
            MQ_00 = M[0, 0] * Q[0, 0] + M[0, 1] * Q[1, 0] + M[0, 2] * Q[2, 0]
            MQ_01 = M[0, 0] * Q[0, 1] + M[0, 1] * Q[1, 1] + M[0, 2] * Q[2, 1]
            MQ_02 = M[0, 0] * Q[0, 2] + M[0, 1] * Q[1, 2] + M[0, 2] * Q[2, 2]
            MQ_10 = M[1, 0] * Q[0, 0] + M[1, 1] * Q[1, 0] + M[1, 2] * Q[2, 0]
            MQ_11 = M[1, 0] * Q[0, 1] + M[1, 1] * Q[1, 1] + M[1, 2] * Q[2, 1]
            MQ_12 = M[1, 0] * Q[0, 2] + M[1, 1] * Q[1, 2] + M[1, 2] * Q[2, 2]
            MQ_20 = M[2, 0] * Q[0, 0] + M[2, 1] * Q[1, 0] + M[2, 2] * Q[2, 0]
            MQ_21 = M[2, 0] * Q[0, 1] + M[2, 1] * Q[1, 1] + M[2, 2] * Q[2, 1]
            MQ_22 = M[2, 0] * Q[0, 2] + M[2, 1] * Q[1, 2] + M[2, 2] * Q[2, 2]
            Qe_00 = MQ_00 * M[0, 0] + MQ_01 * M[0, 1] + MQ_02 * M[0, 2]
            Qe_11 = MQ_10 * M[1, 0] + MQ_11 * M[1, 1] + MQ_12 * M[1, 2]
            Qe_22 = MQ_20 * M[2, 0] + MQ_21 * M[2, 1] + MQ_22 * M[2, 2]
            Qe_01 = MQ_00 * M[1, 0] + MQ_01 * M[1, 1] + MQ_02 * M[1, 2]
            Qe_02 = MQ_00 * M[2, 0] + MQ_01 * M[2, 1] + MQ_02 * M[2, 2]
            Qe_12 = MQ_10 * M[2, 0] + MQ_11 * M[2, 1] + MQ_12 * M[2, 2]

            two = type(charge)(2.0)
            F_frac_0 = (
                Qe_00 * acc_3_xxx
                + Qe_11 * acc_3_yyx
                + Qe_22 * acc_3_zzx
                + two * (Qe_01 * acc_3_xxy + Qe_02 * acc_3_xxz + Qe_12 * acc_3_xyz)
            )
            F_frac_1 = (
                Qe_00 * acc_3_xxy
                + Qe_11 * acc_3_yyy
                + Qe_22 * acc_3_zzy
                + two * (Qe_01 * acc_3_yyx + Qe_02 * acc_3_xyz + Qe_12 * acc_3_yyz)
            )
            F_frac_2 = (
                Qe_00 * acc_3_xxz
                + Qe_11 * acc_3_yyz
                + Qe_22 * acc_3_zzz
                + two * (Qe_01 * acc_3_xyz + Qe_02 * acc_3_zzx + Qe_12 * acc_3_zzy)
            )
            gpos_x = gpos_x + half * (
                M[0, 0] * F_frac_0 + M[1, 0] * F_frac_1 + M[2, 0] * F_frac_2
            )
            gpos_y = gpos_y + half * (
                M[0, 1] * F_frac_0 + M[1, 1] * F_frac_1 + M[2, 1] * F_frac_2
            )
            gpos_z = gpos_z + half * (
                M[0, 2] * F_frac_0 + M[1, 2] * F_frac_1 + M[2, 2] * F_frac_2
            )

        grad_positions[atom_idx] = vec_pos_dtype(gpos_x, gpos_y, gpos_z)

        # Cell gradient ∂L/∂M[c, d] (per system) — same three paths as the
        # single-system backward, accumulated into grad_cell_inv_t[sys_idx].
        r_d_x = position[0]
        r_d_y = position[1]
        r_d_z = position[2]

        gtheta_x = charge * acc_fx
        gtheta_y = charge * acc_fy
        gtheta_z = charge * acc_fz
        if LMAX >= 1:
            gtheta_x = gtheta_x + hmu_x
            gtheta_y = gtheta_y + hmu_y
            gtheta_z = gtheta_z + hmu_z
        if LMAX >= 2:
            half_cell = type(charge)(0.5)
            gtheta_x = gtheta_x + half_cell * F_frac_0
            gtheta_y = gtheta_y + half_cell * F_frac_1
            gtheta_z = gtheta_z + half_cell * F_frac_2

        wp.atomic_add(grad_cell_inv_t, sys_idx, 0, 0, gtheta_x * r_d_x)
        wp.atomic_add(grad_cell_inv_t, sys_idx, 0, 1, gtheta_x * r_d_y)
        wp.atomic_add(grad_cell_inv_t, sys_idx, 0, 2, gtheta_x * r_d_z)
        wp.atomic_add(grad_cell_inv_t, sys_idx, 1, 0, gtheta_y * r_d_x)
        wp.atomic_add(grad_cell_inv_t, sys_idx, 1, 1, gtheta_y * r_d_y)
        wp.atomic_add(grad_cell_inv_t, sys_idx, 1, 2, gtheta_y * r_d_z)
        wp.atomic_add(grad_cell_inv_t, sys_idx, 2, 0, gtheta_z * r_d_x)
        wp.atomic_add(grad_cell_inv_t, sys_idx, 2, 1, gtheta_z * r_d_y)
        wp.atomic_add(grad_cell_inv_t, sys_idx, 2, 2, gtheta_z * r_d_z)

        if LMAX >= 1:
            mu_d_x = dipole[0]
            mu_d_y = dipole[1]
            mu_d_z = dipole[2]
            wp.atomic_add(grad_cell_inv_t, sys_idx, 0, 0, mu_d_x * acc_fx)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 0, 1, mu_d_y * acc_fx)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 0, 2, mu_d_z * acc_fx)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 1, 0, mu_d_x * acc_fy)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 1, 1, mu_d_y * acc_fy)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 1, 2, mu_d_z * acc_fy)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 2, 0, mu_d_x * acc_fz)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 2, 1, mu_d_y * acc_fz)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 2, 2, mu_d_z * acc_fz)

        if LMAX >= 2:
            Q_cell = quadrupoles[atom_idx]
            MQc_00 = (
                M[0, 0] * Q_cell[0, 0] + M[0, 1] * Q_cell[1, 0] + M[0, 2] * Q_cell[2, 0]
            )
            MQc_01 = (
                M[0, 0] * Q_cell[0, 1] + M[0, 1] * Q_cell[1, 1] + M[0, 2] * Q_cell[2, 1]
            )
            MQc_02 = (
                M[0, 0] * Q_cell[0, 2] + M[0, 1] * Q_cell[1, 2] + M[0, 2] * Q_cell[2, 2]
            )
            MQc_10 = (
                M[1, 0] * Q_cell[0, 0] + M[1, 1] * Q_cell[1, 0] + M[1, 2] * Q_cell[2, 0]
            )
            MQc_11 = (
                M[1, 0] * Q_cell[0, 1] + M[1, 1] * Q_cell[1, 1] + M[1, 2] * Q_cell[2, 1]
            )
            MQc_12 = (
                M[1, 0] * Q_cell[0, 2] + M[1, 1] * Q_cell[1, 2] + M[1, 2] * Q_cell[2, 2]
            )
            MQc_20 = (
                M[2, 0] * Q_cell[0, 0] + M[2, 1] * Q_cell[1, 0] + M[2, 2] * Q_cell[2, 0]
            )
            MQc_21 = (
                M[2, 0] * Q_cell[0, 1] + M[2, 1] * Q_cell[1, 1] + M[2, 2] * Q_cell[2, 1]
            )
            MQc_22 = (
                M[2, 0] * Q_cell[0, 2] + M[2, 1] * Q_cell[1, 2] + M[2, 2] * Q_cell[2, 2]
            )
            ahmq_00 = acc_hxx * MQc_00 + acc_hxy * MQc_10 + acc_hxz * MQc_20
            ahmq_01 = acc_hxx * MQc_01 + acc_hxy * MQc_11 + acc_hxz * MQc_21
            ahmq_02 = acc_hxx * MQc_02 + acc_hxy * MQc_12 + acc_hxz * MQc_22
            ahmq_10 = acc_hxy * MQc_00 + acc_hyy * MQc_10 + acc_hyz * MQc_20
            ahmq_11 = acc_hxy * MQc_01 + acc_hyy * MQc_11 + acc_hyz * MQc_21
            ahmq_12 = acc_hxy * MQc_02 + acc_hyy * MQc_12 + acc_hyz * MQc_22
            ahmq_20 = acc_hxz * MQc_00 + acc_hyz * MQc_10 + acc_hzz * MQc_20
            ahmq_21 = acc_hxz * MQc_01 + acc_hyz * MQc_11 + acc_hzz * MQc_21
            ahmq_22 = acc_hxz * MQc_02 + acc_hyz * MQc_12 + acc_hzz * MQc_22
            wp.atomic_add(grad_cell_inv_t, sys_idx, 0, 0, ahmq_00)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 0, 1, ahmq_01)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 0, 2, ahmq_02)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 1, 0, ahmq_10)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 1, 1, ahmq_11)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 1, 2, ahmq_12)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 2, 0, ahmq_20)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 2, 1, ahmq_21)
            wp.atomic_add(grad_cell_inv_t, sys_idx, 2, 2, ahmq_22)

    return kernel


_PER_ORDER_LMAX_BATCH_SPREAD_BACKWARD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _lmax in (0, 1, 2):
        for _scalar, _vec, _mat in (
            (wp.float32, wp.vec3f, wp.mat33f),
            (wp.float64, wp.vec3d, wp.mat33d),
        ):
            _k = _make_batch_pme_multipole_spread_backward_unified_kernel(
                _order, _lmax, _scalar, _vec, _mat
            )
            _PER_ORDER_LMAX_BATCH_SPREAD_BACKWARD_KERNELS[_scalar][(_order, _lmax)] = (
                wp.overload(
                    _k,
                    [
                        wp.array(dtype=_vec),  # positions
                        wp.array(dtype=_scalar),  # charges
                        wp.array(dtype=_vec),  # dipoles
                        wp.array(dtype=_mat),  # quadrupoles
                        wp.array(dtype=wp.int32),  # batch_idx
                        wp.array(dtype=_mat),  # cell_inv_t (B, 3, 3)
                        wp.array(dtype=_scalar, ndim=4),  # grad_mesh (B, nx, ny, nz)
                        wp.array(dtype=_vec),  # grad_positions (out)
                        wp.array(dtype=_scalar),  # grad_charges (out)
                        wp.array(dtype=_vec),  # grad_dipoles (out)
                        wp.array(dtype=_mat),  # grad_quadrupoles (out)
                        wp.array(dtype=_scalar, ndim=3),  # grad_cell_inv_t (B, 3, 3)
                    ],
                )
            )


def _maybe_per_order_lmax_batch_spread_backward_kernel(order: int, lmax: int, wp_dtype):
    """Return the per-(ORDER, LMAX, dtype) batched unified spread-backward kernel."""
    return _PER_ORDER_LMAX_BATCH_SPREAD_BACKWARD_KERNELS.get(wp_dtype, {}).get(
        (order, lmax)
    )


def batch_multipole_pme_spread_backward_unified_launch(
    positions: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    quadrupoles: wp.array,
    batch_idx: wp.array,
    cell_inv_t: wp.array,
    order: int,
    lmax: int,
    grad_mesh: wp.array,
    grad_positions: wp.array,
    grad_charges: wp.array,
    grad_dipoles: wp.array,
    grad_quadrupoles: wp.array,
    grad_cell_inv_t: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> bool:
    r"""Launch the batched unified ``(ORDER, LMAX)`` spread-backward kernel.

    Batched analog of
    :func:`multipole_pme_spread_backward_unified_launch`.
    ``grad_cell_inv_t`` is a ``(B, 3, 3)`` Warp array receiving the
    per-system atomic-accumulated ``∂L/∂M``; caller pre-zeros all outputs.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=vec3f/vec3d
        Concatenated Cartesian atom positions.
    charges : wp.array, shape (N_total,), dtype=wp.float32/float64
        Per-atom monopole charges.
    dipoles : wp.array, shape (N_total,), dtype=vec3f/vec3d
        Per-atom Cartesian dipoles (used when ``lmax >= 1``).
    quadrupoles : wp.array, shape (N_total,), dtype=mat33f/mat33d
        Per-atom Cartesian quadrupoles (used when ``lmax >= 2``).
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        Per-atom system index into the leading mesh axis.
    cell_inv_t : wp.array, shape (B,), dtype=mat33f/mat33d
        Per-system transpose of the inverse cell matrix.
    order : int
        B-spline order (one of ``(3, 4, 5, 6)``).
    lmax : int
        Maximum multipole order (0, 1, or 2).
    grad_mesh : wp.array, shape (B, Nx, Ny, Nz), dtype=wp.float32/float64
        Upstream gradient w.r.t. the per-system spread mesh.
    grad_positions : wp.array, shape (N_total,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Gradient w.r.t. positions.
    grad_charges : wp.array, shape (N_total,), dtype=wp.float32/float64
        OUTPUT, pre-zeroed. Gradient w.r.t. charges.
    grad_dipoles : wp.array, shape (N_total,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Gradient w.r.t. dipoles.
    grad_quadrupoles : wp.array, shape (N_total,), dtype=mat33f/mat33d
        OUTPUT, pre-zeroed. Gradient w.r.t. quadrupoles.
    grad_cell_inv_t : wp.array, shape (B, 3, 3), dtype=wp.float32/float64
        OUTPUT, pre-zeroed. Per-system gradient w.r.t. ``cell_inv_t``.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Returns
    -------
    bool
        ``True`` if the per-order overload ran, ``False`` on cache miss.

    Launch Grid
    -----------
    ``dim = (N_total,)`` — one thread per atom across the batch.
    """
    per_order = _maybe_per_order_lmax_batch_spread_backward_kernel(
        order, lmax, wp_dtype
    )
    if per_order is None:
        return False
    if device is None:
        device = str(positions.device)
    num_atoms = positions.shape[0]
    wp.launch(
        per_order,
        dim=(num_atoms,),
        inputs=[
            positions,
            charges,
            dipoles,
            quadrupoles,
            batch_idx,
            cell_inv_t,
            grad_mesh,
        ],
        outputs=[
            grad_positions,
            grad_charges,
            grad_dipoles,
            grad_quadrupoles,
            grad_cell_inv_t,
        ],
        device=device,
    )
    return True


# =============================================================================
# Octupole (∇³ / ∇⁴) kernels for the spread double-backward (l_max=2)
# =============================================================================
#
# The spread Q channel contributes ``½ Qe : ∇²_frac B`` (Qe = M Q Mᵀ). Its
# directional derivative along a Cartesian position-cotangent ``gg_pos`` is
#   D_gg  = ½ Σ_ijk Qe_ij g_k ∂³_frac B[ijk],   g = M·gg_pos  (frac-frame).
# This is the only new term in ``∂L/∂grad_mesh`` of the l_max=2 spread
# double-backward; everything else reuses the forward/backward spread with
# "effective moments". The per-stencil contraction (Sx, Sy, Sz below) is the
# transpose of the backward's ``F_frac`` rank-3 enumeration — same 10 ∂³
# products.


def _make_pme_octupole_spread_kernel(ORDER, scalar_dtype, vec_pos_dtype, mat33_dtype):
    """Octupole (∇³) spread: mesh += ½ Σ_ijk Qe_ij g_k ∂³_frac B,  g = M gg_pos."""
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_module("octupole_spread", ORDER, scalar_dtype)
    )
    def kernel(
        positions: wp.array(dtype=Any),
        quadrupoles: wp.array(dtype=Any),
        gg_pos: wp.array(dtype=Any),
        cell_inv_t: wp.array(dtype=Any),
        mesh: wp.array3d(dtype=Any),
    ):
        atom_idx = wp.tid()
        mesh_dims = wp.vec3i(mesh.shape[0], mesh.shape[1], mesh.shape[2])
        position = positions[atom_idx]
        base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
        t0 = theta[0]
        half_order = type(t0)(HALF_ORDER_PY)
        half_n_minus_2 = type(t0)(HALF_N_MINUS_2_PY)
        offset_start_x = wp.int32(wp.floor(theta[0] - half_n_minus_2))
        offset_start_y = wp.int32(wp.floor(theta[1] - half_n_minus_2))
        offset_start_z = wp.int32(wp.floor(theta[2] - half_n_minus_2))

        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        dwx = vec_ord()
        dwy = vec_ord()
        dwz = vec_ord()
        d2wx = vec_ord()
        d2wy = vec_ord()
        d2wz = vec_ord()
        d3wx = vec_ord()
        d3wy = vec_ord()
        d3wz = vec_ord()
        md_x = type(t0)(mesh_dims[0])
        md_y = type(t0)(mesh_dims[1])
        md_z = type(t0)(mesh_dims[2])
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            dwx[k] = bspline_derivative(u_x, ORDER) * md_x
            dwy[k] = bspline_derivative(u_y, ORDER) * md_y
            dwz[k] = bspline_derivative(u_z, ORDER) * md_z
            d2wx[k] = bspline_second_derivative(u_x, ORDER) * md_x * md_x
            d2wy[k] = bspline_second_derivative(u_y, ORDER) * md_y * md_y
            d2wz[k] = bspline_second_derivative(u_z, ORDER) * md_z * md_z
            d3wx[k] = bspline_third_derivative(u_x, ORDER) * md_x * md_x * md_x
            d3wy[k] = bspline_third_derivative(u_y, ORDER) * md_y * md_y * md_y
            d3wz[k] = bspline_third_derivative(u_z, ORDER) * md_z * md_z * md_z

        # Qe = M Q Mᵀ (six unique) and g = M gg_pos, hoisted per atom.
        Q = quadrupoles[atom_idx]
        M = cell_inv_t[0]
        gp = gg_pos[atom_idx]
        MQ_00 = M[0, 0] * Q[0, 0] + M[0, 1] * Q[1, 0] + M[0, 2] * Q[2, 0]
        MQ_01 = M[0, 0] * Q[0, 1] + M[0, 1] * Q[1, 1] + M[0, 2] * Q[2, 1]
        MQ_02 = M[0, 0] * Q[0, 2] + M[0, 1] * Q[1, 2] + M[0, 2] * Q[2, 2]
        MQ_10 = M[1, 0] * Q[0, 0] + M[1, 1] * Q[1, 0] + M[1, 2] * Q[2, 0]
        MQ_11 = M[1, 0] * Q[0, 1] + M[1, 1] * Q[1, 1] + M[1, 2] * Q[2, 1]
        MQ_12 = M[1, 0] * Q[0, 2] + M[1, 1] * Q[1, 2] + M[1, 2] * Q[2, 2]
        MQ_20 = M[2, 0] * Q[0, 0] + M[2, 1] * Q[1, 0] + M[2, 2] * Q[2, 0]
        MQ_21 = M[2, 0] * Q[0, 1] + M[2, 1] * Q[1, 1] + M[2, 2] * Q[2, 1]
        MQ_22 = M[2, 0] * Q[0, 2] + M[2, 1] * Q[1, 2] + M[2, 2] * Q[2, 2]
        Qe00 = MQ_00 * M[0, 0] + MQ_01 * M[0, 1] + MQ_02 * M[0, 2]
        Qe11 = MQ_10 * M[1, 0] + MQ_11 * M[1, 1] + MQ_12 * M[1, 2]
        Qe22 = MQ_20 * M[2, 0] + MQ_21 * M[2, 1] + MQ_22 * M[2, 2]
        Qe01 = MQ_00 * M[1, 0] + MQ_01 * M[1, 1] + MQ_02 * M[1, 2]
        Qe02 = MQ_00 * M[2, 0] + MQ_01 * M[2, 1] + MQ_02 * M[2, 2]
        Qe12 = MQ_10 * M[2, 0] + MQ_11 * M[2, 1] + MQ_12 * M[2, 2]
        g0 = M[0, 0] * gp[0] + M[0, 1] * gp[1] + M[0, 2] * gp[2]
        g1 = M[1, 0] * gp[0] + M[1, 1] * gp[1] + M[1, 2] * gp[2]
        g2 = M[2, 0] * gp[0] + M[2, 1] * gp[1] + M[2, 2] * gp[2]

        half = type(t0)(0.5)
        two = type(t0)(2.0)
        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            d2wxi = d2wx[i]
            d3wxi = d3wx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wyj = wy[j]
                dwyj = dwy[j]
                d2wyj = d2wy[j]
                d3wyj = d3wy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    wzk = wz[k]
                    dwzk = dwz[k]
                    d2wzk = d2wz[k]
                    d3wzk = d3wz[k]
                    # 10 distinct ∂³_frac B products.
                    B_xxx = d3wxi * wyj * wzk
                    B_yyy = wxi * d3wyj * wzk
                    B_zzz = wxi * wyj * d3wzk
                    B_xxy = d2wxi * dwyj * wzk
                    B_xxz = d2wxi * wyj * dwzk
                    B_xyy = dwxi * d2wyj * wzk
                    B_yyz = wxi * d2wyj * dwzk
                    B_xzz = dwxi * wyj * d2wzk
                    B_yzz = wxi * dwyj * d2wzk
                    B_xyz = dwxi * dwyj * dwzk
                    Sx = (
                        Qe00 * B_xxx
                        + Qe11 * B_xyy
                        + Qe22 * B_xzz
                        + two * (Qe01 * B_xxy + Qe02 * B_xxz + Qe12 * B_xyz)
                    )
                    Sy = (
                        Qe00 * B_xxy
                        + Qe11 * B_yyy
                        + Qe22 * B_yzz
                        + two * (Qe01 * B_xyy + Qe02 * B_xyz + Qe12 * B_yyz)
                    )
                    Sz = (
                        Qe00 * B_xxz
                        + Qe11 * B_yyz
                        + Qe22 * B_zzz
                        + two * (Qe01 * B_xyz + Qe02 * B_xzz + Qe12 * B_yzz)
                    )
                    contrib = half * (g0 * Sx + g1 * Sy + g2 * Sz)
                    wp.atomic_add(mesh, gx, gy, gz, contrib)

    return kernel


_PER_ORDER_OCTUPOLE_SPREAD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_pme_octupole_spread_kernel(_order, _scalar, _vec, _mat)
        _PER_ORDER_OCTUPOLE_SPREAD_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),  # positions
                wp.array(dtype=_mat),  # quadrupoles
                wp.array(dtype=_vec),  # gg_pos
                wp.array(dtype=_mat),  # cell_inv_t
                wp.array3d(dtype=_scalar),  # mesh
            ],
        )


# =============================================================================
# Double-backward per-atom moment algebra (effective moments + readouts)
# =============================================================================


@wp.kernel
def _pme_effective_moments_kernel(
    charges: wp.array(dtype=Any),  # (N,) scalar
    dipoles: wp.array(dtype=Any),  # (N,) vec3
    gg_pos: wp.array(dtype=Any),  # (N,) vec3
    gg_dipoles: wp.array(dtype=Any),  # (N,) vec3
    gg_quadrupoles: wp.array(dtype=Any),  # (N,) mat33
    eff_d: wp.array(dtype=Any),  # (N,) vec3 OUTPUT
    eff_Q: wp.array(dtype=Any),  # (N,) mat33 OUTPUT
):
    r"""Effective dipole / quadrupole for the spread double-back (per atom).

    ``eff_d = gg_d + q · gg_pos`` and
    ``eff_Q = gg_Q + (gg_pos ⊗ μ) + (gg_pos ⊗ μ)ᵀ`` — promotes the incoming
    2nd-order cotangents to an l_max=2 effective spread.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.

    Parameters
    ----------
    charges : wp.array, shape (N,), dtype=wp.float32/float64
        Per-atom charge (saved forward input).
    dipoles : wp.array, shape (N,), dtype=vec3f/vec3d
        Per-atom dipole μ (saved forward input).
    gg_pos : wp.array, shape (N,), dtype=vec3f/vec3d
        Incoming position cotangent.
    gg_dipoles : wp.array, shape (N,), dtype=vec3f/vec3d
        Incoming dipole cotangent.
    gg_quadrupoles : wp.array, shape (N,), dtype=mat33f/mat33d
        Incoming quadrupole cotangent.
    eff_d : wp.array, shape (N,), dtype=vec3f/vec3d
        OUTPUT. Effective dipole.
    eff_Q : wp.array, shape (N,), dtype=mat33f/mat33d
        OUTPUT. Effective (symmetric) quadrupole.
    """
    i = wp.tid()
    gp = gg_pos[i]
    om = wp.outer(gp, dipoles[i])  # gg_pos ⊗ μ
    eff_d[i] = gg_dipoles[i] + charges[i] * gp
    eff_Q[i] = gg_quadrupoles[i] + om + wp.transpose(om)


def _pme_effective_moments_sig(v, t):
    """Signature builder for :func:`_pme_effective_moments_kernel`."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=v),  # gg_pos
        wp.array(dtype=v),  # gg_dipoles
        wp.array(dtype=mat),  # gg_quadrupoles
        wp.array(dtype=v),  # eff_d (output)
        wp.array(dtype=mat),  # eff_Q (output)
    ]


_pme_effective_moments_overloads = register_overloads(
    _pme_effective_moments_kernel, _pme_effective_moments_sig
)


def pme_effective_moments_launch(
    charges,
    dipoles,
    gg_pos,
    gg_dipoles,
    gg_quadrupoles,
    eff_d,
    eff_Q,
    wp_dtype,
    device=None,
):
    r"""Launch :func:`_pme_effective_moments_kernel` (one thread per atom)."""
    if device is None:
        device = str(charges.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_effective_moments_overloads[vec_dtype],
        dim=(charges.shape[0],),
        inputs=[charges, dipoles, gg_pos, gg_dipoles, gg_quadrupoles],
        outputs=[eff_d, eff_Q],
        device=device,
    )


@wp.kernel
def _pme_fractionalize_kernel(
    positions: wp.array(dtype=Any),  # (N,) vec3
    cell_inv_t: wp.array(dtype=Any),  # (B,) mat33  (M = inv(cell)ᵀ)
    batch_idx: wp.array(dtype=wp.int32),  # (N,) per-atom system index
    dipoles: wp.array(dtype=Any),  # (N,) vec3
    quadrupoles: wp.array(dtype=Any),  # (N,) mat33
    p_out: wp.array(dtype=Any),  # (N,) vec3 OUTPUT — cell-fractional coord
    df_out: wp.array(dtype=Any),  # (N,) vec3 OUTPUT — fractional dipole
    Qf_out: wp.array(dtype=Any),  # (N,) mat33 OUTPUT — fractional quadrupole
):
    r"""Map Cartesian (positions, moments) to the unitless cell-fractional frame.

    Per atom, with :math:`M = \text{cell\_inv\_t}[b_i]`:

    .. math::

        p_i = M\, r_i, \quad
        d^{\text{frac}}_i = M\, \mu_i, \quad
        Q^{\text{frac}}_i = M\, Q_i\, M^{\mathsf T}.

    This factors ALL ``cell_inv_t`` coupling out of the spread/gather kernels:
    feeding them ``p`` + fractional moments with an identity ``cell_inv_t``
    reproduces the cell-coupled spread bit-for-bit (``compute_fractional_coords``
    multiplies by the mesh internally; the moment rotations become no-ops), so
    the cell-stress 2nd-order composes entirely through this multilinear map's
    autograd (cheap, exact). One thread per atom; ``cell_inv_t[batch_idx[i]]``
    is read per atom so batched systems need no host-side per-atom cell gather.
    """
    i = wp.tid()
    b = batch_idx[i]
    m_mat = cell_inv_t[b]
    p_out[i] = m_mat * positions[i]
    df_out[i] = m_mat * dipoles[i]
    Qf_out[i] = m_mat * quadrupoles[i] * wp.transpose(m_mat)


def _pme_fractionalize_sig(v, t):
    """Signature builder for :func:`_pme_fractionalize_kernel`."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=mat),  # cell_inv_t
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=mat),  # quadrupoles
        wp.array(dtype=v),  # p_out
        wp.array(dtype=v),  # df_out
        wp.array(dtype=mat),  # Qf_out
    ]


_pme_fractionalize_overloads = register_overloads(
    _pme_fractionalize_kernel, _pme_fractionalize_sig
)


def pme_fractionalize_launch(
    positions,
    cell_inv_t,
    batch_idx,
    dipoles,
    quadrupoles,
    p_out,
    df_out,
    Qf_out,
    wp_dtype,
    device=None,
):
    r"""Launch :func:`_pme_fractionalize_kernel` (one thread per atom)."""
    if device is None:
        device = str(positions.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_fractionalize_overloads[vec_dtype],
        dim=(positions.shape[0],),
        inputs=[positions, cell_inv_t, batch_idx, dipoles, quadrupoles],
        outputs=[p_out, df_out, Qf_out],
        device=device,
    )


@wp.kernel
def _pme_fractionalize_backward_kernel(
    positions: wp.array(dtype=Any),  # (N,) vec3
    cell_inv_t: wp.array(dtype=Any),  # (B,) mat33
    batch_idx: wp.array(dtype=wp.int32),  # (N,)
    dipoles: wp.array(dtype=Any),  # (N,) vec3
    quadrupoles: wp.array(dtype=Any),  # (N,) mat33
    gp: wp.array(dtype=Any),  # (N,) vec3  — cotangent on p
    gdf: wp.array(dtype=Any),  # (N,) vec3  — cotangent on d_frac
    gQf: wp.array(dtype=Any),  # (N,) mat33 — cotangent on Q_frac
    grad_positions: wp.array(dtype=Any),  # (N,) vec3 OUTPUT
    grad_cell_inv_t: wp.array(dtype=Any),  # (B,) mat33 OUTPUT (atomic accum)
    grad_dipoles: wp.array(dtype=Any),  # (N,) vec3 OUTPUT
    grad_quadrupoles: wp.array(dtype=Any),  # (N,) mat33 OUTPUT
):
    r"""Adjoint of :func:`_pme_fractionalize_kernel` (the map is multilinear).

    With ``M = cell_inv_t[b]``:
    ``grad_r = Mᵀ·gp``; ``grad_μ = Mᵀ·gdf``; ``grad_Q = Mᵀ·gQf·M``; and
    ``grad_M = gp⊗r + gdf⊗μ + gQf·M·Qᵀ + gQfᵀ·M·Q`` (atomic-added per system).
    """
    i = wp.tid()
    b = batch_idx[i]
    m_mat = cell_inv_t[b]
    r = positions[i]
    gp_i = gp[i]
    m_t = wp.transpose(m_mat)
    grad_positions[i] = m_t * gp_i
    grad_dipoles[i] = m_t * gdf[i]
    gQf_i = gQf[i]
    grad_quadrupoles[i] = m_t * gQf_i * m_mat
    q_i = quadrupoles[i]
    grad_m = (
        wp.outer(gp_i, r)
        + wp.outer(gdf[i], dipoles[i])
        + gQf_i * m_mat * wp.transpose(q_i)
        + wp.transpose(gQf_i) * m_mat * q_i
    )
    wp.atomic_add(grad_cell_inv_t, b, grad_m)


def _pme_fractionalize_backward_sig(v, t):
    """Signature builder for :func:`_pme_fractionalize_backward_kernel`."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=mat),  # cell_inv_t
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=mat),  # quadrupoles
        wp.array(dtype=v),  # gp
        wp.array(dtype=v),  # gdf
        wp.array(dtype=mat),  # gQf
        wp.array(dtype=v),  # grad_positions
        wp.array(dtype=mat),  # grad_cell_inv_t
        wp.array(dtype=v),  # grad_dipoles
        wp.array(dtype=mat),  # grad_quadrupoles
    ]


_pme_fractionalize_backward_overloads = register_overloads(
    _pme_fractionalize_backward_kernel, _pme_fractionalize_backward_sig
)


def pme_fractionalize_backward_launch(
    positions,
    cell_inv_t,
    batch_idx,
    dipoles,
    quadrupoles,
    gp,
    gdf,
    gQf,
    grad_positions,
    grad_cell_inv_t,
    grad_dipoles,
    grad_quadrupoles,
    wp_dtype,
    device=None,
):
    r"""Launch :func:`_pme_fractionalize_backward_kernel` (one thread per atom)."""
    if device is None:
        device = str(positions.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_fractionalize_backward_overloads[vec_dtype],
        dim=(positions.shape[0],),
        inputs=[
            positions,
            cell_inv_t,
            batch_idx,
            dipoles,
            quadrupoles,
            gp,
            gdf,
            gQf,
        ],
        outputs=[grad_positions, grad_cell_inv_t, grad_dipoles, grad_quadrupoles],
        device=device,
    )


@wp.kernel
def _pme_fractionalize_double_backward_kernel(
    positions: wp.array(dtype=Any),  # (N,) vec3
    cell_inv_t: wp.array(dtype=Any),  # (B,) mat33
    batch_idx: wp.array(dtype=wp.int32),  # (N,)
    dipoles: wp.array(dtype=Any),  # (N,) vec3
    quadrupoles: wp.array(dtype=Any),  # (N,) mat33
    gp: wp.array(dtype=Any),  # (N,) vec3  — cotangent on p
    gdf: wp.array(dtype=Any),  # (N,) vec3  — cotangent on d_frac
    gQf: wp.array(dtype=Any),  # (N,) mat33 — cotangent on Q_frac
    g_pos: wp.array(dtype=Any),  # (N,) vec3  — cotangent on grad_positions
    g_cell: wp.array(dtype=Any),  # (B,) mat33 — cotangent on grad_cell_inv_t
    g_dip: wp.array(dtype=Any),  # (N,) vec3  — cotangent on grad_dipoles
    g_quad: wp.array(dtype=Any),  # (N,) mat33 — cotangent on grad_quadrupoles
    grad_gp: wp.array(dtype=Any),  # (N,) vec3 OUTPUT
    grad_gdf: wp.array(dtype=Any),  # (N,) vec3 OUTPUT
    grad_gQf: wp.array(dtype=Any),  # (N,) mat33 OUTPUT
    grad_positions: wp.array(dtype=Any),  # (N,) vec3 OUTPUT
    grad_cell_inv_t: wp.array(dtype=Any),  # (B,) mat33 OUTPUT (atomic)
    grad_dipoles: wp.array(dtype=Any),  # (N,) vec3 OUTPUT
    grad_quadrupoles: wp.array(dtype=Any),  # (N,) mat33 OUTPUT
):
    r"""Double-backward of :func:`_pme_fractionalize_kernel` (stress-loss).

    The forward map is multilinear, so the second-order is closed-form. With
    ``M = cell_inv_t[b]`` and the four incoming cotangents ``(Gr, GM, Gμ, GQ)``
    on the backward outputs
    ``(grad_positions, grad_cell_inv_t, grad_dipoles, grad_quadrupoles)``, the
    grads w.r.t. the backward inputs are:

    - ``grad_gp = M·Gr + GM·r`` (gp enters BOTH grad_positions and grad_cell)
    - ``grad_gdf = GM·μ + M·Gμ``
    - ``grad_gQf = GM·Q·Mᵀ + M·Q·GMᵀ + M·GQ·Mᵀ``
    - ``grad_r = GMᵀ·gp``
    - ``grad_μ = GMᵀ·gdf``
    - ``grad_Q = GMᵀ·gQf·M + Mᵀ·gQf·GM``
    - ``grad_M = gp⊗Gr + gdf⊗Gμ + gQfᵀ·GM·Q + gQf·GM·Qᵀ
      + gQf·M·GQᵀ + gQfᵀ·M·GQ`` (atomic per system).
    """
    i = wp.tid()
    b = batch_idx[i]
    m_mat = cell_inv_t[b]
    m_t = wp.transpose(m_mat)
    r = positions[i]
    mu = dipoles[i]
    q_i = quadrupoles[i]
    gp_i = gp[i]
    gdf_i = gdf[i]
    gQf_i = gQf[i]
    gr = g_pos[i]
    gm = g_cell[b]
    gm_t = wp.transpose(gm)
    gmu = g_dip[i]
    gq = g_quad[i]
    # ∂/∂(incoming cotangents). gp enters BOTH grad_positions (via gr=Mᵀ·gp)
    # AND grad_cell_inv_t (via the gp⊗r term), so both feed grad_gp.
    grad_gp[i] = m_mat * gr + gm * r
    grad_gdf[i] = gm * mu + m_mat * gmu
    grad_gQf[i] = gm * q_i * m_t + m_mat * q_i * gm_t + m_mat * gq * m_t
    # ∂/∂(forward inputs) — the genuine cell×{pos,moment} cross terms.
    grad_positions[i] = gm_t * gp_i
    grad_dipoles[i] = gm_t * gdf_i
    grad_quadrupoles[i] = gm_t * gQf_i * m_mat + m_t * gQf_i * gm
    grad_m = (
        wp.outer(gp_i, gr)
        + wp.outer(gdf_i, gmu)
        + wp.transpose(gQf_i) * gm * q_i
        + gQf_i * gm * wp.transpose(q_i)
        + gQf_i * m_mat * wp.transpose(gq)
        + wp.transpose(gQf_i) * m_mat * gq
    )
    wp.atomic_add(grad_cell_inv_t, b, grad_m)


def _pme_fractionalize_double_backward_sig(v, t):
    """Signature builder for :func:`_pme_fractionalize_double_backward_kernel`."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # positions
        wp.array(dtype=mat),  # cell_inv_t
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=mat),  # quadrupoles
        wp.array(dtype=v),  # gp
        wp.array(dtype=v),  # gdf
        wp.array(dtype=mat),  # gQf
        wp.array(dtype=v),  # g_pos
        wp.array(dtype=mat),  # g_cell
        wp.array(dtype=v),  # g_dip
        wp.array(dtype=mat),  # g_quad
        wp.array(dtype=v),  # grad_gp
        wp.array(dtype=v),  # grad_gdf
        wp.array(dtype=mat),  # grad_gQf
        wp.array(dtype=v),  # grad_positions
        wp.array(dtype=mat),  # grad_cell_inv_t
        wp.array(dtype=v),  # grad_dipoles
        wp.array(dtype=mat),  # grad_quadrupoles
    ]


_pme_fractionalize_double_backward_overloads = register_overloads(
    _pme_fractionalize_double_backward_kernel, _pme_fractionalize_double_backward_sig
)


def pme_fractionalize_double_backward_launch(
    positions,
    cell_inv_t,
    batch_idx,
    dipoles,
    quadrupoles,
    gp,
    gdf,
    gQf,
    g_pos,
    g_cell,
    g_dip,
    g_quad,
    grad_gp,
    grad_gdf,
    grad_gQf,
    grad_positions,
    grad_cell_inv_t,
    grad_dipoles,
    grad_quadrupoles,
    wp_dtype,
    device=None,
):
    r"""Launch :func:`_pme_fractionalize_double_backward_kernel` (one thread/atom)."""
    if device is None:
        device = str(positions.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_fractionalize_double_backward_overloads[vec_dtype],
        dim=(positions.shape[0],),
        inputs=[
            positions,
            cell_inv_t,
            batch_idx,
            dipoles,
            quadrupoles,
            gp,
            gdf,
            gQf,
            g_pos,
            g_cell,
            g_dip,
            g_quad,
        ],
        outputs=[
            grad_gp,
            grad_gdf,
            grad_gQf,
            grad_positions,
            grad_cell_inv_t,
            grad_dipoles,
            grad_quadrupoles,
        ],
        device=device,
    )


@wp.kernel
def _pme_spread_dbwd_readout_kernel(
    gg_pos: wp.array(dtype=Any),  # (N,) vec3
    gd2: wp.array(dtype=Any),  # (N,) vec3
    gQ2: wp.array(dtype=Any),  # (N,) mat33
    d_charges: wp.array(dtype=Any),  # (N,) scalar OUTPUT
    d_dipoles: wp.array(dtype=Any),  # (N,) vec3 OUTPUT
):
    r"""Readout combinations of the spread double-back (per atom).

    ``d_charges = gg_pos · gd2`` and ``d_dipoles = 2 · gQ2 · gg_pos`` — the
    moment-independent field readouts contracted against the incoming position
    cotangent.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.

    Parameters
    ----------
    gg_pos : wp.array, shape (N,), dtype=vec3f/vec3d
        Incoming position cotangent.
    gd2 : wp.array, shape (N,), dtype=vec3f/vec3d
        Spread-backward dipole readout (Mᵀ acc_f).
    gQ2 : wp.array, shape (N,), dtype=mat33f/mat33d
        Spread-backward quadrupole readout (½ Mᵀ acc_H M).
    d_charges : wp.array, shape (N,), dtype=wp.float32/float64
        OUTPUT. ∂L/∂charges.
    d_dipoles : wp.array, shape (N,), dtype=vec3f/vec3d
        OUTPUT. ∂L/∂dipoles.
    """
    i = wp.tid()
    gp = gg_pos[i]
    d_charges[i] = wp.dot(gp, gd2[i])
    gv = gQ2[i] * gp  # gQ2 · gg_pos
    d_dipoles[i] = gv + gv  # 2 · gQ2 · gg_pos


def _pme_spread_dbwd_readout_sig(v, t):
    """Signature builder for :func:`_pme_spread_dbwd_readout_kernel`."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=v),  # gg_pos
        wp.array(dtype=v),  # gd2
        wp.array(dtype=mat),  # gQ2
        wp.array(dtype=t),  # d_charges (output)
        wp.array(dtype=v),  # d_dipoles (output)
    ]


_pme_spread_dbwd_readout_overloads = register_overloads(
    _pme_spread_dbwd_readout_kernel, _pme_spread_dbwd_readout_sig
)


def pme_spread_dbwd_readout_launch(
    gg_pos,
    gd2,
    gQ2,
    d_charges,
    d_dipoles,
    wp_dtype,
    device=None,
):
    r"""Launch :func:`_pme_spread_dbwd_readout_kernel` (one thread per atom)."""
    if device is None:
        device = str(gg_pos.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_spread_dbwd_readout_overloads[vec_dtype],
        dim=(gg_pos.shape[0],),
        inputs=[gg_pos, gd2, gQ2],
        outputs=[d_charges, d_dipoles],
        device=device,
    )


def multipole_pme_octupole_spread_launch(
    positions,
    quadrupoles,
    gg_pos,
    cell_inv_t,
    order,
    mesh,
    wp_dtype,
    device=None,
):
    r"""Launch the octupole (∇³) spread kernel (l_max=2 double-back).

    Spreads the octupole-like density
    :math:`\rho[g] \mathrel{+}= \tfrac{1}{2}\sum_{ijk} Q^e_{ij}\,g_k\,
    \partial^3_{\text{frac}} B` arising from the second derivative of the
    quadrupole spread w.r.t. position (needed for create_graph at
    ``l_max=2``).

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=vec3f/vec3d
        Cartesian atom positions.
    quadrupoles : wp.array, shape (N,), dtype=mat33f/mat33d
        Per-atom Cartesian quadrupole tensors.
    gg_pos : wp.array, shape (N,), dtype=vec3f/vec3d
        Incoming position-gradient direction (the ``gg`` vector contracted
        into the octupole spread; :math:`g = M\,\text{gg\_pos}`).
    cell_inv_t : wp.array, shape (1,), dtype=mat33f/mat33d
        Transpose of the inverse cell matrix :math:`M`.
    order : int
        B-spline order.
    mesh : wp.array, shape (Nx, Ny, Nz), dtype=wp.float32/float64
        OUTPUT. Density mesh, accumulated via atomics.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Returns
    -------
    bool
        ``True`` if the per-order kernel ran, ``False`` on cache miss.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    per_order = _PER_ORDER_OCTUPOLE_SPREAD_KERNELS.get(wp_dtype, {}).get(order)
    if per_order is None:
        return False
    if device is None:
        device = str(positions.device)
    wp.launch(
        per_order,
        dim=(positions.shape[0],),
        inputs=[positions, quadrupoles, gg_pos, cell_inv_t],
        outputs=[mesh],
        device=device,
    )
    return True


def _make_pme_octupole_backward_kernel(ORDER, scalar_dtype, vec_pos_dtype, mat33_dtype):
    r"""Octupole backward for the spread double-back (l_max=2).

    Reads ``grad_mesh`` and emits the two input-Q octupole VJP slots:

      * ``grad_positions`` (∂L/∂r octupole term, ∇⁴):
            grad_pos = ½ Mᵀ (P4 · g),   g = M·gg_pos,
            P4[k,l] = Σ_ij Qe_ij acc4[i,j,k,l],   acc4 = Σ_g gm ∂⁴_frac B.
      * ``grad_quadrupoles`` (∂L/∂Q octupole term, ∇³):
            grad_Q = ½ Mᵀ R M,   R_ij = Σ_k g_k acc3[i,j,k],
            acc3 = Σ_g gm ∂³_frac B.
    Both accumulate ATOMIC-add into the supplied output buffers (the caller
    sums them with the l≤1 effective-moment reuse results).
    """
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_module("octupole_backward", ORDER, scalar_dtype)
    )
    def kernel(
        positions: wp.array(dtype=Any),
        quadrupoles: wp.array(dtype=Any),
        gg_pos: wp.array(dtype=Any),
        cell_inv_t: wp.array(dtype=Any),
        grad_mesh: wp.array3d(dtype=Any),
        grad_positions: wp.array(dtype=Any),
        grad_quadrupoles: wp.array(dtype=Any),
    ):
        atom_idx = wp.tid()
        mesh_dims = wp.vec3i(grad_mesh.shape[0], grad_mesh.shape[1], grad_mesh.shape[2])
        position = positions[atom_idx]
        base_grid, theta = compute_fractional_coords(position, cell_inv_t[0], mesh_dims)
        t0 = theta[0]
        half_order = type(t0)(HALF_ORDER_PY)
        half_n_minus_2 = type(t0)(HALF_N_MINUS_2_PY)
        zero = type(t0)(0.0)
        offset_start_x = wp.int32(wp.floor(theta[0] - half_n_minus_2))
        offset_start_y = wp.int32(wp.floor(theta[1] - half_n_minus_2))
        offset_start_z = wp.int32(wp.floor(theta[2] - half_n_minus_2))

        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        dwx = vec_ord()
        dwy = vec_ord()
        dwz = vec_ord()
        d2wx = vec_ord()
        d2wy = vec_ord()
        d2wz = vec_ord()
        d3wx = vec_ord()
        d3wy = vec_ord()
        d3wz = vec_ord()
        d4wx = vec_ord()
        d4wy = vec_ord()
        d4wz = vec_ord()
        md_x = type(t0)(mesh_dims[0])
        md_y = type(t0)(mesh_dims[1])
        md_z = type(t0)(mesh_dims[2])
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            dwx[k] = bspline_derivative(u_x, ORDER) * md_x
            dwy[k] = bspline_derivative(u_y, ORDER) * md_y
            dwz[k] = bspline_derivative(u_z, ORDER) * md_z
            d2wx[k] = bspline_second_derivative(u_x, ORDER) * md_x * md_x
            d2wy[k] = bspline_second_derivative(u_y, ORDER) * md_y * md_y
            d2wz[k] = bspline_second_derivative(u_z, ORDER) * md_z * md_z
            d3wx[k] = bspline_third_derivative(u_x, ORDER) * md_x * md_x * md_x
            d3wy[k] = bspline_third_derivative(u_y, ORDER) * md_y * md_y * md_y
            d3wz[k] = bspline_third_derivative(u_z, ORDER) * md_z * md_z * md_z
            d4wx[k] = bspline_fourth_derivative(u_x, ORDER) * md_x * md_x * md_x * md_x
            d4wy[k] = bspline_fourth_derivative(u_y, ORDER) * md_y * md_y * md_y * md_y
            d4wz[k] = bspline_fourth_derivative(u_z, ORDER) * md_z * md_z * md_z * md_z

        # Rank-3 readout (10 unique) and rank-4 readout (15 unique).
        a3_xxx = zero
        a3_yyy = zero
        a3_zzz = zero
        a3_xxy = zero
        a3_xxz = zero
        a3_xyy = zero
        a3_yyz = zero
        a3_xzz = zero
        a3_yzz = zero
        a3_xyz = zero
        a4_xxxx = zero
        a4_yyyy = zero
        a4_zzzz = zero
        a4_xxxy = zero
        a4_xxxz = zero
        a4_xyyy = zero
        a4_yyyz = zero
        a4_xzzz = zero
        a4_yzzz = zero
        a4_xxyy = zero
        a4_xxzz = zero
        a4_yyzz = zero
        a4_xxyz = zero
        a4_xyyz = zero
        a4_xyzz = zero

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            d2wxi = d2wx[i]
            d3wxi = d3wx[i]
            d4wxi = d4wx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wyj = wy[j]
                dwyj = dwy[j]
                d2wyj = d2wy[j]
                d3wyj = d3wy[j]
                d4wyj = d4wy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    gm = grad_mesh[gx, gy, gz]
                    wzk = wz[k]
                    dwzk = dwz[k]
                    d2wzk = d2wz[k]
                    d3wzk = d3wz[k]
                    d4wzk = d4wz[k]
                    # rank-3
                    a3_xxx += gm * d3wxi * wyj * wzk
                    a3_yyy += gm * wxi * d3wyj * wzk
                    a3_zzz += gm * wxi * wyj * d3wzk
                    a3_xxy += gm * d2wxi * dwyj * wzk
                    a3_xxz += gm * d2wxi * wyj * dwzk
                    a3_xyy += gm * dwxi * d2wyj * wzk
                    a3_yyz += gm * wxi * d2wyj * dwzk
                    a3_xzz += gm * dwxi * wyj * d2wzk
                    a3_yzz += gm * wxi * dwyj * d2wzk
                    a3_xyz += gm * dwxi * dwyj * dwzk
                    # rank-4
                    a4_xxxx += gm * d4wxi * wyj * wzk
                    a4_yyyy += gm * wxi * d4wyj * wzk
                    a4_zzzz += gm * wxi * wyj * d4wzk
                    a4_xxxy += gm * d3wxi * dwyj * wzk
                    a4_xxxz += gm * d3wxi * wyj * dwzk
                    a4_xyyy += gm * dwxi * d3wyj * wzk
                    a4_yyyz += gm * wxi * d3wyj * dwzk
                    a4_xzzz += gm * dwxi * wyj * d3wzk
                    a4_yzzz += gm * wxi * dwyj * d3wzk
                    a4_xxyy += gm * d2wxi * d2wyj * wzk
                    a4_xxzz += gm * d2wxi * wyj * d2wzk
                    a4_yyzz += gm * wxi * d2wyj * d2wzk
                    a4_xxyz += gm * d2wxi * dwyj * dwzk
                    a4_xyyz += gm * dwxi * d2wyj * dwzk
                    a4_xyzz += gm * dwxi * dwyj * d2wzk

        # Qe = M Q Mᵀ, g = M gg_pos.
        Q = quadrupoles[atom_idx]
        M = cell_inv_t[0]
        gpv = gg_pos[atom_idx]
        MQ_00 = M[0, 0] * Q[0, 0] + M[0, 1] * Q[1, 0] + M[0, 2] * Q[2, 0]
        MQ_01 = M[0, 0] * Q[0, 1] + M[0, 1] * Q[1, 1] + M[0, 2] * Q[2, 1]
        MQ_02 = M[0, 0] * Q[0, 2] + M[0, 1] * Q[1, 2] + M[0, 2] * Q[2, 2]
        MQ_10 = M[1, 0] * Q[0, 0] + M[1, 1] * Q[1, 0] + M[1, 2] * Q[2, 0]
        MQ_11 = M[1, 0] * Q[0, 1] + M[1, 1] * Q[1, 1] + M[1, 2] * Q[2, 1]
        MQ_12 = M[1, 0] * Q[0, 2] + M[1, 1] * Q[1, 2] + M[1, 2] * Q[2, 2]
        MQ_20 = M[2, 0] * Q[0, 0] + M[2, 1] * Q[1, 0] + M[2, 2] * Q[2, 0]
        MQ_21 = M[2, 0] * Q[0, 1] + M[2, 1] * Q[1, 1] + M[2, 2] * Q[2, 1]
        MQ_22 = M[2, 0] * Q[0, 2] + M[2, 1] * Q[1, 2] + M[2, 2] * Q[2, 2]
        Qe00 = MQ_00 * M[0, 0] + MQ_01 * M[0, 1] + MQ_02 * M[0, 2]
        Qe11 = MQ_10 * M[1, 0] + MQ_11 * M[1, 1] + MQ_12 * M[1, 2]
        Qe22 = MQ_20 * M[2, 0] + MQ_21 * M[2, 1] + MQ_22 * M[2, 2]
        Qe01 = MQ_00 * M[1, 0] + MQ_01 * M[1, 1] + MQ_02 * M[1, 2]
        Qe02 = MQ_00 * M[2, 0] + MQ_01 * M[2, 1] + MQ_02 * M[2, 2]
        Qe12 = MQ_10 * M[2, 0] + MQ_11 * M[2, 1] + MQ_12 * M[2, 2]
        g0 = M[0, 0] * gpv[0] + M[0, 1] * gpv[1] + M[0, 2] * gpv[2]
        g1 = M[1, 0] * gpv[0] + M[1, 1] * gpv[1] + M[1, 2] * gpv[2]
        g2 = M[2, 0] * gpv[0] + M[2, 1] * gpv[1] + M[2, 2] * gpv[2]
        half = type(t0)(0.5)
        two = type(t0)(2.0)

        # --- (c) grad_Q = ½ Mᵀ R M, R_ij = Σ_k g_k acc3[i,j,k] (symmetric) ---
        R00 = g0 * a3_xxx + g1 * a3_xxy + g2 * a3_xxz
        R11 = g0 * a3_xyy + g1 * a3_yyy + g2 * a3_yyz
        R22 = g0 * a3_xzz + g1 * a3_yzz + g2 * a3_zzz
        R01 = g0 * a3_xxy + g1 * a3_xyy + g2 * a3_xyz
        R02 = g0 * a3_xxz + g1 * a3_xyz + g2 * a3_xzz
        R12 = g0 * a3_xyz + g1 * a3_yyz + g2 * a3_yzz
        # MR = Mᵀ R  (Mᵀ_ai R_ij with Mᵀ_ai = M[i,a]); then (MᵀR)M.
        MR00 = M[0, 0] * R00 + M[1, 0] * R01 + M[2, 0] * R02
        MR01 = M[0, 0] * R01 + M[1, 0] * R11 + M[2, 0] * R12
        MR02 = M[0, 0] * R02 + M[1, 0] * R12 + M[2, 0] * R22
        MR10 = M[0, 1] * R00 + M[1, 1] * R01 + M[2, 1] * R02
        MR11 = M[0, 1] * R01 + M[1, 1] * R11 + M[2, 1] * R12
        MR12 = M[0, 1] * R02 + M[1, 1] * R12 + M[2, 1] * R22
        MR20 = M[0, 2] * R00 + M[1, 2] * R01 + M[2, 2] * R02
        MR21 = M[0, 2] * R01 + M[1, 2] * R11 + M[2, 2] * R12
        MR22 = M[0, 2] * R02 + M[1, 2] * R12 + M[2, 2] * R22
        gq00 = half * (MR00 * M[0, 0] + MR01 * M[1, 0] + MR02 * M[2, 0])
        gq11 = half * (MR10 * M[0, 1] + MR11 * M[1, 1] + MR12 * M[2, 1])
        gq22 = half * (MR20 * M[0, 2] + MR21 * M[1, 2] + MR22 * M[2, 2])
        gq01 = half * (MR00 * M[0, 1] + MR01 * M[1, 1] + MR02 * M[2, 1])
        gq02 = half * (MR00 * M[0, 2] + MR01 * M[1, 2] + MR02 * M[2, 2])
        gq12 = half * (MR10 * M[0, 2] + MR11 * M[1, 2] + MR12 * M[2, 2])
        wp.atomic_add(
            grad_quadrupoles,
            atom_idx,
            mat33_dtype(gq00, gq01, gq02, gq01, gq11, gq12, gq02, gq12, gq22),
        )

        # --- (b) grad_pos = ½ Mᵀ (P4 g), P4[k,l]=Σ_ij Qe_ij acc4[i,j,k,l] ---
        P00 = (
            Qe00 * a4_xxxx
            + Qe11 * a4_xxyy
            + Qe22 * a4_xxzz
            + two * (Qe01 * a4_xxxy + Qe02 * a4_xxxz + Qe12 * a4_xxyz)
        )
        P11 = (
            Qe00 * a4_xxyy
            + Qe11 * a4_yyyy
            + Qe22 * a4_yyzz
            + two * (Qe01 * a4_xyyy + Qe02 * a4_xyyz + Qe12 * a4_yyyz)
        )
        P22 = (
            Qe00 * a4_xxzz
            + Qe11 * a4_yyzz
            + Qe22 * a4_zzzz
            + two * (Qe01 * a4_xyzz + Qe02 * a4_xzzz + Qe12 * a4_yzzz)
        )
        P01 = (
            Qe00 * a4_xxxy
            + Qe11 * a4_xyyy
            + Qe22 * a4_xyzz
            + two * (Qe01 * a4_xxyy + Qe02 * a4_xxyz + Qe12 * a4_xyyz)
        )
        P02 = (
            Qe00 * a4_xxxz
            + Qe11 * a4_xyyz
            + Qe22 * a4_xzzz
            + two * (Qe01 * a4_xxyz + Qe02 * a4_xxzz + Qe12 * a4_xyzz)
        )
        P12 = (
            Qe00 * a4_xxyz
            + Qe11 * a4_yyyz
            + Qe22 * a4_yzzz
            + two * (Qe01 * a4_xyyz + Qe02 * a4_xyzz + Qe12 * a4_yyzz)
        )
        W0 = P00 * g0 + P01 * g1 + P02 * g2
        W1 = P01 * g0 + P11 * g1 + P12 * g2
        W2 = P02 * g0 + P12 * g1 + P22 * g2
        gpx = half * (M[0, 0] * W0 + M[1, 0] * W1 + M[2, 0] * W2)
        gpy = half * (M[0, 1] * W0 + M[1, 1] * W1 + M[2, 1] * W2)
        gpz = half * (M[0, 2] * W0 + M[1, 2] * W1 + M[2, 2] * W2)
        wp.atomic_add(grad_positions, atom_idx, vec_pos_dtype(gpx, gpy, gpz))

    return kernel


_PER_ORDER_OCTUPOLE_BACKWARD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_pme_octupole_backward_kernel(_order, _scalar, _vec, _mat)
        _PER_ORDER_OCTUPOLE_BACKWARD_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),  # positions
                wp.array(dtype=_mat),  # quadrupoles
                wp.array(dtype=_vec),  # gg_pos
                wp.array(dtype=_mat),  # cell_inv_t
                wp.array3d(dtype=_scalar),  # grad_mesh
                wp.array(dtype=_vec),  # grad_positions (out)
                wp.array(dtype=_mat),  # grad_quadrupoles (out)
            ],
        )


def multipole_pme_octupole_backward_launch(
    positions,
    quadrupoles,
    gg_pos,
    cell_inv_t,
    grad_mesh,
    grad_positions,
    grad_quadrupoles,
    order,
    wp_dtype,
    device=None,
):
    r"""Launch the octupole backward kernel (l_max=2 double-back).

    Backward of :func:`multipole_pme_octupole_spread_launch`: emits the
    octupole VJP slots for positions (∇⁴ term) and quadrupoles (∇³ term).

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=vec3f/vec3d
        Cartesian atom positions.
    quadrupoles : wp.array, shape (N,), dtype=mat33f/mat33d
        Per-atom Cartesian quadrupole tensors.
    gg_pos : wp.array, shape (N,), dtype=vec3f/vec3d
        Incoming position-gradient direction (:math:`g = M\,\text{gg\_pos}`).
    cell_inv_t : wp.array, shape (1,), dtype=mat33f/mat33d
        Transpose of the inverse cell matrix :math:`M`.
    grad_mesh : wp.array, shape (Nx, Ny, Nz), dtype=wp.float32/float64
        Upstream gradient w.r.t. the octupole spread mesh.
    grad_positions : wp.array, shape (N,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Octupole ``∂L/∂r`` contribution (atomic add).
    grad_quadrupoles : wp.array, shape (N,), dtype=mat33f/mat33d
        OUTPUT, pre-zeroed. Octupole ``∂L/∂Q`` contribution (atomic add).
    order : int
        B-spline order.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Returns
    -------
    bool
        ``True`` if the per-order kernel ran, ``False`` on cache miss.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    per_order = _PER_ORDER_OCTUPOLE_BACKWARD_KERNELS.get(wp_dtype, {}).get(order)
    if per_order is None:
        return False
    if device is None:
        device = str(positions.device)
    wp.launch(
        per_order,
        dim=(positions.shape[0],),
        inputs=[positions, quadrupoles, gg_pos, cell_inv_t, grad_mesh],
        outputs=[grad_positions, grad_quadrupoles],
        device=device,
    )
    return True


# =============================================================================
# Batched octupole (∇³ / ∇⁴) kernels — batched spread double-back
# =============================================================================
# ``sys_idx = batch_idx[atom]`` selects the per-system grid slice +
# ``cell_inv_t[sys_idx]``.


def _make_batch_pme_octupole_spread_kernel(
    ORDER, scalar_dtype, vec_pos_dtype, mat33_dtype
):
    """Batched octupole (∇³) spread."""
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_module(
            "batch_octupole_spread", ORDER, scalar_dtype
        )
    )
    def kernel(
        positions: wp.array(dtype=Any),
        quadrupoles: wp.array(dtype=Any),
        gg_pos: wp.array(dtype=Any),
        batch_idx: wp.array(dtype=wp.int32),
        cell_inv_t: wp.array(dtype=Any),
        mesh: wp.array(dtype=Any, ndim=4),
    ):
        atom_idx = wp.tid()
        sys_idx = batch_idx[atom_idx]
        mesh_dims = wp.vec3i(mesh.shape[1], mesh.shape[2], mesh.shape[3])
        position = positions[atom_idx]
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
        dwx = vec_ord()
        dwy = vec_ord()
        dwz = vec_ord()
        d2wx = vec_ord()
        d2wy = vec_ord()
        d2wz = vec_ord()
        d3wx = vec_ord()
        d3wy = vec_ord()
        d3wz = vec_ord()
        md_x = type(t0)(mesh_dims[0])
        md_y = type(t0)(mesh_dims[1])
        md_z = type(t0)(mesh_dims[2])
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            dwx[k] = bspline_derivative(u_x, ORDER) * md_x
            dwy[k] = bspline_derivative(u_y, ORDER) * md_y
            dwz[k] = bspline_derivative(u_z, ORDER) * md_z
            d2wx[k] = bspline_second_derivative(u_x, ORDER) * md_x * md_x
            d2wy[k] = bspline_second_derivative(u_y, ORDER) * md_y * md_y
            d2wz[k] = bspline_second_derivative(u_z, ORDER) * md_z * md_z
            d3wx[k] = bspline_third_derivative(u_x, ORDER) * md_x * md_x * md_x
            d3wy[k] = bspline_third_derivative(u_y, ORDER) * md_y * md_y * md_y
            d3wz[k] = bspline_third_derivative(u_z, ORDER) * md_z * md_z * md_z

        Q = quadrupoles[atom_idx]
        M = cell_inv_t[sys_idx]
        gp = gg_pos[atom_idx]
        MQ_00 = M[0, 0] * Q[0, 0] + M[0, 1] * Q[1, 0] + M[0, 2] * Q[2, 0]
        MQ_01 = M[0, 0] * Q[0, 1] + M[0, 1] * Q[1, 1] + M[0, 2] * Q[2, 1]
        MQ_02 = M[0, 0] * Q[0, 2] + M[0, 1] * Q[1, 2] + M[0, 2] * Q[2, 2]
        MQ_10 = M[1, 0] * Q[0, 0] + M[1, 1] * Q[1, 0] + M[1, 2] * Q[2, 0]
        MQ_11 = M[1, 0] * Q[0, 1] + M[1, 1] * Q[1, 1] + M[1, 2] * Q[2, 1]
        MQ_12 = M[1, 0] * Q[0, 2] + M[1, 1] * Q[1, 2] + M[1, 2] * Q[2, 2]
        MQ_20 = M[2, 0] * Q[0, 0] + M[2, 1] * Q[1, 0] + M[2, 2] * Q[2, 0]
        MQ_21 = M[2, 0] * Q[0, 1] + M[2, 1] * Q[1, 1] + M[2, 2] * Q[2, 1]
        MQ_22 = M[2, 0] * Q[0, 2] + M[2, 1] * Q[1, 2] + M[2, 2] * Q[2, 2]
        Qe00 = MQ_00 * M[0, 0] + MQ_01 * M[0, 1] + MQ_02 * M[0, 2]
        Qe11 = MQ_10 * M[1, 0] + MQ_11 * M[1, 1] + MQ_12 * M[1, 2]
        Qe22 = MQ_20 * M[2, 0] + MQ_21 * M[2, 1] + MQ_22 * M[2, 2]
        Qe01 = MQ_00 * M[1, 0] + MQ_01 * M[1, 1] + MQ_02 * M[1, 2]
        Qe02 = MQ_00 * M[2, 0] + MQ_01 * M[2, 1] + MQ_02 * M[2, 2]
        Qe12 = MQ_10 * M[2, 0] + MQ_11 * M[2, 1] + MQ_12 * M[2, 2]
        g0 = M[0, 0] * gp[0] + M[0, 1] * gp[1] + M[0, 2] * gp[2]
        g1 = M[1, 0] * gp[0] + M[1, 1] * gp[1] + M[1, 2] * gp[2]
        g2 = M[2, 0] * gp[0] + M[2, 1] * gp[1] + M[2, 2] * gp[2]

        half = type(t0)(0.5)
        two = type(t0)(2.0)
        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            d2wxi = d2wx[i]
            d3wxi = d3wx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wyj = wy[j]
                dwyj = dwy[j]
                d2wyj = d2wy[j]
                d3wyj = d3wy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    wzk = wz[k]
                    dwzk = dwz[k]
                    d2wzk = d2wz[k]
                    d3wzk = d3wz[k]
                    B_xxx = d3wxi * wyj * wzk
                    B_yyy = wxi * d3wyj * wzk
                    B_zzz = wxi * wyj * d3wzk
                    B_xxy = d2wxi * dwyj * wzk
                    B_xxz = d2wxi * wyj * dwzk
                    B_xyy = dwxi * d2wyj * wzk
                    B_yyz = wxi * d2wyj * dwzk
                    B_xzz = dwxi * wyj * d2wzk
                    B_yzz = wxi * dwyj * d2wzk
                    B_xyz = dwxi * dwyj * dwzk
                    Sx = (
                        Qe00 * B_xxx
                        + Qe11 * B_xyy
                        + Qe22 * B_xzz
                        + two * (Qe01 * B_xxy + Qe02 * B_xxz + Qe12 * B_xyz)
                    )
                    Sy = (
                        Qe00 * B_xxy
                        + Qe11 * B_yyy
                        + Qe22 * B_yzz
                        + two * (Qe01 * B_xyy + Qe02 * B_xyz + Qe12 * B_yyz)
                    )
                    Sz = (
                        Qe00 * B_xxz
                        + Qe11 * B_yyz
                        + Qe22 * B_zzz
                        + two * (Qe01 * B_xyz + Qe02 * B_xzz + Qe12 * B_yzz)
                    )
                    contrib = half * (g0 * Sx + g1 * Sy + g2 * Sz)
                    wp.atomic_add(mesh, sys_idx, gx, gy, gz, contrib)

    return kernel


_PER_ORDER_BATCH_OCTUPOLE_SPREAD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_batch_pme_octupole_spread_kernel(_order, _scalar, _vec, _mat)
        _PER_ORDER_BATCH_OCTUPOLE_SPREAD_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),  # positions
                wp.array(dtype=_mat),  # quadrupoles
                wp.array(dtype=_vec),  # gg_pos
                wp.array(dtype=wp.int32),  # batch_idx
                wp.array(dtype=_mat),  # cell_inv_t (B, 3, 3)
                wp.array(dtype=_scalar, ndim=4),  # mesh (B, nx, ny, nz)
            ],
        )


def batch_multipole_pme_octupole_spread_launch(
    positions,
    quadrupoles,
    gg_pos,
    batch_idx,
    cell_inv_t,
    order,
    mesh,
    wp_dtype,
    device=None,
):
    r"""Launch the batched octupole (∇³) spread kernel.

    Batched analog of :func:`multipole_pme_octupole_spread_launch`;
    ``sys_idx = batch_idx[atom]`` selects the per-system mesh slice and
    ``cell_inv_t[sys_idx]``.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=vec3f/vec3d
        Concatenated Cartesian atom positions.
    quadrupoles : wp.array, shape (N_total,), dtype=mat33f/mat33d
        Per-atom Cartesian quadrupole tensors.
    gg_pos : wp.array, shape (N_total,), dtype=vec3f/vec3d
        Incoming position-gradient direction.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        Per-atom system index into the leading mesh axis.
    cell_inv_t : wp.array, shape (B,), dtype=mat33f/mat33d
        Per-system transpose of the inverse cell matrix.
    order : int
        B-spline order.
    mesh : wp.array, shape (B, Nx, Ny, Nz), dtype=wp.float32/float64
        OUTPUT. Per-system density mesh, accumulated via atomics.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Returns
    -------
    bool
        ``True`` if the per-order kernel ran, ``False`` on cache miss.

    Launch Grid
    -----------
    ``dim = (N_total,)`` — one thread per atom across the batch.
    """
    per_order = _PER_ORDER_BATCH_OCTUPOLE_SPREAD_KERNELS.get(wp_dtype, {}).get(order)
    if per_order is None:
        return False
    if device is None:
        device = str(positions.device)
    wp.launch(
        per_order,
        dim=(positions.shape[0],),
        inputs=[positions, quadrupoles, gg_pos, batch_idx, cell_inv_t],
        outputs=[mesh],
        device=device,
    )
    return True


def _make_batch_pme_octupole_backward_kernel(
    ORDER, scalar_dtype, vec_pos_dtype, mat33_dtype
):
    """Batched octupole backward (∇⁴ grad_positions + ∇³ grad_quadrupoles)."""
    HALF_ORDER_PY = float(ORDER) * 0.5
    HALF_N_MINUS_2_PY = float(ORDER - 2) * 0.5
    vec_ord = _PER_ORDER_VEC[(ORDER, scalar_dtype)]

    @wp.kernel(
        module=_pme_multipole_per_order_module(
            "batch_octupole_backward", ORDER, scalar_dtype
        )
    )
    def kernel(
        positions: wp.array(dtype=Any),
        quadrupoles: wp.array(dtype=Any),
        gg_pos: wp.array(dtype=Any),
        batch_idx: wp.array(dtype=wp.int32),
        cell_inv_t: wp.array(dtype=Any),
        grad_mesh: wp.array(dtype=Any, ndim=4),
        grad_positions: wp.array(dtype=Any),
        grad_quadrupoles: wp.array(dtype=Any),
    ):
        atom_idx = wp.tid()
        sys_idx = batch_idx[atom_idx]
        mesh_dims = wp.vec3i(grad_mesh.shape[1], grad_mesh.shape[2], grad_mesh.shape[3])
        position = positions[atom_idx]
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

        wx = vec_ord()
        wy = vec_ord()
        wz = vec_ord()
        dwx = vec_ord()
        dwy = vec_ord()
        dwz = vec_ord()
        d2wx = vec_ord()
        d2wy = vec_ord()
        d2wz = vec_ord()
        d3wx = vec_ord()
        d3wy = vec_ord()
        d3wz = vec_ord()
        d4wx = vec_ord()
        d4wy = vec_ord()
        d4wz = vec_ord()
        md_x = type(t0)(mesh_dims[0])
        md_y = type(t0)(mesh_dims[1])
        md_z = type(t0)(mesh_dims[2])
        for k in range(ORDER):
            u_x = half_order + theta[0] - type(t0)(offset_start_x + k)
            u_y = half_order + theta[1] - type(t0)(offset_start_y + k)
            u_z = half_order + theta[2] - type(t0)(offset_start_z + k)
            wx[k] = bspline_weight(u_x, ORDER)
            wy[k] = bspline_weight(u_y, ORDER)
            wz[k] = bspline_weight(u_z, ORDER)
            dwx[k] = bspline_derivative(u_x, ORDER) * md_x
            dwy[k] = bspline_derivative(u_y, ORDER) * md_y
            dwz[k] = bspline_derivative(u_z, ORDER) * md_z
            d2wx[k] = bspline_second_derivative(u_x, ORDER) * md_x * md_x
            d2wy[k] = bspline_second_derivative(u_y, ORDER) * md_y * md_y
            d2wz[k] = bspline_second_derivative(u_z, ORDER) * md_z * md_z
            d3wx[k] = bspline_third_derivative(u_x, ORDER) * md_x * md_x * md_x
            d3wy[k] = bspline_third_derivative(u_y, ORDER) * md_y * md_y * md_y
            d3wz[k] = bspline_third_derivative(u_z, ORDER) * md_z * md_z * md_z
            d4wx[k] = bspline_fourth_derivative(u_x, ORDER) * md_x * md_x * md_x * md_x
            d4wy[k] = bspline_fourth_derivative(u_y, ORDER) * md_y * md_y * md_y * md_y
            d4wz[k] = bspline_fourth_derivative(u_z, ORDER) * md_z * md_z * md_z * md_z

        a3_xxx = zero
        a3_yyy = zero
        a3_zzz = zero
        a3_xxy = zero
        a3_xxz = zero
        a3_xyy = zero
        a3_yyz = zero
        a3_xzz = zero
        a3_yzz = zero
        a3_xyz = zero
        a4_xxxx = zero
        a4_yyyy = zero
        a4_zzzz = zero
        a4_xxxy = zero
        a4_xxxz = zero
        a4_xyyy = zero
        a4_yyyz = zero
        a4_xzzz = zero
        a4_yzzz = zero
        a4_xxyy = zero
        a4_xxzz = zero
        a4_yyzz = zero
        a4_xxyz = zero
        a4_xyyz = zero
        a4_xyzz = zero

        for i in range(ORDER):
            gx = wrap_grid_index(base_grid[0] + offset_start_x + i, mesh_dims[0])
            wxi = wx[i]
            dwxi = dwx[i]
            d2wxi = d2wx[i]
            d3wxi = d3wx[i]
            d4wxi = d4wx[i]
            for j in range(ORDER):
                gy = wrap_grid_index(base_grid[1] + offset_start_y + j, mesh_dims[1])
                wyj = wy[j]
                dwyj = dwy[j]
                d2wyj = d2wy[j]
                d3wyj = d3wy[j]
                d4wyj = d4wy[j]
                for k in range(ORDER):
                    gz = wrap_grid_index(
                        base_grid[2] + offset_start_z + k, mesh_dims[2]
                    )
                    gm = grad_mesh[sys_idx, gx, gy, gz]
                    wzk = wz[k]
                    dwzk = dwz[k]
                    d2wzk = d2wz[k]
                    d3wzk = d3wz[k]
                    d4wzk = d4wz[k]
                    a3_xxx += gm * d3wxi * wyj * wzk
                    a3_yyy += gm * wxi * d3wyj * wzk
                    a3_zzz += gm * wxi * wyj * d3wzk
                    a3_xxy += gm * d2wxi * dwyj * wzk
                    a3_xxz += gm * d2wxi * wyj * dwzk
                    a3_xyy += gm * dwxi * d2wyj * wzk
                    a3_yyz += gm * wxi * d2wyj * dwzk
                    a3_xzz += gm * dwxi * wyj * d2wzk
                    a3_yzz += gm * wxi * dwyj * d2wzk
                    a3_xyz += gm * dwxi * dwyj * dwzk
                    a4_xxxx += gm * d4wxi * wyj * wzk
                    a4_yyyy += gm * wxi * d4wyj * wzk
                    a4_zzzz += gm * wxi * wyj * d4wzk
                    a4_xxxy += gm * d3wxi * dwyj * wzk
                    a4_xxxz += gm * d3wxi * wyj * dwzk
                    a4_xyyy += gm * dwxi * d3wyj * wzk
                    a4_yyyz += gm * wxi * d3wyj * dwzk
                    a4_xzzz += gm * dwxi * wyj * d3wzk
                    a4_yzzz += gm * wxi * dwyj * d3wzk
                    a4_xxyy += gm * d2wxi * d2wyj * wzk
                    a4_xxzz += gm * d2wxi * wyj * d2wzk
                    a4_yyzz += gm * wxi * d2wyj * d2wzk
                    a4_xxyz += gm * d2wxi * dwyj * dwzk
                    a4_xyyz += gm * dwxi * d2wyj * dwzk
                    a4_xyzz += gm * dwxi * dwyj * d2wzk

        Q = quadrupoles[atom_idx]
        M = cell_inv_t[sys_idx]
        gpv = gg_pos[atom_idx]
        MQ_00 = M[0, 0] * Q[0, 0] + M[0, 1] * Q[1, 0] + M[0, 2] * Q[2, 0]
        MQ_01 = M[0, 0] * Q[0, 1] + M[0, 1] * Q[1, 1] + M[0, 2] * Q[2, 1]
        MQ_02 = M[0, 0] * Q[0, 2] + M[0, 1] * Q[1, 2] + M[0, 2] * Q[2, 2]
        MQ_10 = M[1, 0] * Q[0, 0] + M[1, 1] * Q[1, 0] + M[1, 2] * Q[2, 0]
        MQ_11 = M[1, 0] * Q[0, 1] + M[1, 1] * Q[1, 1] + M[1, 2] * Q[2, 1]
        MQ_12 = M[1, 0] * Q[0, 2] + M[1, 1] * Q[1, 2] + M[1, 2] * Q[2, 2]
        MQ_20 = M[2, 0] * Q[0, 0] + M[2, 1] * Q[1, 0] + M[2, 2] * Q[2, 0]
        MQ_21 = M[2, 0] * Q[0, 1] + M[2, 1] * Q[1, 1] + M[2, 2] * Q[2, 1]
        MQ_22 = M[2, 0] * Q[0, 2] + M[2, 1] * Q[1, 2] + M[2, 2] * Q[2, 2]
        Qe00 = MQ_00 * M[0, 0] + MQ_01 * M[0, 1] + MQ_02 * M[0, 2]
        Qe11 = MQ_10 * M[1, 0] + MQ_11 * M[1, 1] + MQ_12 * M[1, 2]
        Qe22 = MQ_20 * M[2, 0] + MQ_21 * M[2, 1] + MQ_22 * M[2, 2]
        Qe01 = MQ_00 * M[1, 0] + MQ_01 * M[1, 1] + MQ_02 * M[1, 2]
        Qe02 = MQ_00 * M[2, 0] + MQ_01 * M[2, 1] + MQ_02 * M[2, 2]
        Qe12 = MQ_10 * M[2, 0] + MQ_11 * M[2, 1] + MQ_12 * M[2, 2]
        g0 = M[0, 0] * gpv[0] + M[0, 1] * gpv[1] + M[0, 2] * gpv[2]
        g1 = M[1, 0] * gpv[0] + M[1, 1] * gpv[1] + M[1, 2] * gpv[2]
        g2 = M[2, 0] * gpv[0] + M[2, 1] * gpv[1] + M[2, 2] * gpv[2]
        half = type(t0)(0.5)
        two = type(t0)(2.0)

        R00 = g0 * a3_xxx + g1 * a3_xxy + g2 * a3_xxz
        R11 = g0 * a3_xyy + g1 * a3_yyy + g2 * a3_yyz
        R22 = g0 * a3_xzz + g1 * a3_yzz + g2 * a3_zzz
        R01 = g0 * a3_xxy + g1 * a3_xyy + g2 * a3_xyz
        R02 = g0 * a3_xxz + g1 * a3_xyz + g2 * a3_xzz
        R12 = g0 * a3_xyz + g1 * a3_yyz + g2 * a3_yzz
        MR00 = M[0, 0] * R00 + M[1, 0] * R01 + M[2, 0] * R02
        MR01 = M[0, 0] * R01 + M[1, 0] * R11 + M[2, 0] * R12
        MR02 = M[0, 0] * R02 + M[1, 0] * R12 + M[2, 0] * R22
        MR10 = M[0, 1] * R00 + M[1, 1] * R01 + M[2, 1] * R02
        MR11 = M[0, 1] * R01 + M[1, 1] * R11 + M[2, 1] * R12
        MR12 = M[0, 1] * R02 + M[1, 1] * R12 + M[2, 1] * R22
        MR20 = M[0, 2] * R00 + M[1, 2] * R01 + M[2, 2] * R02
        MR21 = M[0, 2] * R01 + M[1, 2] * R11 + M[2, 2] * R12
        MR22 = M[0, 2] * R02 + M[1, 2] * R12 + M[2, 2] * R22
        gq00 = half * (MR00 * M[0, 0] + MR01 * M[1, 0] + MR02 * M[2, 0])
        gq11 = half * (MR10 * M[0, 1] + MR11 * M[1, 1] + MR12 * M[2, 1])
        gq22 = half * (MR20 * M[0, 2] + MR21 * M[1, 2] + MR22 * M[2, 2])
        gq01 = half * (MR00 * M[0, 1] + MR01 * M[1, 1] + MR02 * M[2, 1])
        gq02 = half * (MR00 * M[0, 2] + MR01 * M[1, 2] + MR02 * M[2, 2])
        gq12 = half * (MR10 * M[0, 2] + MR11 * M[1, 2] + MR12 * M[2, 2])
        wp.atomic_add(
            grad_quadrupoles,
            atom_idx,
            mat33_dtype(gq00, gq01, gq02, gq01, gq11, gq12, gq02, gq12, gq22),
        )

        P00 = (
            Qe00 * a4_xxxx
            + Qe11 * a4_xxyy
            + Qe22 * a4_xxzz
            + two * (Qe01 * a4_xxxy + Qe02 * a4_xxxz + Qe12 * a4_xxyz)
        )
        P11 = (
            Qe00 * a4_xxyy
            + Qe11 * a4_yyyy
            + Qe22 * a4_yyzz
            + two * (Qe01 * a4_xyyy + Qe02 * a4_xyyz + Qe12 * a4_yyyz)
        )
        P22 = (
            Qe00 * a4_xxzz
            + Qe11 * a4_yyzz
            + Qe22 * a4_zzzz
            + two * (Qe01 * a4_xyzz + Qe02 * a4_xzzz + Qe12 * a4_yzzz)
        )
        P01 = (
            Qe00 * a4_xxxy
            + Qe11 * a4_xyyy
            + Qe22 * a4_xyzz
            + two * (Qe01 * a4_xxyy + Qe02 * a4_xxyz + Qe12 * a4_xyyz)
        )
        P02 = (
            Qe00 * a4_xxxz
            + Qe11 * a4_xyyz
            + Qe22 * a4_xzzz
            + two * (Qe01 * a4_xxyz + Qe02 * a4_xxzz + Qe12 * a4_xyzz)
        )
        P12 = (
            Qe00 * a4_xxyz
            + Qe11 * a4_yyyz
            + Qe22 * a4_yzzz
            + two * (Qe01 * a4_xyyz + Qe02 * a4_xyzz + Qe12 * a4_yyzz)
        )
        W0 = P00 * g0 + P01 * g1 + P02 * g2
        W1 = P01 * g0 + P11 * g1 + P12 * g2
        W2 = P02 * g0 + P12 * g1 + P22 * g2
        gpx = half * (M[0, 0] * W0 + M[1, 0] * W1 + M[2, 0] * W2)
        gpy = half * (M[0, 1] * W0 + M[1, 1] * W1 + M[2, 1] * W2)
        gpz = half * (M[0, 2] * W0 + M[1, 2] * W1 + M[2, 2] * W2)
        wp.atomic_add(grad_positions, atom_idx, vec_pos_dtype(gpx, gpy, gpz))

    return kernel


_PER_ORDER_BATCH_OCTUPOLE_BACKWARD_KERNELS: dict = {wp.float32: {}, wp.float64: {}}
for _order in _PER_ORDER_SUPPORTED:
    for _scalar, _vec, _mat in (
        (wp.float32, wp.vec3f, wp.mat33f),
        (wp.float64, wp.vec3d, wp.mat33d),
    ):
        _k = _make_batch_pme_octupole_backward_kernel(_order, _scalar, _vec, _mat)
        _PER_ORDER_BATCH_OCTUPOLE_BACKWARD_KERNELS[_scalar][_order] = wp.overload(
            _k,
            [
                wp.array(dtype=_vec),  # positions
                wp.array(dtype=_mat),  # quadrupoles
                wp.array(dtype=_vec),  # gg_pos
                wp.array(dtype=wp.int32),  # batch_idx
                wp.array(dtype=_mat),  # cell_inv_t (B, 3, 3)
                wp.array(dtype=_scalar, ndim=4),  # grad_mesh (B, nx, ny, nz)
                wp.array(dtype=_vec),  # grad_positions (out)
                wp.array(dtype=_mat),  # grad_quadrupoles (out)
            ],
        )


def batch_multipole_pme_octupole_backward_launch(
    positions,
    quadrupoles,
    gg_pos,
    batch_idx,
    cell_inv_t,
    grad_mesh,
    grad_positions,
    grad_quadrupoles,
    order,
    wp_dtype,
    device=None,
):
    r"""Launch the batched octupole backward kernel (l_max=2 double-back).

    Batched analog of :func:`multipole_pme_octupole_backward_launch`.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=vec3f/vec3d
        Concatenated Cartesian atom positions.
    quadrupoles : wp.array, shape (N_total,), dtype=mat33f/mat33d
        Per-atom Cartesian quadrupole tensors.
    gg_pos : wp.array, shape (N_total,), dtype=vec3f/vec3d
        Incoming position-gradient direction.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        Per-atom system index into the leading mesh axis.
    cell_inv_t : wp.array, shape (B,), dtype=mat33f/mat33d
        Per-system transpose of the inverse cell matrix.
    grad_mesh : wp.array, shape (B, Nx, Ny, Nz), dtype=wp.float32/float64
        Upstream gradient w.r.t. the per-system octupole spread mesh.
    grad_positions : wp.array, shape (N_total,), dtype=vec3f/vec3d
        OUTPUT, pre-zeroed. Octupole ``∂L/∂r`` contribution (atomic add).
    grad_quadrupoles : wp.array, shape (N_total,), dtype=mat33f/mat33d
        OUTPUT, pre-zeroed. Octupole ``∂L/∂Q`` contribution (atomic add).
    order : int
        B-spline order.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64`` — selects the registered overload.
    device : str, optional
        Warp device string. Defaults to ``positions.device``.

    Returns
    -------
    bool
        ``True`` if the per-order kernel ran, ``False`` on cache miss.

    Launch Grid
    -----------
    ``dim = (N_total,)`` — one thread per atom across the batch.
    """
    per_order = _PER_ORDER_BATCH_OCTUPOLE_BACKWARD_KERNELS.get(wp_dtype, {}).get(order)
    if per_order is None:
        return False
    if device is None:
        device = str(positions.device)
    wp.launch(
        per_order,
        dim=(positions.shape[0],),
        inputs=[positions, quadrupoles, gg_pos, batch_idx, cell_inv_t, grad_mesh],
        outputs=[grad_positions, grad_quadrupoles],
        device=device,
    )
    return True


# =============================================================================
# GTO-Ewald multipole self + background energy corrections (per-atom)
# =============================================================================
#
# Per-atom correction (subtracted from the raw reciprocal PME energy):
#
#   corr_i = c_self_q · q_i²                       (l = 0 self)
#          + c_self_mu · |μ_i|²                    (l = 1 self, optional)
#          + c_self_Q · |Q_i|_F²                   (l = 2 self, optional)
#          + (c_bg_no_v / V) · q_i · Q_total       (background share)
#
# where Σ_i (c_bg_no_v / V) · q_i · Q_total = (c_bg_no_v / V) · Q_total².


@wp.kernel(enable_backward=False)
def _pme_multipole_corrections_kernel(
    charges: wp.array(dtype=Any),  # (N,)
    dipoles: wp.array(dtype=Any),  # (N,) of vec3
    quadrupoles: wp.array(dtype=Any),  # (N,) of mat33
    volume: wp.array(dtype=Any),  # (1,)
    total_charge: wp.array(dtype=Any),  # (1,)
    c_self_q: wp.array(dtype=Any),  # (1,)
    c_self_mu: wp.array(dtype=Any),  # (1,)
    c_self_q2: wp.array(dtype=Any),  # (1,)  -- l=2 Frobenius coeff
    c_bg_no_v: wp.array(dtype=Any),  # (1,)
    has_dipoles: wp.int32,
    has_quadrupoles: wp.int32,
    corrections: wp.array(dtype=Any),  # (N,) OUTPUT
):
    r"""Single-system per-atom GTO-Ewald self + background correction.

    Each thread computes one atom's correction term; the torch wrapper
    reduces with ``.sum()``. See module-level note for the formula.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    i = wp.tid()
    q = charges[i]

    corr = c_self_q[0] * q * q

    if has_dipoles != 0:
        mu = dipoles[i]
        corr += c_self_mu[0] * (mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2])

    if has_quadrupoles != 0:
        Q = quadrupoles[i]
        q_fro = (
            Q[0, 0] * Q[0, 0]
            + Q[0, 1] * Q[0, 1]
            + Q[0, 2] * Q[0, 2]
            + Q[1, 0] * Q[1, 0]
            + Q[1, 1] * Q[1, 1]
            + Q[1, 2] * Q[1, 2]
            + Q[2, 0] * Q[2, 0]
            + Q[2, 1] * Q[2, 1]
            + Q[2, 2] * Q[2, 2]
        )
        corr += c_self_q2[0] * q_fro

    # Background share: (c_bg_no_v / V) · q_i · Q_total.
    corr += (c_bg_no_v[0] / volume[0]) * q * total_charge[0]

    corrections[i] = corr


@wp.kernel(enable_backward=False)
def _batch_pme_multipole_corrections_kernel(
    charges: wp.array(dtype=Any),  # (N_total,)
    dipoles: wp.array(dtype=Any),  # (N_total,) of vec3
    quadrupoles: wp.array(dtype=Any),  # (N_total,) of mat33
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    volumes: wp.array(dtype=Any),  # (B,)
    total_charges: wp.array(dtype=Any),  # (B,)
    c_self_q: wp.array(dtype=Any),  # (1,)
    c_self_mu: wp.array(dtype=Any),  # (1,)
    c_self_q2: wp.array(dtype=Any),  # (1,)
    c_bg_no_v: wp.array(dtype=Any),  # (1,)
    has_dipoles: wp.int32,
    has_quadrupoles: wp.int32,
    corrections: wp.array(dtype=Any),  # (N_total,) OUTPUT
):
    r"""Batched per-atom GTO-Ewald self + background correction.

    Per-system ``volume`` / ``total_charge`` are looked up via
    ``batch_idx``; the torch wrapper reduces with ``scatter_add``.

    Launch Grid
    -----------
    ``dim = (N_total,)`` — one thread per atom across the batch.
    """
    i = wp.tid()
    s = batch_idx[i]
    q = charges[i]

    corr = c_self_q[0] * q * q

    if has_dipoles != 0:
        mu = dipoles[i]
        corr += c_self_mu[0] * (mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2])

    if has_quadrupoles != 0:
        Q = quadrupoles[i]
        q_fro = (
            Q[0, 0] * Q[0, 0]
            + Q[0, 1] * Q[0, 1]
            + Q[0, 2] * Q[0, 2]
            + Q[1, 0] * Q[1, 0]
            + Q[1, 1] * Q[1, 1]
            + Q[1, 2] * Q[1, 2]
            + Q[2, 0] * Q[2, 0]
            + Q[2, 1] * Q[2, 1]
            + Q[2, 2] * Q[2, 2]
        )
        corr += c_self_q2[0] * q_fro

    corr += (c_bg_no_v[0] / volumes[s]) * q * total_charges[s]

    corrections[i] = corr


@wp.kernel(enable_backward=False)
def _pme_multipole_corrections_backward_kernel(
    grad_out: wp.array(dtype=Any),  # (1,) upstream scalar grad
    charges: wp.array(dtype=Any),  # (N,)
    dipoles: wp.array(dtype=Any),  # (N,) of vec3
    quadrupoles: wp.array(dtype=Any),  # (N,) of mat33
    volume: wp.array(dtype=Any),  # (1,)
    total_charge: wp.array(dtype=Any),  # (1,)
    c_self_q: wp.array(dtype=Any),  # (1,)
    c_self_mu: wp.array(dtype=Any),  # (1,)
    c_self_q2: wp.array(dtype=Any),  # (1,)
    c_bg_no_v: wp.array(dtype=Any),  # (1,)
    has_dipoles: wp.int32,
    has_quadrupoles: wp.int32,
    grad_charges: wp.array(dtype=Any),  # (N,) OUTPUT
    grad_dipoles: wp.array(dtype=Any),  # (N,) of vec3 OUTPUT
    grad_quadrupoles: wp.array(dtype=Any),  # (N,) of mat33 OUTPUT
    grad_volume: wp.array(dtype=Any),  # (1,) OUTPUT (atomic accumulate)
):
    r"""Single-system backward of :func:`_pme_multipole_corrections_kernel`.

    Given upstream scalar cotangent ``g`` (the gradient of the reduced
    ``corrections.sum()``):

    .. math::

        \partial L/\partial q_i &= g\,[2\,c_q\,q_i + (c_{bg}/V)\,Q_\text{tot}], \\
        \partial L/\partial \mu_i &= g\,2\,c_\mu\,\mu_i, \\
        \partial L/\partial Q_i &= g\,2\,c_{Q}\,Q_i, \\
        \partial L/\partial V &= -g\,(c_{bg}/V^2)\,Q_\text{tot}^2 .

    The ``q_i·Q_total`` background term contributes to ``∂L/∂q_j`` through
    BOTH the explicit ``q_i`` and the shared ``Q_total = Σ_k q_k``; summed
    over atoms this yields the symmetric ``2·(c_bg/V)·Q_total`` per atom.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom; the volume grad is atomic-summed.
    """
    i = wp.tid()
    g = grad_out[0]
    q = charges[i]
    two = type(q)(2.0)
    qtot = total_charge[0]
    vol = volume[0]
    cbg = c_bg_no_v[0]

    grad_charges[i] = g * (two * c_self_q[0] * q + two * (cbg / vol) * qtot)

    if has_dipoles != 0:
        mu = dipoles[i]
        cm = g * two * c_self_mu[0]
        grad_dipoles[i] = type(mu)(cm * mu[0], cm * mu[1], cm * mu[2])

    if has_quadrupoles != 0:
        Q = quadrupoles[i]
        cq = g * two * c_self_q2[0]
        grad_quadrupoles[i] = cq * Q

    # ∂L/∂V = -g · (c_bg / V²) · q_i · Q_total (atomic-summed → Q_total²).
    wp.atomic_add(grad_volume, 0, -g * (cbg / (vol * vol)) * q * qtot)


@wp.kernel(enable_backward=False)
def _batch_pme_multipole_corrections_backward_kernel(
    grad_out: wp.array(dtype=Any),  # (B,) upstream per-system grad
    charges: wp.array(dtype=Any),  # (N_total,)
    dipoles: wp.array(dtype=Any),  # (N_total,) of vec3
    quadrupoles: wp.array(dtype=Any),  # (N_total,) of mat33
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    volumes: wp.array(dtype=Any),  # (B,)
    total_charges: wp.array(dtype=Any),  # (B,)
    c_self_q: wp.array(dtype=Any),  # (1,)
    c_self_mu: wp.array(dtype=Any),  # (1,)
    c_self_q2: wp.array(dtype=Any),  # (1,)
    c_bg_no_v: wp.array(dtype=Any),  # (1,)
    has_dipoles: wp.int32,
    has_quadrupoles: wp.int32,
    grad_charges: wp.array(dtype=Any),  # (N_total,) OUTPUT
    grad_dipoles: wp.array(dtype=Any),  # (N_total,) of vec3 OUTPUT
    grad_quadrupoles: wp.array(dtype=Any),  # (N_total,) of mat33 OUTPUT
    grad_volumes: wp.array(dtype=Any),  # (B,) OUTPUT (atomic accumulate)
):
    r"""Batched backward of :func:`_batch_pme_multipole_corrections_kernel`.

    Per-atom grads use the upstream per-system cotangent ``g = grad_out[s]``
    with ``s = batch_idx[i]``; identical formulas to the single-system
    backward but with per-system ``volume`` / ``total_charge``.

    Launch Grid
    -----------
    ``dim = (N_total,)`` — one thread per atom; volume grads atomic-summed.
    """
    i = wp.tid()
    s = batch_idx[i]
    g = grad_out[s]
    q = charges[i]
    two = type(q)(2.0)
    qtot = total_charges[s]
    vol = volumes[s]
    cbg = c_bg_no_v[0]

    grad_charges[i] = g * (two * c_self_q[0] * q + two * (cbg / vol) * qtot)

    if has_dipoles != 0:
        mu = dipoles[i]
        cm = g * two * c_self_mu[0]
        grad_dipoles[i] = type(mu)(cm * mu[0], cm * mu[1], cm * mu[2])

    if has_quadrupoles != 0:
        Q = quadrupoles[i]
        cq = g * two * c_self_q2[0]
        grad_quadrupoles[i] = cq * Q

    wp.atomic_add(grad_volumes, s, -g * (cbg / (vol * vol)) * q * qtot)


@wp.kernel(enable_backward=False)
def _pme_multipole_corrections_double_backward_kernel(
    gg_charges: wp.array(dtype=Any),  # (N,) upstream cotangent on grad_charges
    gg_dipoles: wp.array(dtype=Any),  # (N,) of vec3
    gg_quadrupoles: wp.array(dtype=Any),  # (N,) of mat33
    gg_volume: wp.array(dtype=Any),  # (1,) upstream cotangent on grad_volume
    grad_out: wp.array(dtype=Any),  # (1,)
    charges: wp.array(dtype=Any),  # (N,)
    dipoles: wp.array(dtype=Any),  # (N,) of vec3
    quadrupoles: wp.array(dtype=Any),  # (N,) of mat33
    total_charge: wp.array(dtype=Any),  # (1,)
    sum_gg_charges: wp.array(dtype=Any),  # (1,) Σ_i gg_charges_i
    volume: wp.array(dtype=Any),  # (1,)
    c_self_q: wp.array(dtype=Any),  # (1,)
    c_self_mu: wp.array(dtype=Any),  # (1,)
    c_self_q2: wp.array(dtype=Any),  # (1,)
    c_bg_no_v: wp.array(dtype=Any),  # (1,)
    has_dipoles: wp.int32,
    has_quadrupoles: wp.int32,
    grad_grad_out: wp.array(dtype=Any),  # (1,) OUTPUT (atomic)
    grad_charges: wp.array(dtype=Any),  # (N,) OUTPUT
    grad_dipoles: wp.array(dtype=Any),  # (N,) of vec3 OUTPUT
    grad_quadrupoles: wp.array(dtype=Any),  # (N,) of mat33 OUTPUT
    grad_volume: wp.array(dtype=Any),  # (1,) OUTPUT (atomic)
):
    r"""Single-system double-backward of the corrections op.

    The first-order backward is linear in ``(grad_out, q, μ, Q, V)``;
    this kernel propagates upstream cotangents ``gg_*`` (one per backward
    output) to ``(grad_out, q, μ, Q, V)``. Needed for moment-moment HVPs
    (e.g. the l=2 quadrupole-quadrupole Hessian) through the PME composite.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom; ``grad_grad_out`` / ``grad_volume``
    are atomic-summed.
    """
    i = wp.tid()
    g = grad_out[0]
    two = type(g)(2.0)
    q = charges[i]
    qt = total_charge[0]
    vol = volume[0]
    cb = c_bg_no_v[0]
    cbv = cb / vol
    cbv2 = cb / (vol * vol)
    ggq = gg_charges[i]
    ggv = gg_volume[0]
    sg = sum_gg_charges[0]
    cq = c_self_q[0]

    # ∂L₂/∂grad_out (per-atom contributions, atomic-summed).
    contrib = ggq * (two * cq * q + two * cbv * qt)
    contrib -= ggv * cbv2 * q * qt
    if has_dipoles != 0:
        mu = dipoles[i]
        ggmu = gg_dipoles[i]
        contrib += (
            two * c_self_mu[0] * (ggmu[0] * mu[0] + ggmu[1] * mu[1] + ggmu[2] * mu[2])
        )
    if has_quadrupoles != 0:
        Q = quadrupoles[i]
        ggQ = gg_quadrupoles[i]
        ddot = (
            ggQ[0, 0] * Q[0, 0]
            + ggQ[0, 1] * Q[0, 1]
            + ggQ[0, 2] * Q[0, 2]
            + ggQ[1, 0] * Q[1, 0]
            + ggQ[1, 1] * Q[1, 1]
            + ggQ[1, 2] * Q[1, 2]
            + ggQ[2, 0] * Q[2, 0]
            + ggQ[2, 1] * Q[2, 1]
            + ggQ[2, 2] * Q[2, 2]
        )
        contrib += two * c_self_q2[0] * ddot
    wp.atomic_add(grad_grad_out, 0, contrib)

    # ∂L₂/∂q_i.
    grad_charges[i] = (
        g * two * cq * ggq + g * two * cbv * sg - ggv * g * two * cbv2 * qt
    )

    # ∂L₂/∂μ_i.
    if has_dipoles != 0:
        ggmu = gg_dipoles[i]
        cm = g * two * c_self_mu[0]
        grad_dipoles[i] = type(ggmu)(cm * ggmu[0], cm * ggmu[1], cm * ggmu[2])

    # ∂L₂/∂Q_i.
    if has_quadrupoles != 0:
        ggQ = gg_quadrupoles[i]
        grad_quadrupoles[i] = (g * two * c_self_q2[0]) * ggQ

    # ∂L₂/∂V (per-atom contributions, atomic-summed).
    gv = -g * two * cbv2 * qt * ggq + ggv * g * two * (cb / (vol * vol * vol)) * q * qt
    wp.atomic_add(grad_volume, 0, gv)


@wp.kernel(enable_backward=False)
def _batch_pme_multipole_corrections_double_backward_kernel(
    gg_charges: wp.array(dtype=Any),  # (N_total,)
    gg_dipoles: wp.array(dtype=Any),  # (N_total,) of vec3
    gg_quadrupoles: wp.array(dtype=Any),  # (N_total,) of mat33
    gg_volumes: wp.array(dtype=Any),  # (B,)
    grad_out: wp.array(dtype=Any),  # (B,)
    charges: wp.array(dtype=Any),  # (N_total,)
    dipoles: wp.array(dtype=Any),  # (N_total,) of vec3
    quadrupoles: wp.array(dtype=Any),  # (N_total,) of mat33
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    total_charges: wp.array(dtype=Any),  # (B,)
    sum_gg_charges: wp.array(dtype=Any),  # (B,)
    volumes: wp.array(dtype=Any),  # (B,)
    c_self_q: wp.array(dtype=Any),  # (1,)
    c_self_mu: wp.array(dtype=Any),  # (1,)
    c_self_q2: wp.array(dtype=Any),  # (1,)
    c_bg_no_v: wp.array(dtype=Any),  # (1,)
    has_dipoles: wp.int32,
    has_quadrupoles: wp.int32,
    grad_grad_out: wp.array(dtype=Any),  # (B,) OUTPUT (atomic)
    grad_charges: wp.array(dtype=Any),  # (N_total,) OUTPUT
    grad_dipoles: wp.array(dtype=Any),  # (N_total,) of vec3 OUTPUT
    grad_quadrupoles: wp.array(dtype=Any),  # (N_total,) of mat33 OUTPUT
    grad_volumes: wp.array(dtype=Any),  # (B,) OUTPUT (atomic)
):
    r"""Batched double-backward of the corrections op (per-system params).

    Launch Grid
    -----------
    ``dim = (N_total,)`` — one thread per atom; ``grad_grad_out`` /
    ``grad_volumes`` are atomic-summed per system.
    """
    i = wp.tid()
    s = batch_idx[i]
    g = grad_out[s]
    two = type(g)(2.0)
    q = charges[i]
    qt = total_charges[s]
    vol = volumes[s]
    cb = c_bg_no_v[0]
    cbv = cb / vol
    cbv2 = cb / (vol * vol)
    ggq = gg_charges[i]
    ggv = gg_volumes[s]
    sg = sum_gg_charges[s]
    cq = c_self_q[0]

    contrib = ggq * (two * cq * q + two * cbv * qt)
    contrib -= ggv * cbv2 * q * qt
    if has_dipoles != 0:
        mu = dipoles[i]
        ggmu = gg_dipoles[i]
        contrib += (
            two * c_self_mu[0] * (ggmu[0] * mu[0] + ggmu[1] * mu[1] + ggmu[2] * mu[2])
        )
    if has_quadrupoles != 0:
        Q = quadrupoles[i]
        ggQ = gg_quadrupoles[i]
        ddot = (
            ggQ[0, 0] * Q[0, 0]
            + ggQ[0, 1] * Q[0, 1]
            + ggQ[0, 2] * Q[0, 2]
            + ggQ[1, 0] * Q[1, 0]
            + ggQ[1, 1] * Q[1, 1]
            + ggQ[1, 2] * Q[1, 2]
            + ggQ[2, 0] * Q[2, 0]
            + ggQ[2, 1] * Q[2, 1]
            + ggQ[2, 2] * Q[2, 2]
        )
        contrib += two * c_self_q2[0] * ddot
    wp.atomic_add(grad_grad_out, s, contrib)

    grad_charges[i] = (
        g * two * cq * ggq + g * two * cbv * sg - ggv * g * two * cbv2 * qt
    )

    if has_dipoles != 0:
        ggmu = gg_dipoles[i]
        cm = g * two * c_self_mu[0]
        grad_dipoles[i] = type(ggmu)(cm * ggmu[0], cm * ggmu[1], cm * ggmu[2])

    if has_quadrupoles != 0:
        ggQ = gg_quadrupoles[i]
        grad_quadrupoles[i] = (g * two * c_self_q2[0]) * ggQ

    gv = -g * two * cbv2 * qt * ggq + ggv * g * two * (cb / (vol * vol * vol)) * q * qt
    wp.atomic_add(grad_volumes, s, gv)


def _pme_multipole_corrections_sig(v, t):
    """Signature builder for :func:`_pme_multipole_corrections_kernel`."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=mat),  # quadrupoles
        wp.array(dtype=t),  # volume
        wp.array(dtype=t),  # total_charge
        wp.array(dtype=t),  # c_self_q
        wp.array(dtype=t),  # c_self_mu
        wp.array(dtype=t),  # c_self_q2
        wp.array(dtype=t),  # c_bg_no_v
        wp.int32,  # has_dipoles
        wp.int32,  # has_quadrupoles
        wp.array(dtype=t),  # corrections (output)
    ]


def _batch_pme_multipole_corrections_sig(v, t):
    """Signature builder for :func:`_batch_pme_multipole_corrections_kernel`."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=mat),  # quadrupoles
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array(dtype=t),  # volumes
        wp.array(dtype=t),  # total_charges
        wp.array(dtype=t),  # c_self_q
        wp.array(dtype=t),  # c_self_mu
        wp.array(dtype=t),  # c_self_q2
        wp.array(dtype=t),  # c_bg_no_v
        wp.int32,  # has_dipoles
        wp.int32,  # has_quadrupoles
        wp.array(dtype=t),  # corrections (output)
    ]


def _pme_multipole_corrections_backward_sig(v, t):
    """Signature builder for the single-system corrections backward kernel."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=t),  # grad_out
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=mat),  # quadrupoles
        wp.array(dtype=t),  # volume
        wp.array(dtype=t),  # total_charge
        wp.array(dtype=t),  # c_self_q
        wp.array(dtype=t),  # c_self_mu
        wp.array(dtype=t),  # c_self_q2
        wp.array(dtype=t),  # c_bg_no_v
        wp.int32,  # has_dipoles
        wp.int32,  # has_quadrupoles
        wp.array(dtype=t),  # grad_charges (out)
        wp.array(dtype=v),  # grad_dipoles (out)
        wp.array(dtype=mat),  # grad_quadrupoles (out)
        wp.array(dtype=t),  # grad_volume (out)
    ]


def _batch_pme_multipole_corrections_backward_sig(v, t):
    """Signature builder for the batched corrections backward kernel."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=t),  # grad_out
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=mat),  # quadrupoles
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array(dtype=t),  # volumes
        wp.array(dtype=t),  # total_charges
        wp.array(dtype=t),  # c_self_q
        wp.array(dtype=t),  # c_self_mu
        wp.array(dtype=t),  # c_self_q2
        wp.array(dtype=t),  # c_bg_no_v
        wp.int32,  # has_dipoles
        wp.int32,  # has_quadrupoles
        wp.array(dtype=t),  # grad_charges (out)
        wp.array(dtype=v),  # grad_dipoles (out)
        wp.array(dtype=mat),  # grad_quadrupoles (out)
        wp.array(dtype=t),  # grad_volumes (out)
    ]


def _pme_multipole_corrections_double_backward_sig(v, t):
    """Signature builder for the single-system corrections double-backward."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=t),  # gg_charges
        wp.array(dtype=v),  # gg_dipoles
        wp.array(dtype=mat),  # gg_quadrupoles
        wp.array(dtype=t),  # gg_volume
        wp.array(dtype=t),  # grad_out
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=mat),  # quadrupoles
        wp.array(dtype=t),  # total_charge
        wp.array(dtype=t),  # sum_gg_charges
        wp.array(dtype=t),  # volume
        wp.array(dtype=t),  # c_self_q
        wp.array(dtype=t),  # c_self_mu
        wp.array(dtype=t),  # c_self_q2
        wp.array(dtype=t),  # c_bg_no_v
        wp.int32,  # has_dipoles
        wp.int32,  # has_quadrupoles
        wp.array(dtype=t),  # grad_grad_out (out)
        wp.array(dtype=t),  # grad_charges (out)
        wp.array(dtype=v),  # grad_dipoles (out)
        wp.array(dtype=mat),  # grad_quadrupoles (out)
        wp.array(dtype=t),  # grad_volume (out)
    ]


def _batch_pme_multipole_corrections_double_backward_sig(v, t):
    """Signature builder for the batched corrections double-backward."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=t),  # gg_charges
        wp.array(dtype=v),  # gg_dipoles
        wp.array(dtype=mat),  # gg_quadrupoles
        wp.array(dtype=t),  # gg_volumes
        wp.array(dtype=t),  # grad_out
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=mat),  # quadrupoles
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array(dtype=t),  # total_charges
        wp.array(dtype=t),  # sum_gg_charges
        wp.array(dtype=t),  # volumes
        wp.array(dtype=t),  # c_self_q
        wp.array(dtype=t),  # c_self_mu
        wp.array(dtype=t),  # c_self_q2
        wp.array(dtype=t),  # c_bg_no_v
        wp.int32,  # has_dipoles
        wp.int32,  # has_quadrupoles
        wp.array(dtype=t),  # grad_grad_out (out)
        wp.array(dtype=t),  # grad_charges (out)
        wp.array(dtype=v),  # grad_dipoles (out)
        wp.array(dtype=mat),  # grad_quadrupoles (out)
        wp.array(dtype=t),  # grad_volumes (out)
    ]


_pme_multipole_corrections_overloads = register_overloads(
    _pme_multipole_corrections_kernel, _pme_multipole_corrections_sig
)
_batch_pme_multipole_corrections_overloads = register_overloads(
    _batch_pme_multipole_corrections_kernel, _batch_pme_multipole_corrections_sig
)
_pme_multipole_corrections_backward_overloads = register_overloads(
    _pme_multipole_corrections_backward_kernel,
    _pme_multipole_corrections_backward_sig,
)
_batch_pme_multipole_corrections_backward_overloads = register_overloads(
    _batch_pme_multipole_corrections_backward_kernel,
    _batch_pme_multipole_corrections_backward_sig,
)
_pme_multipole_corrections_double_backward_overloads = register_overloads(
    _pme_multipole_corrections_double_backward_kernel,
    _pme_multipole_corrections_double_backward_sig,
)
_batch_pme_multipole_corrections_double_backward_overloads = register_overloads(
    _batch_pme_multipole_corrections_double_backward_kernel,
    _batch_pme_multipole_corrections_double_backward_sig,
)


def multipole_pme_corrections_launch(
    charges,
    dipoles,
    quadrupoles,
    volume,
    total_charge,
    c_self_q,
    c_self_mu,
    c_self_q2,
    c_bg_no_v,
    has_dipoles,
    has_quadrupoles,
    corrections,
    wp_dtype,
    device=None,
):
    r"""Single-system launcher for :func:`_pme_multipole_corrections_kernel`.

    Computes the per-atom GTO-Ewald self + background-share correction.
    All scalar coefficients are fp64 length-1 arrays precomputed by the
    torch wrapper. ``has_dipoles`` / ``has_quadrupoles`` are ``0``/``1``
    flags gating the l=1 / l=2 self-energy channels.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    if device is None:
        device = str(charges.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_multipole_corrections_overloads[vec_dtype],
        dim=(charges.shape[0],),
        inputs=[
            charges,
            dipoles,
            quadrupoles,
            volume,
            total_charge,
            c_self_q,
            c_self_mu,
            c_self_q2,
            c_bg_no_v,
            int(has_dipoles),
            int(has_quadrupoles),
        ],
        outputs=[corrections],
        device=device,
    )


def batch_multipole_pme_corrections_launch(
    charges,
    dipoles,
    quadrupoles,
    batch_idx,
    volumes,
    total_charges,
    c_self_q,
    c_self_mu,
    c_self_q2,
    c_bg_no_v,
    has_dipoles,
    has_quadrupoles,
    corrections,
    wp_dtype,
    device=None,
):
    r"""Batched launcher for :func:`_batch_pme_multipole_corrections_kernel`.

    Launch Grid
    -----------
    ``dim = (N_total,)`` — one thread per atom across the batch.
    """
    if device is None:
        device = str(charges.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _batch_pme_multipole_corrections_overloads[vec_dtype],
        dim=(charges.shape[0],),
        inputs=[
            charges,
            dipoles,
            quadrupoles,
            batch_idx,
            volumes,
            total_charges,
            c_self_q,
            c_self_mu,
            c_self_q2,
            c_bg_no_v,
            int(has_dipoles),
            int(has_quadrupoles),
        ],
        outputs=[corrections],
        device=device,
    )


def multipole_pme_corrections_backward_launch(
    grad_out,
    charges,
    dipoles,
    quadrupoles,
    volume,
    total_charge,
    c_self_q,
    c_self_mu,
    c_self_q2,
    c_bg_no_v,
    has_dipoles,
    has_quadrupoles,
    grad_charges,
    grad_dipoles,
    grad_quadrupoles,
    grad_volume,
    wp_dtype,
    device=None,
):
    r"""Single-system backward launcher (corrections grads).

    ``grad_charges`` / ``grad_dipoles`` / ``grad_quadrupoles`` are written
    directly; ``grad_volume`` is pre-zeroed and atomic-accumulated.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    if device is None:
        device = str(charges.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_multipole_corrections_backward_overloads[vec_dtype],
        dim=(charges.shape[0],),
        inputs=[
            grad_out,
            charges,
            dipoles,
            quadrupoles,
            volume,
            total_charge,
            c_self_q,
            c_self_mu,
            c_self_q2,
            c_bg_no_v,
            int(has_dipoles),
            int(has_quadrupoles),
        ],
        outputs=[grad_charges, grad_dipoles, grad_quadrupoles, grad_volume],
        device=device,
    )


def batch_multipole_pme_corrections_backward_launch(
    grad_out,
    charges,
    dipoles,
    quadrupoles,
    batch_idx,
    volumes,
    total_charges,
    c_self_q,
    c_self_mu,
    c_self_q2,
    c_bg_no_v,
    has_dipoles,
    has_quadrupoles,
    grad_charges,
    grad_dipoles,
    grad_quadrupoles,
    grad_volumes,
    wp_dtype,
    device=None,
):
    r"""Batched backward launcher (corrections grads).

    Launch Grid
    -----------
    ``dim = (N_total,)`` — one thread per atom across the batch.
    """
    if device is None:
        device = str(charges.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _batch_pme_multipole_corrections_backward_overloads[vec_dtype],
        dim=(charges.shape[0],),
        inputs=[
            grad_out,
            charges,
            dipoles,
            quadrupoles,
            batch_idx,
            volumes,
            total_charges,
            c_self_q,
            c_self_mu,
            c_self_q2,
            c_bg_no_v,
            int(has_dipoles),
            int(has_quadrupoles),
        ],
        outputs=[grad_charges, grad_dipoles, grad_quadrupoles, grad_volumes],
        device=device,
    )


def multipole_pme_corrections_double_backward_launch(
    gg_charges,
    gg_dipoles,
    gg_quadrupoles,
    gg_volume,
    grad_out,
    charges,
    dipoles,
    quadrupoles,
    total_charge,
    sum_gg_charges,
    volume,
    c_self_q,
    c_self_mu,
    c_self_q2,
    c_bg_no_v,
    has_dipoles,
    has_quadrupoles,
    grad_grad_out,
    grad_charges,
    grad_dipoles,
    grad_quadrupoles,
    grad_volume,
    wp_dtype,
    device=None,
):
    r"""Single-system double-backward launcher (corrections HVP).

    ``grad_grad_out`` / ``grad_volume`` are pre-zeroed and atomic-summed.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    if device is None:
        device = str(charges.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _pme_multipole_corrections_double_backward_overloads[vec_dtype],
        dim=(charges.shape[0],),
        inputs=[
            gg_charges,
            gg_dipoles,
            gg_quadrupoles,
            gg_volume,
            grad_out,
            charges,
            dipoles,
            quadrupoles,
            total_charge,
            sum_gg_charges,
            volume,
            c_self_q,
            c_self_mu,
            c_self_q2,
            c_bg_no_v,
            int(has_dipoles),
            int(has_quadrupoles),
        ],
        outputs=[
            grad_grad_out,
            grad_charges,
            grad_dipoles,
            grad_quadrupoles,
            grad_volume,
        ],
        device=device,
    )


def batch_multipole_pme_corrections_double_backward_launch(
    gg_charges,
    gg_dipoles,
    gg_quadrupoles,
    gg_volumes,
    grad_out,
    charges,
    dipoles,
    quadrupoles,
    batch_idx,
    total_charges,
    sum_gg_charges,
    volumes,
    c_self_q,
    c_self_mu,
    c_self_q2,
    c_bg_no_v,
    has_dipoles,
    has_quadrupoles,
    grad_grad_out,
    grad_charges,
    grad_dipoles,
    grad_quadrupoles,
    grad_volumes,
    wp_dtype,
    device=None,
):
    r"""Batched double-backward launcher (corrections HVP).

    Launch Grid
    -----------
    ``dim = (N_total,)`` — one thread per atom across the batch.
    """
    if device is None:
        device = str(charges.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _batch_pme_multipole_corrections_double_backward_overloads[vec_dtype],
        dim=(charges.shape[0],),
        inputs=[
            gg_charges,
            gg_dipoles,
            gg_quadrupoles,
            gg_volumes,
            grad_out,
            charges,
            dipoles,
            quadrupoles,
            batch_idx,
            total_charges,
            sum_gg_charges,
            volumes,
            c_self_q,
            c_self_mu,
            c_self_q2,
            c_bg_no_v,
            int(has_dipoles),
            int(has_quadrupoles),
        ],
        outputs=[
            grad_grad_out,
            grad_charges,
            grad_dipoles,
            grad_quadrupoles,
            grad_volumes,
        ],
        device=device,
    )


# =============================================================================
# Reciprocal-space per-k energy from the density rho(k)
# =============================================================================
#
# The reciprocal-space energy is assembled from the per-k density rho(k)
# (a real/imag pair) and the position-independent per-k factor:
#
#   e_k = 2 * per_k_factor[k] * (rho[k, 0]^2 + rho[k, 1]^2)
#
# The torch wrapper applies the ``0.5 * V / (2 pi)^6`` scale after reduction.


@wp.kernel(enable_backward=False)
def _multipole_reciprocal_rho_energy_kernel(
    rho: wp.array(dtype=Any),  # (M,) of vec2 (real, imag)
    per_k_factor: wp.array(dtype=Any),  # (M,)
    per_k_energy: wp.array(dtype=Any),  # (M,) OUTPUT
):
    r"""Per-k reciprocal energy contribution from the density rho.

    Each thread computes one k-point's contribution
    :math:`e_k = 2\,f_k\,(\rho_{k,0}^2 + \rho_{k,1}^2)`; the torch wrapper
    reduces and applies the ``0.5 V / (2\pi)^6`` scale.

    Launch Grid
    -----------
    ``dim = (M,)`` — one thread per (flattened) k-point.
    """
    k = wp.tid()
    r = rho[k]
    two = type(r[0])(2.0)
    per_k_energy[k] = two * per_k_factor[k] * (r[0] * r[0] + r[1] * r[1])


@wp.kernel(enable_backward=False)
def _multipole_reciprocal_rho_energy_backward_kernel(
    grad_out: wp.array(dtype=Any),  # (M,) cotangent on per_k_energy
    rho: wp.array(dtype=Any),  # (M,) of vec2
    per_k_factor: wp.array(dtype=Any),  # (M,)
    grad_rho: wp.array(dtype=Any),  # (M,) of vec2 OUTPUT
    grad_per_k_factor: wp.array(dtype=Any),  # (M,) OUTPUT
):
    r"""Backward of :func:`_multipole_reciprocal_rho_energy_kernel`.

    .. math::

        \partial L/\partial \rho_{k,c} &= g_k\,4\,f_k\,\rho_{k,c}, \\
        \partial L/\partial f_k &= g_k\,2\,|\rho_k|^2,

    where ``g_k = grad_out[k]`` is the per-k cotangent. ``per_k_factor`` is
    differentiable because it depends on the cell (cell-grad / stress path).

    Launch Grid
    -----------
    ``dim = (M,)`` — one thread per (flattened) k-point.
    """
    k = wp.tid()
    r = rho[k]
    g = grad_out[k]
    coeff = g * type(r[0])(4.0) * per_k_factor[k]
    grad_rho[k] = type(r)(coeff * r[0], coeff * r[1])
    grad_per_k_factor[k] = g * type(r[0])(2.0) * (r[0] * r[0] + r[1] * r[1])


@wp.kernel(enable_backward=False)
def _multipole_reciprocal_rho_energy_double_backward_kernel(
    gg_rho: wp.array(dtype=Any),  # (M,) of vec2 cotangent on grad_rho
    gg_per_k_factor: wp.array(dtype=Any),  # (M,) cotangent on grad_per_k_factor
    grad_out: wp.array(dtype=Any),  # (M,)
    rho: wp.array(dtype=Any),  # (M,) of vec2
    per_k_factor: wp.array(dtype=Any),  # (M,)
    grad_grad_out: wp.array(dtype=Any),  # (M,) OUTPUT
    grad_rho: wp.array(dtype=Any),  # (M,) of vec2 OUTPUT
    grad_per_k_factor: wp.array(dtype=Any),  # (M,) OUTPUT
):
    r"""Double-backward of the per-k energy op (constant per-k Hessian).

    The first-order backward
    ``grad_rho_{k,c} = g_k 4 f_k rho_{k,c}``,
    ``grad_f_k = g_k 2 |rho_k|^2``
    is bilinear in ``(g_k, rho, f_k)``; with upstream cotangents
    ``(gg_rho, gg_f)``:

    .. math::

        \partial L_2/\partial g_k &= 4 f_k \sum_c gg\_rho_{k,c}\,\rho_{k,c}
            + 2\,gg\_f_k\,|\rho_k|^2, \\
        \partial L_2/\partial \rho_{k,c} &= g_k\,4\,f_k\,gg\_rho_{k,c}
            + g_k\,4\,gg\_f_k\,\rho_{k,c}, \\
        \partial L_2/\partial f_k &= 4\,g_k \sum_c gg\_rho_{k,c}\,\rho_{k,c}.

    Needed for moment-moment / position HVPs (force-loss) since rho depends
    on positions and the assembly is quadratic in rho.

    Launch Grid
    -----------
    ``dim = (M,)`` — one thread per (flattened) k-point.
    """
    k = wp.tid()
    r = rho[k]
    gr = gg_rho[k]
    ggf = gg_per_k_factor[k]
    g = grad_out[k]
    f = per_k_factor[k]
    two = type(r[0])(2.0)
    four = type(r[0])(4.0)
    gr_dot_r = gr[0] * r[0] + gr[1] * r[1]
    r2 = r[0] * r[0] + r[1] * r[1]

    grad_grad_out[k] = four * f * gr_dot_r + two * ggf * r2
    cr = g * four * f
    cf = g * four * ggf
    grad_rho[k] = type(r)(cr * gr[0] + cf * r[0], cr * gr[1] + cf * r[1])
    grad_per_k_factor[k] = four * g * gr_dot_r


def _multipole_reciprocal_rho_energy_sig(v, t):
    """Signature builder for :func:`_multipole_reciprocal_rho_energy_kernel`."""
    vec2 = wp.vec2d if t == wp.float64 else wp.vec2f
    return [
        wp.array(dtype=vec2),  # rho
        wp.array(dtype=t),  # per_k_factor
        wp.array(dtype=t),  # per_k_energy (output)
    ]


def _multipole_reciprocal_rho_energy_backward_sig(v, t):
    """Signature builder for the per-k energy backward kernel."""
    vec2 = wp.vec2d if t == wp.float64 else wp.vec2f
    return [
        wp.array(dtype=t),  # grad_out
        wp.array(dtype=vec2),  # rho
        wp.array(dtype=t),  # per_k_factor
        wp.array(dtype=vec2),  # grad_rho (output)
        wp.array(dtype=t),  # grad_per_k_factor (output)
    ]


def _multipole_reciprocal_rho_energy_double_backward_sig(v, t):
    """Signature builder for the per-k energy double-backward kernel."""
    vec2 = wp.vec2d if t == wp.float64 else wp.vec2f
    return [
        wp.array(dtype=vec2),  # gg_rho
        wp.array(dtype=t),  # gg_per_k_factor
        wp.array(dtype=t),  # grad_out
        wp.array(dtype=vec2),  # rho
        wp.array(dtype=t),  # per_k_factor
        wp.array(dtype=t),  # grad_grad_out (output)
        wp.array(dtype=vec2),  # grad_rho (output)
        wp.array(dtype=t),  # grad_per_k_factor (output)
    ]


_multipole_reciprocal_rho_energy_overloads = register_overloads(
    _multipole_reciprocal_rho_energy_kernel, _multipole_reciprocal_rho_energy_sig
)
_multipole_reciprocal_rho_energy_backward_overloads = register_overloads(
    _multipole_reciprocal_rho_energy_backward_kernel,
    _multipole_reciprocal_rho_energy_backward_sig,
)
_multipole_reciprocal_rho_energy_double_backward_overloads = register_overloads(
    _multipole_reciprocal_rho_energy_double_backward_kernel,
    _multipole_reciprocal_rho_energy_double_backward_sig,
)


def multipole_reciprocal_rho_energy_launch(
    rho,
    per_k_factor,
    per_k_energy,
    wp_dtype,
    device=None,
):
    r"""Launcher for :func:`_multipole_reciprocal_rho_energy_kernel`.

    Computes the per-k energy ``2 f_k |rho_k|^2`` over a flat ``(M,)`` grid
    (``M = K`` single / ``M = B * K`` batched). The torch wrapper owns the
    reduction + scale.

    Launch Grid
    -----------
    ``dim = (M,)`` — one thread per (flattened) k-point.
    """
    if device is None:
        device = str(rho.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _multipole_reciprocal_rho_energy_overloads[vec_dtype],
        dim=(rho.shape[0],),
        inputs=[rho, per_k_factor],
        outputs=[per_k_energy],
        device=device,
    )


def multipole_reciprocal_rho_energy_backward_launch(
    grad_out,
    rho,
    per_k_factor,
    grad_rho,
    grad_per_k_factor,
    wp_dtype,
    device=None,
):
    r"""Backward launcher for the per-k reciprocal energy op.

    Launch Grid
    -----------
    ``dim = (M,)`` — one thread per (flattened) k-point.
    """
    if device is None:
        device = str(rho.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _multipole_reciprocal_rho_energy_backward_overloads[vec_dtype],
        dim=(rho.shape[0],),
        inputs=[grad_out, rho, per_k_factor],
        outputs=[grad_rho, grad_per_k_factor],
        device=device,
    )


def multipole_reciprocal_rho_energy_double_backward_launch(
    gg_rho,
    gg_per_k_factor,
    grad_out,
    rho,
    per_k_factor,
    grad_grad_out,
    grad_rho,
    grad_per_k_factor,
    wp_dtype,
    device=None,
):
    r"""Double-backward launcher for the per-k reciprocal energy op.

    Launch Grid
    -----------
    ``dim = (M,)`` — one thread per (flattened) k-point.
    """
    if device is None:
        device = str(rho.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _multipole_reciprocal_rho_energy_double_backward_overloads[vec_dtype],
        dim=(rho.shape[0],),
        inputs=[gg_rho, gg_per_k_factor, grad_out, rho, per_k_factor],
        outputs=[grad_grad_out, grad_rho, grad_per_k_factor],
        device=device,
    )


# =============================================================================
# Multipole self-energy: weighted sum of per-atom moment squares
# =============================================================================
#
# The reciprocal self-energy subtraction is a per-atom weighted sum of
# moment squares (no background / no volume term):
#
#   e_self_i = c0 * q_i^2 + c1 * |mu_i|^2 + c2 * |Q_i|_F^2
#
# with the optional l=1 / l=2 channels gated by flags.


@wp.kernel(enable_backward=False)
def _multipole_self_energy_kernel(
    charges: wp.array(dtype=Any),  # (N,)
    dipoles: wp.array(dtype=Any),  # (N,) of vec3
    quadrupoles: wp.array(dtype=Any),  # (N,) of mat33
    c_self_q: wp.array(dtype=Any),  # (1,)
    c_self_mu: wp.array(dtype=Any),  # (1,)
    c_self_q2: wp.array(dtype=Any),  # (1,)
    has_dipoles: wp.int32,
    has_quadrupoles: wp.int32,
    self_energy: wp.array(dtype=Any),  # (N,) OUTPUT
):
    r"""Per-atom weighted sum of moment squares (self-energy density).

    Each thread computes
    :math:`e_i = c_q q_i^2 + c_\mu |\mu_i|^2 + c_Q |Q_i|_F^2`; the l=1 / l=2
    channels are gated by ``has_dipoles`` / ``has_quadrupoles``. The torch
    wrapper reduces (``.sum()`` / ``scatter_add``).

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    i = wp.tid()
    q = charges[i]
    e = c_self_q[0] * q * q

    if has_dipoles != 0:
        mu = dipoles[i]
        e += c_self_mu[0] * (mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2])

    if has_quadrupoles != 0:
        Q = quadrupoles[i]
        q_fro = (
            Q[0, 0] * Q[0, 0]
            + Q[0, 1] * Q[0, 1]
            + Q[0, 2] * Q[0, 2]
            + Q[1, 0] * Q[1, 0]
            + Q[1, 1] * Q[1, 1]
            + Q[1, 2] * Q[1, 2]
            + Q[2, 0] * Q[2, 0]
            + Q[2, 1] * Q[2, 1]
            + Q[2, 2] * Q[2, 2]
        )
        e += c_self_q2[0] * q_fro

    self_energy[i] = e


@wp.kernel(enable_backward=False)
def _multipole_self_energy_backward_kernel(
    grad_out: wp.array(dtype=Any),  # (N,) cotangent on self_energy
    charges: wp.array(dtype=Any),  # (N,)
    dipoles: wp.array(dtype=Any),  # (N,) of vec3
    quadrupoles: wp.array(dtype=Any),  # (N,) of mat33
    c_self_q: wp.array(dtype=Any),  # (1,)
    c_self_mu: wp.array(dtype=Any),  # (1,)
    c_self_q2: wp.array(dtype=Any),  # (1,)
    has_dipoles: wp.int32,
    has_quadrupoles: wp.int32,
    grad_charges: wp.array(dtype=Any),  # (N,) OUTPUT
    grad_dipoles: wp.array(dtype=Any),  # (N,) of vec3 OUTPUT
    grad_quadrupoles: wp.array(dtype=Any),  # (N,) of mat33 OUTPUT
):
    r"""Backward of :func:`_multipole_self_energy_kernel`.

    .. math::

        \partial L/\partial q_i &= g_i\,2\,c_q\,q_i, \\
        \partial L/\partial \mu_i &= g_i\,2\,c_\mu\,\mu_i, \\
        \partial L/\partial Q_i &= g_i\,2\,c_Q\,Q_i,

    where ``g_i = grad_out[i]`` is the per-atom cotangent.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    i = wp.tid()
    q = charges[i]
    g = grad_out[i]
    two = type(q)(2.0)
    grad_charges[i] = g * two * c_self_q[0] * q

    if has_dipoles != 0:
        mu = dipoles[i]
        cm = g * two * c_self_mu[0]
        grad_dipoles[i] = type(mu)(cm * mu[0], cm * mu[1], cm * mu[2])

    if has_quadrupoles != 0:
        Q = quadrupoles[i]
        grad_quadrupoles[i] = (g * two * c_self_q2[0]) * Q


@wp.kernel(enable_backward=False)
def _multipole_self_energy_double_backward_kernel(
    gg_charges: wp.array(dtype=Any),  # (N,) cotangent on grad_charges
    gg_dipoles: wp.array(dtype=Any),  # (N,) of vec3
    gg_quadrupoles: wp.array(dtype=Any),  # (N,) of mat33
    grad_out: wp.array(dtype=Any),  # (N,)
    charges: wp.array(dtype=Any),  # (N,)
    dipoles: wp.array(dtype=Any),  # (N,) of vec3
    quadrupoles: wp.array(dtype=Any),  # (N,) of mat33
    c_self_q: wp.array(dtype=Any),  # (1,)
    c_self_mu: wp.array(dtype=Any),  # (1,)
    c_self_q2: wp.array(dtype=Any),  # (1,)
    has_dipoles: wp.int32,
    has_quadrupoles: wp.int32,
    grad_grad_out: wp.array(dtype=Any),  # (N,) OUTPUT
    grad_charges: wp.array(dtype=Any),  # (N,) OUTPUT
    grad_dipoles: wp.array(dtype=Any),  # (N,) of vec3 OUTPUT
    grad_quadrupoles: wp.array(dtype=Any),  # (N,) of mat33 OUTPUT
):
    r"""Double-backward of the self-energy op (constant per-atom Hessian).

    The first-order backward is bilinear in ``(grad_out, q, mu, Q)``; with
    upstream cotangents ``gg_*`` this propagates to ``(grad_out, q, mu, Q)``.
    Needed for moment-moment HVPs (force-loss) through the reciprocal
    composite.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    i = wp.tid()
    q = charges[i]
    g = grad_out[i]
    ggq = gg_charges[i]
    two = type(q)(2.0)
    cq = c_self_q[0]

    contrib = two * cq * q * ggq
    grad_charges[i] = g * two * cq * ggq

    if has_dipoles != 0:
        mu = dipoles[i]
        ggmu = gg_dipoles[i]
        cm = two * c_self_mu[0]
        contrib += cm * (ggmu[0] * mu[0] + ggmu[1] * mu[1] + ggmu[2] * mu[2])
        gc = g * cm
        grad_dipoles[i] = type(mu)(gc * ggmu[0], gc * ggmu[1], gc * ggmu[2])

    if has_quadrupoles != 0:
        Q = quadrupoles[i]
        ggQ = gg_quadrupoles[i]
        cQ = two * c_self_q2[0]
        ddot = (
            ggQ[0, 0] * Q[0, 0]
            + ggQ[0, 1] * Q[0, 1]
            + ggQ[0, 2] * Q[0, 2]
            + ggQ[1, 0] * Q[1, 0]
            + ggQ[1, 1] * Q[1, 1]
            + ggQ[1, 2] * Q[1, 2]
            + ggQ[2, 0] * Q[2, 0]
            + ggQ[2, 1] * Q[2, 1]
            + ggQ[2, 2] * Q[2, 2]
        )
        contrib += cQ * ddot
        grad_quadrupoles[i] = (g * cQ) * ggQ

    grad_grad_out[i] = contrib


def _multipole_self_energy_sig(v, t):
    """Signature builder for :func:`_multipole_self_energy_kernel`."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=mat),  # quadrupoles
        wp.array(dtype=t),  # c_self_q
        wp.array(dtype=t),  # c_self_mu
        wp.array(dtype=t),  # c_self_q2
        wp.int32,  # has_dipoles
        wp.int32,  # has_quadrupoles
        wp.array(dtype=t),  # self_energy (output)
    ]


def _multipole_self_energy_backward_sig(v, t):
    """Signature builder for the self-energy backward kernel."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=t),  # grad_out
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=mat),  # quadrupoles
        wp.array(dtype=t),  # c_self_q
        wp.array(dtype=t),  # c_self_mu
        wp.array(dtype=t),  # c_self_q2
        wp.int32,  # has_dipoles
        wp.int32,  # has_quadrupoles
        wp.array(dtype=t),  # grad_charges (output)
        wp.array(dtype=v),  # grad_dipoles (output)
        wp.array(dtype=mat),  # grad_quadrupoles (output)
    ]


def _multipole_self_energy_double_backward_sig(v, t):
    """Signature builder for the self-energy double-backward kernel."""
    mat = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=t),  # gg_charges
        wp.array(dtype=v),  # gg_dipoles
        wp.array(dtype=mat),  # gg_quadrupoles
        wp.array(dtype=t),  # grad_out
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=mat),  # quadrupoles
        wp.array(dtype=t),  # c_self_q
        wp.array(dtype=t),  # c_self_mu
        wp.array(dtype=t),  # c_self_q2
        wp.int32,  # has_dipoles
        wp.int32,  # has_quadrupoles
        wp.array(dtype=t),  # grad_grad_out (output)
        wp.array(dtype=t),  # grad_charges (output)
        wp.array(dtype=v),  # grad_dipoles (output)
        wp.array(dtype=mat),  # grad_quadrupoles (output)
    ]


_multipole_self_energy_overloads = register_overloads(
    _multipole_self_energy_kernel, _multipole_self_energy_sig
)
_multipole_self_energy_backward_overloads = register_overloads(
    _multipole_self_energy_backward_kernel, _multipole_self_energy_backward_sig
)
_multipole_self_energy_double_backward_overloads = register_overloads(
    _multipole_self_energy_double_backward_kernel,
    _multipole_self_energy_double_backward_sig,
)


def multipole_self_energy_launch(
    charges,
    dipoles,
    quadrupoles,
    c_self_q,
    c_self_mu,
    c_self_q2,
    has_dipoles,
    has_quadrupoles,
    self_energy,
    wp_dtype,
    device=None,
):
    r"""Launcher for :func:`_multipole_self_energy_kernel`.

    Computes the per-atom weighted sum of moment squares; the torch wrapper
    owns the reduction + overlap-constant coefficients.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    if device is None:
        device = str(charges.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _multipole_self_energy_overloads[vec_dtype],
        dim=(charges.shape[0],),
        inputs=[
            charges,
            dipoles,
            quadrupoles,
            c_self_q,
            c_self_mu,
            c_self_q2,
            int(has_dipoles),
            int(has_quadrupoles),
        ],
        outputs=[self_energy],
        device=device,
    )


def multipole_self_energy_backward_launch(
    grad_out,
    charges,
    dipoles,
    quadrupoles,
    c_self_q,
    c_self_mu,
    c_self_q2,
    has_dipoles,
    has_quadrupoles,
    grad_charges,
    grad_dipoles,
    grad_quadrupoles,
    wp_dtype,
    device=None,
):
    r"""Backward launcher for the self-energy op.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    if device is None:
        device = str(charges.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _multipole_self_energy_backward_overloads[vec_dtype],
        dim=(charges.shape[0],),
        inputs=[
            grad_out,
            charges,
            dipoles,
            quadrupoles,
            c_self_q,
            c_self_mu,
            c_self_q2,
            int(has_dipoles),
            int(has_quadrupoles),
        ],
        outputs=[grad_charges, grad_dipoles, grad_quadrupoles],
        device=device,
    )


def multipole_self_energy_double_backward_launch(
    gg_charges,
    gg_dipoles,
    gg_quadrupoles,
    grad_out,
    charges,
    dipoles,
    quadrupoles,
    c_self_q,
    c_self_mu,
    c_self_q2,
    has_dipoles,
    has_quadrupoles,
    grad_grad_out,
    grad_charges,
    grad_dipoles,
    grad_quadrupoles,
    wp_dtype,
    device=None,
):
    r"""Double-backward launcher for the self-energy op.

    Launch Grid
    -----------
    ``dim = (N,)`` — one thread per atom.
    """
    if device is None:
        device = str(charges.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _multipole_self_energy_double_backward_overloads[vec_dtype],
        dim=(charges.shape[0],),
        inputs=[
            gg_charges,
            gg_dipoles,
            gg_quadrupoles,
            grad_out,
            charges,
            dipoles,
            quadrupoles,
            c_self_q,
            c_self_mu,
            c_self_q2,
            int(has_dipoles),
            int(has_quadrupoles),
        ],
        outputs=[grad_grad_out, grad_charges, grad_dipoles, grad_quadrupoles],
        device=device,
    )


# =============================================================================
# PME mesh inner product: per-grid-point ``rho_grid * phi_grid``
# =============================================================================
#
# The PME reciprocal energy is the mesh inner product
#
#   E = Σ_g rho_grid[g] * phi_grid[g]
#
# The per-grid-point product physics ``e_g = rho_grid[g] * phi_grid[g]`` lives
# in this kernel; the torch wrapper owns the reduction + ``F/(4 pi)`` scale.


@wp.kernel(enable_backward=False)
def _multipole_pme_mesh_inner_product_kernel(
    rho_grid: wp.array(dtype=Any),  # (M,) fp64
    phi_grid: wp.array(dtype=Any),  # (M,) fp64
    per_grid_energy: wp.array(dtype=Any),  # (M,) OUTPUT
):
    r"""Per-grid-point mesh-inner-product contribution ``e_g = rho_g * phi_g``.

    Each thread multiplies one (flattened) grid point; the torch wrapper
    reduces (single ``.sum()`` / batched per-system ``sum``) and applies the
    ``F/(4 pi)`` Coulomb scale.

    Launch Grid
    -----------
    ``dim = (M,)`` — one thread per (flattened) grid point.
    """
    g = wp.tid()
    per_grid_energy[g] = rho_grid[g] * phi_grid[g]


@wp.kernel(enable_backward=False)
def _multipole_pme_mesh_inner_product_backward_kernel(
    grad_out: wp.array(dtype=Any),  # (M,) cotangent on per_grid_energy
    rho_grid: wp.array(dtype=Any),  # (M,) fp64
    phi_grid: wp.array(dtype=Any),  # (M,) fp64
    grad_rho_grid: wp.array(dtype=Any),  # (M,) OUTPUT
    grad_phi_grid: wp.array(dtype=Any),  # (M,) OUTPUT
):
    r"""Backward of :func:`_multipole_pme_mesh_inner_product_kernel`.

    The product ``e_g = rho_g phi_g`` is bilinear, so with the per-grid-point
    cotangent ``g_g = grad_out[g]``:

    .. math::

        \partial L/\partial \rho_g = g_g\,\phi_g, \qquad
        \partial L/\partial \phi_g = g_g\,\rho_g .

    Both grids are autograd-connected (they depend on positions / moments via
    spread + convolve), so both gradients are produced.

    Launch Grid
    -----------
    ``dim = (M,)`` — one thread per (flattened) grid point.
    """
    g = wp.tid()
    gg = grad_out[g]
    grad_rho_grid[g] = gg * phi_grid[g]
    grad_phi_grid[g] = gg * rho_grid[g]


@wp.kernel(enable_backward=False)
def _multipole_pme_mesh_inner_product_double_backward_kernel(
    gg_rho_grid: wp.array(dtype=Any),  # (M,) cotangent on grad_rho_grid
    gg_phi_grid: wp.array(dtype=Any),  # (M,) cotangent on grad_phi_grid
    grad_out: wp.array(dtype=Any),  # (M,)
    rho_grid: wp.array(dtype=Any),  # (M,) fp64
    phi_grid: wp.array(dtype=Any),  # (M,) fp64
    grad_grad_out: wp.array(dtype=Any),  # (M,) OUTPUT
    grad_rho_grid: wp.array(dtype=Any),  # (M,) OUTPUT
    grad_phi_grid: wp.array(dtype=Any),  # (M,) OUTPUT
):
    r"""Double-backward of the mesh-inner-product op (constant cross-Hessian).

    The first-order backward
    ``grad_rho_g = g_g phi_g``, ``grad_phi_g = g_g rho_g`` is bilinear in
    ``(g_g, rho_g, phi_g)``; with upstream cotangents ``(gg_rho, gg_phi)``:

    .. math::

        \partial L_2/\partial g_g &= gg\_rho_g\,\phi_g + gg\_phi_g\,\rho_g, \\
        \partial L_2/\partial \rho_g &= g_g\,gg\_phi_g, \\
        \partial L_2/\partial \phi_g &= g_g\,gg\_rho_g.

    Needed for the PME force-loss / stress HVPs since both grids depend on
    positions / moments / cell and the assembly is bilinear in them.

    Launch Grid
    -----------
    ``dim = (M,)`` — one thread per (flattened) grid point.
    """
    g = wp.tid()
    ggr = gg_rho_grid[g]
    ggp = gg_phi_grid[g]
    go = grad_out[g]
    rho = rho_grid[g]
    phi = phi_grid[g]
    grad_grad_out[g] = ggr * phi + ggp * rho
    grad_rho_grid[g] = go * ggp
    grad_phi_grid[g] = go * ggr


def _multipole_pme_mesh_inner_product_sig(v, t):
    """Signature builder for :func:`_multipole_pme_mesh_inner_product_kernel`."""
    del v
    return [
        wp.array(dtype=t),  # rho_grid
        wp.array(dtype=t),  # phi_grid
        wp.array(dtype=t),  # per_grid_energy (output)
    ]


def _multipole_pme_mesh_inner_product_backward_sig(v, t):
    """Signature builder for the mesh-inner-product backward kernel."""
    del v
    return [
        wp.array(dtype=t),  # grad_out
        wp.array(dtype=t),  # rho_grid
        wp.array(dtype=t),  # phi_grid
        wp.array(dtype=t),  # grad_rho_grid (output)
        wp.array(dtype=t),  # grad_phi_grid (output)
    ]


def _multipole_pme_mesh_inner_product_double_backward_sig(v, t):
    """Signature builder for the mesh-inner-product double-backward kernel."""
    del v
    return [
        wp.array(dtype=t),  # gg_rho_grid
        wp.array(dtype=t),  # gg_phi_grid
        wp.array(dtype=t),  # grad_out
        wp.array(dtype=t),  # rho_grid
        wp.array(dtype=t),  # phi_grid
        wp.array(dtype=t),  # grad_grad_out (output)
        wp.array(dtype=t),  # grad_rho_grid (output)
        wp.array(dtype=t),  # grad_phi_grid (output)
    ]


_multipole_pme_mesh_inner_product_overloads = register_overloads(
    _multipole_pme_mesh_inner_product_kernel,
    _multipole_pme_mesh_inner_product_sig,
)
_multipole_pme_mesh_inner_product_backward_overloads = register_overloads(
    _multipole_pme_mesh_inner_product_backward_kernel,
    _multipole_pme_mesh_inner_product_backward_sig,
)
_multipole_pme_mesh_inner_product_double_backward_overloads = register_overloads(
    _multipole_pme_mesh_inner_product_double_backward_kernel,
    _multipole_pme_mesh_inner_product_double_backward_sig,
)


def multipole_pme_mesh_inner_product_launch(
    rho_grid,
    phi_grid,
    per_grid_energy,
    wp_dtype,
    device=None,
):
    r"""Launcher for :func:`_multipole_pme_mesh_inner_product_kernel`.

    Computes the per-grid-point product ``rho_grid * phi_grid`` over a flat
    ``(M,)`` grid (``M = Nx*Ny*Nz`` single / ``M = B*Nx*Ny*Nz`` batched). The
    torch wrapper owns the reduction + ``F/(4 pi)`` scale.

    Launch Grid
    -----------
    ``dim = (M,)`` — one thread per (flattened) grid point.
    """
    if device is None:
        device = str(rho_grid.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _multipole_pme_mesh_inner_product_overloads[vec_dtype],
        dim=(rho_grid.shape[0],),
        inputs=[rho_grid, phi_grid],
        outputs=[per_grid_energy],
        device=device,
    )


def multipole_pme_mesh_inner_product_backward_launch(
    grad_out,
    rho_grid,
    phi_grid,
    grad_rho_grid,
    grad_phi_grid,
    wp_dtype,
    device=None,
):
    r"""Backward launcher for the PME mesh-inner-product op.

    Launch Grid
    -----------
    ``dim = (M,)`` — one thread per (flattened) grid point.
    """
    if device is None:
        device = str(rho_grid.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _multipole_pme_mesh_inner_product_backward_overloads[vec_dtype],
        dim=(rho_grid.shape[0],),
        inputs=[grad_out, rho_grid, phi_grid],
        outputs=[grad_rho_grid, grad_phi_grid],
        device=device,
    )


def multipole_pme_mesh_inner_product_double_backward_launch(
    gg_rho_grid,
    gg_phi_grid,
    grad_out,
    rho_grid,
    phi_grid,
    grad_grad_out,
    grad_rho_grid,
    grad_phi_grid,
    wp_dtype,
    device=None,
):
    r"""Double-backward launcher for the PME mesh-inner-product op.

    Launch Grid
    -----------
    ``dim = (M,)`` — one thread per (flattened) grid point.
    """
    if device is None:
        device = str(rho_grid.device)
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    wp.launch(
        _multipole_pme_mesh_inner_product_double_backward_overloads[vec_dtype],
        dim=(rho_grid.shape[0],),
        inputs=[gg_rho_grid, gg_phi_grid, grad_out, rho_grid, phi_grid],
        outputs=[grad_grad_out, grad_rho_grid, grad_phi_grid],
        device=device,
    )
