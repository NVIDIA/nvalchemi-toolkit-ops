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

"""JAX Particle Mesh Ewald (PME) implementation.

This module provides JAX bindings for PME long-range electrostatics calculations.
PME achieves O(N log N) scaling through FFT-based reciprocal space computation
combined with real-space Ewald summation.

The implementation uses:
- JAX FFT operations (jnp.fft.rfftn/irfftn)
- B-spline interpolation from nvalchemiops.jax.spline
- Ewald real-space from nvalchemiops.jax.interactions.electrostatics.ewald
- Warp kernels for Green's function and energy corrections

Key Functions
-------------
particle_mesh_ewald : Complete PME calculation (real + reciprocal space)
pme_reciprocal_space : Reciprocal-space component only
pme_green_structure_factor : Green's function and structure factor
pme_energy_corrections : Self-energy and background corrections

See Also
--------
nvalchemiops.jax.interactions.electrostatics.ewald : Ewald real-space
nvalchemiops.jax.spline : B-spline interpolation
"""

from __future__ import annotations

import functools
import math
import warnings

import jax
import jax.numpy as jnp
import warp as wp
from jax.interpreters import ad as jax_ad
from warp.jax_experimental import GraphMode, jax_callable

from nvalchemiops.interactions.electrostatics.pme_factory import get_pme_kernel
from nvalchemiops.interactions.electrostatics.pme_kernels import (
    _batch_pme_green_structure_factor_kernel_overload,
    _pme_green_structure_factor_kernel_overload,
    _pme_virial_bg_apply_kernel_overload,
    _pme_virial_bg_reduce_kernel_overload,
)
from nvalchemiops.jax.interactions.electrostatics._autograd import (
    _inject_charge_grad,
)
from nvalchemiops.jax.interactions.electrostatics._lazy_jax_kernels import (
    _make_jax_kernel_factory,
    _make_jax_kernels,
)
from nvalchemiops.jax.interactions.electrostatics._utils import (
    _build_electrostatic_result,
    _component_direct_output_deprecation_msg,
    _direct_output_deprecation_msg,
    _normalize_dtype,
    _prepare_cell,
)
from nvalchemiops.jax.interactions.electrostatics.ewald import (
    ewald_real_space,
)
from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
    generate_k_vectors_pme,
)
from nvalchemiops.jax.interactions.electrostatics.parameters import (
    estimate_pme_mesh_dimensions,
    estimate_pme_parameters,
    mesh_spacing_to_dimensions,
)
from nvalchemiops.jax.interactions.electrostatics.slab import (
    _prepare_pbc_for_slab,
    _slab_correction_energy_autodiff,
)
from nvalchemiops.jax.interactions.electrostatics.slab import (
    compute_slab_correction as _compute_slab_correction,
)
from nvalchemiops.jax.spline import (
    _spline_gather_gradient_position_hessian,
    _spline_gather_with_force,
    _spline_spread_gradient_weights,
    spline_gather,
    spline_gather_gradient,
    spline_spread,
)

__all__ = [
    "particle_mesh_ewald",
    "pme_reciprocal_space",
    "pme_green_structure_factor",
    "pme_energy_corrections",
    "pme_energy_corrections_with_charge_grad",
    "compute_bspline_moduli_1d",
]


# ==============================================================================
# Helper Function for JAX Kernel Creation
# ==============================================================================

# ``_make_jax_kernels`` returns a lazy dict (see _lazy_jax_kernels) that
# materializes its ``jax_kernel`` entries on first __getitem__. Prefer
# ``jax_kernel`` for single-launch ops; use ``jax_callable`` only when
# fusing multiple wp.launch calls into one FFI thunk
# (see :func:`_make_jax_pme_virial_bg_fused`).


def _jax_pme_factory_component(
    component: str,
    output_names: list[str],
    *,
    batched: bool = False,
    charge_grad: bool = False,
):
    """Return a lazy JAX wrapper for a factory-backed PME component."""
    return _make_jax_kernel_factory(
        lambda wp_dtype: get_pme_kernel(
            wp_dtype,
            component=component,
            batched=batched,
            charge_grad=charge_grad,
        ),
        len(output_names),
        output_names,
    )


# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# Single-system kernels
_jax_pme_green_sf = _make_jax_kernels(
    _pme_green_structure_factor_kernel_overload,
    2,
    ["green_function", "structure_factor_sq"],
)

_jax_pme_energy_corrections = _jax_pme_factory_component(
    "pme_corrections",
    ["corrected_energies"],
)

_jax_pme_energy_corrections_charge_grad = _jax_pme_factory_component(
    "pme_corrections",
    ["corrected_energies", "charge_gradients"],
    charge_grad=True,
)

# Batch kernels
_jax_batch_pme_green_sf = _make_jax_kernels(
    _batch_pme_green_structure_factor_kernel_overload,
    2,
    ["green_function", "structure_factor_sq"],
)

_jax_batch_pme_energy_corrections = _jax_pme_factory_component(
    "pme_corrections",
    ["corrected_energies"],
    batched=True,
)

_jax_batch_pme_energy_corrections_charge_grad = _jax_pme_factory_component(
    "pme_corrections",
    ["corrected_energies", "charge_gradients"],
    batched=True,
    charge_grad=True,
)

# Fused convolve — replaces the older Green's-function + multiply path with
# a single warp kernel that computes G(k), the B-spline structure factor
# correction C^2(k), and multiplies mesh_fft → convolved_mesh in one launch.
# (mirrors the fused-convolve path in the torch bindings.)
_jax_pme_convolve = _jax_pme_factory_component(
    "pme_convolve",
    ["convolved_mesh"],
)

_jax_batch_pme_convolve = _jax_pme_factory_component(
    "pme_convolve",
    ["convolved_mesh"],
    batched=True,
)

# Two-pass virial background correction. Pass 1 reduces per-atom
# charges into per-system total charges (atomic_add). Pass 2 computes
# E_bg = π Q² / (2 α² V) per system and subtracts it from the three diagonal
# entries of ``virial_in``. Mirrors the torch path's ``pme_virial_bg_correction``.
_jax_pme_virial_bg_reduce = _make_jax_kernels(
    _pme_virial_bg_reduce_kernel_overload,
    1,
    ["total_charges"],
)

_jax_pme_virial_bg_apply = _make_jax_kernels(
    _pme_virial_bg_apply_kernel_overload,
    1,
    ["virial_out"],
)


# Fuse the two-pass virial bg correction (reduce + apply) into one XLA FFI
# call via jax_callable so JAX can CUDA-graph it with the surrounding warp
# ops + FFTs. Per-pass jax_kernel thunks above remain for direct test use.
def _make_jax_pme_virial_bg_fused(wp_dtype):
    reduce_overload = _pme_virial_bg_reduce_kernel_overload[wp_dtype]
    apply_overload = _pme_virial_bg_apply_kernel_overload[wp_dtype]

    def _fn(
        # inputs
        charges: wp.array(dtype=wp_dtype),
        batch_idx: wp.array(dtype=wp.int32),
        cell: wp.array3d(dtype=wp_dtype),
        alpha: wp.array(dtype=wp_dtype),
        virial_in: wp.array3d(dtype=wp_dtype),
        # in-out: zero-initialized by caller, scatter-add target for pass 1
        total_charges: wp.array(dtype=wp_dtype),
        # outputs
        virial_out: wp.array3d(dtype=wp_dtype),
    ):
        # Reference closure-captured names so they survive as
        # ``__closure__`` cells. ``from __future__ import annotations``
        # stringifies the annotations, so they don't on their own pull
        # ``wp_dtype`` into the closure -- and warp's annotation eval
        # then can't resolve it. Touching the names in the body fixes that.
        _ = wp_dtype
        wp.launch(
            reduce_overload,
            dim=charges.shape,
            inputs=[charges, batch_idx],
            outputs=[total_charges],
        )
        wp.launch(
            apply_overload,
            dim=total_charges.shape,
            inputs=[total_charges, cell, alpha, virial_in],
            outputs=[virial_out],
        )

    return jax_callable(
        _fn,
        num_outputs=2,
        in_out_argnames=["total_charges"],
        graph_mode=GraphMode.JAX,
    )


_jax_pme_virial_bg_fused = {
    jnp.float32: _make_jax_pme_virial_bg_fused(wp.float32),
    jnp.float64: _make_jax_pme_virial_bg_fused(wp.float64),
}


# ==============================================================================
# Public API Functions
# ==============================================================================


def compute_bspline_moduli_1d(
    miller_indices: jax.Array,
    mesh_N: int,
    spline_order: int,
) -> jax.Array:
    """Precompute a 1D B-spline modulus LUT for one PME mesh axis.

    Returns ``b[i] = sinc(m_i / N)^spline_order`` for each Miller index
    ``m_i`` (with ``sinc(x) = sin(pi*x)/(pi*x)``, ``sinc(0) = 1``). The
    three-axis product ``b_x[i] * b_y[j] * b_z[k]`` is the B-spline
    structure factor consumed by ``pme_fused_convolve``. Precomputing the
    LUT lets the convolve kernel replace three sinc transcendentals + an
    order-dependent power loop per (i, j, k) thread with three reads + two
    multiplies.
    """
    # sinc(x) for x in [-0.5, 0.5] is bounded in [2/pi, 1], so s^spline_order
    # (for orders 2-6) stays well within fp32 range. Stay in the input dtype
    # to avoid an fp32 -> fp64 -> fp32 round-trip per call. jax.numpy.sinc uses
    # the normalized convention sinc(pi*x)/(pi*x); matches torch.
    arg = miller_indices / float(mesh_N)
    s = jnp.sinc(arg)
    return s**spline_order


def pme_fused_convolve(
    mesh_fft: jax.Array,
    k_squared: jax.Array,
    moduli_x: jax.Array,
    moduli_y: jax.Array,
    moduli_z: jax.Array,
    alpha: jax.Array,
    volume: jax.Array,
    is_batch: bool,
) -> jax.Array:
    """Fused Green's function + structure-factor multiply, single launch.

    Replaces (compute G(k), compute C^2(k), divide mesh_fft by C^2, multiply
    by G(k)) with a single warp kernel. ``moduli_x/y/z`` are precomputed
    1D B-spline modulus LUTs (``sinc(m/N)^spline_order`` per axis); see
    ``compute_bspline_moduli_1d``.

    Parameters
    ----------
    mesh_fft : complex64 or complex128
        FFT of the charge mesh. Shape (Nx, Ny, Nz_rfft) for single system
        or (B, Nx, Ny, Nz_rfft) for batch.
    k_squared : float32 or float64
        |k|^2 at each grid point. Same leading shape as mesh_fft.
    moduli_x, moduli_y, moduli_z : float32 or float64
        Per-axis B-spline modulus LUTs.
    alpha : float32 or float64
        Ewald splitting parameter. Shape (1,) or (B,).
    volume : float32 or float64
        Cell volume. Shape (1,) or (B,).
    is_batch : bool
        Whether this is a batched call.

    Returns
    -------
    convolved_mesh : complex64 or complex128, same shape as mesh_fft.
    """
    real_dtype = jnp.float32 if mesh_fft.dtype == jnp.complex64 else jnp.float64
    complex_dtype = mesh_fft.dtype
    input_dtype = _normalize_dtype(real_dtype)

    # generate_k_vectors_pme squeezes the batch dim when B=1 — restore it
    # for the batch kernel which expects (B, nx, ny, nz_r).
    squeeze_output = False
    if is_batch and k_squared.ndim == 3:
        k_squared = k_squared[jnp.newaxis, ...]
    if is_batch and mesh_fft.ndim == 3:
        mesh_fft = mesh_fft[jnp.newaxis, ...]
        squeeze_output = True

    # Reinterpret complex (N, ..., M) -> real (N, ..., M, 2) for the
    # vec2-typed warp kernel. jax.lax.bitcast_convert_type doesn't accept
    # complex→float, so use .view() (doubles the trailing dim) and then
    # reshape to add the explicit vec2 axis.
    mesh_fft_real = mesh_fft.view(real_dtype).reshape(*mesh_fft.shape, 2)

    # Ensure alpha / volume are 1-D arrays of the right dtype.
    alpha = alpha.astype(real_dtype)
    volume = volume.astype(real_dtype)
    if alpha.ndim == 0:
        alpha = alpha.reshape(1)
    if volume.ndim == 0:
        volume = volume.reshape(1)

    moduli_x = moduli_x.astype(real_dtype)
    moduli_y = moduli_y.astype(real_dtype)
    moduli_z = moduli_z.astype(real_dtype)
    k_squared = k_squared.astype(real_dtype)

    # Pre-allocate output (same shape as mesh_fft_real, treated as in-out
    # by the warp kernel).
    convolved_real = jnp.zeros_like(mesh_fft_real)

    if is_batch:
        kernel = _jax_batch_pme_convolve[input_dtype]
    else:
        kernel = _jax_pme_convolve[input_dtype]

    # Launch dims match the spectrum shape (drop the trailing length-2 vec2 dim).
    launch_dims = mesh_fft.shape
    (convolved_real,) = kernel(
        mesh_fft_real,
        k_squared,
        moduli_x,
        moduli_y,
        moduli_z,
        alpha,
        volume,
        convolved_real,
        launch_dims=launch_dims,
    )

    # Reverse the reshape+view: collapse the trailing vec2 dim, then view
    # as complex (which halves the trailing dim back to the original).
    convolved_flat = convolved_real.reshape(*mesh_fft.shape[:-1], -1)
    convolved = convolved_flat.view(complex_dtype)
    if squeeze_output:
        convolved = convolved.squeeze(0)
    return convolved


def pme_green_structure_factor(
    k_squared: jax.Array,
    mesh_dimensions: tuple[int, int, int],
    alpha: jax.Array,
    cell: jax.Array,
    spline_order: int = 4,
    batch_idx: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute Green's function and B-spline structure factor correction.

    Computes the Coulomb Green's function with volume normalization and the
    B-spline aliasing correction factor for PME.

    Green's function (volume-normalized):
        G(k) = (2π/V) * exp(-k²/(4α²)) / k²

    Structure factor correction (for B-spline deconvolution):
        C²(k) = [sinc(m_x/N_x) · sinc(m_y/N_y) · sinc(m_z/N_z)]^(2p)

    where p is the spline order.

    Parameters
    ----------
    k_squared : jax.Array
        |k|² values at each FFT grid point.
        - Single-system: shape (Nx, Ny, Nz_rfft)
        - Batch: shape (B, Nx, Ny, Nz_rfft)
    mesh_dimensions : tuple[int, int, int]
        Full mesh dimensions (Nx, Ny, Nz) before rfft.
    alpha : jax.Array
        Ewald splitting parameter.
        - Single-system: shape (1,) or scalar
        - Batch: shape (B,)
    cell : jax.Array
        Unit cell matrices.
        - Single-system: shape (3, 3) or (1, 3, 3)
        - Batch: shape (B, 3, 3)
    spline_order : int, default=4
        B-spline interpolation order (typically 4 for cubic B-splines).
    batch_idx : jax.Array | None, default=None
        If provided, dispatches to batch kernels.

    Returns
    -------
    green_function : jax.Array
        Volume-normalized Green's function G(k).
        - Single-system: shape (Nx, Ny, Nz_rfft)
        - Batch: shape (B, Nx, Ny, Nz_rfft)
    structure_factor_sq : jax.Array
        Squared structure factor C²(k) for B-spline deconvolution.
        Shape (Nx, Ny, Nz_rfft), shared across batch.

    Notes
    -----
    - G(k=0) is set to zero to avoid singularity
    - The volume normalization in G(k) eliminates later divisions
    - Structure factor is mesh-dependent only, so shared across batch
    - This compatibility helper returns raw ``G(k)``. The fused convolve path
      folds deconvolution internally as ``G(k) / C^2(k)``.
    """
    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
    input_dtype = _normalize_dtype(k_squared.dtype)

    # Ensure cell is correct shape
    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    volume = jnp.abs(jnp.linalg.det(cell)).astype(input_dtype)

    # Generate Miller indices using JAX FFT frequency functions
    # Use d=1.0/n to get integer Miller indices
    miller_x = jnp.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx).astype(input_dtype)
    miller_y = jnp.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny).astype(input_dtype)
    miller_z = jnp.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz).astype(input_dtype)

    # Ensure alpha is 1D array
    if alpha.ndim == 0:
        alpha = alpha.reshape(1)
    alpha = alpha.astype(input_dtype)

    # Get kernel for input dtype
    if batch_idx is None:
        # Single system
        kernel = _jax_pme_green_sf[input_dtype]

        # Allocate outputs
        green_function = jnp.zeros(
            (mesh_nx, mesh_ny, mesh_nz // 2 + 1), dtype=input_dtype
        )
        structure_factor_sq = jnp.zeros(
            (mesh_nx, mesh_ny, mesh_nz // 2 + 1), dtype=input_dtype
        )

        # Launch kernel
        green_out, sf_out = kernel(
            k_squared.astype(input_dtype),
            miller_x,
            miller_y,
            miller_z,
            alpha,
            volume,
            int(mesh_nx),
            int(mesh_ny),
            int(mesh_nz),
            int(spline_order),
            green_function,
            structure_factor_sq,
            launch_dims=(mesh_nx, mesh_ny, mesh_nz // 2 + 1),
        )
        return green_out, sf_out
    else:
        # Batch
        num_systems = cell.shape[0]
        kernel = _jax_batch_pme_green_sf[input_dtype]

        # Ensure k_squared has batch dimension for batch kernels
        k_sq = k_squared.astype(input_dtype)
        if k_sq.ndim == 3:
            k_sq = jnp.broadcast_to(
                k_sq[jnp.newaxis], (num_systems, mesh_nx, mesh_ny, mesh_nz // 2 + 1)
            )

        # Allocate outputs
        green_function = jnp.zeros(
            (num_systems, mesh_nx, mesh_ny, mesh_nz // 2 + 1), dtype=input_dtype
        )
        structure_factor_sq = jnp.zeros(
            (mesh_nx, mesh_ny, mesh_nz // 2 + 1), dtype=input_dtype
        )

        # Launch kernel
        green_out, sf_out = kernel(
            k_sq,
            miller_x,
            miller_y,
            miller_z,
            alpha,
            volume,
            int(mesh_nx),
            int(mesh_ny),
            int(mesh_nz),
            int(spline_order),
            green_function,
            structure_factor_sq,
            launch_dims=(num_systems, mesh_nx, mesh_ny, mesh_nz // 2 + 1),
        )
        return green_out, sf_out


def pme_virial_bg_correction(
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    virial: jax.Array,
    batch_idx: jax.Array | None = None,
    volume: jax.Array | None = None,
) -> jax.Array:
    """Apply non-neutral background virial correction in a single Warp launch.

    Two-pass fused kernel:
      1. Reduce per-atom ``charges`` into per-system totals (atomic_add).
      2. Compute ``E_bg = π Q² / (2 α² V)`` per system and subtract it from
         the three diagonal entries of ``virial`` (off-diagonal unchanged).

    Single-system inputs are fanned out via ``batch_idx`` filled with zeros.

    Parameters
    ----------
    charges : jax.Array, shape (N,)
        Per-atom charges.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell. Single-system 2D is promoted to (1, 3, 3).
    alpha : jax.Array, shape () / (1,) / (B,)
        Per-system Ewald splitting parameter.
    virial : jax.Array, shape (1, 3, 3) or (B, 3, 3)
        Virial tensor to correct in place (functional return).
    batch_idx : jax.Array | None, shape (N,), dtype=int32, optional
        System index per atom. If None, every atom maps to system 0.
    volume : jax.Array | None, optional
        Precomputed per-system cell volume. Treated as static metadata when
        supplied.

    Returns
    -------
    virial_out : jax.Array, same shape as ``virial``
        Background-corrected virial.
    """
    input_dtype = _normalize_dtype(charges.dtype)

    cell_w = cell.astype(input_dtype)
    if cell_w.ndim == 2:
        cell_w = cell_w[jnp.newaxis, :, :]
    num_systems = cell_w.shape[0]
    num_atoms = charges.shape[0]

    alpha_w = alpha.astype(input_dtype)
    if alpha_w.ndim == 0:
        alpha_w = alpha_w.reshape(1)
    if alpha_w.shape[0] == 1 and num_systems > 1:
        alpha_w = jnp.broadcast_to(alpha_w, (num_systems,))

    if batch_idx is None:
        bidx = jnp.zeros(num_atoms, dtype=jnp.int32)
    else:
        bidx = batch_idx.astype(jnp.int32)

    virial_w = virial.astype(input_dtype)
    if virial_w.ndim == 2:
        virial_w = virial_w[jnp.newaxis, :, :]

    total_charges = jnp.zeros(num_systems, dtype=input_dtype)
    if volume is not None:
        total_charges = total_charges.at[bidx].add(charges.astype(input_dtype))
        volume_w = volume.astype(input_dtype)
        if volume_w.ndim == 0:
            volume_w = volume_w.reshape(1)
        if volume_w.shape[0] == 1 and num_systems > 1:
            volume_w = jnp.broadcast_to(volume_w, (num_systems,))
        bg_energy = (
            jnp.pi
            * total_charges
            * total_charges
            / (2.0 * alpha_w * alpha_w * volume_w)
        )
        eye = jnp.eye(3, dtype=input_dtype)
        diag_delta = bg_energy[:, None, None] * eye[jnp.newaxis, :, :]
        return virial_w - diag_delta

    # Fused two-pass via jax_callable: a single XLA FFI thunk that runs both
    # the scatter-add reduce (pass 1) and the per-system E_bg apply (pass 2).
    # This collapses what used to be 2 jax_kernel thunks into 1 and lets JAX
    # CUDA-graph-capture both passes together.
    fused = _jax_pme_virial_bg_fused[input_dtype]
    _total_charges, virial_out = fused(
        charges.astype(input_dtype),
        bidx,
        cell_w,
        alpha_w,
        virial_w,
        total_charges,  # in/out — zero-initialized scatter-add target
        output_dims={"virial_out": virial_w.shape},
    )
    return virial_out


def pme_energy_corrections(
    raw_energies: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    batch_idx: jax.Array | None = None,
    volume: jax.Array | None = None,
) -> jax.Array:
    """Apply self-energy and background corrections to PME energies.

    Converts raw interpolated potential to energy and subtracts corrections:

        E_i = q_i φ_i - E_self,i - E_background,i

    Self-energy correction (removes Gaussian self-interaction):
        E_self,i = (α/√π) q_i²

    Background correction (for non-neutral systems):
        E_background,i = (π/(2α²V)) q_i Q_total

    Parameters
    ----------
    raw_energies : jax.Array, shape (N,) or (N_total,)
        Raw potential values φ_i from mesh interpolation.
    charges : jax.Array, shape (N,) or (N_total,)
        Atomic charges.
    cell : jax.Array
        Unit cell matrices.
        - Single-system: shape (3, 3) or (1, 3, 3)
        - Batch: shape (B, 3, 3)
    alpha : jax.Array
        Ewald splitting parameter.
        - Single-system: shape (1,) or scalar
        - Batch: shape (B,)
    batch_idx : jax.Array | None, default=None
        System index for each atom. If provided, uses batch kernels. Atoms must
        be grouped by system: ``batch_idx`` must be contiguous, nondecreasing,
        and use system IDs ``0..B-1``.
    volume : jax.Array | None, optional
        Precomputed per-system cell volume. Treated as static setup metadata
        when supplied.

    Returns
    -------
    corrected_energies : jax.Array, shape (N,) or (N_total,)
        Final per-atom reciprocal-space energy with corrections applied.

    Notes
    -----
    - For neutral systems, background correction is zero
    - Supports both float32 and float64 dtypes
    """
    input_dtype = _normalize_dtype(raw_energies.dtype)
    num_atoms = raw_energies.shape[0]

    # Ensure alpha is 1D array
    if alpha.ndim == 0:
        alpha = alpha.reshape(1)
    alpha = alpha.astype(input_dtype)

    if batch_idx is None:
        # Single system
        kernel = _jax_pme_energy_corrections[input_dtype]

        # Ensure cell is correct shape
        if cell.ndim == 2:
            cell = cell[jnp.newaxis, :, :]
        if volume is None:
            volume = jnp.abs(jnp.linalg.det(cell)).astype(input_dtype)
        else:
            volume = volume.astype(input_dtype)
            if volume.ndim == 0:
                volume = volume.reshape(1)
        total_charge = charges.sum().reshape(1).astype(input_dtype)

        # Allocate output
        corrected_energies = jnp.zeros(num_atoms, dtype=input_dtype)
        batch_idx_dummy = jnp.zeros((num_atoms,), dtype=jnp.int32)
        charge_gradients_dummy = jnp.zeros(num_atoms, dtype=input_dtype)

        # Launch kernel
        (corrected_out,) = kernel(
            raw_energies.astype(input_dtype),
            charges.astype(input_dtype),
            batch_idx_dummy,
            volume,
            alpha,
            total_charge,
            corrected_energies,
            charge_gradients_dummy,
            launch_dims=(num_atoms,),
        )
        return corrected_out
    else:
        # Batch
        kernel = _jax_batch_pme_energy_corrections[input_dtype]
        num_systems = cell.shape[0] if cell.ndim == 3 else 1

        if cell.ndim == 2:
            cell = cell[jnp.newaxis, :, :]
        if volume is None:
            volumes = jnp.abs(jnp.linalg.det(cell)).astype(input_dtype)
        else:
            volumes = volume.astype(input_dtype)
            if volumes.ndim == 0:
                volumes = volumes.reshape(1)
            if volumes.shape[0] == 1 and num_systems > 1:
                volumes = jnp.broadcast_to(volumes, (num_systems,))

        # Compute total charge per system
        total_charges = jnp.zeros(num_systems, dtype=input_dtype)
        total_charges = total_charges.at[batch_idx].add(charges.astype(input_dtype))

        # Allocate output
        corrected_energies = jnp.zeros(num_atoms, dtype=input_dtype)
        charge_gradients_dummy = jnp.zeros(num_atoms, dtype=input_dtype)

        # Launch kernel
        (corrected_out,) = kernel(
            raw_energies.astype(input_dtype),
            charges.astype(input_dtype),
            batch_idx.astype(jnp.int32),
            volumes,
            alpha,
            total_charges,
            corrected_energies,
            charge_gradients_dummy,
            launch_dims=(num_atoms,),
        )
        return corrected_out


def pme_energy_corrections_with_charge_grad(
    raw_energies: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    batch_idx: jax.Array | None = None,
    volume: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Apply energy corrections and compute charge gradients.

    Same as pme_energy_corrections but also returns dE/dq for each atom.

    Parameters
    ----------
    raw_energies : jax.Array, shape (N,) or (N_total,)
        Raw potential values φ_i from mesh interpolation.
    charges : jax.Array, shape (N,) or (N_total,)
        Atomic charges.
    cell : jax.Array
        Unit cell matrices.
        - Single-system: shape (3, 3) or (1, 3, 3)
        - Batch: shape (B, 3, 3)
    alpha : jax.Array
        Ewald splitting parameter.
        - Single-system: shape (1,) or scalar
        - Batch: shape (B,)
    batch_idx : jax.Array | None, default=None
        System index for each atom. If provided, uses batch kernels. Atoms must
        be grouped by system: ``batch_idx`` must be contiguous, nondecreasing,
        and use system IDs ``0..B-1``.
    volume : jax.Array | None, optional
        Precomputed per-system cell volume. Treated as static setup metadata
        when supplied.

    Returns
    -------
    corrected_energies : jax.Array, shape (N,) or (N_total,)
        Final per-atom reciprocal-space energy with corrections applied.
    charge_gradients : jax.Array, shape (N,) or (N_total,)
        Per-atom charge gradients dE/dq.

    Notes
    -----
    - Useful for training models that predict partial charges
    - Supports both float32 and float64 dtypes
    """
    input_dtype = _normalize_dtype(raw_energies.dtype)
    num_atoms = raw_energies.shape[0]

    # Ensure alpha is 1D array
    if alpha.ndim == 0:
        alpha = alpha.reshape(1)
    alpha = alpha.astype(input_dtype)

    if batch_idx is None:
        # Single system
        kernel = _jax_pme_energy_corrections_charge_grad[input_dtype]

        # Ensure cell is correct shape
        if cell.ndim == 2:
            cell = cell[jnp.newaxis, :, :]
        if volume is None:
            volume = jnp.abs(jnp.linalg.det(cell)).astype(input_dtype)
        else:
            volume = volume.astype(input_dtype)
            if volume.ndim == 0:
                volume = volume.reshape(1)
        total_charge = charges.sum().reshape(1).astype(input_dtype)

        # Allocate outputs
        corrected_energies = jnp.zeros(num_atoms, dtype=input_dtype)
        charge_gradients = jnp.zeros(num_atoms, dtype=input_dtype)
        batch_idx_dummy = jnp.zeros((num_atoms,), dtype=jnp.int32)

        # Launch kernel
        corrected_out, charge_grad_out = kernel(
            raw_energies.astype(input_dtype),
            charges.astype(input_dtype),
            batch_idx_dummy,
            volume,
            alpha,
            total_charge,
            corrected_energies,
            charge_gradients,
            launch_dims=(num_atoms,),
        )
        return corrected_out, charge_grad_out
    else:
        # Batch
        kernel = _jax_batch_pme_energy_corrections_charge_grad[input_dtype]
        num_systems = cell.shape[0] if cell.ndim == 3 else 1

        if cell.ndim == 2:
            cell = cell[jnp.newaxis, :, :]
        if volume is None:
            volumes = jnp.abs(jnp.linalg.det(cell)).astype(input_dtype)
        else:
            volumes = volume.astype(input_dtype)
            if volumes.ndim == 0:
                volumes = volumes.reshape(1)
            if volumes.shape[0] == 1 and num_systems > 1:
                volumes = jnp.broadcast_to(volumes, (num_systems,))

        # Compute total charge per system
        total_charges = jnp.zeros(num_systems, dtype=input_dtype)
        total_charges = total_charges.at[batch_idx].add(charges.astype(input_dtype))

        # Allocate outputs
        corrected_energies = jnp.zeros(num_atoms, dtype=input_dtype)
        charge_gradients = jnp.zeros(num_atoms, dtype=input_dtype)

        # Launch kernel
        corrected_out, charge_grad_out = kernel(
            raw_energies.astype(input_dtype),
            charges.astype(input_dtype),
            batch_idx.astype(jnp.int32),
            volumes,
            alpha,
            total_charges,
            corrected_energies,
            charge_gradients,
            launch_dims=(num_atoms,),
        )
        return corrected_out, charge_grad_out


def _compute_pme_reciprocal_virial(
    mesh_fft_raw: jax.Array,
    convolved_mesh: jax.Array,
    k_vectors: jax.Array,
    k_squared: jax.Array,
    alpha: jax.Array,
    mesh_dimensions: tuple[int, int, int],
    is_batch: bool,
) -> jax.Array:
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
    mesh_fft_raw : jax.Array
        Raw rfftn output before B-spline deconvolution.
        Shape (nx, ny, nz//2+1) or (B, nx, ny, nz//2+1), complex.
    convolved_mesh : jax.Array
        Deconvolved mesh FFT multiplied by Green's function: (mesh_fft/B^2)*G.
        Shape matching mesh_fft_raw.
    k_vectors : jax.Array
        k-vectors on the mesh. Shape (..., nx, ny, nz//2+1, 3).
    k_squared : jax.Array
        |k|^2. Shape (..., nx, ny, nz//2+1).
    alpha : jax.Array
        Ewald splitting parameter.
    mesh_dimensions : tuple
        (nx, ny, nz).
    is_batch : bool
        Whether this is a batched calculation.

    Returns
    -------
    virial : jax.Array, shape (B, 3, 3) or (1, 3, 3)
        Per-system virial tensor.
    """
    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions

    # Determine accumulation dtype from k_squared (float32 or float64)
    acc_dtype = _normalize_dtype(k_squared.dtype)
    complex_dtype = jnp.complex64 if acc_dtype == jnp.float32 else jnp.complex128

    # Per-k energy density from exact pipeline spectral pair.
    # Re(mesh_fft_raw * convolved_mesh*) = |mesh_fft_raw|^2 * G / B^2
    fft_raw_cast = mesh_fft_raw.astype(complex_dtype)
    conv_cast = convolved_mesh.astype(complex_dtype)
    energy_density = (fft_raw_cast * jnp.conj(conv_cast)).real

    # Weight for rfft symmetry: 2 for interior k_z, 1 for boundary
    weight = jnp.full_like(energy_density, 2.0)
    weight = weight.at[..., 0].set(1.0)  # k_z = 0
    if mesh_nz % 2 == 0:
        weight = weight.at[..., -1].set(1.0)  # k_z = nz//2 (Nyquist)

    # Weighted energy density
    weighted_energy = weight * energy_density

    # Virial W = -dE/dε, so sigma_ab = delta_ab - 2*k_a*k_b/k^2 * (1 + k^2/(4*alpha^2))
    k_sq_acc = k_squared.astype(acc_dtype)
    alpha_acc = alpha.astype(acc_dtype)

    # generate_k_vectors_pme squeezes the batch dim when B=1; restore it so
    # the batched einsum and sum_dims=(1,2,3) operate on the correct axes.
    if is_batch and k_sq_acc.ndim == 3:
        k_sq_acc = jnp.expand_dims(k_sq_acc, axis=0)

    # Handle alpha broadcasting: alpha may be (B,) for batch
    if is_batch and alpha_acc.ndim == 1:
        alpha_view = alpha_acc.reshape(-1, 1, 1, 1)
    else:
        alpha_view = alpha_acc.reshape(-1) if alpha_acc.ndim == 0 else alpha_acc

    exp_factor = 0.25 / (alpha_view**2)

    # Avoid division by zero at k=0
    safe_k_sq = jnp.maximum(k_sq_acc, 1e-30)
    k_factor = 2.0 * (1.0 + k_sq_acc * exp_factor) / safe_k_sq

    # Zero out k=0 contribution (no virial from k=0)
    k_mask = k_sq_acc > 1e-10

    # Six per-component weighted reductions instead of einsum: XLA lowers
    # einsum(k,k,m) to a slow small-MN / large-K cuBLAS GEMM.
    k_vecs_acc = k_vectors.astype(acc_dtype)  # (..., nx, ny, nz//2+1, 3)
    if is_batch and k_vecs_acc.ndim == 4:
        k_vecs_acc = jnp.expand_dims(k_vecs_acc, axis=0)

    masked_energy = weighted_energy * k_mask  # (..., nx, ny, nz//2+1)
    masked_energy_kf = masked_energy * k_factor  # (..., nx, ny, nz//2+1)

    if is_batch:
        sum_dims = (1, 2, 3)
    else:
        sum_dims = (0, 1, 2)

    trace_term = masked_energy.sum(axis=sum_dims)  # scalar or (B,)

    kx = k_vecs_acc[..., 0]
    ky = k_vecs_acc[..., 1]
    kz = k_vecs_acc[..., 2]
    xx = (kx * kx * masked_energy_kf).sum(axis=sum_dims)
    yy = (ky * ky * masked_energy_kf).sum(axis=sum_dims)
    zz = (kz * kz * masked_energy_kf).sum(axis=sum_dims)
    xy = (kx * ky * masked_energy_kf).sum(axis=sum_dims)
    xz = (kx * kz * masked_energy_kf).sum(axis=sum_dims)
    yz = (ky * kz * masked_energy_kf).sum(axis=sum_dims)

    eye = jnp.eye(3, dtype=acc_dtype)
    if is_batch:
        kk_term = jnp.stack(
            [
                jnp.stack([xx, xy, xz], axis=-1),
                jnp.stack([xy, yy, yz], axis=-1),
                jnp.stack([xz, yz, zz], axis=-1),
            ],
            axis=-2,
        )
        virial = eye * trace_term[:, jnp.newaxis, jnp.newaxis] - kk_term  # (B, 3, 3)
    else:
        kk_term = jnp.stack(
            [jnp.stack([xx, xy, xz]), jnp.stack([xy, yy, yz]), jnp.stack([xz, yz, zz])],
        )  # (3, 3)
        virial = (eye * trace_term - kk_term)[jnp.newaxis, :, :]  # (1, 3, 3)

    return virial.astype(acc_dtype)


def _pme_reciprocal_space_impl(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    mesh_dimensions: tuple[int, int, int] | None = None,
    mesh_spacing: float | None = None,
    spline_order: int = 4,
    batch_idx: jax.Array | None = None,
    k_vectors: jax.Array | None = None,
    k_squared: jax.Array | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    hybrid_forces: bool = False,
    volume: jax.Array | None = None,
    cell_inv_t: jax.Array | None = None,
    moduli_x: jax.Array | None = None,
    moduli_y: jax.Array | None = None,
    moduli_z: jax.Array | None = None,
) -> (
    jax.Array
    | tuple[jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Compute PME reciprocal-space contribution implementation.

    Implements the FFT-based long-range component of PME using B-spline
    interpolation and convolution with the Green's function.

    Pipeline:
        1. Spread charges to mesh (spline_spread)
        2. FFT → frequency space
        3. Compute Green's function and structure factor
        4. Convolve: mesh_fft * G(k) / C²(k)
        5. IFFT → potential mesh
        6. Gather potential at atoms (spline_gather)
        7. Apply self-energy and background corrections
        8. (Optional) Compute forces via Fourier gradient

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    charges : jax.Array, shape (N,)
        Atomic partial charges.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrices with lattice vectors as rows.
    alpha : jax.Array
        Ewald splitting parameter.
        - Single-system: shape (1,) or scalar
        - Batch: shape (B,)
    mesh_dimensions : tuple[int, int, int], optional
        FFT mesh dimensions (nx, ny, nz).
    mesh_spacing : float, optional
        Target mesh spacing. Used to compute mesh_dimensions if not provided.
    spline_order : int, default=4
        B-spline interpolation order (4 = cubic).
    batch_idx : jax.Array | None, default=None
        System index for each atom. When provided, atoms must be grouped by
        system: ``batch_idx`` must be contiguous, nondecreasing, and use system
        IDs ``0..B-1``.
    k_vectors : jax.Array, optional
        Precomputed k-vectors from generate_k_vectors_pme.
    k_squared : jax.Array, optional
        Precomputed k² values from generate_k_vectors_pme.
    compute_forces : bool, default=False
        If True, compute forces via Fourier gradient.
    compute_charge_gradients : bool, default=False
        If True, compute charge gradients dE/dq.
    compute_virial : bool, default=False
        If True, compute the virial tensor ``W = -dE/d(displacement)`` for the
        row-vector displacement recipe.
        Stress = -virial / volume.
    hybrid_forces : bool, default=False
        If True, detach ``positions``/``charges``/``cell`` from the autograd
        graph through the spline/FFT chain (forces and virial become
        forward-only), and inject analytical ∂E/∂q via a custom-VJP straight-
        through trick so ``jax.grad`` w.r.t. charges propagates correctly.
        Forces ``compute_charge_gradients=True``.

    Returns
    -------
    energies : jax.Array, shape (N,)
        Per-atom reciprocal-space energies.
    forces : jax.Array, shape (N, 3), optional
        Per-atom forces (only if compute_forces=True).
    charge_gradients : jax.Array, shape (N,), optional
        Per-atom charge gradients (only if compute_charge_gradients=True).
    virial : jax.Array, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor (only if compute_virial=True). Always last in the return tuple.

    Notes
    -----
    - Output dtype for energy/forces matches the input positions dtype
    - FFT/convolution and spline operations all respect the input dtype
    - Automatically determines mesh_dimensions if not provided
    - Virial is computed in k-space and uses the same dtype as k_squared
    - Energy-derived gradients are supported for positions, charges, and
      strain-first virials. Reverse-mode higher-order reciprocal position and
      charge losses use the private PME mesh HVP path.
    """
    num_atoms = positions.shape[0]
    input_dtype = _normalize_dtype(positions.dtype)
    is_batch = batch_idx is not None
    fft_dims = (1, 2, 3) if is_batch else (0, 1, 2)
    reciprocal_metadata_is_supplied = k_vectors is not None and k_squared is not None
    volume_is_supplied = volume is not None

    # hybrid_forces: forward-only spline/FFT chain. We sever ∂/∂{positions,
    # cell} (and the spline/FFT path through charges) via lax.stop_gradient,
    # then re-attach the analytical ∂E/∂q at the end. Charges still need to
    # be a tracer for the custom-VJP injector to see them, so save the
    # original handle.
    charges_orig = charges
    need_charge_gradients = compute_charge_gradients or hybrid_forces
    if hybrid_forces:
        positions = jax.lax.stop_gradient(positions)
        charges = jax.lax.stop_gradient(charges)
        cell = jax.lax.stop_gradient(cell)
        alpha = jax.lax.stop_gradient(alpha)
        if cell_inv_t is not None:
            cell_inv_t = jax.lax.stop_gradient(cell_inv_t)

    # Ensure cell is correct shape for num_systems calculation
    if cell.ndim == 2:
        num_systems = 1
    else:
        num_systems = cell.shape[0]

    # Handle empty systems
    if num_atoms == 0:
        energies = jnp.zeros(num_atoms, dtype=input_dtype)
        forces = (
            jnp.zeros((num_atoms, 3), dtype=input_dtype) if compute_forces else None
        )
        charge_grads = (
            jnp.zeros(num_atoms, dtype=input_dtype) if need_charge_gradients else None
        )
        virial = (
            jnp.zeros((num_systems, 3, 3), dtype=input_dtype)
            if compute_virial
            else None
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

    _require_explicit_mesh_dimensions_in_tracing(
        mesh_dimensions=mesh_dimensions,
        cell=cell,
        alpha=alpha,
        batch_idx=batch_idx,
    )

    # Determine mesh dimensions
    if mesh_dimensions is None:
        if mesh_spacing is not None:
            mesh_dimensions = mesh_spacing_to_dimensions(cell, mesh_spacing)
        else:
            # Default estimation
            mesh_dimensions = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-6)

    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions

    # cell_inv_t cache: MD callers can pass cell_inv_t in (NVT case)
    # to skip the per-step linalg.inv + transpose. We compute it once here
    # and forward to spline_spread / spline_gather(_with_force), so any
    # work upstream of the kernel doesn't get duplicated.
    cell_3d = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
    if cell_inv_t is None:
        cell_inv = jnp.linalg.inv(cell_3d)
        cell_inv_t = jnp.transpose(cell_inv, (0, 2, 1)).astype(input_dtype)
    else:
        cell_inv_t = cell_inv_t.astype(input_dtype)
        if cell_inv_t.ndim == 2:
            cell_inv_t = cell_inv_t[jnp.newaxis, :, :]
        cell_inv_t = jax.lax.stop_gradient(cell_inv_t)
        cell_inv = jnp.transpose(cell_inv_t, (0, 2, 1))

    # Step 1: Spread charges to mesh
    mesh_grid = spline_spread(
        positions,
        charges,
        cell,
        mesh_dims=mesh_dimensions,
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t,
    )

    # Step 2: FFT of charge mesh
    mesh_fft = jnp.fft.rfftn(mesh_grid, axes=fft_dims, norm="backward")

    # Step 3: Generate k-space grid and compute Green's function + structure factor.
    # When cell_inv_t is supplied, derive reciprocal_cell = 2π · cell_inv from
    # the cached transpose so generate_k_vectors_pme skips its own inv.
    if k_vectors is None or k_squared is None:
        reciprocal_cell = (2.0 * jnp.pi) * cell_inv
        k_vectors, k_squared = generate_k_vectors_pme(
            cell,
            mesh_dimensions,
            reciprocal_cell=reciprocal_cell,
        )
    if hybrid_forces or reciprocal_metadata_is_supplied:
        k_vectors = jax.lax.stop_gradient(k_vectors)
        k_squared = jax.lax.stop_gradient(k_squared)

    # Step 4: Fused Green's function + B-spline deconvolution + multiply in a
    # single warp kernel. Replaces the prior 2-pass path that
    # called pme_green_structure_factor then divided/multiplied in JAX.
    # Caller can supply moduli_x/y/z to skip the per-call fftfreq + sinc^p
    # rebuild (they only depend on mesh + spline_order).
    if moduli_x is None or moduli_y is None or moduli_z is None:
        miller_x = jnp.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx).astype(input_dtype)
        miller_y = jnp.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny).astype(input_dtype)
        miller_z = jnp.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz).astype(input_dtype)
        moduli_x = compute_bspline_moduli_1d(miller_x, mesh_nx, spline_order)
        moduli_y = compute_bspline_moduli_1d(miller_y, mesh_ny, spline_order)
        moduli_z = compute_bspline_moduli_1d(miller_z, mesh_nz, spline_order)
    else:
        moduli_x = jax.lax.stop_gradient(moduli_x)
        moduli_y = jax.lax.stop_gradient(moduli_y)
        moduli_z = jax.lax.stop_gradient(moduli_z)
    # Caller-supplied volume= short-circuits the linalg.det.
    if volume is None:
        volume = jnp.abs(jnp.linalg.det(cell_3d)).astype(input_dtype)
    else:
        volume = volume.astype(input_dtype)
        if volume.ndim == 0:
            volume = volume.reshape(1)
    if hybrid_forces or volume_is_supplied:
        volume = jax.lax.stop_gradient(volume)

    complex_dtype = jnp.complex64 if input_dtype == jnp.float32 else jnp.complex128
    mesh_fft = mesh_fft.astype(complex_dtype)
    # Raw FFT (before convolve) is what the virial path needs.
    mesh_fft_raw = mesh_fft if compute_virial else None

    convolved_mesh = pme_fused_convolve(
        mesh_fft,
        k_squared.astype(input_dtype),
        moduli_x,
        moduli_y,
        moduli_z,
        alpha,
        volume,
        is_batch,
    )

    # Step 5: Compute virial before forces to allow early release of mesh_fft_raw
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
        )
        del mesh_fft_raw  # Free before force field meshes are allocated

        # Fused 2-pass warp kernel: reduce per-atom charges → per-system Q,
        # then subtract E_bg = π Q² / (2 α² V) from the virial diagonal.
        # Matches the torch ``pme_virial_bg_correction`` path. Single-system
        # is fanned out internally via batch_idx=zeros.
        virial = pme_virial_bg_correction(
            charges=charges,
            cell=cell,
            alpha=alpha,
            virial=virial,
            batch_idx=batch_idx,
            volume=volume,
        )

    # Step 6: Inverse FFT to get potential mesh
    potential_mesh = jnp.fft.irfftn(
        convolved_mesh, s=mesh_dimensions, axes=fft_dims, norm="forward"
    )

    # Step 6: Interpolate potential to atomic positions. With forces requested,
    # use the fused gather kernel that walks the spline stencil
    # ONCE per atom and emits both potential AND spline-derivative force,
    # avoiding the 3 extra IFFTs + spline_gather_vec3 of the Fourier-gradient
    # path. Matches the torch reciprocal-space path.
    if compute_forces:
        raw_energies, gathered_force = _spline_gather_with_force(
            positions,
            charges,
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

    # Step 7: Apply corrections
    if need_charge_gradients:
        energies, charge_grads = pme_energy_corrections_with_charge_grad(
            raw_energies, charges, cell, alpha, batch_idx, volume=volume
        )
    else:
        energies = pme_energy_corrections(
            raw_energies, charges, cell, alpha, batch_idx, volume=volume
        )
        charge_grads = None

    # Step 8: Forces from the fused gather above. The 2× scaling absorbs the
    # 1/2 pair-counting factor baked into the Green's function
    # (G = 2π/(V k²) instead of 4π/(V k²)).
    forces = 2.0 * gathered_force if compute_forces else None

    # Hybrid-forces: route ∂E/∂q through the analytical kernel-computed
    # ``charge_grads`` via a custom-VJP straight-through, using the original
    # (non-detached) ``charges`` so jax.grad reaches them.
    if hybrid_forces:
        bidx_for_inject = (
            batch_idx
            if batch_idx is not None
            else jnp.zeros(num_atoms, dtype=jnp.int32)
        )
        energies = _inject_charge_grad(
            energies,
            charges_orig,
            charge_grads,
            batch_idx is not None,
            bidx_for_inject,
            num_systems,
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


def _stop_optional(value: jax.Array | None) -> jax.Array | None:
    """Stop gradients through an optional residual."""
    if value is None:
        return None
    return jax.lax.stop_gradient(value)


def _is_traced_array(value) -> bool:
    """Return whether ``value`` is a JAX tracer inside transformations."""
    return isinstance(value, jax.core.Tracer)


def _require_explicit_mesh_dimensions_in_tracing(
    *,
    mesh_dimensions: tuple[int, int, int] | None,
    cell: jax.Array,
    alpha: jax.Array | float | None,
    batch_idx: jax.Array | None = None,
) -> None:
    """Reject auto PME mesh sizing under JAX tracing with a clear message."""
    is_traced = (
        _is_traced_array(cell) or _is_traced_array(alpha) or _is_traced_array(batch_idx)
    )
    if is_traced and alpha is None and mesh_dimensions is None:
        raise ValueError(
            "JAX PME requires explicit alpha and explicit mesh_dimensions inside "
            "jax.jit or other JAX transformations. Compute PME parameters outside "
            "the transformed function and pass alpha and mesh_dimensions=(nx, ny, nz) "
            "explicitly."
        )
    if is_traced and alpha is None:
        raise ValueError(
            "JAX PME requires explicit alpha inside jax.jit or other JAX "
            "transformations. Compute PME parameters outside the transformed "
            "function and pass alpha explicitly."
        )
    if mesh_dimensions is not None:
        return
    if is_traced:
        raise ValueError(
            "JAX PME requires explicit mesh_dimensions inside jax.jit or other "
            "JAX transformations. Compute mesh_spacing/accuracy-based mesh sizing "
            "outside the transformed function and pass mesh_dimensions=(nx, ny, nz)."
        )


def _tangent_or_zeros(tangent, primal: jax.Array, dtype=None) -> jax.Array:
    """Materialize a custom-JVP tangent, replacing symbolic zeros."""
    out_dtype = primal.dtype if dtype is None else dtype
    if _is_symbolic_zero(tangent):
        return jnp.zeros(primal.shape, dtype=out_dtype)
    return tangent.astype(out_dtype)


def _is_symbolic_zero(tangent) -> bool:
    """Return whether a custom-JVP tangent is JAX's symbolic zero sentinel."""
    return (
        tangent is None
        or isinstance(tangent, jax_ad.Zero)
        or tangent.__class__.__name__ == "SymbolicZero"
    )


def _bspline_weight_reference(u: jax.Array, order: int) -> jax.Array:
    """Pure-JAX cardinal B-spline basis for custom-JVP reference tangents."""
    dtype = u.dtype
    if order == 1:
        return jnp.where((u >= 0.0) & (u < 1.0), 1.0, 0.0).astype(dtype)

    result = jnp.zeros_like(u)
    for j in range(order + 1):
        coeff = (-1.0 if j % 2 else 1.0) * float(math.comb(order, j))
        result = result + jnp.asarray(coeff, dtype=dtype) * jnp.maximum(
            u - jnp.asarray(float(j), dtype=dtype), 0.0
        ) ** (order - 1)
    return result / jnp.asarray(float(math.factorial(order - 1)), dtype=dtype)


def _reference_cell_inv_t(
    cell: jax.Array,
    cell_inv_t: jax.Array | None,
    dtype,
) -> jax.Array:
    """Return cell inverse-transpose using supplied static metadata if present."""
    if cell_inv_t is None:
        cell_3d = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
        return jnp.transpose(jnp.linalg.inv(cell_3d.astype(dtype)), (0, 2, 1))

    cell_inv_t = jax.lax.stop_gradient(cell_inv_t).astype(dtype)
    if cell_inv_t.ndim == 2:
        cell_inv_t = cell_inv_t[jnp.newaxis, :, :]
    return cell_inv_t


def _reference_atom_systems(
    positions: jax.Array,
    batch_idx: jax.Array | None,
) -> jax.Array:
    """Return an int32 system id per atom."""
    if batch_idx is None:
        return jnp.zeros((positions.shape[0],), dtype=jnp.int32)
    return batch_idx.astype(jnp.int32)


def _spline_spread_reference(
    positions: jax.Array,
    values: jax.Array,
    cell_inv_t: jax.Array,
    mesh_dimensions: tuple[int, int, int],
    spline_order: int,
    batch_idx: jax.Array | None,
    num_systems: int,
) -> jax.Array:
    """Pure-JAX charge spread with the production B-spline stencil."""
    dtype = _normalize_dtype(positions.dtype)
    nx, ny, nz = mesh_dimensions
    atom_system = _reference_atom_systems(positions, batch_idx)
    dims = jnp.asarray(mesh_dimensions, dtype=dtype)
    frac = jnp.einsum(
        "nij,nj->ni",
        cell_inv_t[atom_system].astype(dtype),
        positions.astype(dtype),
    )
    mesh_coords = frac * dims
    base = jnp.floor(mesh_coords).astype(jnp.int32)
    theta = mesh_coords - base.astype(dtype)

    mesh_shape = (nx, ny, nz) if batch_idx is None else (num_systems, nx, ny, nz)
    mesh = jnp.zeros(mesh_shape, dtype=dtype)
    half_n_minus_1 = jnp.asarray(0.5 * float(spline_order - 2), dtype=dtype)
    half_order = jnp.asarray(0.5 * float(spline_order), dtype=dtype)
    starts = jnp.floor(theta - half_n_minus_1).astype(jnp.int32)

    for ox in range(spline_order):
        off_x = starts[:, 0] + ox
        gx = jnp.mod(base[:, 0] + off_x, nx)
        wx = _bspline_weight_reference(
            half_order + theta[:, 0] - off_x.astype(dtype), spline_order
        )
        for oy in range(spline_order):
            off_y = starts[:, 1] + oy
            gy = jnp.mod(base[:, 1] + off_y, ny)
            wy = _bspline_weight_reference(
                half_order + theta[:, 1] - off_y.astype(dtype), spline_order
            )
            for oz in range(spline_order):
                off_z = starts[:, 2] + oz
                gz = jnp.mod(base[:, 2] + off_z, nz)
                wz = _bspline_weight_reference(
                    half_order + theta[:, 2] - off_z.astype(dtype), spline_order
                )
                contrib = values.astype(dtype) * wx * wy * wz
                if batch_idx is None:
                    mesh = mesh.at[gx, gy, gz].add(contrib)
                else:
                    mesh = mesh.at[atom_system, gx, gy, gz].add(contrib)
    return mesh


def _spline_gather_reference(
    positions: jax.Array,
    mesh: jax.Array,
    cell_inv_t: jax.Array,
    spline_order: int,
    batch_idx: jax.Array | None,
) -> jax.Array:
    """Pure-JAX mesh gather with the production B-spline stencil."""
    dtype = _normalize_dtype(positions.dtype)
    nx, ny, nz = mesh.shape[-3:]
    atom_system = _reference_atom_systems(positions, batch_idx)
    dims = jnp.asarray((nx, ny, nz), dtype=dtype)
    frac = jnp.einsum(
        "nij,nj->ni",
        cell_inv_t[atom_system].astype(dtype),
        positions.astype(dtype),
    )
    mesh_coords = frac * dims
    base = jnp.floor(mesh_coords).astype(jnp.int32)
    theta = mesh_coords - base.astype(dtype)

    output = jnp.zeros((positions.shape[0],), dtype=dtype)
    half_n_minus_1 = jnp.asarray(0.5 * float(spline_order - 2), dtype=dtype)
    half_order = jnp.asarray(0.5 * float(spline_order), dtype=dtype)
    starts = jnp.floor(theta - half_n_minus_1).astype(jnp.int32)

    for ox in range(spline_order):
        off_x = starts[:, 0] + ox
        gx = jnp.mod(base[:, 0] + off_x, nx)
        wx = _bspline_weight_reference(
            half_order + theta[:, 0] - off_x.astype(dtype), spline_order
        )
        for oy in range(spline_order):
            off_y = starts[:, 1] + oy
            gy = jnp.mod(base[:, 1] + off_y, ny)
            wy = _bspline_weight_reference(
                half_order + theta[:, 1] - off_y.astype(dtype), spline_order
            )
            for oz in range(spline_order):
                off_z = starts[:, 2] + oz
                gz = jnp.mod(base[:, 2] + off_z, nz)
                wz = _bspline_weight_reference(
                    half_order + theta[:, 2] - off_z.astype(dtype), spline_order
                )
                if batch_idx is None:
                    mesh_values = mesh[gx, gy, gz]
                else:
                    mesh_values = mesh[atom_system, gx, gy, gz]
                output = output + mesh_values.astype(dtype) * wx * wy * wz
    return output


def _pme_energy_corrections_reference(
    raw_energies: jax.Array,
    charges: jax.Array,
    alpha: jax.Array,
    volume: jax.Array,
    batch_idx: jax.Array | None,
    num_systems: int,
) -> jax.Array:
    """Pure-JAX PME self/background correction for reference tangents."""
    dtype = _normalize_dtype(raw_energies.dtype)
    charges = charges.astype(jnp.float64)
    raw = raw_energies.astype(jnp.float64)
    alpha_arr = _pme_alpha_array(alpha, jnp.float64, num_systems)
    volume = volume.astype(jnp.float64)
    if volume.ndim == 0:
        volume = volume.reshape(1)
    if volume.shape[0] == 1 and num_systems > 1:
        volume = jnp.broadcast_to(volume, (num_systems,))

    atom_system = _reference_atom_systems(raw_energies, batch_idx)
    total_charges = _system_sum_from_atoms(charges, batch_idx, num_systems)
    alpha_atom = alpha_arr[atom_system]
    volume_atom = volume[atom_system]
    total_atom = total_charges[atom_system]

    self_energy = alpha_atom * charges * charges / jnp.sqrt(jnp.pi)
    background = (
        jnp.pi * charges * total_atom / (2.0 * alpha_atom * alpha_atom * volume_atom)
    )
    return (raw * charges - self_energy - background).astype(dtype)


def _pme_convolve_reference(
    mesh_fft: jax.Array,
    k_squared: jax.Array,
    moduli_x: jax.Array,
    moduli_y: jax.Array,
    moduli_z: jax.Array,
    alpha: jax.Array,
    volume: jax.Array,
    is_batch: bool,
) -> jax.Array:
    """Pure-JAX equivalent of the fused PME convolve multiplier."""
    real_dtype = jnp.float32 if mesh_fft.dtype == jnp.complex64 else jnp.float64
    alpha = _pme_alpha_array(alpha, real_dtype, volume.shape[0])
    volume = volume.astype(real_dtype)
    if volume.ndim == 0:
        volume = volume.reshape(1)

    k_sq = k_squared.astype(real_dtype)
    if is_batch and k_sq.ndim == 3:
        k_sq = k_sq[jnp.newaxis, ...]
    safe_k_sq = jnp.where(k_sq > 1e-10, k_sq, 1.0)
    if is_batch:
        alpha_view = alpha.reshape(-1, 1, 1, 1)
        volume_view = volume.reshape(-1, 1, 1, 1)
    else:
        alpha_view = alpha.reshape(-1)[0]
        volume_view = volume.reshape(-1)[0]

    green = (
        2.0
        * jnp.pi
        * jnp.exp(-safe_k_sq / (4.0 * alpha_view * alpha_view))
        / (volume_view * safe_k_sq)
    )
    green = jnp.where(k_sq > 1e-10, green, 0.0)
    sf = (
        moduli_x.astype(real_dtype)[:, None, None]
        * moduli_y.astype(real_dtype)[None, :, None]
        * moduli_z.astype(real_dtype)[None, None, :]
    )
    sf_sq = jnp.maximum(sf * sf, jnp.asarray(1e-10, dtype=real_dtype))
    return mesh_fft * (green / sf_sq).astype(mesh_fft.dtype)


def _pme_reciprocal_energy_reference(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    mesh_dimensions: tuple[int, int, int] | None,
    mesh_spacing: float | None,
    spline_order: int,
    batch_idx: jax.Array | None,
    k_vectors: jax.Array | None,
    k_squared: jax.Array | None,
    volume: jax.Array | None,
    cell_inv_t: jax.Array | None,
    moduli_x: jax.Array | None,
    moduli_y: jax.Array | None,
    moduli_z: jax.Array | None,
) -> jax.Array:
    """Pure-JAX PME reciprocal per-atom energy for weighted-loss tangents."""
    dtype = _normalize_dtype(positions.dtype)
    cell_3d = cell.astype(dtype)
    if cell_3d.ndim == 2:
        cell_3d = cell_3d[jnp.newaxis, :, :]
    num_systems = cell_3d.shape[0]
    is_batch = batch_idx is not None
    if mesh_dimensions is None:
        if mesh_spacing is None:
            mesh_dimensions = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-6)
        else:
            mesh_dimensions = mesh_spacing_to_dimensions(cell, mesh_spacing)

    cell_inv_t_ref = _reference_cell_inv_t(cell, cell_inv_t, dtype)
    cell_inv_ref = jnp.transpose(cell_inv_t_ref, (0, 2, 1))

    mesh_grid = _spline_spread_reference(
        positions,
        charges,
        cell_inv_t_ref,
        mesh_dimensions,
        spline_order,
        batch_idx,
        num_systems,
    )
    fft_dims = (1, 2, 3) if is_batch else (0, 1, 2)
    complex_dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    mesh_fft = jnp.fft.rfftn(mesh_grid, axes=fft_dims, norm="backward").astype(
        complex_dtype
    )

    if k_vectors is None or k_squared is None:
        reciprocal_cell = (2.0 * jnp.pi) * cell_inv_ref
        _k_vectors, k_squared = generate_k_vectors_pme(
            cell,
            mesh_dimensions,
            reciprocal_cell=reciprocal_cell,
        )
    else:
        k_squared = jax.lax.stop_gradient(k_squared)

    nx, ny, nz = mesh_dimensions
    if moduli_x is None or moduli_y is None or moduli_z is None:
        miller_x = jnp.fft.fftfreq(nx, d=1.0 / nx).astype(dtype)
        miller_y = jnp.fft.fftfreq(ny, d=1.0 / ny).astype(dtype)
        miller_z = jnp.fft.rfftfreq(nz, d=1.0 / nz).astype(dtype)
        moduli_x = compute_bspline_moduli_1d(miller_x, nx, spline_order)
        moduli_y = compute_bspline_moduli_1d(miller_y, ny, spline_order)
        moduli_z = compute_bspline_moduli_1d(miller_z, nz, spline_order)
    else:
        moduli_x = jax.lax.stop_gradient(moduli_x)
        moduli_y = jax.lax.stop_gradient(moduli_y)
        moduli_z = jax.lax.stop_gradient(moduli_z)

    if volume is None:
        volume_ref = jnp.abs(jnp.linalg.det(cell_3d)).astype(dtype)
    else:
        volume_ref = jax.lax.stop_gradient(volume).astype(dtype)
        if volume_ref.ndim == 0:
            volume_ref = volume_ref.reshape(1)
        if volume_ref.shape[0] == 1 and num_systems > 1:
            volume_ref = jnp.broadcast_to(volume_ref, (num_systems,))

    convolved = _pme_convolve_reference(
        mesh_fft,
        k_squared,
        moduli_x,
        moduli_y,
        moduli_z,
        alpha,
        volume_ref,
        is_batch,
    )
    potential_mesh = jnp.fft.irfftn(
        convolved, s=mesh_dimensions, axes=fft_dims, norm="forward"
    ).astype(dtype)
    raw_energies = _spline_gather_reference(
        positions,
        potential_mesh,
        cell_inv_t_ref,
        spline_order,
        batch_idx,
    )
    return _pme_energy_corrections_reference(
        raw_energies,
        charges,
        alpha,
        volume_ref,
        batch_idx,
        num_systems,
    )


@jax.custom_jvp
def _unsupported_pme_cell_hvp_gradient(grad_cell: jax.Array) -> jax.Array:
    """Identity wrapper that rejects cell/stress higher-order PME derivatives."""
    return grad_cell


@_unsupported_pme_cell_hvp_gradient.defjvp
def _unsupported_pme_cell_hvp_gradient_jvp(
    primals: tuple[jax.Array],
    tangents: tuple[jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Reject JVPs through the first-order PME cell-gradient adapter."""
    del tangents
    (grad_cell,) = primals
    raise NotImplementedError(
        "JAX PME stress/cell/strain HVPs are unsupported. Differentiate "
        "position or charge losses, or use first-order cell/strain gradients only."
    )


@jax.custom_jvp
def _unsupported_pme_cell_hvp_primal(cell: jax.Array) -> jax.Array:
    """Identity wrapper that marks PME cell JVP state as first-order only."""
    return cell


@_unsupported_pme_cell_hvp_primal.defjvp
def _unsupported_pme_cell_hvp_primal_jvp(
    primals: tuple[jax.Array],
    tangents: tuple[jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Reject JVPs through a PME cell-gradient computation."""
    del tangents
    (cell,) = primals
    raise NotImplementedError(
        "JAX PME stress/cell/strain HVPs are unsupported. Differentiate "
        "position or charge losses, or use first-order cell/strain gradients only."
    )


def _system_sum_from_atoms(
    values: jax.Array,
    batch_idx: jax.Array | None,
    num_systems: int,
) -> jax.Array:
    """Sum per-atom scalar values into one scalar per system."""
    if batch_idx is None:
        return values.sum(keepdims=True)
    return (
        jnp.zeros((num_systems,), dtype=values.dtype)
        .at[batch_idx.astype(jnp.int32)]
        .add(values)
    )


def _per_system_atom_counts(
    batch_idx: jax.Array | None,
    num_systems: int,
    num_atoms: int,
) -> jax.Array:
    """Return per-system atom counts as float64 for tangent redistribution."""
    if batch_idx is None:
        return jnp.full((num_systems,), float(num_atoms), dtype=jnp.float64)
    return (
        jnp.zeros((num_systems,), dtype=jnp.float64)
        .at[batch_idx.astype(jnp.int32)]
        .add(jnp.ones((num_atoms,), dtype=jnp.float64))
    )


def _distribute_system_values(
    system_values: jax.Array,
    batch_idx: jax.Array | None,
    num_atoms: int,
) -> jax.Array:
    """Distribute per-system values uniformly over each system's atoms."""
    if batch_idx is None:
        if num_atoms == 0:
            return jnp.zeros((0,), dtype=system_values.dtype)
        return jnp.full(
            (num_atoms,), system_values[0] / num_atoms, dtype=system_values.dtype
        )

    counts = _per_system_atom_counts(batch_idx, system_values.shape[0], num_atoms)
    return (system_values / jnp.maximum(counts, 1.0))[batch_idx.astype(jnp.int32)]


def _cell_tangent_system_values(
    grad_cell: jax.Array,
    tangent_cell,
) -> jax.Array:
    """Contract a cell cotangent with a cell tangent per system."""
    tcell = _tangent_or_zeros(tangent_cell, grad_cell, dtype=jnp.float64)
    values = grad_cell.astype(jnp.float64) * tcell.astype(jnp.float64)
    if values.ndim == 2:
        return jnp.array([values.sum()], dtype=jnp.float64)
    return values.sum(axis=(1, 2))


def _per_atom_cell_inv_t_matvec(
    cell_inv_t: jax.Array,
    vectors: jax.Array,
    batch_idx: jax.Array | None,
) -> jax.Array:
    """Apply the per-system ``cell_inv_t`` matrix to per-atom vectors."""
    if batch_idx is None:
        return jnp.einsum("ij,nj->ni", cell_inv_t[0], vectors)
    return jnp.einsum(
        "nij,nj->ni",
        cell_inv_t[batch_idx.astype(jnp.int32)],
        vectors,
    )


def _pme_alpha_array(
    alpha: jax.Array,
    dtype,
    num_systems: int,
) -> jax.Array:
    """Return PME alpha as a length-``num_systems`` array."""
    alpha_arr = alpha.astype(dtype)
    if alpha_arr.ndim == 0:
        alpha_arr = alpha_arr.reshape(1)
    if alpha_arr.shape[0] == 1 and num_systems > 1:
        alpha_arr = jnp.broadcast_to(alpha_arr, (num_systems,))
    return alpha_arr


def _pme_charge_background_hvp(
    h_charges: jax.Array,
    alpha: jax.Array,
    volume: jax.Array,
    batch_idx: jax.Array | None,
    num_systems: int,
) -> tuple[jax.Array, jax.Array]:
    """Return per-atom ``alpha`` and background charge-Hessian coefficient."""
    d_qtotal = _system_sum_from_atoms(h_charges, batch_idx, num_systems)
    if batch_idx is None:
        alpha_atom = alpha[0]
        bg_coeff = jnp.full_like(
            h_charges, jnp.pi / (alpha_atom * alpha_atom * volume[0])
        )
        dqtotal_atom = jnp.full_like(h_charges, d_qtotal[0])
        return alpha_atom, bg_coeff * dqtotal_atom

    batch_i32 = batch_idx.astype(jnp.int32)
    bg_coeff = jnp.pi / (alpha * alpha * volume)
    return alpha[batch_i32], bg_coeff[batch_i32] * d_qtotal[batch_i32]


def _pme_reciprocal_hvp_state(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    mesh_dimensions: tuple[int, int, int] | None,
    mesh_spacing: float | None,
    spline_order: int,
    batch_idx: jax.Array | None,
    k_vectors: jax.Array | None,
    k_squared: jax.Array | None,
    volume: jax.Array | None,
    cell_inv_t: jax.Array | None,
    moduli_x: jax.Array | None,
    moduli_y: jax.Array | None,
    moduli_z: jax.Array | None,
) -> tuple[jax.Array, ...]:
    """Build reusable PME reciprocal state for fixed-cell HVP evaluation."""
    if mesh_dimensions is None:
        if mesh_spacing is None:
            raise ValueError("mesh_dimensions must be resolved before PME HVP")
        mesh_dimensions = mesh_spacing_to_dimensions(cell, mesh_spacing)

    dtype = _normalize_dtype(positions.dtype)
    positions_cast = positions.astype(dtype)
    charges_cast = charges.astype(dtype)
    cell_cast = cell.astype(dtype)
    cell_3d = cell_cast if cell_cast.ndim == 3 else cell_cast[jnp.newaxis, :, :]
    num_systems = cell_3d.shape[0]
    is_batch = batch_idx is not None
    fft_dims = (1, 2, 3) if is_batch else (0, 1, 2)

    if cell_inv_t is None:
        cell_inv = jnp.linalg.inv(cell_3d)
        cell_inv_t_work = jnp.transpose(cell_inv, (0, 2, 1)).astype(dtype)
    else:
        cell_inv_t_work = cell_inv_t.astype(dtype)
        if cell_inv_t_work.ndim == 2:
            cell_inv_t_work = cell_inv_t_work[jnp.newaxis, :, :]
        cell_inv = jnp.transpose(cell_inv_t_work, (0, 2, 1))

    if volume is None:
        volume_work = jnp.abs(jnp.linalg.det(cell_3d)).astype(dtype)
    else:
        volume_work = volume.astype(dtype)
        if volume_work.ndim == 0:
            volume_work = volume_work.reshape(1)
    alpha_work = _pme_alpha_array(alpha, dtype, num_systems)

    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
    if k_squared is None:
        if k_vectors is not None:
            k_vectors_work = k_vectors.astype(dtype)
            k_squared_work = jnp.sum(k_vectors_work * k_vectors_work, axis=-1)
        else:
            reciprocal_cell = (2.0 * jnp.pi) * cell_inv
            _k_vectors, k_squared_work = generate_k_vectors_pme(
                cell_cast,
                mesh_dimensions,
                reciprocal_cell=reciprocal_cell,
            )
    else:
        k_squared_work = k_squared.astype(dtype)

    if moduli_x is None or moduli_y is None or moduli_z is None:
        miller_x = jnp.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx).astype(dtype)
        miller_y = jnp.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny).astype(dtype)
        miller_z = jnp.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz).astype(dtype)
        moduli_x_work = compute_bspline_moduli_1d(miller_x, mesh_nx, spline_order)
        moduli_y_work = compute_bspline_moduli_1d(miller_y, mesh_ny, spline_order)
        moduli_z_work = compute_bspline_moduli_1d(miller_z, mesh_nz, spline_order)
    else:
        moduli_x_work = moduli_x.astype(dtype)
        moduli_y_work = moduli_y.astype(dtype)
        moduli_z_work = moduli_z.astype(dtype)

    mesh_grid = spline_spread(
        positions_cast,
        charges_cast,
        cell_cast,
        mesh_dims=mesh_dimensions,
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t_work,
    )
    complex_dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    mesh_fft = jnp.fft.rfftn(mesh_grid, axes=fft_dims, norm="backward").astype(
        complex_dtype
    )
    convolved_mesh = pme_fused_convolve(
        mesh_fft,
        k_squared_work,
        moduli_x_work,
        moduli_y_work,
        moduli_z_work,
        alpha_work,
        volume_work,
        is_batch,
    )
    potential_mesh = jnp.fft.irfftn(
        convolved_mesh,
        s=mesh_dimensions,
        axes=fft_dims,
        norm="forward",
    )
    return (
        positions_cast,
        charges_cast,
        cell_cast,
        alpha_work,
        batch_idx,
        cell_inv_t_work,
        volume_work,
        k_squared_work,
        moduli_x_work,
        moduli_y_work,
        moduli_z_work,
        potential_mesh,
    )


def _pme_reciprocal_energy_hvp_from_state(
    v_positions: jax.Array,
    v_charges: jax.Array,
    positions_cast: jax.Array,
    charges_cast: jax.Array,
    cell_cast: jax.Array,
    alpha_work: jax.Array,
    batch_idx: jax.Array | None,
    cell_inv_t_work: jax.Array,
    volume_work: jax.Array,
    k_squared_work: jax.Array,
    moduli_x_work: jax.Array,
    moduli_y_work: jax.Array,
    moduli_z_work: jax.Array,
    potential_mesh: jax.Array,
    spline_order: int,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate the linear PME reciprocal HVP from saved mesh state."""
    dtype = _normalize_dtype(positions_cast.dtype)
    is_batch = batch_idx is not None
    mesh_dimensions = (
        tuple(potential_mesh.shape[1:]) if is_batch else tuple(potential_mesh.shape)
    )
    fft_dims = (1, 2, 3) if is_batch else (0, 1, 2)
    v_positions_cast = v_positions.astype(dtype)
    v_charges_cast = v_charges.astype(dtype)
    complex_dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    cell_3d = cell_cast if cell_cast.ndim == 3 else cell_cast[jnp.newaxis, :, :]
    num_systems = cell_3d.shape[0]

    v_frac = _per_atom_cell_inv_t_matvec(
        cell_inv_t_work,
        v_positions_cast,
        batch_idx,
    )
    dmesh_charge = spline_spread(
        positions_cast,
        v_charges_cast,
        cell_cast,
        mesh_dims=mesh_dimensions,
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t_work,
    )
    dmesh_position = _spline_spread_gradient_weights(
        positions_cast,
        charges_cast[:, jnp.newaxis] * v_frac,
        cell_cast,
        mesh_dimensions,
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t_work,
    )
    dmesh = dmesh_charge + dmesh_position
    dmesh_fft = jnp.fft.rfftn(dmesh, axes=fft_dims, norm="backward").astype(
        complex_dtype
    )
    dconvolved_mesh = pme_fused_convolve(
        dmesh_fft,
        k_squared_work,
        moduli_x_work,
        moduli_y_work,
        moduli_z_work,
        alpha_work,
        volume_work,
        is_batch,
    )
    dpotential_mesh = jnp.fft.irfftn(
        dconvolved_mesh,
        s=mesh_dimensions,
        axes=fft_dims,
        norm="forward",
    )

    grad_raw_cart = spline_gather_gradient(
        positions_cast,
        -jnp.ones_like(charges_cast),
        potential_mesh,
        cell_cast,
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t_work,
    )
    dforce_mesh = spline_gather_gradient(
        positions_cast,
        charges_cast,
        dpotential_mesh,
        cell_cast,
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t_work,
    )
    dforce_position = _spline_gather_gradient_position_hessian(
        positions_cast,
        charges_cast,
        v_frac,
        cell_cast,
        potential_mesh,
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t_work,
    )
    hvp_positions = (
        2.0 * v_charges_cast[:, jnp.newaxis] * grad_raw_cart
        - 2.0 * dforce_mesh
        - 2.0 * dforce_position
    )

    draw_mesh = spline_gather(
        positions_cast,
        dpotential_mesh,
        cell_cast,
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t_work,
    )
    draw_position = (grad_raw_cart * v_positions_cast).sum(axis=1)
    alpha_atom, background_hvp = _pme_charge_background_hvp(
        v_charges_cast,
        alpha_work,
        volume_work,
        batch_idx,
        num_systems,
    )
    hvp_charges = (
        2.0 * (draw_mesh + draw_position)
        - 2.0 * alpha_atom / jnp.sqrt(jnp.pi) * v_charges_cast
        - background_hvp
    )
    return (
        hvp_positions.astype(positions_cast.dtype),
        hvp_charges.astype(charges_cast.dtype),
    )


def _pme_reciprocal_energy_hvp_raw(
    v_positions: jax.Array,
    v_charges: jax.Array,
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    mesh_dimensions: tuple[int, int, int] | None,
    mesh_spacing: float | None,
    spline_order: int,
    batch_idx: jax.Array | None,
    k_vectors: jax.Array | None,
    k_squared: jax.Array | None,
    volume: jax.Array | None,
    cell_inv_t: jax.Array | None,
    moduli_x: jax.Array | None,
    moduli_y: jax.Array | None,
    moduli_z: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate fixed-cell PME reciprocal HVPs for positions and charges."""
    if positions.shape[0] == 0:
        dtype = _normalize_dtype(positions.dtype)
        return (
            jnp.zeros_like(positions, dtype=dtype),
            jnp.zeros_like(charges, dtype=dtype),
        )
    state = _pme_reciprocal_hvp_state(
        positions,
        charges,
        cell,
        alpha,
        mesh_dimensions,
        mesh_spacing,
        spline_order,
        batch_idx,
        k_vectors,
        k_squared,
        volume,
        cell_inv_t,
        moduli_x,
        moduli_y,
        moduli_z,
    )
    return _pme_reciprocal_energy_hvp_from_state(
        v_positions,
        v_charges,
        *state,
        spline_order,
    )


def _pme_reciprocal_energy_derivative_values(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    mesh_dimensions: tuple[int, int, int] | None,
    mesh_spacing: float | None,
    spline_order: int,
    batch_idx: jax.Array | None,
    k_vectors: jax.Array | None,
    k_squared: jax.Array | None,
    volume: jax.Array | None,
    cell_inv_t: jax.Array | None,
    moduli_x: jax.Array | None,
    moduli_y: jax.Array | None,
    moduli_z: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Return raw PME reciprocal ``dE/dR`` and ``dE/dq`` direct outputs."""
    _energy, forces, charge_grads = _pme_reciprocal_space_impl(
        positions=positions,
        charges=charges,
        cell=cell,
        alpha=alpha,
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
        compute_forces=True,
        compute_charge_gradients=True,
        compute_virial=False,
        hybrid_forces=False,
    )
    return -forces, charge_grads


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _pme_reciprocal_energy_derivatives(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    mesh_dimensions: tuple[int, int, int] | None,
    mesh_spacing: float | None,
    spline_order: int,
    batch_idx: jax.Array | None,
    k_vectors: jax.Array | None,
    k_squared: jax.Array | None,
    volume: jax.Array | None,
    cell_inv_t: jax.Array | None,
    moduli_x: jax.Array | None,
    moduli_y: jax.Array | None,
    moduli_z: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """PME reciprocal ``(dE/dR, dE/dq)`` first-derivative values.

    Primal returns the PME *mesh* first derivatives (so forces/charge gradients are
    bit-identical to the direct-output path). The custom VJP below supplies the
    private PME-native HVP needed for reverse-mode higher-order
    position/charge losses.
    """
    dpos, charge_grads = _pme_reciprocal_energy_derivative_values(
        positions,
        charges,
        cell,
        alpha,
        mesh_dimensions,
        mesh_spacing,
        spline_order,
        batch_idx,
        k_vectors,
        k_squared,
        volume,
        cell_inv_t,
        moduli_x,
        moduli_y,
        moduli_z,
    )
    return jax.lax.stop_gradient(dpos), jax.lax.stop_gradient(charge_grads)


def _pme_reciprocal_energy_derivatives_fwd(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    mesh_dimensions: tuple[int, int, int] | None,
    mesh_spacing: float | None,
    spline_order: int,
    batch_idx: jax.Array | None,
    k_vectors: jax.Array | None,
    k_squared: jax.Array | None,
    volume: jax.Array | None,
    cell_inv_t: jax.Array | None,
    moduli_x: jax.Array | None,
    moduli_y: jax.Array | None,
    moduli_z: jax.Array | None,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]]:
    """Forward rule for PME reciprocal first derivatives."""
    primal_out = _pme_reciprocal_energy_derivatives(
        positions,
        charges,
        cell,
        alpha,
        mesh_dimensions,
        mesh_spacing,
        spline_order,
        batch_idx,
        k_vectors,
        k_squared,
        volume,
        cell_inv_t,
        moduli_x,
        moduli_y,
        moduli_z,
    )
    state = _pme_reciprocal_hvp_state(
        positions,
        charges,
        cell,
        alpha,
        mesh_dimensions,
        mesh_spacing,
        spline_order,
        batch_idx,
        k_vectors,
        k_squared,
        volume,
        cell_inv_t,
        moduli_x,
        moduli_y,
        moduli_z,
    )
    return primal_out, state


def _pme_reciprocal_energy_derivatives_bwd(
    mesh_dimensions: tuple[int, int, int] | None,
    mesh_spacing: float | None,
    spline_order: int,
    residuals: tuple[jax.Array, ...],
    ct_out: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array | None, ...]:
    """Backward rule for PME reciprocal first derivatives."""
    del mesh_dimensions, mesh_spacing
    positions_cast, charges_cast, *_rest = residuals
    ct_positions, ct_charges = ct_out
    grad_positions, grad_charges = _pme_reciprocal_energy_hvp_from_state(
        _tangent_or_zeros(
            ct_positions,
            positions_cast,
            dtype=positions_cast.dtype,
        ),
        _tangent_or_zeros(
            ct_charges,
            charges_cast,
            dtype=charges_cast.dtype,
        ),
        *residuals,
        spline_order,
    )
    return (
        grad_positions,
        grad_charges,
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


_pme_reciprocal_energy_derivatives.defvjp(
    _pme_reciprocal_energy_derivatives_fwd,
    _pme_reciprocal_energy_derivatives_bwd,
)


@functools.partial(jax.custom_jvp, nondiff_argnums=(4, 5, 6))
def _pme_reciprocal_energy_jvp(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    mesh_dimensions: tuple[int, int, int] | None,
    mesh_spacing: float | None,
    spline_order: int,
    batch_idx: jax.Array | None,
    k_vectors: jax.Array | None,
    k_squared: jax.Array | None,
    volume: jax.Array | None,
    cell_inv_t: jax.Array | None,
    moduli_x: jax.Array | None,
    moduli_y: jax.Array | None,
    moduli_z: jax.Array | None,
) -> jax.Array:
    """Energy-only PME reciprocal wrapper with a custom JVP."""
    energy = _pme_reciprocal_space_impl(
        positions=positions,
        charges=charges,
        cell=cell,
        alpha=alpha,
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
        compute_forces=False,
        compute_charge_gradients=False,
        compute_virial=False,
        hybrid_forces=False,
    )
    return jax.lax.stop_gradient(energy)


def _pme_reciprocal_energy_jvp_rule(
    mesh_dimensions: tuple[int, int, int] | None,
    mesh_spacing: float | None,
    spline_order: int,
    primals: tuple[jax.Array | None, ...],
    tangents: tuple[jax.Array | None, ...],
) -> tuple[jax.Array, jax.Array]:
    """JVP rule for the reciprocal PME per-atom energy vector."""
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
    ) = primals
    (
        t_positions,
        t_charges,
        t_cell,
        _t_alpha,
        _t_batch_idx,
        _t_k_vectors,
        _t_k_squared,
        _t_volume,
        _t_cell_inv_t,
        _t_moduli_x,
        _t_moduli_y,
        _t_moduli_z,
    ) = tangents

    del (
        _t_alpha,
        _t_batch_idx,
        _t_k_vectors,
        _t_k_squared,
        _t_volume,
        _t_cell_inv_t,
        _t_moduli_x,
        _t_moduli_y,
        _t_moduli_z,
    )
    primal_out = _pme_reciprocal_energy_jvp(
        positions,
        charges,
        cell,
        alpha,
        mesh_dimensions,
        mesh_spacing,
        spline_order,
        batch_idx,
        k_vectors,
        k_squared,
        volume,
        cell_inv_t,
        moduli_x,
        moduli_y,
        moduli_z,
    )

    tpos = _tangent_or_zeros(t_positions, positions, dtype=positions.dtype)
    tq = _tangent_or_zeros(t_charges, charges, dtype=charges.dtype)
    tcell = _tangent_or_zeros(t_cell, cell, dtype=cell.dtype)
    charges_ref = charges.astype(jnp.float64)
    tq_ref = tq.astype(jnp.float64)
    # Keep the energy tangent in pure JAX. The primal still uses the Warp PME
    # mesh path above, but JAX cannot safely transpose the current Warp spline
    # FFI boundary for PME cell/strain HVPs. Position and charge tangents are
    # therefore evaluated through a pure-JAX PME reference, while nonzero cell
    # tangents are tagged below so higher-order cell/strain requests reject
    # explicitly instead of silently dropping terms or falling back to Ewald.
    reference_cell = (
        _unsupported_pme_cell_hvp_primal(cell)
        if not _is_symbolic_zero(t_cell)
        else cell
    )
    _reference_out, tangent_out = jax.jvp(
        lambda p, q, c: _pme_reciprocal_energy_reference(
            p,
            q,
            c,
            alpha,
            mesh_dimensions,
            mesh_spacing,
            spline_order,
            batch_idx,
            k_vectors,
            k_squared,
            volume,
            cell_inv_t,
            moduli_x,
            moduli_y,
            moduli_z,
        ),
        (positions, charges_ref, reference_cell),
        (tpos, tq_ref, tcell),
    )
    if not _is_symbolic_zero(t_cell):
        # First-order cell gradients are allowed, but differentiating those
        # gradients again requires a native transposable PME cell HVP.
        tangent_out = _unsupported_pme_cell_hvp_gradient(tangent_out)
    return primal_out, tangent_out.astype(primal_out.dtype)


_pme_reciprocal_energy_jvp.defjvp(
    _pme_reciprocal_energy_jvp_rule,
    symbolic_zeros=True,
)


def pme_reciprocal_space(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    mesh_dimensions: tuple[int, int, int] | None = None,
    mesh_spacing: float | None = None,
    spline_order: int = 4,
    batch_idx: jax.Array | None = None,
    k_vectors: jax.Array | None = None,
    k_squared: jax.Array | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    hybrid_forces: bool = False,
    *,
    cell_inv_t: jax.Array | None = None,
    volume: jax.Array | None = None,
    moduli_x: jax.Array | None = None,
    moduli_y: jax.Array | None = None,
    moduli_z: jax.Array | None = None,
) -> (
    jax.Array
    | tuple[jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Compute PME reciprocal-space contribution.

    Energy-only calls use a custom JVP so JAX does not attempt to
    differentiate the Warp spline/FFT FFI path. ``compute_forces=True`` remains
    a forward/direct escape hatch for no-autograd MD/inference loops; charge
    gradients, virial, and hybrid direct outputs are deprecated training-style
    outputs and warn.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    charges : jax.Array, shape (N,)
        Atomic partial charges.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrices with lattice vectors as rows.
    alpha : jax.Array
        Ewald splitting parameter.
    mesh_dimensions : tuple[int, int, int] or None, default=None
        Explicit FFT mesh dimensions. Required when ``cell``, ``alpha``, or
        batch metadata are traced by ``jax.jit`` or other JAX transformations.
    mesh_spacing : float or None, default=None
        Target mesh spacing for eager-only mesh-size inference.
    spline_order : int, default=4
        B-spline interpolation order.
    batch_idx : jax.Array or None, default=None
        System index for each atom. When provided, atoms must be grouped by
        system: ``batch_idx`` must be contiguous, nondecreasing, and use system
        IDs ``0..B-1``.
    k_vectors, k_squared : jax.Array or None
        Optional precomputed reciprocal grid values. These are setup constants
        for the JAX custom-JVP path; tangents through them are ignored. When
        supplied while differentiating with respect to ``cell``, they are
        assumed to correspond to the current ``cell``.
    compute_forces, compute_charge_gradients, compute_virial : bool
        Direct-output flags. ``compute_forces=True`` remains supported for
        no-autograd MD/inference use; charge-gradient and virial direct outputs
        are deprecated for differentiable training.
    hybrid_forces : bool, default=False
        Deprecated charge-gradient injection mode for compatibility.
    cell_inv_t, volume, moduli_x, moduli_y, moduli_z : jax.Array or None
        Optional precomputed PME intermediates. These are setup constants for
        JAX and are not differentiable inputs. Cell-derived metadata such as
        ``cell_inv_t`` and ``volume`` is accepted while differentiating with
        respect to ``cell`` and is assumed to correspond to the current
        ``cell``.

    Returns
    -------
    jax.Array or tuple[jax.Array, ...]
        Per-atom reciprocal energies, plus direct outputs when
        requested.

    Notes
    -----
    When ``cell`` or batch metadata are traced by ``jax.jit`` or other JAX
    transformations, pass explicit ``mesh_dimensions``. If ``alpha`` would
    otherwise be estimated, precompute and pass it explicitly as well.
    ``mesh_spacing`` and accuracy-based parameter estimation depend on concrete
    setup values.

    JAX PME higher-order support is limited to tested position and charge
    losses. Stress/cell/strain HVPs, alpha HVPs, and precomputed-metadata HVPs
    are unsupported until explicitly implemented and tested.
    """
    _require_explicit_mesh_dimensions_in_tracing(
        mesh_dimensions=mesh_dimensions,
        cell=cell,
        alpha=alpha,
        batch_idx=batch_idx,
    )
    component_deprecated_flags = tuple(
        name
        for name, enabled in (
            ("compute_charge_gradients", compute_charge_gradients),
            ("compute_virial", compute_virial),
            ("hybrid_forces", hybrid_forces),
        )
        if enabled
    )
    if component_deprecated_flags:
        warnings.warn(
            _component_direct_output_deprecation_msg(
                "pme_reciprocal_space", component_deprecated_flags
            ),
            DeprecationWarning,
            stacklevel=2,
        )

    if compute_forces or compute_charge_gradients or compute_virial or hybrid_forces:
        return _pme_reciprocal_space_impl(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
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
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
            hybrid_forces=hybrid_forces,
        )
    if mesh_dimensions is None and mesh_spacing is not None:
        mesh_dimensions = mesh_spacing_to_dimensions(cell, mesh_spacing)
        mesh_spacing = None
    return _pme_reciprocal_energy_jvp(
        positions,
        charges,
        cell,
        alpha,
        mesh_dimensions,
        mesh_spacing,
        spline_order,
        batch_idx,
        k_vectors,
        k_squared,
        volume,
        cell_inv_t,
        moduli_x,
        moduli_y,
        moduli_z,
    )


def _particle_mesh_ewald_impl(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: float | jax.Array | None = None,
    mesh_spacing: float | None = None,
    mesh_dimensions: tuple[int, int, int] | None = None,
    spline_order: int = 4,
    batch_idx: jax.Array | None = None,
    k_vectors: jax.Array | None = None,
    k_squared: jax.Array | None = None,
    neighbor_list: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    neighbor_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    mask_value: int | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    accuracy: float = 1e-6,
    pbc: jax.Array | None = None,
    slab_correction: bool = False,
    hybrid_forces: bool = False,
    volume: jax.Array | None = None,
    cell_inv_t: jax.Array | None = None,
    moduli_x: jax.Array | None = None,
    moduli_y: jax.Array | None = None,
    moduli_z: jax.Array | None = None,
) -> (
    jax.Array
    | tuple[jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Complete Particle Mesh Ewald (PME) calculation for long-range electrostatics.

    Computes total Coulomb energy using the PME method, which achieves O(N log N)
    scaling through FFT-based reciprocal space calculations. Combines:
    1. Real-space contribution (short-range, erfc-damped)
    2. Reciprocal-space contribution (long-range, FFT + B-spline interpolation)
    3. Self-energy and background corrections

    Total Energy Formula:
        E_total = E_real + E_reciprocal - E_self - E_background

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    charges : jax.Array, shape (N,)
        Atomic partial charges in elementary charge units.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrices with lattice vectors as rows. Shape (3, 3) is
        automatically promoted to (1, 3, 3) for single-system mode.
    alpha : float, jax.Array, or None, default=None
        Ewald splitting parameter controlling real/reciprocal space balance.
        - float: Same α for all systems
        - Array shape (B,): Per-system α values
        - None: Automatically estimated using Kolafa-Perram formula
    mesh_spacing : float, optional
        Target mesh spacing. Mesh dimensions computed as ceil(cell_length / mesh_spacing).
    mesh_dimensions : tuple[int, int, int], optional
        Explicit FFT mesh dimensions (nx, ny, nz). Power-of-2 values recommended.
    spline_order : int, default=4
        B-spline interpolation order (4 = cubic B-splines, recommended).
    batch_idx : jax.Array, shape (N,), dtype=int32, optional
        System index for each atom (0 to B-1). Determines execution mode:
        - None: Single-system optimized kernels
        - Provided: Batched kernels for multiple independent systems
        When provided, atoms must be grouped by system: ``batch_idx`` must be
        contiguous, nondecreasing, and use system IDs ``0..B-1``.
    k_vectors : jax.Array, optional
        Precomputed k-vectors from generate_k_vectors_pme. Providing this
        along with k_squared skips k-vector generation.
    k_squared : jax.Array, optional
        Precomputed k² values from generate_k_vectors_pme.
    neighbor_list : jax.Array, optional
        CSR-format neighbor list indices. See ewald_real_space.
    neighbor_ptr : jax.Array, optional
        CSR-format neighbor list pointers. See ewald_real_space.
    neighbor_shifts : jax.Array, optional
        Periodic image shifts for neighbor list. See ewald_real_space.
    neighbor_matrix : jax.Array, optional
        Dense neighbor matrix. Alternative to CSR format.
    neighbor_matrix_shifts : jax.Array, optional
        Shifts for dense neighbor matrix.
    mask_value : int, optional
        Mask value for invalid neighbors in dense format.
    compute_forces : bool, default=False
        If True, compute per-atom forces.
    compute_charge_gradients : bool, default=False
        If True, compute per-atom charge gradients dE/dq.
    compute_virial : bool, default=False
        If True, compute the virial tensor ``W = -dE/d(displacement)`` for the
        row-vector displacement recipe.
        Stress = -virial / volume.
    accuracy : float, default=1e-6
        Target accuracy for automatic parameter estimation.
    pbc : jax.Array, shape (3,) or (B, 3), dtype=bool, optional
        Per-system periodic boundary conditions. Required when
        ``slab_correction=True``. True marks periodic directions and False
        marks the non-periodic slab direction.
    slab_correction : bool, default=False
        If True, add the Yeh-Berkowitz/Ballenegger slab correction to the
        3D-periodic PME outputs.

    Returns
    -------
    energies : jax.Array, shape (N,)
        Per-atom total electrostatic energies.
    forces : jax.Array, shape (N, 3), optional
        Per-atom forces (only if compute_forces=True).
    charge_gradients : jax.Array, shape (N,), optional
        Per-atom charge gradients (only if compute_charge_gradients=True).
    virial : jax.Array, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor (only if compute_virial=True). Always last in the return tuple.

    Notes
    -----
    Automatic Parameter Estimation (when alpha is None):
        Uses Kolafa-Perram formula for optimal α and mesh dimensions based on
        requested accuracy.

    Energy-derived first-order gradients are supported. Higher-order PME
    reverse-mode higher-order position and charge losses use the private PME
    mesh HVP path.

    When ``cell`` or batch metadata are traced by ``jax.jit`` or other JAX
    transformations, pass explicit ``mesh_dimensions``. When ``alpha`` would be
    estimated from traced inputs, precompute it outside the transformation and
    pass it explicitly. ``mesh_spacing`` and accuracy-based mesh sizing depend
    on concrete setup values.

    Examples
    --------
    Basic usage:

        >>> energies = particle_mesh_ewald(
        ...     positions, charges, cell, alpha=0.3,
        ...     mesh_dimensions=(32, 32, 32),
        ...     neighbor_list=nl, neighbor_ptr=ptr, neighbor_shifts=shifts,
        ... )

    With forces and automatic parameters:

        >>> energies, forces = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     mesh_spacing=1.0, accuracy=1e-5,
        ...     neighbor_list=nl, neighbor_ptr=ptr, neighbor_shifts=shifts,
        ...     compute_forces=True,
        ... )

    Batched systems:

        >>> energies = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     batch_idx=batch_idx,
        ...     neighbor_list=nl, neighbor_ptr=ptr, neighbor_shifts=shifts,
        ... )

    See Also
    --------
    pme_reciprocal_space : Reciprocal-space component only
    ewald_real_space : Real-space component
    estimate_pme_parameters : Automatic parameter estimation
    """
    num_atoms = positions.shape[0]

    # Prepare cell and slab pbc
    cell, num_systems = _prepare_cell(cell)
    if batch_idx is not None:
        batch_idx = batch_idx.astype(jnp.int32)
    if slab_correction:
        pbc = _prepare_pbc_for_slab(pbc, num_systems)

    _require_explicit_mesh_dimensions_in_tracing(
        mesh_dimensions=mesh_dimensions,
        cell=cell,
        alpha=alpha,
        batch_idx=batch_idx,
    )

    # Estimate parameters if not provided
    if alpha is None:
        params = estimate_pme_parameters(positions, cell, batch_idx, accuracy)
        alpha = params.alpha
        if mesh_dimensions is None and mesh_spacing is None:
            # Convert to explicit tuple[int, int, int]
            md = params.mesh_dimensions
            mesh_dimensions = (int(md[0]), int(md[1]), int(md[2]))

    # Prepare alpha
    if isinstance(alpha, (int, float)):
        alpha = jnp.array([alpha] * num_systems, dtype=positions.dtype)
    elif alpha.ndim == 0:
        alpha = alpha.reshape(1)

    if mask_value is None:
        mask_value = num_atoms

    # Determine mesh dimensions
    if mesh_dimensions is None:
        if mesh_spacing is not None:
            mesh_dimensions = mesh_spacing_to_dimensions(cell, mesh_spacing)
        else:
            mesh_dimensions = estimate_pme_mesh_dimensions(cell, alpha, accuracy)

    charges_orig = charges
    need_charge_gradients = compute_charge_gradients or hybrid_forces
    if hybrid_forces:
        positions = jax.lax.stop_gradient(positions)
        charges = jax.lax.stop_gradient(charges)
        cell = jax.lax.stop_gradient(cell)
        alpha = jax.lax.stop_gradient(alpha)
        if k_vectors is not None:
            k_vectors = jax.lax.stop_gradient(k_vectors)
        if k_squared is not None:
            k_squared = jax.lax.stop_gradient(k_squared)
        if volume is not None:
            volume = jax.lax.stop_gradient(volume)
        if cell_inv_t is not None:
            cell_inv_t = jax.lax.stop_gradient(cell_inv_t)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The component direct-output flag\(s\).*",
            category=DeprecationWarning,
        )
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
            compute_charge_gradients=need_charge_gradients,
            compute_virial=compute_virial,
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
            compute_charge_gradients=need_charge_gradients,
            compute_virial=compute_virial,
            k_vectors=k_vectors,
            k_squared=k_squared,
            hybrid_forces=False,
            volume=volume,
            cell_inv_t=cell_inv_t,
            moduli_x=moduli_x,
            moduli_y=moduli_y,
            moduli_z=moduli_z,
        )

    slab = None
    if slab_correction:
        if compute_forces or need_charge_gradients or compute_virial:
            slab = _compute_slab_correction(
                positions,
                charges,
                cell,
                pbc,
                batch_idx=batch_idx,
                compute_forces=compute_forces,
                compute_charge_gradients=need_charge_gradients,
                compute_virial=compute_virial,
            )
        else:
            slab = _slab_correction_energy_autodiff(
                positions,
                charges,
                cell,
                pbc,
                batch_idx=batch_idx,
            )

    component_tuples = [
        rs if isinstance(rs, tuple) else (rs,),
        rec if isinstance(rec, tuple) else (rec,),
    ]
    if slab is not None:
        component_tuples.append(slab if isinstance(slab, tuple) else (slab,))

    def _sum_component(tuple_index: int) -> jax.Array:
        total = component_tuples[0][tuple_index]
        for component in component_tuples[1:]:
            total = total + component[tuple_index]
        return total

    tuple_index = 0
    total_energies = _sum_component(tuple_index)
    tuple_index += 1
    total_charge_grads = None
    results: tuple[jax.Array, ...] = (total_energies,)

    if compute_forces:
        total_forces = _sum_component(tuple_index)
        results += (total_forces,)
        tuple_index += 1

    if need_charge_gradients:
        total_charge_grads = _sum_component(tuple_index)
        tuple_index += 1
        if compute_charge_gradients:
            results += (total_charge_grads,)

    if compute_virial:
        total_virial = _sum_component(tuple_index)
        results += (total_virial,)

    if hybrid_forces and total_charge_grads is not None:
        bidx_for_inject = (
            batch_idx
            if batch_idx is not None
            else jnp.zeros(num_atoms, dtype=jnp.int32)
        )
        total_energies = _inject_charge_grad(
            total_energies,
            charges_orig,
            total_charge_grads,
            batch_idx is not None,
            bidx_for_inject,
            num_systems,
        )
        results = (total_energies, *results[1:])

    return results[0] if len(results) == 1 else results


def _resolve_particle_mesh_ewald_parameters(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: float | jax.Array | None,
    mesh_spacing: float | None,
    mesh_dimensions: tuple[int, int, int] | None,
    batch_idx: jax.Array | None,
    accuracy: float,
) -> tuple[jax.Array, jax.Array, tuple[int, int, int]]:
    """Resolve PME ``cell``, ``alpha``, and mesh dimensions for custom rules."""
    _require_explicit_mesh_dimensions_in_tracing(
        mesh_dimensions=mesh_dimensions,
        cell=cell,
        alpha=alpha,
        batch_idx=batch_idx,
    )
    cell_3d = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
    num_systems = cell_3d.shape[0]

    if alpha is None:
        params = estimate_pme_parameters(positions, cell_3d, batch_idx, accuracy)
        alpha = params.alpha
        if mesh_dimensions is None and mesh_spacing is None:
            md = params.mesh_dimensions
            mesh_dimensions = (int(md[0]), int(md[1]), int(md[2]))

    if isinstance(alpha, (int, float)):
        alpha = jnp.array([alpha] * num_systems, dtype=positions.dtype)
    elif alpha.ndim == 0:
        alpha = alpha.reshape(1)

    if mesh_dimensions is None:
        if mesh_spacing is not None:
            mesh_dimensions = mesh_spacing_to_dimensions(cell_3d, mesh_spacing)
        else:
            mesh_dimensions = estimate_pme_mesh_dimensions(cell_3d, alpha, accuracy)

    return cell_3d, alpha, mesh_dimensions


def particle_mesh_ewald(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: float | jax.Array | None = None,
    mesh_spacing: float | None = None,
    mesh_dimensions: tuple[int, int, int] | None = None,
    spline_order: int = 4,
    batch_idx: jax.Array | None = None,
    k_vectors: jax.Array | None = None,
    k_squared: jax.Array | None = None,
    neighbor_list: jax.Array | None = None,
    neighbor_ptr: jax.Array | None = None,
    neighbor_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    mask_value: int | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    accuracy: float = 1e-6,
    hybrid_forces: bool = False,
    pbc: jax.Array | None = None,
    slab_correction: bool = False,
    *,
    cell_inv_t: jax.Array | None = None,
    volume: jax.Array | None = None,
    moduli_x: jax.Array | None = None,
    moduli_y: jax.Array | None = None,
    moduli_z: jax.Array | None = None,
) -> (
    jax.Array
    | tuple[jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Complete Particle Mesh Ewald calculation for long-range electrostatics.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    charges : jax.Array, shape (N,)
        Atomic partial charges.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrices with lattice vectors as rows.
    alpha : float, jax.Array, or None, default=None
        Ewald splitting parameter. If ``None``, estimated automatically.
    mesh_spacing : float or None, default=None
        Target mesh spacing used when ``mesh_dimensions`` is omitted.
    mesh_dimensions : tuple[int, int, int] or None, default=None
        Explicit FFT mesh dimensions.
    spline_order : int, default=4
        B-spline interpolation order.
    batch_idx : jax.Array or None, default=None
        System index for each atom. When provided, atoms must be grouped by
        system: ``batch_idx`` must be contiguous, nondecreasing, and use system
        IDs ``0..B-1``.
    k_vectors, k_squared : jax.Array or None
        Precomputed PME reciprocal grid values.
    neighbor_list, neighbor_ptr, neighbor_shifts : jax.Array or None
        CSR neighbor-list inputs for the real-space component.
    neighbor_matrix, neighbor_matrix_shifts : jax.Array or None
        Dense neighbor-matrix inputs for the real-space component.
    mask_value : int or None, default=None
        Sentinel value for invalid neighbor-matrix entries.
    compute_forces, compute_charge_gradients, compute_virial : bool
        Deprecated direct-output flags. Compute energy and use JAX autodiff for
        differentiable forces, charge gradients, and strain virials.
    accuracy : float, default=1e-6
        Target accuracy for automatic parameter estimation.
    hybrid_forces : bool, default=False
        Deprecated Torch-compatibility escape hatch for charge-gradient routing.
    pbc : jax.Array, optional
        Per-system periodic boundary conditions for slab correction.
    slab_correction : bool, default=False
        If True, add the Yeh-Berkowitz/Ballenegger slab correction.
    volume, cell_inv_t, moduli_x, moduli_y, moduli_z : jax.Array or None
        Optional precomputed PME intermediates. Cell-derived values supplied
        while differentiating with respect to ``cell`` are treated as static
        metadata that corresponds to the current ``cell``.

    Returns
    -------
    jax.Array or tuple[jax.Array, ...]
        Per-atom energy, plus deprecated direct outputs when requested.

    Notes
    -----
    When ``cell``, ``alpha``, or batch metadata are traced by ``jax.jit`` or
    other JAX transformations, pass explicit ``mesh_dimensions``.
    ``mesh_spacing`` and accuracy-based mesh sizing depend on concrete mesh
    setup values. If ``alpha`` would otherwise be estimated from traced inputs,
    precompute it outside the transformation and pass it explicitly.
    """
    if compute_forces or compute_virial or compute_charge_gradients or hybrid_forces:
        warnings.warn(
            _direct_output_deprecation_msg("particle_mesh_ewald"),
            DeprecationWarning,
            stacklevel=2,
        )

    if compute_forces or compute_charge_gradients or compute_virial or hybrid_forces:
        return _particle_mesh_ewald_impl(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            mesh_spacing=mesh_spacing,
            mesh_dimensions=mesh_dimensions,
            spline_order=spline_order,
            batch_idx=batch_idx,
            k_vectors=k_vectors,
            k_squared=k_squared,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=mask_value,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
            accuracy=accuracy,
            hybrid_forces=hybrid_forces,
            pbc=pbc,
            slab_correction=slab_correction,
            volume=volume,
            cell_inv_t=cell_inv_t,
            moduli_x=moduli_x,
            moduli_y=moduli_y,
            moduli_z=moduli_z,
        )

    cell_3d, alpha_arr, mesh_dims = _resolve_particle_mesh_ewald_parameters(
        positions=positions,
        charges=charges,
        cell=cell,
        alpha=alpha,
        mesh_spacing=mesh_spacing,
        mesh_dimensions=mesh_dimensions,
        batch_idx=batch_idx,
        accuracy=accuracy,
    )
    if mask_value is None:
        mask_value = positions.shape[0]

    # Energy-only path: call the impl directly so the full energy is the sum of
    # real-space and reciprocal terms. Component custom derivative rules provide
    # energy gradients and reverse-mode position/charge higher-order losses.
    return _particle_mesh_ewald_impl(
        positions=positions,
        charges=charges,
        cell=cell_3d,
        alpha=alpha_arr,
        mesh_dimensions=mesh_dims,
        spline_order=spline_order,
        batch_idx=batch_idx,
        k_vectors=k_vectors,
        k_squared=k_squared,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        mask_value=mask_value,
        compute_forces=False,
        compute_charge_gradients=False,
        compute_virial=False,
        accuracy=accuracy,
        hybrid_forces=False,
        pbc=pbc,
        slab_correction=slab_correction,
        volume=volume,
        cell_inv_t=cell_inv_t,
        moduli_x=moduli_x,
        moduli_y=moduli_y,
        moduli_z=moduli_z,
    )
