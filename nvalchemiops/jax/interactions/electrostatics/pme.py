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

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import GraphMode, jax_callable

from nvalchemiops.interactions.electrostatics.pme_kernels import (
    _batch_pme_convolve_kernel_overload,
    _batch_pme_energy_corrections_kernel_overload,
    _batch_pme_energy_corrections_with_charge_grad_kernel_overload,
    _batch_pme_green_structure_factor_kernel_overload,
    _pme_convolve_kernel_overload,
    _pme_energy_corrections_kernel_overload,
    _pme_energy_corrections_with_charge_grad_kernel_overload,
    _pme_green_structure_factor_kernel_overload,
    _pme_virial_bg_apply_kernel_overload,
    _pme_virial_bg_reduce_kernel_overload,
)
from nvalchemiops.jax.interactions.electrostatics._lazy_jax_kernels import (
    make_jax_kernels as _make_jax_kernels,
)
from nvalchemiops.jax.interactions.electrostatics._utils import (
    _build_electrostatic_result,
    _combine_electrostatic_outputs,
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
)
from nvalchemiops.jax.interactions.electrostatics.slab import (
    compute_slab_correction as _compute_slab_correction,
)
from nvalchemiops.jax.spline import (
    spline_gather,
    spline_gather_with_force,
    spline_spread,
)

__all__ = [
    "particle_mesh_ewald",
    "pme_reciprocal_space",
    "pme_green_structure_factor",
    "pme_fused_convolve",
    "pme_energy_corrections",
    "pme_energy_corrections_with_charge_grad",
    "pme_virial_bg_correction",
    "compute_bspline_moduli_1d",
]


# ==============================================================================
# Helper Function for JAX Kernel Creation
# ==============================================================================

# ``_make_jax_kernels`` is imported from ``_lazy_jax_kernels``: each entry is a
# :func:`warp.jax_experimental.jax_kernel` binding for the matching warp
# overload, built lazily on first ``__getitem__`` access. Single-kernel
# ``jax_kernel`` wraps are preferred here because they outperform
# single-launch ``jax_callable`` wraps -- profiling showed converting 1:1 to
# ``jax_callable`` added ~15% per-iter overhead vs ``jax_kernel`` (the
# additional bookkeeping ``FfiCallable`` does isn't amortized when there's
# only one wp.launch). Use ``jax_callable`` only when you have multiple
# wp.launch calls to fuse into one FFI thunk (see
# :func:`_make_jax_pme_virial_bg_fused` for an example).


def _normalize_dtype(dtype):
    """Normalize dtype for kernel dictionary lookup.

    Parameters
    ----------
    dtype : dtype-like
        Input dtype from a JAX array.

    Returns
    -------
    jnp.float32 or jnp.float64
        Normalized JAX dtype for kernel lookup.
    """
    if dtype == jnp.float32 or str(dtype) == "float32":
        return jnp.float32
    elif dtype == jnp.float64 or str(dtype) == "float64":
        return jnp.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# Single-system kernels
_jax_pme_green_sf = _make_jax_kernels(
    _pme_green_structure_factor_kernel_overload,
    2,
    ["green_function", "structure_factor_sq"],
)

_jax_pme_energy_corrections = _make_jax_kernels(
    _pme_energy_corrections_kernel_overload,
    1,
    ["corrected_energies"],
)

_jax_pme_energy_corrections_charge_grad = _make_jax_kernels(
    _pme_energy_corrections_with_charge_grad_kernel_overload,
    2,
    ["corrected_energies", "charge_gradients"],
)

# Batch kernels
_jax_batch_pme_green_sf = _make_jax_kernels(
    _batch_pme_green_structure_factor_kernel_overload,
    2,
    ["green_function", "structure_factor_sq"],
)

_jax_batch_pme_energy_corrections = _make_jax_kernels(
    _batch_pme_energy_corrections_kernel_overload,
    1,
    ["corrected_energies"],
)

_jax_batch_pme_energy_corrections_charge_grad = _make_jax_kernels(
    _batch_pme_energy_corrections_with_charge_grad_kernel_overload,
    2,
    ["corrected_energies", "charge_gradients"],
)

# Fused convolve — replaces the older Green's-function + multiply path with
# a single warp kernel that computes G(k), the B-spline structure factor
# correction C^2(k), and multiplies mesh_fft → convolved_mesh in one launch.
# (mirrors the fused-convolve path in the torch bindings.)
_jax_pme_convolve = _make_jax_kernels(
    _pme_convolve_kernel_overload,
    1,
    ["convolved_mesh"],
)

_jax_batch_pme_convolve = _make_jax_kernels(
    _batch_pme_convolve_kernel_overload,
    1,
    ["convolved_mesh"],
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


# Fused two-pass virial bg correction via warp.jax_experimental.jax_callable.
# Wraps both the scatter-add reduce and the per-system E_bg apply as a SINGLE
# XLA FFI call. With GraphMode.JAX (default) plus the
# ``--xla_gpu_graph_min_graph_size=1`` flag set in ``nvalchemiops/jax/__init__.py``,
# JAX can capture this together with the surrounding warp ops + FFTs into one
# CUDA graph, eliminating per-launch dispatch gaps that dominated iter time at
# large mesh sizes. The fallback to the two ``jax_kernel`` thunks above is
# kept intact for tests that exercise the per-pass kernels directly.
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
    return s ** spline_order


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
    real_dtype = (
        jnp.float32 if mesh_fft.dtype == jnp.complex64 else jnp.float64
    )
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
        mesh_fft_real, k_squared, moduli_x, moduli_y, moduli_z,
        alpha, volume, convolved_real,
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
        System index for each atom. If provided, uses batch kernels.

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
        volume = jnp.abs(jnp.linalg.det(cell)).astype(input_dtype)
        total_charge = charges.sum().reshape(1).astype(input_dtype)

        # Allocate output
        corrected_energies = jnp.zeros(num_atoms, dtype=input_dtype)

        # Launch kernel
        (corrected_out,) = kernel(
            raw_energies.astype(input_dtype),
            charges.astype(input_dtype),
            volume,
            alpha,
            total_charge,
            corrected_energies,
            launch_dims=(num_atoms,),
        )
        return corrected_out
    else:
        # Batch
        kernel = _jax_batch_pme_energy_corrections[input_dtype]
        num_systems = cell.shape[0] if cell.ndim == 3 else 1

        if cell.ndim == 2:
            cell = cell[jnp.newaxis, :, :]
        volumes = jnp.abs(jnp.linalg.det(cell)).astype(input_dtype)

        # Compute total charge per system
        total_charges = jnp.zeros(num_systems, dtype=input_dtype)
        total_charges = total_charges.at[batch_idx].add(charges.astype(input_dtype))

        # Allocate output
        corrected_energies = jnp.zeros(num_atoms, dtype=input_dtype)

        # Launch kernel
        (corrected_out,) = kernel(
            raw_energies.astype(input_dtype),
            charges.astype(input_dtype),
            batch_idx.astype(jnp.int32),
            volumes,
            alpha,
            total_charges,
            corrected_energies,
            launch_dims=(num_atoms,),
        )
        return corrected_out


def pme_energy_corrections_with_charge_grad(
    raw_energies: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    alpha: jax.Array,
    batch_idx: jax.Array | None = None,
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
        System index for each atom. If provided, uses batch kernels.

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
        volume = jnp.abs(jnp.linalg.det(cell)).astype(input_dtype)
        total_charge = charges.sum().reshape(1).astype(input_dtype)

        # Allocate outputs
        corrected_energies = jnp.zeros(num_atoms, dtype=input_dtype)
        charge_gradients = jnp.zeros(num_atoms, dtype=input_dtype)

        # Launch kernel
        corrected_out, charge_grad_out = kernel(
            raw_energies.astype(input_dtype),
            charges.astype(input_dtype),
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
        volumes = jnp.abs(jnp.linalg.det(cell)).astype(input_dtype)

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

    # Replace einsum with six explicit weighted reductions — mirrors the torch
    # path. XLA lowers ``einsum("...i,...j,...->ij", k, k, m)`` to a batched
    # GEMM with M=N=3, K=mesh-size, which hits cuBLAS's small-MN / large-K
    # weak spot (single largest kernel cost in the profiled timed window).
    # Six per-component sum-of-products avoid the GEMM and are bandwidth-
    # bound.
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
            [jnp.stack([xx, xy, xz], axis=-1),
             jnp.stack([xy, yy, yz], axis=-1),
             jnp.stack([xz, yz, zz], axis=-1)],
            axis=-2,
        )
        virial = eye * trace_term[:, jnp.newaxis, jnp.newaxis] - kk_term  # (B, 3, 3)
    else:
        kk_term = jnp.stack(
            [jnp.stack([xx, xy, xz]),
             jnp.stack([xy, yy, yz]),
             jnp.stack([xz, yz, zz])],
        )  # (3, 3)
        virial = (eye * trace_term - kk_term)[jnp.newaxis, :, :]  # (1, 3, 3)

    return virial.astype(acc_dtype)


# Custom-VJP straight-through injector for hybrid-forces mode.
# Forward returns ``energy`` unchanged. Backward routes ``grad_energy`` to
# ``charges`` via the precomputed analytical ``charge_grad`` (∂E/∂q from the
# kernel-side ``pme_energy_corrections_with_charge_grad``), while never
# propagating into the spline/FFT chain. ``has_batch_idx`` is non-diff so
# we can branch on Python ``None`` without breaking tracing.
@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _inject_charge_grad(energy, charges, charge_grad, has_batch_idx, batch_idx):
    # ``charges`` is referenced only to carry an autograd edge; the forward
    # is a no-op on energy.
    del charges, charge_grad, has_batch_idx, batch_idx
    return energy


def _inject_charge_grad_fwd(energy, charges, charge_grad, has_batch_idx, batch_idx):
    del charges
    return energy, (charge_grad, batch_idx)


def _inject_charge_grad_bwd(has_batch_idx, residuals, grad_energy):
    charge_grad, batch_idx = residuals
    if has_batch_idx:
        # Per-atom grad_energy lookup via system index.
        atom_grad = grad_energy[batch_idx]
    else:
        # Single-system: ``grad_energy`` is per-atom already; passing it
        # through unchanged matches torch's ``grad_energy.squeeze(0)`` when
        # the energy is already per-atom.
        atom_grad = grad_energy
    grad_charges = charge_grad * atom_grad
    # (energy, charges, charge_grad, batch_idx) — no grad for charge_grad,
    # batch_idx; has_batch_idx is non-diff so not returned here.
    return grad_energy, grad_charges, jnp.zeros_like(charge_grad), jnp.zeros_like(batch_idx)


_inject_charge_grad.defvjp(_inject_charge_grad_fwd, _inject_charge_grad_bwd)


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
    volume: jax.Array | None = None,
    cell_inv_t: jax.Array | None = None,
    moduli_x: jax.Array | None = None,
    moduli_y: jax.Array | None = None,
    moduli_z: jax.Array | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    hybrid_forces: bool = False,
) -> (
    jax.Array
    | tuple[jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Compute PME reciprocal-space contribution.

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
        System index for each atom.
    k_vectors : jax.Array, optional
        Precomputed k-vectors from generate_k_vectors_pme.
    k_squared : jax.Array, optional
        Precomputed k² values from generate_k_vectors_pme.
    compute_forces : bool, default=False
        If True, compute forces via Fourier gradient.
    compute_charge_gradients : bool, default=False
        If True, compute charge gradients dE/dq.
    compute_virial : bool, default=False
        If True, compute the virial tensor W = -dE/d(epsilon).
        Stress = virial / volume.
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
    """
    num_atoms = positions.shape[0]
    input_dtype = _normalize_dtype(positions.dtype)
    is_batch = batch_idx is not None
    fft_dims = (1, 2, 3) if is_batch else (0, 1, 2)

    # hybrid_forces: forward-only spline/FFT chain. We sever ∂/∂{positions,
    # cell} (and the spline/FFT path through charges) via lax.stop_gradient,
    # then re-attach the analytical ∂E/∂q at the end. Charges still need to
    # be a tracer for the custom-VJP injector to see them, so save the
    # original handle.
    charges_orig = charges
    if hybrid_forces:
        compute_charge_gradients = True
        positions = jax.lax.stop_gradient(positions)
        charges = jax.lax.stop_gradient(charges)
        cell = jax.lax.stop_gradient(cell)
        alpha = jax.lax.stop_gradient(alpha)

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
            jnp.zeros(num_atoms, dtype=input_dtype)
            if compute_charge_gradients
            else None
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
            cell, mesh_dimensions, reciprocal_cell=reciprocal_cell,
        )

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
    # Caller-supplied volume= short-circuits the linalg.det.
    if volume is None:
        volume = jnp.abs(jnp.linalg.det(cell_3d)).astype(input_dtype)
    else:
        volume = volume.astype(input_dtype)
        if volume.ndim == 0:
            volume = volume.reshape(1)

    complex_dtype = jnp.complex64 if input_dtype == jnp.float32 else jnp.complex128
    mesh_fft = mesh_fft.astype(complex_dtype)
    # Raw FFT (before convolve) is what the virial path needs.
    mesh_fft_raw = mesh_fft if compute_virial else None

    convolved_mesh = pme_fused_convolve(
        mesh_fft, k_squared.astype(input_dtype),
        moduli_x, moduli_y, moduli_z,
        alpha, volume, is_batch,
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
        raw_energies, gathered_force = spline_gather_with_force(
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
    if compute_charge_gradients:
        energies, charge_grads = pme_energy_corrections_with_charge_grad(
            raw_energies, charges, cell, alpha, batch_idx
        )
    else:
        energies = pme_energy_corrections(raw_energies, charges, cell, alpha, batch_idx)
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
            energies, charges_orig, charge_grads, batch_idx is not None,
            bidx_for_inject,
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
        If True, compute the virial tensor W = -dE/d(epsilon).
        Stress = virial / volume.
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
        hybrid_forces=hybrid_forces,
        volume=volume,
        cell_inv_t=cell_inv_t,
        moduli_x=moduli_x,
        moduli_y=moduli_y,
        moduli_z=moduli_z,
    )

    slab = None
    if slab_correction:
        slab = _compute_slab_correction(
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
        slab,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )
