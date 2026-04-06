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

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel

from nvalchemiops.interactions.electrostatics.pme_kernels import (
    _batch_pme_energy_corrections_kernel_overload,
    _batch_pme_energy_corrections_with_charge_grad_kernel_overload,
    _batch_pme_green_structure_factor_kernel_overload,
    _pme_energy_corrections_kernel_overload,
    _pme_energy_corrections_with_charge_grad_kernel_overload,
    _pme_green_structure_factor_kernel_overload,
)
from nvalchemiops.jax.interactions.electrostatics.ewald import ewald_real_space
from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
    generate_k_vectors_pme,
)
from nvalchemiops.jax.interactions.electrostatics.parameters import (
    estimate_pme_mesh_dimensions,
    estimate_pme_parameters,
    mesh_spacing_to_dimensions,
)
from nvalchemiops.jax.spline import (
    spline_gather,
    spline_gather_vec3,
    spline_spread,
)

__all__ = [
    "particle_mesh_ewald",
    "pme_reciprocal_space",
    "pme_green_structure_factor",
    "pme_energy_corrections",
    "pme_energy_corrections_with_charge_grad",
]


# ==============================================================================
# Helper Function for JAX Kernel Creation
# ==============================================================================


def _make_jax_kernels(
    wp_overload_dict: dict,
    num_outputs: int,
    in_out_argnames: list[str],
) -> dict:
    """Maps JAX data types to Warp kernel overloads.

    Parameters
    ----------
    wp_overload_dict : dict
        Warp kernel overload dictionary keyed by wp.float32/wp.float64.
    num_outputs : int
        Number of output arrays returned by the kernel.
    in_out_argnames : list of str
        Names of in-place output arguments.

    Returns
    -------
    dict
        Dictionary mapping jnp.float32/jnp.float64 to jax_kernel instances.
    """
    _JAX_TO_WP = {jnp.float32: wp.float32, jnp.float64: wp.float64}
    return {
        jax_dtype: jax_kernel(
            wp_overload_dict[wp_dtype],
            num_outputs=num_outputs,
            in_out_argnames=in_out_argnames,
            enable_backward=False,
        )
        for jax_dtype, wp_dtype in _JAX_TO_WP.items()
    }


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


# ==============================================================================
# Public API Functions
# ==============================================================================


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

    # Vectorized virial computation using einsum
    # virial_ab = sum_k weighted_energy * (delta_ab - k_factor * k_a * k_b) * k_mask
    # = delta_ab * sum_k (weighted_energy * k_mask) - sum_k (weighted_energy * k_mask * k_factor) * k_a * k_b
    k_vecs_acc = k_vectors.astype(acc_dtype)  # (..., nx, ny, nz//2+1, 3)
    if is_batch and k_vecs_acc.ndim == 4:
        k_vecs_acc = jnp.expand_dims(k_vecs_acc, axis=0)

    masked_energy = weighted_energy * k_mask  # (..., nx, ny, nz//2+1)
    masked_energy_kf = masked_energy * k_factor  # (..., nx, ny, nz//2+1)

    # Sum dimensions depend on batch vs single
    if is_batch:
        sum_dims = (1, 2, 3)  # sum over (nx, ny, nz//2+1)
    else:
        sum_dims = (0, 1, 2)  # sum over (nx, ny, nz//2+1)

    # Trace term: delta_ab * sum_k masked_energy
    trace_term = masked_energy.sum(axis=sum_dims)  # scalar or (B,)

    # kk term: sum_k masked_energy_kf * k_a * k_b
    # k_vecs_acc has shape (..., nx, ny, nz//2+1, 3)
    # masked_energy_kf has shape (..., nx, ny, nz//2+1)
    # Use einsum for vectorized outer product + reduction
    if is_batch:
        # k_vecs: (B, nx, ny, nz_half, 3), masked_energy_kf: (B, nx, ny, nz_half)
        kk_term = jnp.einsum(
            "b...i,b...j,b...->bij", k_vecs_acc, k_vecs_acc, masked_energy_kf
        )  # (B, 3, 3)
        eye = jnp.eye(3, dtype=acc_dtype)
        virial = eye * trace_term[:, jnp.newaxis, jnp.newaxis] - kk_term  # (B, 3, 3)
    else:
        kk_term = jnp.einsum(
            "...i,...j,...->ij", k_vecs_acc, k_vecs_acc, masked_energy_kf
        )  # (3, 3)
        eye = jnp.eye(3, dtype=acc_dtype)
        virial = (eye * trace_term - kk_term)[jnp.newaxis, :, :]  # (1, 3, 3)

    return virial.astype(acc_dtype)


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
        # Build return tuple based on flags
        result = [energies]
        if compute_forces:
            result.append(forces)
        if compute_charge_gradients:
            result.append(charge_grads)
        if compute_virial:
            result.append(virial)
        if len(result) == 1:
            return result[0]
        return tuple(result)

    # Determine mesh dimensions
    if mesh_dimensions is None:
        if mesh_spacing is not None:
            mesh_dimensions = mesh_spacing_to_dimensions(cell, mesh_spacing)
        else:
            # Default estimation
            mesh_dimensions = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-6)

    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions

    # Step 1: Spread charges to mesh
    mesh_grid = spline_spread(
        positions,
        charges,
        cell,
        mesh_dims=mesh_dimensions,
        spline_order=spline_order,
        batch_idx=batch_idx,
    )

    # Step 2: FFT of charge mesh
    mesh_fft = jnp.fft.rfftn(mesh_grid, axes=fft_dims, norm="backward")

    # Step 3: Generate k-space grid and compute Green's function + structure factor
    if k_vectors is None or k_squared is None:
        k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dimensions)

    green_function, structure_factor_sq = pme_green_structure_factor(
        k_squared,
        mesh_dimensions,
        alpha,
        cell,
        spline_order,
        batch_idx,
    )

    # Save reference to raw FFT before deconvolution (needed for virial).
    # No copy needed: the reassignment below creates a new array.
    mesh_fft_raw = mesh_fft if compute_virial else None

    # Step 4: Apply B-spline deconvolution and convolve with Green's function
    # Upcast to the complex equivalent of input_dtype to preserve imaginary part.
    # spline_spread now returns the same dtype as input positions.
    # rfftn then produces complex64 (float32 input) or complex128 (float64 input).
    # (casting complex to real silently drops the imaginary component).
    complex_dtype = jnp.complex64 if input_dtype == jnp.float32 else jnp.complex128
    mesh_fft = mesh_fft.astype(complex_dtype) / structure_factor_sq
    convolved_mesh = mesh_fft * green_function

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

    # Step 6: Inverse FFT to get potential mesh
    potential_mesh = jnp.fft.irfftn(
        convolved_mesh, s=mesh_dimensions, axes=fft_dims, norm="forward"
    )

    # Step 6: Interpolate potential to atomic positions (dtype matches positions)
    raw_energies = spline_gather(
        positions,
        potential_mesh,
        cell,
        spline_order=spline_order,
        batch_idx=batch_idx,
    )

    # Step 7: Apply corrections
    if compute_charge_gradients:
        energies, charge_grads = pme_energy_corrections_with_charge_grad(
            raw_energies, charges, cell, alpha, batch_idx
        )
    else:
        energies = pme_energy_corrections(raw_energies, charges, cell, alpha, batch_idx)
        charge_grads = None

    # Step 8: Compute forces if needed
    forces = None
    if compute_forces:
        # Compute electric field by taking gradient in Fourier space
        Ex_fft = -1j * k_vectors[..., 0] * convolved_mesh
        Ey_fft = -1j * k_vectors[..., 1] * convolved_mesh
        Ez_fft = -1j * k_vectors[..., 2] * convolved_mesh

        Ex = jnp.fft.irfftn(Ex_fft, s=mesh_dimensions, axes=fft_dims, norm="forward")
        Ey = jnp.fft.irfftn(Ey_fft, s=mesh_dimensions, axes=fft_dims, norm="forward")
        Ez = jnp.fft.irfftn(Ez_fft, s=mesh_dimensions, axes=fft_dims, norm="forward")

        electric_field_mesh = jnp.stack([Ex, Ey, Ez], axis=-1)

        # Interpolate electric field to atomic positions (dtype matches positions)
        interpolated_field = spline_gather_vec3(
            positions,
            charges,
            electric_field_mesh,
            cell,
            spline_order=spline_order,
            batch_idx=batch_idx,
        )

        # Compute forces: F = 2 * q * E
        forces = 2.0 * interpolated_field

    # Build return tuple based on flags
    # Order: energies, [forces], [charge_grads], [virial] (virial always last)
    if compute_forces and compute_charge_gradients and compute_virial:
        return energies, forces, charge_grads, virial
    elif compute_forces and compute_charge_gradients:
        return energies, forces, charge_grads
    elif compute_forces and compute_virial:
        return energies, forces, virial
    elif compute_charge_gradients and compute_virial:
        return energies, charge_grads, virial
    elif compute_forces:
        return energies, forces
    elif compute_charge_gradients:
        return energies, charge_grads
    elif compute_virial:
        return energies, virial
    else:
        return energies


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

    # Prepare cell
    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    num_systems = cell.shape[0]

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
    )

    # Normalize return tuples for easy combination
    # Both rs and rec return: energies, [forces], [charge_grads], [virial]
    # where virial is always last if present
    rs_tuple = rs if isinstance(rs, tuple) else (rs,)
    rec_tuple = rec if isinstance(rec, tuple) else (rec,)

    # The number of outputs should match between rs and rec
    # Combine element-wise
    results = tuple(r + s for r, s in zip(rs_tuple, rec_tuple))

    if len(results) == 1:
        return results[0]
    return results
