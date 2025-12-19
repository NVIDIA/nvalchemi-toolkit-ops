# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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

REFERENCES
==========

- Essmann et al. (1995). J. Chem. Phys. 103, 8577 (SPME paper)
- Darden et al. (1993). J. Chem. Phys. 98, 10089 (Original PME)
- torchpme: https://github.com/lab-cosmo/torch-pme (Reference implementation)
"""

import math

import warp as wp
from warp.types import Any

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
    2. Structure factor squared: |B(k)|² for B-spline dealiasing

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
        OUTPUT: |B(k)|² structure factor squared at each grid point.

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

    # Green's function: G(k) = 2*pi * exp(-k^2/(4*alpha^2)) / (k^2 * V)
    if k_sq < threshold:
        green_function[i, j, k] = zero
    else:
        exp_factor = wp_exp_kernel(k_sq, one / (four * alpha_ * alpha_))
        green_function[i, j, k] = twopi * exp_factor / volume_

    if i == 0 and j == 0 and k == 0:
        green_function[i, j, k] = zero

    # Structure factor: sinc(mi_x/Nx) * sinc(mi_y/Ny) * sinc(mi_z/Nz)
    sinc_x = compute_sinc(mi_x / type(mi_x)(mesh_nx))
    sinc_y = compute_sinc(mi_y / type(mi_y)(mesh_ny))
    sinc_z = compute_sinc(mi_z / type(mi_z)(mesh_nz))

    sinc_product = sinc_x * sinc_y * sinc_z

    # Raise to spline_order power
    sf = sinc_product
    for _ in range(1, 4):  # Max order 4
        if _ < spline_order:
            sf = sf * sinc_product

    # Clamp to avoid division by zero
    if sf < clamp_threshold:
        sf = clamp_threshold

    structure_factor_sq[i, j, k] = sf * sf


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
    Structure factor: |B(k)|² (computed once, shared across systems)

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
        OUTPUT: |B(k)|² structure factor squared (computed only at batch_idx=0).

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

    # Green's function: G(k) = 2*pi * exp(-k^2/(4*alpha^2)) / (k^2 * V)
    if k_sq < threshold:
        green_function[batch_idx, i, j, k] = zero
    else:
        exp_factor = wp_exp_kernel(k_sq, one / (four * system_alpha * system_alpha))
        green_function[batch_idx, i, j, k] = twopi * exp_factor / system_volume

    if i == 0 and j == 0 and k == 0:
        green_function[batch_idx, i, j, k] = zero

    # Structure factor (only compute once per k-point, at batch_idx=0)
    if batch_idx == wp.int32(0):
        sinc_x = compute_sinc(mi_x / type(mi_x)(mesh_nx))
        sinc_y = compute_sinc(mi_y / type(mi_y)(mesh_ny))
        sinc_z = compute_sinc(mi_z / type(mi_z)(mesh_nz))

        sinc_product = sinc_x * sinc_y * sinc_z

        sf = sinc_product
        for _ in range(1, 4):
            if _ < spline_order:
                sf = sf * sinc_product

        if sf < clamp_threshold:
            sf = clamp_threshold

        structure_factor_sq[i, j, k] = sf * sf


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
########################### Kernel Overloads for Dtype Flexibility ########################
###########################################################################################

# Type lists for creating overloads
_T = [wp.float32, wp.float64]

# Single-system kernel overloads
_pme_green_structure_factor_kernel_overload = {}
_pme_energy_corrections_kernel_overload = {}
_pme_energy_corrections_with_charge_grad_kernel_overload = {}

# Batch kernel overloads
_batch_pme_green_structure_factor_kernel_overload = {}
_batch_pme_energy_corrections_kernel_overload = {}
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
