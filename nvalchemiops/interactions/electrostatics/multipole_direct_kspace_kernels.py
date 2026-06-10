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
Direct k-space Multipole Electrostatics Kernels
================================================

Warp kernels for the direct-k-space (no Ewald splitting) multipolar
electrostatics pipeline. These kernels are the building blocks for the
``l_max = 1`` energy, the atom-centered electrostatic features, and the
SCF-loop precomputation cache — they match the decomposition used by the
customer reference ``graph_longrange``, enabling bit-for-bit parity tests.

Pipeline outline
----------------

For a system of ``N`` atoms in a periodic box with k-space cutoff, the direct
k-space pipeline is:

1. **Structure factor table** (:func:`build_structure_factor_table`): dense
   :math:`(\cos(\mathbf{k}\cdot\mathbf{r}_i),\ \sin(\mathbf{k}\cdot\mathbf{r}_i))`
   for every ``(k, atom)`` pair.
2. **GTO Fourier coefficients** (:func:`eval_gto_fourier_dipole`): the per-k,
   per-``(l, m)`` complex coefficients
   :math:`\hat\phi_{l,m}^{\sigma}(\mathbf{k})` of the GTO basis.
3. **Density assembly** (:func:`assemble_rho_k_dipole`): combines (1) and (2)
   with per-atom multipole moments to produce :math:`\rho(\mathbf{k})`.
4. **Per-k multiplier** (:func:`apply_per_k_factor`): generic
   :math:`V(\mathbf{k}) = f(\mathbf{k}) \cdot \rho(\mathbf{k})` kernel.
   The direct k-space sum passes ``f(k) = F / k²`` with the k=0 mode zeroed;
   the Ewald reciprocal sum passes ``f(k) = F · exp(-k²/4α²) / k²``. The kernel
   is factor-agnostic.
5. **Energy product / feature projection** (follow-ups).

Each stage factors out cleanly so that the Phase 7a Ewald-reciprocal kernel
can reuse the same machinery with a different per-k multiplier
(:math:`\exp(-k^2/4\alpha^2)/k^2` instead of :math:`1/k^2`).

Conventions
-----------

* **k-space phase:** :math:`\hat\phi_{l,m}(\mathbf{k}) \propto i^{-l}
  \cdot Y_l^m(\hat{\mathbf{k}}) \cdot \text{radial}(|\mathbf{k}|)`. Real and
  imaginary parts are stored side-by-side in the last dimension of the
  output (``[..., 0]`` = real, ``[..., 1]`` = imag). For ``l=0`` the result
  is purely real; for ``l=1`` it is purely imaginary with sign ``-1``.
* **``m`` ordering** within each ``l`` block is
  :math:`m = -l, \ldots, +l` in physics order; ``Y_1^m`` maps to
  ``(k_y, k_z, k_x)`` via :mod:`nvalchemiops.math.spherical_harmonics`.
* **Normalization.** The ``inv_cl`` scaling factor (host-side, computed by
  :func:`nvalchemiops.torch.math.gto.inv_cl`) is passed in per-``l`` as a
  kernel argument. The kernel itself is normalization-agnostic.
"""

from __future__ import annotations

import math
from typing import Any

import warp as wp

from nvalchemiops.math.spherical_harmonics import (
    Y00_COEFF,
    Y1_COEFF,
    Y2_0_COEFF,
    Y2_M1_COEFF,
    Y2_M2_COEFF,
    Y2_P1_COEFF,
    Y2_P2_COEFF,
)
from nvalchemiops.warp_dispatch import register_overloads

# 4π · √(π/2), the common radial prefactor in φ̂_{l,m}^σ(k).
_FOUR_PI_SQRT_PI_OVER_2 = wp.constant(
    wp.float64(4.0 * math.pi * math.sqrt(math.pi / 2.0))
)

# (2π)^3 — the Fourier-series normalization that ships ρ(k) at its physical
# density scale (matches ``graph_longrange.features.assemble_fourier_series_batch``).
_TWO_PI_CUBED = wp.constant(wp.float64((2.0 * math.pi) ** 3))

# 1 / (2π)^3 — the inverse-Fourier scaling for the feature projection.
_INV_TWO_PI_CUBED = wp.constant(wp.float64(1.0 / (2.0 * math.pi) ** 3))


# =============================================================================
# (cos, sin) structure-factor table
# =============================================================================


@wp.kernel
def _build_structure_factor_table_kernel(
    k_vectors: wp.array(dtype=wp.vec3d),
    positions: wp.array(dtype=Any),
    cos_table: wp.array2d(dtype=wp.float64),
    sin_table: wp.array2d(dtype=wp.float64),
):
    r"""Compute :math:`(\cos(\mathbf{k}\cdot\mathbf{r}),\ \sin(\mathbf{k}\cdot\mathbf{r}))` for every ``(k, atom)`` pair.

    Launch Grid
    -----------
    dim = [N_k, N_atoms] — one thread per ``(k, atom)`` pair.

    Parameters
    ----------
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-lattice k-vectors. Always float64 — k-vector precision is
        propagated straight through to the energy accumulation, so downgrading
        them to float32 is never worth the noise.
    positions : wp.array, shape (N_atoms,), dtype wp.vec3d or wp.vec3f
        Atomic positions. Both dtypes are supported; the inner product is
        always computed in float64.
    cos_table, sin_table : wp.array2d(dtype=wp.float64), shape (N_k, N_atoms)
        OUTPUT tables. Do not need to be pre-zeroed (this kernel writes every
        entry unconditionally).
    """
    k_idx, atom_idx = wp.tid()

    k_vec = k_vectors[k_idx]
    pos = positions[atom_idx]

    # Always compute the dot product in float64 regardless of position dtype.
    kr = (
        k_vec[0] * wp.float64(pos[0])
        + k_vec[1] * wp.float64(pos[1])
        + k_vec[2] * wp.float64(pos[2])
    )

    cos_table[k_idx, atom_idx] = wp.cos(kr)
    sin_table[k_idx, atom_idx] = wp.sin(kr)


def _structure_factor_sig(v, t):
    """Signature builder: only ``positions`` carries the polymorphic ``v`` dtype."""
    return [
        wp.array(dtype=wp.vec3d),  # k_vectors
        wp.array(dtype=v),  # positions
        wp.array2d(dtype=wp.float64),  # cos_table
        wp.array2d(dtype=wp.float64),  # sin_table
    ]


_build_structure_factor_table_overloads = register_overloads(
    _build_structure_factor_table_kernel, _structure_factor_sig
)


def build_structure_factor_table(
    k_vectors: wp.array,
    positions: wp.array,
    cos_table: wp.array,
    sin_table: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_build_structure_factor_table_kernel`.

    Produces a dense ``(N_k, N_atoms)`` pair of float64 tables of
    :math:`\cos(\mathbf{k}\cdot\mathbf{r}_i)` and
    :math:`\sin(\mathbf{k}\cdot\mathbf{r}_i)`. This is the "geometry-only"
    half of the per-k structure factor; multiplying by atomic moments comes
    in a later kernel.

    Parameters
    ----------
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-lattice k-vectors.
    positions : wp.array, shape (N_atoms,), dtype wp.vec3f or wp.vec3d
        Atomic positions. Selects the kernel overload.
    cos_table, sin_table : wp.array, shape (N_k, N_atoms), dtype wp.float64
        Pre-allocated output buffers.
    wp_dtype : type
        Scalar type matching ``positions``: ``wp.float32`` or ``wp.float64``.
    device : str, optional
        Warp device string; defaults to ``k_vectors.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f

    n_k = k_vectors.shape[0]
    n_atoms = positions.shape[0]
    if device is None:
        device = str(k_vectors.device)

    wp.launch(
        _build_structure_factor_table_overloads[vec_dtype],
        dim=(n_k, n_atoms),
        inputs=[k_vectors, positions, cos_table, sin_table],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched structure-factor table
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_build_structure_factor_table_kernel(
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    positions: wp.array(dtype=Any),  # (N_total,)
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    cos_table: wp.array2d(dtype=wp.float64),  # (K_max, N_total) OUTPUT
    sin_table: wp.array2d(dtype=wp.float64),  # (K_max, N_total) OUTPUT
):
    r"""Batched :math:`(\cos(k r), \sin(k r))` table across ``B`` systems.

    Atoms are flat across the batch; ``batch_idx[atom_idx]`` gives the
    system id so each thread picks up the right ``k_vectors[b, k]``.
    Output tables are ``(K_max, N_total)`` matching the existing
    ``batch_ewald_reciprocal_*`` convention. Pad k-vectors
    (``batch_idx`` still picks them up at indices ``>= K_b`` for a given
    system's atoms) are zero, so ``cos = 1`` / ``sin = 0`` — harmless
    because the pad rows are zeroed out via ``per_k_factor`` /
    ``k_factor_proj`` downstream.

    Launch Grid
    -----------
    ``dim = (K_max, N_total)``.


    Parameters
    ----------
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    positions : wp.array, shape (N_total,), dtype Any
        Atomic Cartesian positions.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    cos_table : wp.array2d, shape (K_max, N_total), dtype wp.float64
        OUTPUT: Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sin_table : wp.array2d, shape (K_max, N_total), dtype wp.float64
        OUTPUT: Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    """
    k_idx, atom_idx = wp.tid()

    b = batch_idx[atom_idx]
    k_vec = k_vectors[b, k_idx]
    pos = positions[atom_idx]

    kr = (
        k_vec[0] * wp.float64(pos[0])
        + k_vec[1] * wp.float64(pos[1])
        + k_vec[2] * wp.float64(pos[2])
    )

    cos_table[k_idx, atom_idx] = wp.cos(kr)
    sin_table[k_idx, atom_idx] = wp.sin(kr)


def _batch_structure_factor_sig(v, t):
    """Signature builder for the batched structure-factor kernel."""
    del t
    return [
        wp.array2d(dtype=wp.vec3d),  # k_vectors
        wp.array(dtype=v),  # positions (polymorphic float32/float64)
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array2d(dtype=wp.float64),  # cos_table
        wp.array2d(dtype=wp.float64),  # sin_table
    ]


_batch_build_structure_factor_table_overloads = register_overloads(
    _batch_build_structure_factor_table_kernel, _batch_structure_factor_sig
)


def batch_build_structure_factor_table(
    k_vectors: wp.array,
    positions: wp.array,
    batch_idx: wp.array,
    cos_table: wp.array,
    sin_table: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_build_structure_factor_table_kernel`.

    Parameters
    ----------
    k_vectors : wp.array, shape (B, K_max), dtype wp.vec3d
        Per-system k-vectors, zero-padded to ``K_max`` along axis 1.
    positions : wp.array, shape (N_total,), dtype wp.vec3f/wp.vec3d
        Flat atomic positions across the batch.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        System index for each atom; ``batch_idx[i] ∈ [0, B)``.
    cos_table, sin_table : wp.array, shape (K_max, N_total), dtype wp.float64
        Pre-allocated output buffers.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64``; selects the positions overload.
    device : str, optional
        Defaults to ``positions.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f

    k_max = k_vectors.shape[1]
    n_total = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _batch_build_structure_factor_table_overloads[vec_dtype],
        dim=(k_max, n_total),
        inputs=[k_vectors, positions, batch_idx, cos_table, sin_table],
        device=device,
    )


# =============================================================================
# GTO Fourier coefficients φ̂_{l,m}^σ(k) for l_max ≤ 1
# =============================================================================


@wp.kernel
def _eval_gto_fourier_dipole_kernel(
    k_vectors: wp.array(dtype=wp.vec3d),
    k_norm2: wp.array(dtype=wp.float64),
    sigma: wp.float64,
    inv_cl_l0: wp.float64,
    inv_cl_l1: wp.float64,
    output: wp.array3d(dtype=wp.float64),
):
    r"""Evaluate GTO basis Fourier coefficients :math:`\hat\phi_{l,m}^\sigma(\mathbf{k})` at l_max = 1.

    For each k-vector, writes a ``(4, 2)`` block corresponding to
    ``(Y_0^0, Y_1^{-1}, Y_1^{0}, Y_1^{+1})`` in the last-but-one dim and
    ``(real, imag)`` in the last dim.

    .. math::

        \hat\phi_{l,m}^{\sigma}(\mathbf{k}) \;=\;
            \frac{1}{C_l(\sigma, \text{mode})}
            \cdot 4\pi \sqrt{\tfrac{\pi}{2}} \cdot \sigma^{2l+3}
            \cdot k^l \cdot e^{-k^2 \sigma^2 / 2}
            \cdot Y_l^m(\hat{\mathbf{k}}) \cdot i^{-l}.

    For l=1 the ``k^l · Y_l^m(k̂)`` product is written directly in Cartesian
    form so the ``k = 0`` case falls out analytically (``k_component = 0``
    at the origin → zero without an explicit guard).

    Launch Grid
    -----------
    dim = [N_k] — one thread per k-vector.

    Parameters
    ----------
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-lattice k-vectors (float64).
    k_norm2 : wp.array, shape (N_k,), dtype wp.float64
        Pre-computed :math:`|\mathbf{k}|^2`. Avoids recomputation since the
        caller typically already has it from k-vector generation.
    sigma : wp.float64
        Single-σ density-basis width.
    inv_cl_l0, inv_cl_l1 : wp.float64
        Host-computed :math:`1/C_l(\sigma, \text{mode})` factors from
        :func:`nvalchemiops.torch.math.gto.inv_cl`.
    output : wp.array3d(dtype=wp.float64), shape (N_k, 4, 2)
        OUTPUT. Entry ``[k, lm, 0]`` is the real part, ``[k, lm, 1]`` the
        imaginary part. ``lm`` layout: ``0 = (l=0, m=0)``,
        ``1 = (l=1, m=-1)``, ``2 = (l=1, m=0)``, ``3 = (l=1, m=+1)``.

    Notes
    -----
    For l=0 the result is purely real (imag = 0); for l=1 the result is
    purely imaginary (real = 0) with the ``-1`` sign of the
    ``i^{-1} = -i`` phase already baked in.
    """
    idx = wp.tid()

    k_vec = k_vectors[idx]
    k2 = k_norm2[idx]

    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    sigma5 = sigma3 * sigma2

    gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)
    common_radial = _FOUR_PI_SQRT_PI_OVER_2 * gauss

    # l = 0, m = 0: purely real.
    # φ̂_{0,0}(k) = inv_cl_l0 · 4π√(π/2) · σ³ · exp(-k²σ²/2) · Y_0^0.
    output[idx, 0, 0] = inv_cl_l0 * common_radial * sigma3 * Y00_COEFF
    output[idx, 0, 1] = wp.float64(0.0)

    # l = 1: purely imaginary with sign -1 (i^{-1} = -i).
    # radial_1 · Y_1^m(k̂) = (4π√(π/2) σ⁵ exp) · k · (Y1_COEFF · k_component / k).
    # The k factors cancel, leaving
    #     coeff_l1 = inv_cl_l1 · 4π√(π/2) · σ⁵ · exp(-k²σ²/2) · Y1_COEFF
    # and the imag part is  `-coeff_l1 · k_component`. At k = 0 every Cartesian
    # component is zero, so the whole l=1 block is zero analytically.
    coeff_l1 = -inv_cl_l1 * common_radial * sigma5 * Y1_COEFF

    # m = -1 corresponds to k_y in our physics ordering.
    output[idx, 1, 0] = wp.float64(0.0)
    output[idx, 1, 1] = coeff_l1 * k_vec[1]
    # m = 0 corresponds to k_z.
    output[idx, 2, 0] = wp.float64(0.0)
    output[idx, 2, 1] = coeff_l1 * k_vec[2]
    # m = +1 corresponds to k_x.
    output[idx, 3, 0] = wp.float64(0.0)
    output[idx, 3, 1] = coeff_l1 * k_vec[0]


def eval_gto_fourier_dipole(
    k_vectors: wp.array,
    k_norm2: wp.array,
    sigma: float,
    inv_cl_l0: float,
    inv_cl_l1: float,
    output: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_eval_gto_fourier_dipole_kernel`.

    All array arguments are float64; there is no polymorphism to dispatch on
    for this kernel (everything in k-space stays in float64).

    Parameters
    ----------
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
    k_norm2 : wp.array, shape (N_k,), dtype wp.float64
    sigma : float
        Density-basis Gaussian width.
    inv_cl_l0, inv_cl_l1 : float
        :math:`1 / C_l(\sigma, \text{mode})` factors.
    output : wp.array, shape (N_k, 4, 2), dtype wp.float64
        Pre-allocated output buffer.
    device : str, optional
        Defaults to ``k_vectors.device``.
    """
    n_k = k_vectors.shape[0]
    if device is None:
        device = str(k_vectors.device)
    wp.launch(
        _eval_gto_fourier_dipole_kernel,
        dim=n_k,
        inputs=[
            k_vectors,
            k_norm2,
            wp.float64(sigma),
            wp.float64(inv_cl_l0),
            wp.float64(inv_cl_l1),
            output,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched GTO Fourier coefficients (source basis)
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_eval_gto_fourier_dipole_kernel(
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    k_norm2: wp.array2d(dtype=wp.float64),  # (B, K_max)
    sigma: wp.float64,
    inv_cl_l0: wp.float64,
    inv_cl_l1: wp.float64,
    output: wp.array4d(dtype=wp.float64),  # (B, K_max, 4, 2)
):
    r"""Batched :func:`_eval_gto_fourier_dipole_kernel` across B systems.

    ``sigma`` / ``inv_cl_*`` are shared across the batch — same density
    basis for every system. K-vector values vary per system;
    pad k-vectors at ``k_norm2[b, k] == 0`` for ``k >= K_b`` still
    write a non-zero ``l = 0`` entry (same as real ``k = 0``), which is
    harmless because ``per_k_factor`` at those rows is zero.

    Launch Grid
    -----------
    ``dim = (B, K_max)``.


    Parameters
    ----------
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    k_norm2 : wp.array2d, shape (B, K_max), dtype wp.float64
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigma : wp.float64
        Gaussian (GTO) width parameter.
    inv_cl_l0 : wp.float64
        Inverse :math:`l=0` overlap normalization constant.
    inv_cl_l1 : wp.float64
        Inverse :math:`l=1` overlap normalization constant.
    output : wp.array4d, shape (B, K_max, 4, 2), dtype wp.float64
        OUTPUT: GTO Fourier coefficients :math:`\hat\phi_{l,m}^{\sigma}(k)`.
    """
    b, idx = wp.tid()

    k_vec = k_vectors[b, idx]
    k2 = k_norm2[b, idx]

    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    sigma5 = sigma3 * sigma2

    gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)
    common_radial = _FOUR_PI_SQRT_PI_OVER_2 * gauss

    output[b, idx, 0, 0] = inv_cl_l0 * common_radial * sigma3 * Y00_COEFF
    output[b, idx, 0, 1] = wp.float64(0.0)

    coeff_l1 = -inv_cl_l1 * common_radial * sigma5 * Y1_COEFF

    output[b, idx, 1, 0] = wp.float64(0.0)
    output[b, idx, 1, 1] = coeff_l1 * k_vec[1]
    output[b, idx, 2, 0] = wp.float64(0.0)
    output[b, idx, 2, 1] = coeff_l1 * k_vec[2]
    output[b, idx, 3, 0] = wp.float64(0.0)
    output[b, idx, 3, 1] = coeff_l1 * k_vec[0]


def batch_eval_gto_fourier_dipole(
    k_vectors: wp.array,
    k_norm2: wp.array,
    sigma: float,
    inv_cl_l0: float,
    inv_cl_l1: float,
    output: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_eval_gto_fourier_dipole_kernel`.

    Parameters
    ----------
    k_vectors : wp.array, shape (B, K_max), dtype wp.vec3d
    k_norm2 : wp.array, shape (B, K_max), dtype wp.float64
    sigma, inv_cl_l0, inv_cl_l1 : float
        Shared across batch.
    output : wp.array, shape (B, K_max, 4, 2), dtype wp.float64
        Pre-allocated.
    device : str, optional
    """
    batch_size = k_vectors.shape[0]
    k_max = k_vectors.shape[1]
    if device is None:
        device = str(k_vectors.device)
    wp.launch(
        _batch_eval_gto_fourier_dipole_kernel,
        dim=(batch_size, k_max),
        inputs=[
            k_vectors,
            k_norm2,
            wp.float64(sigma),
            wp.float64(inv_cl_l0),
            wp.float64(inv_cl_l1),
            output,
        ],
        device=device,
    )


# =============================================================================
# ρ(k) assembly for l_max = 1
# =============================================================================


@wp.kernel
def _assemble_rho_k_dipole_kernel(
    charges: wp.array(dtype=Any),
    dipoles: wp.array(dtype=Any),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    gto_fourier: wp.array3d(dtype=wp.float64),
    volume: wp.float64,
    rho: wp.array2d(dtype=wp.float64),
):
    r"""Assemble :math:`\rho(\mathbf{k})` from per-atom multipoles at l_max = 1.

    For each k-vector, computes

    .. math::

        \rho(\mathbf{k}) \;=\; \frac{(2\pi)^3}{V} \sum_i \sum_{l, m}
            Q_{l,m}^{\,i} \cdot \hat\phi_{l,m}^{\sigma}(\mathbf{k})
            \cdot e^{-i \mathbf{k} \cdot \mathbf{r}_i},

    with :math:`e^{-i \mathbf{k} \cdot \mathbf{r}_i} = \cos(\mathbf{k}\cdot\mathbf{r}_i)
    - i \sin(\mathbf{k}\cdot\mathbf{r}_i)` supplied by the precomputed ``cosines``
    and ``sines`` tables.

    The per-atom multipole moments are taken in the form this codebase
    natively uses: a scalar ``charge`` and a Cartesian ``dipole`` vector
    ``(μ_x, μ_y, μ_z)``. The ``Y_1^m`` basis is permuted to ``(y, z, x)``
    internally to line up with ``gto_fourier``'s layout.

    Launch Grid
    -----------
    dim = [N_k] — one thread per k-vector; each thread sweeps all atoms.

    Parameters
    ----------
    charges : wp.array, shape (N_atoms,), dtype wp.float32 or wp.float64
    dipoles : wp.array, shape (N_atoms,), dtype wp.vec3f or wp.vec3d
        Cartesian ``(μ_x, μ_y, μ_z)``.
    cosines, sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Pre-computed :math:`(\cos(\mathbf{k}\cdot\mathbf{r}),\ \sin(\mathbf{k}\cdot\mathbf{r}))`
        from :func:`build_structure_factor_table`.
    gto_fourier : wp.array3d, shape (N_k, 4, 2), dtype wp.float64
        Output of :func:`eval_gto_fourier_dipole` — last two dims are ``(lm, real/imag)``.
    volume : wp.float64
        Periodic-cell volume. Scalar for single-system; a future batched variant
        will take a per-system array.
    rho : wp.array2d, shape (N_k, 2), dtype wp.float64
        OUTPUT. ``[..., 0]`` real, ``[..., 1]`` imag. Does not need pre-zeroing;
        the kernel writes every entry unconditionally.

    Notes
    -----
    Intermediate accumulation is ``float64`` regardless of ``charges`` /
    ``dipoles`` dtype, matching the numerical-stability choice of the
    existing monopole Ewald reciprocal kernel.
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]

    # Per-k accumulators for the four (l, m) components.
    #   lm = 0:  (l=0, m= 0)  → charge
    #   lm = 1:  (l=1, m=-1)  → μ_y
    #   lm = 2:  (l=1, m= 0)  → μ_z
    #   lm = 3:  (l=1, m=+1)  → μ_x
    c0 = wp.float64(0.0)
    c1 = wp.float64(0.0)
    c2 = wp.float64(0.0)
    c3 = wp.float64(0.0)
    s0 = wp.float64(0.0)
    s1 = wp.float64(0.0)
    s2 = wp.float64(0.0)
    s3 = wp.float64(0.0)

    for i in range(n_atoms):
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        q = wp.float64(charges[i])
        mu = dipoles[i]
        mu_x = wp.float64(mu[0])
        mu_y = wp.float64(mu[1])
        mu_z = wp.float64(mu[2])

        c0 += cos_ki * q
        s0 += sin_ki * q
        c1 += cos_ki * mu_y
        s1 += sin_ki * mu_y
        c2 += cos_ki * mu_z
        s2 += sin_ki * mu_z
        c3 += cos_ki * mu_x
        s3 += sin_ki * mu_x

    # Contract (coeff_cos, coeff_sin) with the Fourier basis (φ_r, φ_i):
    #     rho_real = Σ_lm φ_r · coeff_cos + Σ_lm φ_i · coeff_sin
    #     rho_imag = Σ_lm φ_i · coeff_cos - Σ_lm φ_r · coeff_sin
    pr0 = gto_fourier[k_idx, 0, 0]
    pi0 = gto_fourier[k_idx, 0, 1]
    pr1 = gto_fourier[k_idx, 1, 0]
    pi1 = gto_fourier[k_idx, 1, 1]
    pr2 = gto_fourier[k_idx, 2, 0]
    pi2 = gto_fourier[k_idx, 2, 1]
    pr3 = gto_fourier[k_idx, 3, 0]
    pi3 = gto_fourier[k_idx, 3, 1]

    rho_r = (
        pr0 * c0
        + pi0 * s0
        + pr1 * c1
        + pi1 * s1
        + pr2 * c2
        + pi2 * s2
        + pr3 * c3
        + pi3 * s3
    )
    rho_i = (
        pi0 * c0
        - pr0 * s0
        + pi1 * c1
        - pr1 * s1
        + pi2 * c2
        - pr2 * s2
        + pi3 * c3
        - pr3 * s3
    )

    scale = _TWO_PI_CUBED / volume
    rho[k_idx, 0] = rho_r * scale
    rho[k_idx, 1] = rho_i * scale


def _assemble_rho_sig(v, t):
    """Signature builder: charges takes scalar ``t``, dipoles takes vector ``v``."""
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles (Cartesian)
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array3d(dtype=wp.float64),  # gto_fourier
        wp.float64,  # volume
        wp.array2d(dtype=wp.float64),  # rho
    ]


_assemble_rho_k_dipole_overloads = register_overloads(
    _assemble_rho_k_dipole_kernel, _assemble_rho_sig
)


def assemble_rho_k_dipole(
    charges: wp.array,
    dipoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    gto_fourier: wp.array,
    volume: float,
    rho: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_assemble_rho_k_dipole_kernel`.

    Parameters
    ----------
    charges : wp.array, shape (N_atoms,), dtype wp.float32 or wp.float64
    dipoles : wp.array, shape (N_atoms,), dtype wp.vec3f or wp.vec3d
        Cartesian dipole moments.
    cosines, sines : wp.array, shape (N_k, N_atoms), dtype wp.float64
        From :func:`build_structure_factor_table`.
    gto_fourier : wp.array, shape (N_k, 4, 2), dtype wp.float64
        From :func:`eval_gto_fourier_dipole`.
    volume : float
        Periodic-cell volume.
    rho : wp.array, shape (N_k, 2), dtype wp.float64
        Pre-allocated output.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64``; selects the kernel overload.
    device : str, optional
        Defaults to ``cosines.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f

    n_k = cosines.shape[0]
    if device is None:
        device = str(cosines.device)

    wp.launch(
        _assemble_rho_k_dipole_overloads[vec_dtype],
        dim=n_k,
        inputs=[
            charges,
            dipoles,
            cosines,
            sines,
            gto_fourier,
            wp.float64(volume),
            rho,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched ρ(k) assembly
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_assemble_rho_k_dipole_kernel(
    charges: wp.array(dtype=Any),  # (N_total,)
    dipoles: wp.array(dtype=Any),  # (N_total,)
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    gto_fourier: wp.array4d(dtype=wp.float64),  # (B, K_max, 4, 2)
    volume: wp.array(dtype=wp.float64),  # (B,)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    rho: wp.array3d(dtype=wp.float64),  # (B, K_max, 2) OUTPUT
):
    r"""Batched ρ(k) assembly — per-(system, k-vector) thread with atom inner loop.

    Each thread sums over ``[atom_start[b], atom_end[b])`` — the atoms
    belonging to system ``b``. Pad k-vectors (``k ≥ K_b`` for a given
    system) have ``gto_fourier[b, k, :, :] = 0`` in the batched cache,
    so their contributions cancel naturally regardless of what ``c_lm``,
    ``s_lm`` accumulate.

    Launch Grid
    -----------
    ``dim = (B, K_max)``.


    Parameters
    ----------
    charges : wp.array, shape (N_total,), dtype Any
        Per-atom monopole charges.
    dipoles : wp.array, shape (N_total,), dtype Any
        Per-atom Cartesian dipole moments.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    gto_fourier : wp.array4d, shape (B, K_max, 4, 2), dtype wp.float64
        Precomputed GTO Fourier coefficients :math:`\hat\phi_{l,m}^{\sigma}(k)`.
    volume : wp.array, shape (B,), dtype wp.float64
        Unit-cell volume.
    atom_start : wp.array, shape (B,), dtype wp.int32
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array, shape (B,), dtype wp.int32
        Per-system end offset into the flat atom arrays.
    rho : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        OUTPUT: complex reciprocal-space density :math:`\hat\rho(k)` (re, im).
    """
    b, k_idx = wp.tid()

    c0 = wp.float64(0.0)
    c1 = wp.float64(0.0)
    c2 = wp.float64(0.0)
    c3 = wp.float64(0.0)
    s0 = wp.float64(0.0)
    s1 = wp.float64(0.0)
    s2 = wp.float64(0.0)
    s3 = wp.float64(0.0)

    i_lo = atom_start[b]
    i_hi = atom_end[b]
    for i in range(i_lo, i_hi):
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        q = wp.float64(charges[i])
        mu = dipoles[i]
        mu_x = wp.float64(mu[0])
        mu_y = wp.float64(mu[1])
        mu_z = wp.float64(mu[2])

        c0 += cos_ki * q
        s0 += sin_ki * q
        c1 += cos_ki * mu_y
        s1 += sin_ki * mu_y
        c2 += cos_ki * mu_z
        s2 += sin_ki * mu_z
        c3 += cos_ki * mu_x
        s3 += sin_ki * mu_x

    pr0 = gto_fourier[b, k_idx, 0, 0]
    pi0 = gto_fourier[b, k_idx, 0, 1]
    pr1 = gto_fourier[b, k_idx, 1, 0]
    pi1 = gto_fourier[b, k_idx, 1, 1]
    pr2 = gto_fourier[b, k_idx, 2, 0]
    pi2 = gto_fourier[b, k_idx, 2, 1]
    pr3 = gto_fourier[b, k_idx, 3, 0]
    pi3 = gto_fourier[b, k_idx, 3, 1]

    rho_r = (
        pr0 * c0
        + pi0 * s0
        + pr1 * c1
        + pi1 * s1
        + pr2 * c2
        + pi2 * s2
        + pr3 * c3
        + pi3 * s3
    )
    rho_i = (
        pi0 * c0
        - pr0 * s0
        + pi1 * c1
        - pr1 * s1
        + pi2 * c2
        - pr2 * s2
        + pi3 * c3
        - pr3 * s3
    )

    scale = _TWO_PI_CUBED / volume[b]
    rho[b, k_idx, 0] = rho_r * scale
    rho[b, k_idx, 1] = rho_i * scale


def _batch_assemble_rho_sig(v, t):
    """Signature builder for batched ρ assembly."""
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array4d(dtype=wp.float64),  # gto_fourier
        wp.array(dtype=wp.float64),  # volume
        wp.array(dtype=wp.int32),  # atom_start
        wp.array(dtype=wp.int32),  # atom_end
        wp.array3d(dtype=wp.float64),  # rho
    ]


_batch_assemble_rho_k_dipole_overloads = register_overloads(
    _batch_assemble_rho_k_dipole_kernel, _batch_assemble_rho_sig
)


def batch_assemble_rho_k_dipole(
    charges: wp.array,
    dipoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    gto_fourier: wp.array,
    volume: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    rho: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_assemble_rho_k_dipole_kernel`.

    Parameters
    ----------
    charges : wp.array, shape (N_total,), dtype wp.float32/wp.float64
    dipoles : wp.array, shape (N_total,), dtype wp.vec3f/wp.vec3d
    cosines, sines : wp.array, shape (K_max, N_total), dtype wp.float64
    gto_fourier : wp.array, shape (B, K_max, 4, 2), dtype wp.float64
    volume : wp.array, shape (B,), dtype wp.float64
    atom_start, atom_end : wp.array, shape (B,), dtype wp.int32
    rho : wp.array, shape (B, K_max, 2), dtype wp.float64
        Pre-allocated output.
    wp_dtype : type
    device : str, optional
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f

    batch_size = gto_fourier.shape[0]
    k_max = cosines.shape[0]
    if device is None:
        device = str(cosines.device)

    wp.launch(
        _batch_assemble_rho_k_dipole_overloads[vec_dtype],
        dim=(batch_size, k_max),
        inputs=[
            charges,
            dipoles,
            cosines,
            sines,
            gto_fourier,
            volume,
            atom_start,
            atom_end,
            rho,
        ],
        device=device,
    )


# =============================================================================
# ∂ρ(k)/∂r_i — backward of ρ assembly w.r.t. atomic positions (Phase 8b)
# =============================================================================
#
# Closed-form position gradient used by MultipoleRhoFunction.backward:
#
#   ∂L/∂r_{i, α} = (2π)³ / V · Σ_k k_α · [A(k, i) · cos(k·r_i)
#                                         + B(k, i) · sin(k·r_i)]
#
# where
#   P_r(k, i) = Σ_lm  φ̂_r(k, lm) · q_{i, lm}
#   P_i(k, i) = Σ_lm  φ̂_i(k, lm) · q_{i, lm}
#   A(k, i)   =  grad_ρ_r(k) · P_i(k, i) − grad_ρ_i(k) · P_r(k, i)
#   B(k, i)   = −grad_ρ_r(k) · P_r(k, i) − grad_ρ_i(k) · P_i(k, i)
#
# and ``q_{i, lm}`` follows the same e3nn lm layout as ``assemble_rho_k_dipole``
# (``lm = 0`` → charge, ``lm = 1, 2, 3`` → ``μ_y, μ_z, μ_x``).


@wp.kernel
def _position_gradient_from_rhok_kernel(
    charges: wp.array(dtype=Any),
    dipoles: wp.array(dtype=Any),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    source_phi_hat: wp.array3d(dtype=wp.float64),
    grad_rho: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    scale: wp.float64,
    grad_positions: wp.array2d(dtype=wp.float64),
):
    r"""Analytical backward of ρ(k) assembly w.r.t. per-atom positions.

    Each thread owns one atom ``i`` and accumulates the contribution of
    every k-vector to ``∂L/∂r_i``. The kernel layout mirrors
    :func:`_assemble_rho_k_dipole_kernel` (same ``(cosines, sines,
    source_phi_hat)`` input shape and ``(charges, dipoles)`` lm ordering)
    so the two can coexist in the same launch graph with shared
    intermediate state.

    Launch Grid
    -----------
    dim = [N_atoms] — one thread per atom; output has shape
    ``(N_atoms, 3)``.

    Parameters
    ----------
    charges, dipoles
        Per-atom moments, same dtype / shape convention as the forward
        (scalar charges; ``wp.vec3`` Cartesian dipoles). Selects the
        kernel overload.
    cosines, sines : wp.array2d(dtype=wp.float64), shape (N_k, N_atoms)
        Structure-factor tables. Must be the ones computed from the
        same ``positions`` we're taking the gradient w.r.t. (i.e., the
        autograd.Function's forward must have written them in this
        invocation, not reused a stale cache).
    source_phi_hat : wp.array3d(dtype=wp.float64), shape (N_k, 4, 2)
        Source-basis GTO Fourier coefficients (identical to what
        ``assemble_rho_k_dipole`` consumes).
    grad_rho : wp.array2d(dtype=wp.float64), shape (N_k, 2)
        Cotangent ``∂L/∂ρ(k)`` with ``[:, 0]`` / ``[:, 1]`` = (real,
        imag).
    k_vectors : wp.array(dtype=wp.vec3d), shape (N_k,)
        Reciprocal-lattice k-vectors (float64).
    scale : wp.float64
        Forward's prefactor ``(2π)³ / V``. Passed in from the host so
        the kernel doesn't have to know the volume.
    grad_positions : wp.array2d(dtype=wp.float64), shape (N_atoms, 3)
        OUTPUT (Cartesian). Must be zero-initialized by the caller, or
        first-write by design (we overwrite each entry unconditionally
        below).

    Notes
    -----
    The per-k accumulator collects ``(A · cos + B · sin)`` in a single
    float64 scalar (no ``k_α`` factor yet), then multiplies by the three
    Cartesian components of ``k_vectors[k_idx]``. This keeps the
    k-loop's working set small (one float per k) and lets the kernel
    fuse the three-component output write.
    """
    i_idx = wp.tid()

    n_k = cosines.shape[0]

    q = wp.float64(charges[i_idx])
    mu = dipoles[i_idx]
    mu_x = wp.float64(mu[0])
    mu_y = wp.float64(mu[1])
    mu_z = wp.float64(mu[2])

    # Running (x, y, z) accumulators.
    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(n_k):
        k_vec = k_vectors[k_idx]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        gr = grad_rho[k_idx, 0]
        gi = grad_rho[k_idx, 1]

        pr0 = source_phi_hat[k_idx, 0, 0]
        pi0 = source_phi_hat[k_idx, 0, 1]
        pr1 = source_phi_hat[k_idx, 1, 0]
        pi1 = source_phi_hat[k_idx, 1, 1]
        pr2 = source_phi_hat[k_idx, 2, 0]
        pi2 = source_phi_hat[k_idx, 2, 1]
        pr3 = source_phi_hat[k_idx, 3, 0]
        pi3 = source_phi_hat[k_idx, 3, 1]

        # P_r = Σ_lm φ̂_r(k, lm) · q_{i, lm}
        # P_i = Σ_lm φ̂_i(k, lm) · q_{i, lm}
        # lm = 0 → q; lm = 1 → μ_y; lm = 2 → μ_z; lm = 3 → μ_x.
        p_r = pr0 * q + pr1 * mu_y + pr2 * mu_z + pr3 * mu_x
        p_i = pi0 * q + pi1 * mu_y + pi2 * mu_z + pi3 * mu_x

        # A = gr · P_i − gi · P_r
        # B = −(gr · P_r + gi · P_i)
        a_k = gr * p_i - gi * p_r
        b_k = -(gr * p_r + gi * p_i)

        contrib = a_k * cos_ki + b_k * sin_ki
        gx += k_vec[0] * contrib
        gy += k_vec[1] * contrib
        gz += k_vec[2] * contrib

    grad_positions[i_idx, 0] = scale * gx
    grad_positions[i_idx, 1] = scale * gy
    grad_positions[i_idx, 2] = scale * gz


def _position_gradient_sig(v, t):
    """Signature builder: charges takes scalar ``t``, dipoles takes vector ``v``."""
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles (Cartesian)
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array3d(dtype=wp.float64),  # source_phi_hat
        wp.array2d(dtype=wp.float64),  # grad_rho
        wp.array(dtype=wp.vec3d),  # k_vectors
        wp.float64,  # scale
        wp.array2d(dtype=wp.float64),  # grad_positions
    ]


_position_gradient_from_rhok_overloads = register_overloads(
    _position_gradient_from_rhok_kernel, _position_gradient_sig
)


# -----------------------------------------------------------------------------
# Tile-based GPU implementation of position_gradient_from_rhok
# -----------------------------------------------------------------------------
#
# Derivation (same as the per-atom kernel, rearranged):
#
# .. math::
#
#     \partial L / \partial r_{i, \alpha} = \mathrm{scale} \sum_k k_\alpha
#         \bigl[ A(k, i) \cos(k r_i) + B(k, i) \sin(k r_i) \bigr]
#
#     A(k, i) = g^\rho_r \, P_i(k, i) - g^\rho_i \, P_r(k, i)
#     B(k, i) = -(g^\rho_r P_r + g^\rho_i P_i)
#
#     P_{r/i}(k, i) = \sum_{lm} \hat\phi_{r/i}(k, lm) \, q(i, lm)
#
# Substituting the ``P`` expansion and factoring the per-atom moment
# ``q(i, lm)`` out of the k-sum:
#
# .. math::
#
#     \partial L / \partial r_{i, \alpha}
#         = \mathrm{scale} \sum_{lm} q(i, lm) \sum_k k_\alpha \bigl[
#             c(k, lm) \cos(k r_i) + d(k, lm) \sin(k r_i) \bigr]
#
#     c(k, lm) = g^\rho_r \hat\phi_i(k, lm) - g^\rho_i \hat\phi_r(k, lm)
#     d(k, lm) = -(g^\rho_r \hat\phi_r + g^\rho_i \hat\phi_i)
#
# Introduce ``big_cos(k, α*4 + lm) := k_α · c(k, lm)`` and
# ``big_sin(k, α*4 + lm) := k_α · d(k, lm)``. Then the k-sum collapses
# to two dense matmuls:
#
# .. math::
#
#     \mathrm{contribs}(i, \alpha*4+lm)
#         = \sum_k \cos(k, i) \, \mathrm{big\_cos}(k, \alpha*4+lm)
#         + \sum_k \sin(k, i) \, \mathrm{big\_sin}(k, \alpha*4+lm)
#
#     = \cos^\top @ \mathrm{big\_cos} + \sin^\top @ \mathrm{big\_sin}
#
# followed by a tiny ``(N_atoms, 12)`` → ``(N_atoms, 3)`` reduction
# that contracts along ``lm`` against the per-atom moments.
#
# Padding: we allocate ``big_cos`` / ``big_sin`` with ``N_k`` rounded up
# to a multiple of ``TILE_K`` and zero the pad. The output column
# dimension is padded from 12 to 16 (``TILE_J``) for the same reason,
# and we slice the real 12 columns out for the final reduction.

# -----------------------------------------------------------------------------
# Shared tile-matmul primitive
# -----------------------------------------------------------------------------

_TILE_I = wp.constant(8)  # atoms per block (tuned on GB10)
_TILE_K = wp.constant(32)  # k-vectors per inner iteration
_TILE_J_LARGE = wp.constant(16)  # output cols per block
_TILE_BLOCK_DIM = 128


@wp.kernel
def _cossin_native_matmul_kernel(
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms) — NATIVE layout
    sines: wp.array2d(dtype=wp.float64),
    m_cos: wp.array2d(dtype=wp.float64),  # (N_k, N_cols)
    m_sin: wp.array2d(dtype=wp.float64),
    contribs: wp.array2d(dtype=wp.float64),  # (N_atoms, N_cols) OUTPUT
):
    r"""``contribs = cosᵀ @ m_cos + sinᵀ @ m_sin`` reading cos/sin in their
    NATIVE ``(N_k, N_atoms)`` layout — the transpose is an in-kernel
    ``wp.tile_transpose`` per tile, so no ``(N_atoms, N_k)`` transpose+pad copy
    is materialized at the wrapper. ``wp.tile_load`` / ``wp.tile_store``
    bounds-check, so ``N_k`` / ``N_atoms`` / ``N_cols`` need not be tile
    multiples (launch with a ``ceil`` grid; no padding).

    Launch: ``wp.launch_tiled(dim=(ceil(N_atoms/_TILE_I), ceil(N_cols/_TILE_J_LARGE)),
    block_dim=_TILE_BLOCK_DIM)``.
    """
    i_block, j_block = wp.tid()
    n_k = cosines.shape[0]
    acc = wp.tile_zeros(shape=(_TILE_I, _TILE_J_LARGE), dtype=wp.float64)
    i_off = i_block * _TILE_I
    j_off = j_block * _TILE_J_LARGE
    for kk in range(0, n_k, _TILE_K):
        c = wp.tile_load(cosines, shape=(_TILE_K, _TILE_I), offset=(kk, i_off))
        s = wp.tile_load(sines, shape=(_TILE_K, _TILE_I), offset=(kk, i_off))
        mc = wp.tile_load(m_cos, shape=(_TILE_K, _TILE_J_LARGE), offset=(kk, j_off))
        ms = wp.tile_load(m_sin, shape=(_TILE_K, _TILE_J_LARGE), offset=(kk, j_off))
        wp.tile_matmul(wp.tile_transpose(c), mc, acc)
        wp.tile_matmul(wp.tile_transpose(s), ms, acc)
    wp.tile_store(contribs, acc, offset=(i_off, j_off))


def _launch_cossin_native_matmul(cosines, sines, m_cos, m_sin, contribs, device):
    r"""Launch :func:`_cossin_native_matmul_kernel` over a ``ceil`` grid.

    ``contribs[i, j] = Σ_k cos[k, i]·m_cos[k, j] + sin[k, i]·m_sin[k, j]``,
    with cos/sin in native ``(N_k, N_atoms)`` and ``m_cos`` / ``m_sin`` in
    ``(N_k, N_cols)``. No transpose+pad copy; bounds-checked tiles.
    """
    n_atoms = contribs.shape[0]
    n_cols = contribs.shape[1]
    n_i = (n_atoms + int(_TILE_I) - 1) // int(_TILE_I)
    n_j = (n_cols + int(_TILE_J_LARGE) - 1) // int(_TILE_J_LARGE)
    wp.launch_tiled(
        _cossin_native_matmul_kernel,
        dim=(n_i, n_j),
        inputs=[
            cosines,
            sines,
            m_cos,
            m_sin,
            wp.from_torch(contribs, dtype=wp.float64),
        ],
        block_dim=int(_TILE_BLOCK_DIM),
        device=device,
    )


# =============================================================================
# Int32 byte-offset overflow workaround for Warp tile loads
# =============================================================================
#
# Warp 1.12.1's ``tile_global_t::index`` (native/tile.h) accumulates the
# byte offset of a tile element in a signed ``int32``:
#
#     int index = 0;
#     for (i...) index += data.strides[i] * c;   // strides in bytes
#     out = index / sizeof(T);
#
# For a 2D float64 array of shape ``(M, N)`` with row stride ``N * 8``
# bytes, the kernel's last-row tile-load computes
# ``index = (M-1) * N * 8 + ...``. When that exceeds ``2^31 - 1``
# (~2.1 GB) the int32 silently wraps negative, ``data.data[neg_index]``
# reads OOB, and the launch raises ``cudaErrorIllegalAddress``. The
# corrupted CUDA context then breaks every subsequent launch (kernel
# symbol lookup itself fails), so the user-visible error is reported
# against the *next* kernel — confusing but characteristic.
#
# Repro: in the multipole Ewald backward at N≈4400 atoms, BCC layout,
# k_cutoff=3.0, the tile-matmul over the (cos, sin) structure-factor
# table ((4400, 74464) float64) crosses the 2 GB threshold; the next
# launch (``_project_features_postprocess_kernel``) is what surfaces the
# error.
#
# Workaround: split each affected ``wp.launch_tiled`` along the long
# (atom-count or k-count) axis into row-slice sub-launches. A row-slice
# of a contiguous ``(M, N)`` tensor preserves the ``N * 8`` row stride
# but shrinks the visible row count, so the kernel's coordinate range
# becomes ``[0, chunk_rows)`` and the per-launch byte product stays
# under int32. Output tiles in ``contribs`` / ``grc`` / ``grs`` are
# disjoint across chunks (different output rows), so the multi-launch
# result is bit-identical to a single (theoretical) safe launch.

_INT32_BYTE_OFFSET_LIMIT = (1 << 31) - 1


def _safe_chunk_rows(row_stride_bytes: int, axis_tile: int) -> int:
    r"""Largest multiple of ``axis_tile`` such that ``(rows-1)*row_stride_bytes < 2**31-1``.

    Returns at least one tile (``axis_tile``) so the caller always makes
    forward progress even when the row stride is pathological.
    """
    if row_stride_bytes <= 0:
        return axis_tile
    max_rows = _INT32_BYTE_OFFSET_LIMIT // row_stride_bytes
    chunked = (max_rows // axis_tile) * axis_tile
    return max(chunked, axis_tile)


def _launch_v_grad_tile_matmul_chunked(
    kernel,
    *,
    cosines,  # torch.Tensor, (n_k, n_atoms) contiguous float64 — native layout
    sines,  # torch.Tensor, (n_k, n_atoms)
    grad_raw,  # torch.Tensor, (n_atoms, n_sl)
    grc,  # torch.Tensor, (n_k, n_sl) OUTPUT
    grs,  # torch.Tensor, (n_k, n_sl) OUTPUT
    tile_m: int,
    tile_n: int,
    block_dim: int,
    device: str,
) -> None:
    r"""Chunked launch for ``_v_grad_tiled_matmul_kernel`` (no padding).

    The cos/sin v-grad layout is ``(n_k, n_atoms)`` — k-major — so the
    int32-overflow axis is the *k-axis*. This helper chunks along m
    (k-blocks) and slices the GRC/GRS outputs in lockstep. ``wp.tile_load``
    / ``wp.tile_store`` bounds-check, so the matmul runs on the unpadded
    arrays with a ``ceil`` grid (no transpose+pad copy at the wrapper).
    """
    n_k, n_atoms = cosines.shape
    n_sl = grc.shape[1]
    n_n_blocks = (n_sl + tile_n - 1) // tile_n

    row_stride_bytes = n_atoms * 8
    chunk_rows = _safe_chunk_rows(row_stride_bytes, tile_m)

    start = 0
    while start < n_k:
        end = min(start + chunk_rows, n_k)
        n_m_blocks = (end - start + tile_m - 1) // tile_m
        cos_view = cosines[start:end]
        sin_view = sines[start:end]
        grc_view = grc[start:end]
        grs_view = grs[start:end]
        wp.launch_tiled(
            kernel,
            dim=(n_m_blocks, n_n_blocks),
            inputs=[
                wp.from_torch(cos_view, dtype=wp.float64),
                wp.from_torch(sin_view, dtype=wp.float64),
                wp.from_torch(grad_raw, dtype=wp.float64),
                wp.from_torch(grc_view, dtype=wp.float64),
                wp.from_torch(grs_view, dtype=wp.float64),
            ],
            block_dim=block_dim,
            device=device,
        )
        start = end


# Legacy constant aliases — kept so the individual physics sections
# (precompute, reduce, launcher padding math) can reference them
# without a sweeping rename. New callers should prefer ``_TILE_*``.
_RHOK_PG_TILE_I = _TILE_I
_RHOK_PG_TILE_K = _TILE_K
_RHOK_PG_TILE_J = _TILE_J_LARGE
_RHOK_PG_N_COLS = wp.constant(12)  # 3 Cartesian α × 4 e3nn lm
_RHOK_PG_BLOCK_DIM = _TILE_BLOCK_DIM


@wp.kernel
def _rhok_pos_grad_precompute_big_matrices_kernel(
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    source_phi_hat: wp.array3d(dtype=wp.float64),  # (N_k, 4, 2)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    n_k_valid: wp.int32,
    big_cos: wp.array2d(dtype=wp.float64),  # (N_k_pad, 16)
    big_sin: wp.array2d(dtype=wp.float64),  # (N_k_pad, 16)
):
    r"""Elementwise per-k precompute of ``big_cos`` / ``big_sin``.

    Launch dim: ``N_k_pad``. Each thread:

    * Reads ``grad_rho[k, :]``, ``source_phi_hat[k, lm, :]``, ``k_vec``.
    * Writes the 12 real columns (``α * 4 + lm``) per ``big_cos`` /
      ``big_sin`` row and zeros columns ``[12, 16)``.
    * Threads with ``k >= n_k_valid`` (the pad region) write all zeros.


    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx)``; each thread processes one k-vector index.

    Parameters
    ----------
    grad_rho : wp.array2d, shape (N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    source_phi_hat : wp.array3d, shape (N_k, 4, 2), dtype wp.float64
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-space k-vectors.
    n_k_valid : wp.int32
        Number of valid (non-padded) k-vectors.
    big_cos : wp.array2d, shape (N_k_pad, 16), dtype wp.float64
        OUTPUT: large cosine intermediate matrix for the tiled matmul.
    big_sin : wp.array2d, shape (N_k_pad, 16), dtype wp.float64
        OUTPUT: large sine intermediate matrix for the tiled matmul.
    """
    k_idx = wp.tid()

    if k_idx >= n_k_valid:
        # Pad row: clear all 16 columns.
        for j in range(16):
            big_cos[k_idx, j] = wp.float64(0.0)
            big_sin[k_idx, j] = wp.float64(0.0)
        return

    gr = grad_rho[k_idx, 0]
    gi = grad_rho[k_idx, 1]
    k_vec = k_vectors[k_idx]
    kx = k_vec[0]
    ky = k_vec[1]
    kz = k_vec[2]

    for lm in range(4):
        phi_r = source_phi_hat[k_idx, lm, 0]
        phi_i = source_phi_hat[k_idx, lm, 1]
        c_lm = gr * phi_i - gi * phi_r
        d_lm = -(gr * phi_r + gi * phi_i)

        big_cos[k_idx, lm] = kx * c_lm  # α=0
        big_cos[k_idx, 4 + lm] = ky * c_lm  # α=1
        big_cos[k_idx, 8 + lm] = kz * c_lm  # α=2

        big_sin[k_idx, lm] = kx * d_lm
        big_sin[k_idx, 4 + lm] = ky * d_lm
        big_sin[k_idx, 8 + lm] = kz * d_lm

    # Zero the 4 pad columns.
    for j in range(12, 16):
        big_cos[k_idx, j] = wp.float64(0.0)
        big_sin[k_idx, j] = wp.float64(0.0)


# _rhok_pos_grad_tiled_matmul_kernel → replaced by
# :func:`_cossin_native_matmul_kernel` (shared, native layout).
# Private matmul kernel removed as part of the Phase-8 tile-kernel
# consolidation; launcher below points directly at the shared one.


@wp.kernel
def _rhok_pos_grad_tiled_reduce_kernel(
    charges: wp.array(dtype=Any),
    dipoles: wp.array(dtype=Any),
    contribs: wp.array2d(dtype=wp.float64),  # (N_atoms_pad, 16)
    scale: wp.float64,
    n_atoms_valid: wp.int32,
    grad_positions: wp.array2d(dtype=wp.float64),  # (N_atoms, 3) OUTPUT
):
    r"""Final per-atom ``(N_atoms, 12) → (N_atoms, 3)`` reduction.

    ``grad_positions[i, α] = scale · Σ_lm q(i, lm) · contribs(i, α*4+lm)``.
    ``q`` is packed in e3nn layout: ``[q, μ_y, μ_z, μ_x]``.


    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    charges : wp.array, dtype Any
        Per-atom monopole charges.
    dipoles : wp.array, dtype Any
        Per-atom Cartesian dipole moments.
    contribs : wp.array2d, shape (N_atoms_pad, 16), dtype wp.float64
        OUTPUT: per-tile partial contributions awaiting reduction.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    n_atoms_valid : wp.int32
        Number of valid (non-padded) atoms.
    grad_positions : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        OUTPUT: gradient w.r.t. atomic positions.
    """
    i_idx = wp.tid()
    if i_idx >= n_atoms_valid:
        return

    q = wp.float64(charges[i_idx])
    mu = dipoles[i_idx]
    mu_x = wp.float64(mu[0])
    mu_y = wp.float64(mu[1])
    mu_z = wp.float64(mu[2])

    # e3nn lm layout: 0=q, 1=μ_y, 2=μ_z, 3=μ_x.
    for alpha in range(3):
        base = alpha * 4
        total = (
            q * contribs[i_idx, base + 0]
            + mu_y * contribs[i_idx, base + 1]
            + mu_z * contribs[i_idx, base + 2]
            + mu_x * contribs[i_idx, base + 3]
        )
        grad_positions[i_idx, alpha] = scale * total


def _rhok_pg_reduce_sig(v, t):
    """Signature builder for the reduce kernel: charges scalar t, dipoles vec v."""
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array2d(dtype=wp.float64),  # contribs
        wp.float64,  # scale
        wp.int32,  # n_atoms_valid
        wp.array2d(dtype=wp.float64),  # grad_positions
    ]


_rhok_pos_grad_tiled_reduce_overloads = register_overloads(
    _rhok_pos_grad_tiled_reduce_kernel, _rhok_pg_reduce_sig
)


def _position_gradient_from_rhok_tiled_launch(
    charges: wp.array,
    dipoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    grad_rho: wp.array,
    k_vectors: wp.array,
    scale: float,
    grad_positions: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    r"""Tiled-matmul implementation orchestrator — GPU path only.

    Runs three kernels:
    1. :func:`_rhok_pos_grad_precompute_big_matrices_kernel` → ``big_cos`` / ``big_sin``.
    2. :func:`_rhok_pos_grad_tiled_matmul_kernel` → ``contribs``.
    3. :func:`_rhok_pos_grad_tiled_reduce_kernel` → ``grad_positions``.
    """
    import torch  # local import — CPU path has no torch dependency at launcher time

    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f

    n_k = cosines.shape[0]
    n_atoms = cosines.shape[1]

    # ``big_cos`` / ``big_sin``: per-k 16-column precompute. No pad rows — the
    # contraction below is a cuBLAS GEMM, which takes any K and applies the
    # transpose flag directly to the ``(N_k, N_atoms)`` cos/sin tables, so we
    # avoid both the manual ``(N_atoms_pad, N_k_pad)`` transpose+zero-pad copy
    # (the dominant cost at large N_k) and the int32 tile-offset chunking.
    cos_t = wp.to_torch(cosines)  # (N_k, N_atoms) — for .device only
    big_cos = torch.empty((n_k, 16), dtype=torch.float64, device=cos_t.device)
    big_sin = torch.empty((n_k, 16), dtype=torch.float64, device=cos_t.device)

    # -- Launch 1: precompute (one thread per valid k; no pad rows). ----------
    wp.launch(
        _rhok_pos_grad_precompute_big_matrices_kernel,
        dim=n_k,
        inputs=[
            grad_rho,
            source_phi_hat,
            k_vectors,
            wp.int32(n_k),
            wp.from_torch(big_cos, dtype=wp.float64),
            wp.from_torch(big_sin, dtype=wp.float64),
        ],
        device=device,
    )

    # -- Step 2: contribs = cosᵀ @ big_cos + sinᵀ @ big_sin via the native-layout
    #    Warp tile matmul (in-kernel tile_transpose; no transpose+pad copy, no
    #    torch matmul — keeps the path framework-native).
    contribs = torch.empty((n_atoms, 16), dtype=torch.float64, device=cos_t.device)
    _launch_cossin_native_matmul(
        cosines,
        sines,
        wp.from_torch(big_cos, dtype=wp.float64),
        wp.from_torch(big_sin, dtype=wp.float64),
        contribs,
        device,
    )

    # -- Launch 3: final reduction (one thread per atom; no pad atoms). -------
    wp.launch(
        _rhok_pos_grad_tiled_reduce_overloads[vec_dtype],
        dim=n_atoms,
        inputs=[
            charges,
            dipoles,
            wp.from_torch(contribs, dtype=wp.float64),
            wp.float64(scale),
            wp.int32(n_atoms),
            grad_positions,
        ],
        device=device,
    )


def position_gradient_from_rhok(
    charges: wp.array,
    dipoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    grad_rho: wp.array,
    k_vectors: wp.array,
    scale: float,
    grad_positions: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_position_gradient_from_rhok_kernel`.

    Produces the per-atom position gradient ``∂L/∂r_i`` analytically, in
    one kernel launch, given the forward's ``source_phi_hat`` /
    ``cosines`` / ``sines`` and the upstream cotangent
    ``grad_rho = ∂L/∂ρ(k)``.

    Parameters
    ----------
    charges, dipoles
        Same dtype / shape as the forward. Picks the kernel overload.
    cosines, sines : wp.array, shape (N_k, N_atoms), dtype wp.float64
        Must be from the same forward pass as ``grad_rho`` — the gradient
        is only defined w.r.t. the positions those tables were built
        from.
    source_phi_hat : wp.array, shape (N_k, 4, 2), dtype wp.float64
    grad_rho : wp.array, shape (N_k, 2), dtype wp.float64
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
    scale : float
        Forward prefactor ``(2π)³ / V``.
    grad_positions : wp.array, shape (N_atoms, 3), dtype wp.float64
        Pre-allocated output. The kernel overwrites every entry.
    wp_dtype : type
        ``wp.float32`` or ``wp.float64``; selects the overload.
    device : str, optional
        Defaults to ``cosines.device``.
    """
    if device is None:
        device = str(cosines.device)

    # Dispatch: CUDA gets the tile-matmul implementation (driven by
    # ``wp.tile_matmul`` over the inner k-loop), CPU stays on the
    # serial per-atom kernel (where the loop is pre-vectorized by the
    # C++ codegen already).
    if "cuda" in str(device):
        _position_gradient_from_rhok_tiled_launch(
            charges,
            dipoles,
            cosines,
            sines,
            source_phi_hat,
            grad_rho,
            k_vectors,
            scale,
            grad_positions,
            wp_dtype,
            device,
        )
        return

    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    n_atoms = cosines.shape[1]
    wp.launch(
        _position_gradient_from_rhok_overloads[vec_dtype],
        dim=n_atoms,
        inputs=[
            charges,
            dipoles,
            cosines,
            sines,
            source_phi_hat,
            grad_rho,
            k_vectors,
            wp.float64(scale),
            grad_positions,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched position_gradient_from_rhok (flat, no tile in this first pass)
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_position_gradient_from_rhok_kernel(
    charges: wp.array(dtype=Any),  # (N_total,)
    dipoles: wp.array(dtype=Any),  # (N_total,)
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    source_phi_hat: wp.array4d(dtype=wp.float64),  # (B, K_max, 4, 2)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    scale: wp.array(dtype=wp.float64),  # (B,) per-system (2π)³/V_b
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    grad_positions: wp.array2d(dtype=wp.float64),  # (N_total, 3) OUTPUT
):
    r"""Batched analytical backward of ρ(k) assembly w.r.t. per-atom positions.

    One thread per atom, flat across the batch. ``batch_idx[i]`` picks
    the system; per-k quantities (``k_vectors``, ``source_phi_hat``,
    ``grad_rho``, ``scale``) are looked up with that ``b``. Pad
    k-vectors in the batched cache have ``source_phi_hat = 0`` so
    their contributions cancel without explicit bounds checks.

    Launch Grid
    -----------
    ``dim = N_total``.


    Parameters
    ----------
    charges : wp.array, shape (N_total,), dtype Any
        Per-atom monopole charges.
    dipoles : wp.array, shape (N_total,), dtype Any
        Per-atom Cartesian dipole moments.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array4d, shape (B, K_max, 4, 2), dtype wp.float64
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    scale : wp.array, shape (B,), dtype wp.float64
        Per-system (2π)³/V_b.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    grad_positions : wp.array2d, shape (N_total, 3), dtype wp.float64
        OUTPUT: gradient w.r.t. atomic positions.
    """
    i_idx = wp.tid()
    b = batch_idx[i_idx]
    k_max = k_vectors.shape[1]

    q = wp.float64(charges[i_idx])
    mu = dipoles[i_idx]
    mu_x = wp.float64(mu[0])
    mu_y = wp.float64(mu[1])
    mu_z = wp.float64(mu[2])

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(k_max):
        k_vec = k_vectors[b, k_idx]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        gr = grad_rho[b, k_idx, 0]
        gi = grad_rho[b, k_idx, 1]

        pr0 = source_phi_hat[b, k_idx, 0, 0]
        pi0 = source_phi_hat[b, k_idx, 0, 1]
        pr1 = source_phi_hat[b, k_idx, 1, 0]
        pi1 = source_phi_hat[b, k_idx, 1, 1]
        pr2 = source_phi_hat[b, k_idx, 2, 0]
        pi2 = source_phi_hat[b, k_idx, 2, 1]
        pr3 = source_phi_hat[b, k_idx, 3, 0]
        pi3 = source_phi_hat[b, k_idx, 3, 1]

        p_r = pr0 * q + pr1 * mu_y + pr2 * mu_z + pr3 * mu_x
        p_i = pi0 * q + pi1 * mu_y + pi2 * mu_z + pi3 * mu_x

        a_k = gr * p_i - gi * p_r
        b_k = -(gr * p_r + gi * p_i)

        contrib = a_k * cos_ki + b_k * sin_ki
        gx += k_vec[0] * contrib
        gy += k_vec[1] * contrib
        gz += k_vec[2] * contrib

    scale_b = scale[b]
    grad_positions[i_idx, 0] = scale_b * gx
    grad_positions[i_idx, 1] = scale_b * gy
    grad_positions[i_idx, 2] = scale_b * gz


def _batch_position_gradient_sig(v, t):
    """Signature builder for the batched ρ-position-gradient kernel."""
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array4d(dtype=wp.float64),  # source_phi_hat
        wp.array3d(dtype=wp.float64),  # grad_rho
        wp.array2d(dtype=wp.vec3d),  # k_vectors
        wp.array(dtype=wp.float64),  # scale
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array2d(dtype=wp.float64),  # grad_positions
    ]


_batch_position_gradient_from_rhok_overloads = register_overloads(
    _batch_position_gradient_from_rhok_kernel, _batch_position_gradient_sig
)


def batch_position_gradient_from_rhok(
    charges: wp.array,
    dipoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    grad_rho: wp.array,
    k_vectors: wp.array,
    scale: wp.array,
    batch_idx: wp.array,
    grad_positions: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_position_gradient_from_rhok_kernel`.

    ``scale`` is a ``(B,)`` per-system array of ``(2π)³/V_b`` prefactors.
    ``grad_positions`` is a flat ``(N_total, 3)`` output on the batched
    atom axis.


    Parameters
    ----------
    charges : wp.array
        Per-atom monopole charges.
    dipoles : wp.array
        Per-atom Cartesian dipole moments.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    scale : wp.array
        Scalar prefactor applied to the contribution.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    grad_positions : wp.array
        OUTPUT: gradient w.r.t. atomic positions.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f

    n_total = cosines.shape[1]
    if device is None:
        device = str(cosines.device)

    wp.launch(
        _batch_position_gradient_from_rhok_overloads[vec_dtype],
        dim=n_total,
        inputs=[
            charges,
            dipoles,
            cosines,
            sines,
            source_phi_hat,
            grad_rho,
            k_vectors,
            scale,
            batch_idx,
            grad_positions,
        ],
        device=device,
    )


# =============================================================================
# V(k) = per_k_factor · ρ(k) — generic per-k multiplier kernel
# =============================================================================


@wp.kernel
def _apply_per_k_factor_kernel(
    rho: wp.array2d(dtype=wp.float64),
    per_k_factor: wp.array(dtype=wp.float64),
    potential: wp.array2d(dtype=wp.float64),
):
    r"""Elementwise per-k scalar multiply: :math:`V(\mathbf{k}) = f(\mathbf{k}) \cdot \rho(\mathbf{k})`.

    The caller supplies ``per_k_factor[k]`` fully formed — for the direct
    k-space sum this is ``FIELD_CONSTANT / k²`` with the k=0 entry zeroed; for
    the Ewald reciprocal sum it is ``FIELD_CONSTANT · exp(-k²/4α²) / k²``. The
    kernel is agnostic to the physical interpretation.

    Launch Grid
    -----------
    dim = [N_k] — one thread per k-vector.

    Parameters
    ----------
    rho : wp.array2d(dtype=wp.float64), shape (N_k, 2)
        Density in k-space (real, imag) from :func:`assemble_rho_k_dipole`.
    per_k_factor : wp.array(dtype=wp.float64), shape (N_k,)
        Scalar multiplier per k-vector.
    potential : wp.array2d(dtype=wp.float64), shape (N_k, 2)
        OUTPUT. Does not need pre-zeroing.
    """
    k_idx = wp.tid()
    factor = per_k_factor[k_idx]
    potential[k_idx, 0] = rho[k_idx, 0] * factor
    potential[k_idx, 1] = rho[k_idx, 1] * factor


def apply_per_k_factor(
    rho: wp.array,
    per_k_factor: wp.array,
    potential: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_apply_per_k_factor_kernel`.

    Parameters
    ----------
    rho : wp.array, shape (N_k, 2), dtype wp.float64
    per_k_factor : wp.array, shape (N_k,), dtype wp.float64
        Scalar multiplier per k-vector. For the direct k-space sum,
        ``FIELD_CONSTANT / k²`` with k=0 zeroed.
    potential : wp.array, shape (N_k, 2), dtype wp.float64
        Pre-allocated output.
    device : str, optional
        Defaults to ``rho.device``.
    """
    n_k = rho.shape[0]
    if device is None:
        device = str(rho.device)
    wp.launch(
        _apply_per_k_factor_kernel,
        dim=n_k,
        inputs=[rho, per_k_factor, potential],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched per-k factor multiply
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_apply_per_k_factor_kernel(
    rho: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    per_k_factor: wp.array2d(dtype=wp.float64),  # (B, K_max)
    potential: wp.array3d(dtype=wp.float64),  # (B, K_max, 2) OUTPUT
):
    r"""Batched elementwise: ``V[b, k] = per_k_factor[b, k] · ρ[b, k]``.

    Launch Grid
    -----------
    ``dim`` indexes ``(b, k_idx)``; each thread processes one (system, k-vector) work item.

    Parameters
    ----------
    rho : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        OUTPUT: complex reciprocal-space density :math:`\hat\rho(k)` (re, im).
    per_k_factor : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k multiplicative factor (Green/structure factor).
    potential : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        OUTPUT: Per-k reciprocal-space potential factor.
    """
    b, k_idx = wp.tid()
    factor = per_k_factor[b, k_idx]
    potential[b, k_idx, 0] = rho[b, k_idx, 0] * factor
    potential[b, k_idx, 1] = rho[b, k_idx, 1] * factor


def batch_apply_per_k_factor(
    rho: wp.array,
    per_k_factor: wp.array,
    potential: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_apply_per_k_factor_kernel`.

    Parameters
    ----------
    rho : wp.array
        OUTPUT: complex reciprocal-space density :math:`\hat\rho(k)` (re, im).
    per_k_factor : wp.array
        Per-k multiplicative factor (Green/structure factor).
    potential : wp.array
        Per-k reciprocal-space potential factor.
    device : str
        Warp device string; defaults to the input array's device.
    """
    batch_size = rho.shape[0]
    k_max = rho.shape[1]
    if device is None:
        device = str(rho.device)
    wp.launch(
        _batch_apply_per_k_factor_kernel,
        dim=(batch_size, k_max),
        inputs=[rho, per_k_factor, potential],
        device=device,
    )


# =============================================================================
# Per-k energy product (reduction-free)
# =============================================================================


@wp.kernel
def _energy_product_per_k_kernel(
    rho: wp.array2d(dtype=wp.float64),
    potential: wp.array2d(dtype=wp.float64),
    per_k_energy: wp.array(dtype=wp.float64),
):
    r"""Per-k electrostatic-energy contribution :math:`2\,\text{Re}[\rho^{*}(\mathbf{k}) V(\mathbf{k})]`.

    Writes an ``(N_k,)`` array of per-k contributions; the caller is
    responsible for the final reduction and the ``0.5 \cdot V / (2\pi)^6``
    scaling. Keeping the kernel reduction-free lets downstream torch
    bindings do the sum inside the autograd tape (straightforward
    ``torch.sum`` with gradients flowing back to ``rho`` / ``potential``).

    The factor of 2 comes from the real-field Fourier-series convention:
    enumerating only the "positive-half" k-vectors + origin (as
    ``graph_longrange`` does) requires each non-origin term to be counted
    twice, once for ``k`` and once for the conjugate pair ``-k``. The k=0
    term is fine too because the caller zeroes ``potential(k=0)`` via the
    ``per_k_factor`` array passed to :func:`apply_per_k_factor`.

    Launch Grid
    -----------
    dim = [N_k] — one thread per k-vector.

    Parameters
    ----------
    rho : wp.array2d(dtype=wp.float64), shape (N_k, 2)
        Density in k-space (real, imag) from :func:`assemble_rho_k_dipole`.
    potential : wp.array2d(dtype=wp.float64), shape (N_k, 2)
        Potential in k-space from :func:`apply_per_k_factor`.
    per_k_energy : wp.array(dtype=wp.float64), shape (N_k,)
        OUTPUT. Does not need pre-zeroing.
    """
    k_idx = wp.tid()
    per_k_energy[k_idx] = wp.float64(2.0) * (
        rho[k_idx, 0] * potential[k_idx, 0] + rho[k_idx, 1] * potential[k_idx, 1]
    )


def compute_energy_product_per_k(
    rho: wp.array,
    potential: wp.array,
    per_k_energy: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_energy_product_per_k_kernel`.

    After calling this, the caller obtains the total reciprocal-space
    energy by summing ``per_k_energy`` and multiplying by
    :math:`0.5 \cdot V / (2\pi)^6`:

    .. code-block:: python

        compute_energy_product_per_k(rho, potential, per_k_energy)
        raw_reciprocal_energy = (
            float(wp.utils.array_sum(per_k_energy).numpy())
            * 0.5 * volume / (2.0 * math.pi) ** 6
        )

    Self-interaction subtraction (when the source and receiver basis share
    the same ``sigma``) is a separate, binding-layer step — see
    :func:`nvalchemiops.torch.math.gto_self_overlap.compute_overlap_constants`
    for the scaling coefficients it consumes.

    Parameters
    ----------
    rho, potential : wp.array, shape (N_k, 2), dtype wp.float64
    per_k_energy : wp.array, shape (N_k,), dtype wp.float64
        Pre-allocated output.
    device : str, optional
        Defaults to ``rho.device``.
    """
    n_k = rho.shape[0]
    if device is None:
        device = str(rho.device)
    wp.launch(
        _energy_product_per_k_kernel,
        dim=n_k,
        inputs=[rho, potential, per_k_energy],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched per-k energy product
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_energy_product_per_k_kernel(
    rho: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    potential: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    per_k_energy: wp.array2d(dtype=wp.float64),  # (B, K_max) OUTPUT
):
    r"""Batched per-k energy contribution :math:`2\,\mathrm{Re}[\rho^* V]`.

    Identical math to the single-system kernel, just with an extra
    batch dim.


    Launch Grid
    -----------
    ``dim`` indexes ``(b, k_idx)``; each thread processes one (system, k-vector) work item.

    Parameters
    ----------
    rho : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        OUTPUT: complex reciprocal-space density :math:`\hat\rho(k)` (re, im).
    potential : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Per-k reciprocal-space potential factor.
    per_k_energy : wp.array2d, shape (B, K_max), dtype wp.float64
        OUTPUT: per-k energy contribution.
    """
    b, k_idx = wp.tid()
    per_k_energy[b, k_idx] = wp.float64(2.0) * (
        rho[b, k_idx, 0] * potential[b, k_idx, 0]
        + rho[b, k_idx, 1] * potential[b, k_idx, 1]
    )


def batch_compute_energy_product_per_k(
    rho: wp.array,
    potential: wp.array,
    per_k_energy: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_energy_product_per_k_kernel`.

    The caller reduces ``per_k_energy`` over the K axis per-system and
    applies the scalar ``0.5 · V_b / (2π)⁶`` factor to get per-system
    total reciprocal energies.


    Parameters
    ----------
    rho : wp.array
        OUTPUT: complex reciprocal-space density :math:`\hat\rho(k)` (re, im).
    potential : wp.array
        Per-k reciprocal-space potential factor.
    per_k_energy : wp.array
        OUTPUT: per-k energy contribution.
    device : str
        Warp device string; defaults to the input array's device.
    """
    batch_size = rho.shape[0]
    k_max = rho.shape[1]
    if device is None:
        device = str(rho.device)
    wp.launch(
        _batch_energy_product_per_k_kernel,
        dim=(batch_size, k_max),
        inputs=[rho, potential, per_k_energy],
        device=device,
    )


# =============================================================================
# Receiver-basis GTO Fourier coefficients (Phase 4 — features / ACE)
# =============================================================================
#
# Phase 4 adds atom-centered (ACE / LODE) feature projection on top of Phase 3's
# V(k). The two new Warp kernels are designed with fusion in mind so the
# eventual `@warp_custom_op` PyTorch wrapper becomes a short sequence of
# custom-op calls (one per fused kernel) rather than the per-Warp-launch
# orchestration Phase 3's binding uses today. See
# ``feedback_torch_wrapper_fusion.md`` for the design rule.


@wp.kernel
def _eval_receiver_gto_fourier_dipole_kernel(
    k_vectors: wp.array(dtype=wp.vec3d),
    k_norm2: wp.array(dtype=wp.float64),
    sigmas: wp.array(dtype=wp.float64),
    inv_cl_table: wp.array2d(dtype=wp.float64),
    output: wp.array4d(dtype=wp.float64),
):
    r"""Evaluate receiver-basis :math:`\hat\phi_{l,m}^{\sigma_r}(\mathbf{k})` across multi-σ at l_max = 1.

    Same radial form as :func:`_eval_gto_fourier_dipole_kernel`, but iterates
    over a list of receiver ``σ_r`` widths so the output has shape
    ``(N_k, N_σ, 4, 2)``. Used by the feature-projection kernel to get
    :math:`\hat\phi_{l,m}^{\sigma_r}(\mathbf{k})` per-(k, σ) without
    recomputing the k-dependent radial prefactor for every (l, m).

    The per-(σ, l) :math:`1/C_l(\sigma_r, \text{mode})` factors arrive in
    ``inv_cl_table[σ_i, l]`` so the caller (binding layer) picks the
    ``"receiver"`` / ``"multipoles"`` / ``"none"`` convention without this
    kernel needing to know about NormMode.

    Launch Grid
    -----------
    dim = [N_k, N_σ] — one thread per (k-vector, receiver sigma) pair.

    Parameters
    ----------
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-lattice k-vectors (float64).
    k_norm2 : wp.array, shape (N_k,), dtype wp.float64
        Pre-computed :math:`|\mathbf{k}|^2`.
    sigmas : wp.array, shape (N_σ,), dtype wp.float64
        Receiver-basis Gaussian widths. Each σ produces its own 4×2 block
        in the output.
    inv_cl_table : wp.array2d, shape (N_σ, 2), dtype wp.float64
        Host-computed ``1/C_l`` factors. ``inv_cl_table[σ_i, 0]`` is for
        ``l=0``, ``inv_cl_table[σ_i, 1]`` is for ``l=1``.
    output : wp.array4d(dtype=wp.float64), shape (N_k, N_σ, 4, 2)
        OUTPUT. Layout matches the source-side kernel: ``[k, σ, 0, :] =
        (l=0, m=0)``, ``[k, σ, 1, :] = (l=1, m=-1)``, etc. Does not need
        pre-zeroing — every entry is written unconditionally.

    Notes
    -----
    For ``l=0`` the result is purely real (imag = 0). For ``l=1`` the
    result is purely imaginary (real = 0) with the ``i^{-1} = -i`` sign
    already baked in; at ``k = 0`` every Cartesian component is zero, so
    the whole ``l=1`` block is zero analytically without an explicit guard.
    """
    k_idx, s_idx = wp.tid()

    k_vec = k_vectors[k_idx]
    k2 = k_norm2[k_idx]

    sigma = sigmas[s_idx]
    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    sigma5 = sigma3 * sigma2

    gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)
    common_radial = _FOUR_PI_SQRT_PI_OVER_2 * gauss

    inv_cl_l0 = inv_cl_table[s_idx, 0]
    inv_cl_l1 = inv_cl_table[s_idx, 1]

    # l = 0, m = 0: purely real.
    output[k_idx, s_idx, 0, 0] = inv_cl_l0 * common_radial * sigma3 * Y00_COEFF
    output[k_idx, s_idx, 0, 1] = wp.float64(0.0)

    # l = 1: purely imaginary, sign -1 from i^{-1} = -i.
    coeff_l1 = -inv_cl_l1 * common_radial * sigma5 * Y1_COEFF

    # m = -1 → k_y
    output[k_idx, s_idx, 1, 0] = wp.float64(0.0)
    output[k_idx, s_idx, 1, 1] = coeff_l1 * k_vec[1]
    # m = 0 → k_z
    output[k_idx, s_idx, 2, 0] = wp.float64(0.0)
    output[k_idx, s_idx, 2, 1] = coeff_l1 * k_vec[2]
    # m = +1 → k_x
    output[k_idx, s_idx, 3, 0] = wp.float64(0.0)
    output[k_idx, s_idx, 3, 1] = coeff_l1 * k_vec[0]


def eval_receiver_gto_fourier_dipole(
    k_vectors: wp.array,
    k_norm2: wp.array,
    sigmas: wp.array,
    inv_cl_table: wp.array,
    output: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_eval_receiver_gto_fourier_dipole_kernel`.

    Multi-σ receiver-basis :math:`\hat\phi_{l,m}^{\sigma_r}(\mathbf{k})` at
    l_max = 1, output shape ``(N_k, N_σ, 4, 2)``. All arrays are float64 —
    no polymorphism to dispatch on (everything in k-space stays in
    float64).

    Parameters
    ----------
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
    k_norm2 : wp.array, shape (N_k,), dtype wp.float64
    sigmas : wp.array, shape (N_σ,), dtype wp.float64
    inv_cl_table : wp.array2d, shape (N_σ, 2), dtype wp.float64
        Rows are ``[inv_cl_l0, inv_cl_l1]`` for each receiver σ.
    output : wp.array4d, shape (N_k, N_σ, 4, 2), dtype wp.float64
        Pre-allocated output buffer.
    device : str, optional
        Defaults to ``k_vectors.device``.
    """
    n_k = k_vectors.shape[0]
    n_sigma = sigmas.shape[0]
    if device is None:
        device = str(k_vectors.device)
    wp.launch(
        _eval_receiver_gto_fourier_dipole_kernel,
        dim=(n_k, n_sigma),
        inputs=[k_vectors, k_norm2, sigmas, inv_cl_table, output],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched receiver GTO Fourier coefficients
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_eval_receiver_gto_fourier_dipole_kernel(
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    k_norm2: wp.array2d(dtype=wp.float64),  # (B, K_max)
    sigmas: wp.array(dtype=wp.float64),  # (N_σ,)
    inv_cl_table: wp.array2d(dtype=wp.float64),  # (N_σ, 2)
    output: wp.array4d(dtype=wp.vec2d),  # (B, K_max, N_σ, 4) — vec2d = (real, imag)
):
    r"""Batched receiver-basis φ̂ across ``B`` systems.

    Uses ``wp.array4d(dtype=wp.vec2d)`` for the output so the logical
    shape ``(B, K_max, N_σ, 4, 2)`` fits within Warp's 4D-array limit;
    the last axis is the packed ``(real, imag)`` vec2d.

    Launch Grid
    -----------
    ``dim = (B, K_max, N_σ)``.


    Parameters
    ----------
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    k_norm2 : wp.array2d, shape (B, K_max), dtype wp.float64
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigmas : wp.array, shape (N_σ,), dtype wp.float64
        Per-channel Gaussian (GTO) width parameters.
    inv_cl_table : wp.array2d, shape (N_σ, 2), dtype wp.float64
        Per-channel inverse overlap normalization constants.
    output : wp.array4d, shape (B, K_max, N_σ, 4), dtype wp.vec2d
        — vec2d = (real, imag).
    """
    b, k_idx, s_idx = wp.tid()

    k_vec = k_vectors[b, k_idx]
    k2 = k_norm2[b, k_idx]

    sigma = sigmas[s_idx]
    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    sigma5 = sigma3 * sigma2

    gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)
    common_radial = _FOUR_PI_SQRT_PI_OVER_2 * gauss

    inv_cl_l0 = inv_cl_table[s_idx, 0]
    inv_cl_l1 = inv_cl_table[s_idx, 1]

    l0_real = inv_cl_l0 * common_radial * sigma3 * Y00_COEFF
    output[b, k_idx, s_idx, 0] = wp.vec2d(l0_real, wp.float64(0.0))

    coeff_l1 = -inv_cl_l1 * common_radial * sigma5 * Y1_COEFF
    output[b, k_idx, s_idx, 1] = wp.vec2d(wp.float64(0.0), coeff_l1 * k_vec[1])
    output[b, k_idx, s_idx, 2] = wp.vec2d(wp.float64(0.0), coeff_l1 * k_vec[2])
    output[b, k_idx, s_idx, 3] = wp.vec2d(wp.float64(0.0), coeff_l1 * k_vec[0])


def batch_eval_receiver_gto_fourier_dipole(
    k_vectors: wp.array,
    k_norm2: wp.array,
    sigmas: wp.array,
    inv_cl_table: wp.array,
    output: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_eval_receiver_gto_fourier_dipole_kernel`.

    Parameters
    ----------
    k_vectors : wp.array, shape (B, K_max), dtype wp.vec3d
    k_norm2 : wp.array, shape (B, K_max), dtype wp.float64
    sigmas : wp.array, shape (N_σ,), dtype wp.float64
    inv_cl_table : wp.array2d, shape (N_σ, 2), dtype wp.float64
    output : wp.array, shape (B, K_max, N_σ, 4), dtype wp.vec2d
        Pre-allocated. Caller passes the underlying float64 buffer of
        shape ``(B, K_max, N_σ, 4, 2)`` reinterpreted via
        ``wp.from_torch(t, dtype=wp.vec2d)``.
    device : str, optional
    """
    batch_size = k_vectors.shape[0]
    k_max = k_vectors.shape[1]
    n_sigma = sigmas.shape[0]
    if device is None:
        device = str(k_vectors.device)
    wp.launch(
        _batch_eval_receiver_gto_fourier_dipole_kernel,
        dim=(batch_size, k_max, n_sigma),
        inputs=[k_vectors, k_norm2, sigmas, inv_cl_table, output],
        device=device,
    )


# =============================================================================
# Fused feature projection (Phase 4 core)
# =============================================================================


@wp.kernel
def _project_features_dipole_kernel(
    potential: wp.array2d(dtype=wp.float64),
    receiver_phi_hat: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    source_feats_lm: wp.array2d(dtype=wp.float64),
    overlap_constants: wp.array2d(dtype=wp.float64),
    subtract_self: wp.int32,
    out_col_lut: wp.array2d(dtype=wp.int32),
    features: wp.array2d(dtype=wp.float64),
):
    r"""Project :math:`V(\mathbf{k})` onto the receiver basis at every atom, with optional self-interaction subtract and arbitrary output layout.

    Fused Phase 4 core — extended for Phase 5. For each ``(atom i,
    receiver σ σ_r, lm index)`` thread, accumulates

    .. math::

        f_{i, \sigma_r, lm} \;=\; \frac{2}{(2\pi)^3}
            \sum_{\mathbf{k}} w(\mathbf{k}) \Bigl[
                A_{\sigma_r, lm}(\mathbf{k}) \cos(\mathbf{k}\cdot\mathbf{r}_i)
              + B_{\sigma_r, lm}(\mathbf{k}) \sin(\mathbf{k}\cdot\mathbf{r}_i)
            \Bigr]

    (with :math:`A = V_r\,\hat\phi_r + V_i\,\hat\phi_i`,
    :math:`B = V_r\,\hat\phi_i - V_i\,\hat\phi_r`, and :math:`w` =
    ``k_factor_proj``), then — when ``subtract_self != 0`` — subtracts
    :math:`\mathrm{oc}[\sigma_r, l(lm)] \cdot \text{source\_feats\_lm}[i, lm]`
    inside the same thread to produce the self-interaction-corrected
    feature value. The corrected value is finally written to
    ``features[i, out_col_lut[s_idx, lm_idx]]`` — a 2-D output with an
    arbitrary per-``(σ, lm)`` column remap. Pass an identity LUT
    (``lut[s, lm] = s * 4 + lm``) for the natural flat layout; pass the
    ``graph_longrange`` output permutation to write the customer-
    drop-in layout directly.

    Fusing the self-interaction subtract and the output-address remap
    into this kernel eliminates the Python/torch post-processing chain
    that would otherwise run *between* the kernel and the user-facing
    return value (allocating ``src_lm`` / broadcast-multiplying ``oc`` /
    calling ``index_select``). See ``feedback_torch_wrapper_antipatterns.md``
    for the rule.

    Launch Grid
    -----------
    dim = [N_atoms, N_σ, 4] — one thread per ``(i, σ, lm)`` output entry.

    Parameters
    ----------
    potential : wp.array2d(dtype=wp.float64), shape (N_k, 2)
        :math:`V(\mathbf{k})` from :func:`apply_per_k_factor`.
    receiver_phi_hat : wp.array4d(dtype=wp.float64), shape (N_k, N_σ, 4, 2)
        Receiver-basis Fourier coefficients from
        :func:`eval_receiver_gto_fourier_dipole`.
    cosines, sines : wp.array2d(dtype=wp.float64), shape (N_k, N_atoms)
        Structure-factor tables from :func:`build_structure_factor_table`.
    k_factor_proj : wp.array(dtype=wp.float64), shape (N_k,)
        Per-k weight: ``0.5`` at ``k = 0``, ``1`` elsewhere.
    source_feats_lm : wp.array2d(dtype=wp.float64), shape (N_atoms, 4)
        Per-atom source moments in e3nn layout:
        ``[q, μ_y, μ_z, μ_x]``. Ignored when ``subtract_self == 0`` — but
        still must be a valid float64 ``(N_atoms, 4)`` array, as Warp
        does not support optional arguments. Callers can pass a zero
        tensor if they never want self-interaction.
    overlap_constants : wp.array2d(dtype=wp.float64), shape (N_σ, 2)
        Per-σ overlap constants: ``[:, 0]`` = ``l=0`` factor, ``[:, 1]``
        = ``l=1`` factor. From
        :func:`nvalchemiops.torch.math.compute_overlap_constants`, cached
        by the SCF cache. Also ignored when ``subtract_self == 0``.
    subtract_self : wp.int32
        ``0`` to skip the self-interaction subtract; anything else to
        apply it.
    out_col_lut : wp.array2d(dtype=wp.int32), shape (N_σ, 4)
        Per-``(σ, lm)`` flat output column index in ``[0, N_σ * 4)``.
        Pass ``lut[s, lm] = s * 4 + lm`` for the natural row-major
        layout. Pass the ``graph_longrange`` permutation for the
        customer-drop-in layout.
    features : wp.array2d(dtype=wp.float64), shape (N_atoms, N_σ * 4)
        OUTPUT — flat (atom, column) layout; the ``out_col_lut`` picks
        which flat column each per-``(σ, lm)`` result writes to. Does
        not need pre-zeroing **only if every flat column is covered by
        the LUT** (which is true for bijective LUTs like the identity
        and the graph_longrange permutation); pre-zero defensively
        otherwise.
    """
    i_idx, s_idx, lm_idx = wp.tid()

    n_k = potential.shape[0]
    acc = wp.float64(0.0)

    for k_idx in range(n_k):
        v_r = potential[k_idx, 0]
        v_i = potential[k_idx, 1]
        phi_r = receiver_phi_hat[k_idx, s_idx, lm_idx, 0]
        phi_i = receiver_phi_hat[k_idx, s_idx, lm_idx, 1]

        a = v_r * phi_r + v_i * phi_i
        b = v_r * phi_i - v_i * phi_r

        w = k_factor_proj[k_idx]
        c = cosines[k_idx, i_idx]
        s = sines[k_idx, i_idx]
        acc += w * (a * c + b * s)

    value = wp.float64(2.0) * acc * _INV_TWO_PI_CUBED

    if subtract_self != 0:
        # l(lm): 0 when lm == 0, else 1.
        if lm_idx == 0:
            l_idx = wp.int32(0)
        else:
            l_idx = wp.int32(1)
        value = value - overlap_constants[s_idx, l_idx] * source_feats_lm[i_idx, lm_idx]

    out_col = out_col_lut[s_idx, lm_idx]
    features[i_idx, out_col] = value


def project_features_dipole(
    potential: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    source_feats_lm: wp.array,
    overlap_constants: wp.array,
    subtract_self: bool,
    out_col_lut: wp.array,
    features: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_project_features_dipole_kernel`.

    Bit-for-bit parity with ``graph_longrange.features.project_to_features_batch``
    (optionally composed with ``GTOSelfInteractionBlock`` + the customer
    output permutation) at float64 under matched inputs.

    Parameters
    ----------
    potential : wp.array, shape (N_k, 2), dtype wp.float64
    receiver_phi_hat : wp.array, shape (N_k, N_σ, 4, 2), dtype wp.float64
    cosines, sines : wp.array, shape (N_k, N_atoms), dtype wp.float64
    k_factor_proj : wp.array, shape (N_k,), dtype wp.float64
    source_feats_lm : wp.array, shape (N_atoms, 4), dtype wp.float64
        Per-atom source moments in e3nn layout. Used only when
        ``subtract_self=True``.
    overlap_constants : wp.array, shape (N_σ, 2), dtype wp.float64
        Per-σ overlap constants. Used only when ``subtract_self=True``.
    subtract_self : bool
        Whether to subtract the self-interaction term inside the kernel.
    out_col_lut : wp.array, shape (N_σ, 4), dtype wp.int32
        Per-``(σ, lm)`` flat output column index.
    features : wp.array, shape (N_atoms, N_σ * 4), dtype wp.float64
        Pre-allocated output (flat).
    device : str, optional
        Defaults to ``potential.device``.
    """
    n_k = potential.shape[0]
    n_sigma = receiver_phi_hat.shape[1]
    n_atoms = cosines.shape[1]
    if device is None:
        device = str(potential.device)
    if cosines.shape[0] != n_k or sines.shape[0] != n_k:
        raise ValueError(
            f"cosines/sines must have N_k={n_k} rows, got "
            f"cosines.shape={tuple(cosines.shape)}, sines.shape={tuple(sines.shape)}"
        )
    if source_feats_lm.shape != (n_atoms, 4):
        raise ValueError(
            f"source_feats_lm must have shape (N_atoms={n_atoms}, 4), "
            f"got {tuple(source_feats_lm.shape)}"
        )
    if overlap_constants.shape != (n_sigma, 2):
        raise ValueError(
            f"overlap_constants must have shape (N_σ={n_sigma}, 2), "
            f"got {tuple(overlap_constants.shape)}"
        )
    if out_col_lut.shape != (n_sigma, 4):
        raise ValueError(
            f"out_col_lut must have shape (N_σ={n_sigma}, 4), "
            f"got {tuple(out_col_lut.shape)}"
        )
    if features.shape != (n_atoms, n_sigma * 4):
        raise ValueError(
            f"features must have shape (N_atoms={n_atoms}, N_σ * 4={n_sigma * 4}), "
            f"got {tuple(features.shape)}"
        )

    # Dispatch: CUDA gets the three-phase tile-matmul rewrite, CPU
    # stays on the serial per-(i, σ, lm) kernel.
    if "cuda" in str(device):
        _project_features_dipole_tiled_launch(
            potential,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            source_feats_lm,
            overlap_constants,
            subtract_self,
            out_col_lut,
            features,
            device,
        )
        return

    wp.launch(
        _project_features_dipole_kernel,
        dim=(n_atoms, n_sigma, 4),
        inputs=[
            potential,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            source_feats_lm,
            overlap_constants,
            wp.int32(1 if subtract_self else 0),
            out_col_lut,
            features,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched project_features_dipole (no tile — flat launch across the batch)
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_project_features_dipole_kernel(
    potential: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    receiver_phi_hat: wp.array4d(
        dtype=wp.vec2d
    ),  # (B, K_max, N_σ, 4) vec2d=(real,imag)
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_factor_proj: wp.array2d(dtype=wp.float64),  # (B, K_max)
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    source_feats_lm: wp.array2d(dtype=wp.float64),  # (N_total, 4) e3nn layout
    overlap_constants: wp.array2d(dtype=wp.float64),  # (N_σ, 2) — shared
    subtract_self: wp.int32,
    out_col_lut: wp.array2d(dtype=wp.int32),  # (N_σ, 4)
    features: wp.array2d(dtype=wp.float64),  # (N_total, N_σ*4) OUTPUT
):
    r"""Batched feature projection — one thread per ``(atom_i, σ, lm)``.

    Same math as the single-system kernel. Per-atom lookup of
    ``batch_idx[i] = b`` gives the system; the k-sum then reads
    ``potential[b, k]``, ``receiver_phi_hat[b, k, s, lm]``,
    ``k_factor_proj[b, k]``. Pad k-vectors have ``k_factor_proj = 0``
    in the batched cache, so they contribute zero without an explicit
    bound check.

    ``overlap_constants`` / ``out_col_lut`` are shared across the batch
    (same σ for every system).

    Launch Grid
    -----------
    ``dim = (N_total, N_σ, 4)``.


    Parameters
    ----------
    potential : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Per-k reciprocal-space potential factor.
    receiver_phi_hat : wp.array4d, dtype wp.vec2d
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    source_feats_lm : wp.array2d, shape (N_total, 4), dtype wp.float64
        E3nn layout.
    overlap_constants : wp.array2d, shape (N_σ, 2), dtype wp.float64
        — shared.
    subtract_self : wp.int32
        Flag: subtract the self-interaction term when nonzero.
    out_col_lut : wp.array2d, shape (N_σ, 4), dtype wp.int32
        Lookup table mapping channels to output feature columns.
    features : wp.array2d, shape (N_total, N_σ*4), dtype wp.float64
        OUTPUT: projected per-atom features.
    """
    i_idx, s_idx, lm_idx = wp.tid()
    b = batch_idx[i_idx]

    n_k_max = potential.shape[1]
    acc = wp.float64(0.0)

    for k_idx in range(n_k_max):
        v_r = potential[b, k_idx, 0]
        v_i = potential[b, k_idx, 1]
        phi = receiver_phi_hat[b, k_idx, s_idx, lm_idx]
        phi_r = phi[0]
        phi_i = phi[1]

        a = v_r * phi_r + v_i * phi_i
        b_term = v_r * phi_i - v_i * phi_r

        w = k_factor_proj[b, k_idx]
        c = cosines[k_idx, i_idx]
        s = sines[k_idx, i_idx]
        acc += w * (a * c + b_term * s)

    value = wp.float64(2.0) * acc * _INV_TWO_PI_CUBED

    if subtract_self != 0:
        if lm_idx == 0:
            l_idx = wp.int32(0)
        else:
            l_idx = wp.int32(1)
        value = value - overlap_constants[s_idx, l_idx] * source_feats_lm[i_idx, lm_idx]

    out_col = out_col_lut[s_idx, lm_idx]
    features[i_idx, out_col] = value


def batch_project_features_dipole(
    potential: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    batch_idx: wp.array,
    source_feats_lm: wp.array,
    overlap_constants: wp.array,
    subtract_self: bool,
    out_col_lut: wp.array,
    features: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_project_features_dipole_kernel`.

    Parameters
    ----------
    potential : wp.array, shape (B, K_max, 2), dtype wp.float64
    receiver_phi_hat : wp.array, shape (B, K_max, N_σ, 4), dtype wp.vec2d
        Caller supplies the underlying torch tensor of shape
        ``(B, K_max, N_σ, 4, 2)`` via ``wp.from_torch(t, dtype=wp.vec2d)``
        — see :func:`batch_eval_receiver_gto_fourier_dipole` for the
        producer.
    cosines, sines : wp.array, shape (K_max, N_total), dtype wp.float64
    k_factor_proj : wp.array, shape (B, K_max), dtype wp.float64
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
    source_feats_lm : wp.array, shape (N_total, 4), dtype wp.float64
    overlap_constants : wp.array, shape (N_σ, 2), dtype wp.float64
    subtract_self : bool
    out_col_lut : wp.array, shape (N_σ, 4), dtype wp.int32
    features : wp.array, shape (N_total, N_σ*4), dtype wp.float64
        Pre-allocated output.
    device : str, optional
    """
    batch_size = potential.shape[0]
    k_max = potential.shape[1]
    n_sigma = receiver_phi_hat.shape[2]
    n_total = cosines.shape[1]
    if device is None:
        device = str(cosines.device)

    # --- Basic shape / dtype validation -----------------------------------
    if receiver_phi_hat.shape[0] != batch_size or receiver_phi_hat.shape[1] != k_max:
        raise ValueError(
            "receiver_phi_hat must have shape (B, K_max, N_σ, 4) — got "
            f"{tuple(receiver_phi_hat.shape)}"
        )
    if k_factor_proj.shape != (batch_size, k_max):
        raise ValueError(
            f"k_factor_proj must be (B={batch_size}, K_max={k_max}), "
            f"got {tuple(k_factor_proj.shape)}"
        )
    if source_feats_lm.shape != (n_total, 4):
        raise ValueError(
            f"source_feats_lm must be (N_total={n_total}, 4), "
            f"got {tuple(source_feats_lm.shape)}"
        )
    if overlap_constants.shape != (n_sigma, 2):
        raise ValueError(
            f"overlap_constants must be (N_σ={n_sigma}, 2), "
            f"got {tuple(overlap_constants.shape)}"
        )
    if out_col_lut.shape != (n_sigma, 4):
        raise ValueError(
            f"out_col_lut must be (N_σ={n_sigma}, 4), got {tuple(out_col_lut.shape)}"
        )
    if features.shape != (n_total, n_sigma * 4):
        raise ValueError(
            f"features must be (N_total={n_total}, N_σ*4={n_sigma * 4}), "
            f"got {tuple(features.shape)}"
        )

    wp.launch(
        _batch_project_features_dipole_kernel,
        dim=(n_total, n_sigma, 4),
        inputs=[
            potential,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            batch_idx,
            source_feats_lm,
            overlap_constants,
            wp.int32(1 if subtract_self else 0),
            out_col_lut,
            features,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Tile-based GPU implementation of project_features_dipole
# -----------------------------------------------------------------------------
#
# Same tile-matmul strategy as :func:`position_gradient_from_rhok`:
# refactor the inner k-sum into a dense GEMM that ``wp.tile_matmul``
# can dispatch to Tensor Cores. Derivation:
#
# .. math::
#
#     f_{i, \sigma, lm}
#         = \frac{2}{(2\pi)^3} \sum_k w(k) \bigl[
#             A(k, \sigma, lm) \cos(k r_i) + B(k, \sigma, lm) \sin(k r_i)
#           \bigr]
#         - [\text{self-int correction when enabled}]
#
#     A(k, \sigma, lm) = V_r \hat\phi_r + V_i \hat\phi_i
#     B(k, \sigma, lm) = V_r \hat\phi_i - V_i \hat\phi_r
#
# Let ``sl = σ * 4 + lm``. Define the pre-scaled weighted matrices:
#
# .. math::
#
#     a_{\text{flat}}(k, sl) &= \tfrac{2}{(2\pi)^3} \, w(k) \, A(k, \sigma, lm) \\
#     b_{\text{flat}}(k, sl) &= \tfrac{2}{(2\pi)^3} \, w(k) \, B(k, \sigma, lm)
#
# The inner sum becomes a matmul:
#
# .. math::
#
#     \text{contribs}(i, sl)
#         = \sum_k \bigl[ \cos(k, i) \, a_{\text{flat}} + \sin(k, i) \, b_{\text{flat}} \bigr]

_PROJ_TILE_I = wp.constant(8)  # atoms per block (matches pos_grad sweet spot)
_PROJ_TILE_K = wp.constant(32)  # k-vectors per inner iteration
_PROJ_TILE_J = wp.constant(4)  # σ*4 cols per block (exact fit for N_σ=1 case)
_PROJ_BLOCK_DIM = 128


@wp.kernel
def _project_features_precompute_ab_kernel(
    potential: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    receiver_phi_hat: wp.array4d(dtype=wp.float64),  # (N_k, N_σ, 4, 2)
    k_factor_proj: wp.array(dtype=wp.float64),  # (N_k,)
    n_k_valid: wp.int32,
    n_sl_valid: wp.int32,  # = N_σ * 4 (before padding)
    a_flat: wp.array2d(dtype=wp.float64),  # (N_k_pad, N_sl_pad) OUTPUT
    b_flat: wp.array2d(dtype=wp.float64),  # (N_k_pad, N_sl_pad) OUTPUT
):
    r"""Build the ``a_flat`` / ``b_flat`` matrices with ``2/(2π)³ · kfp`` folded in.

    Launch dim: ``(N_k_pad, N_σ)``. Each thread writes the 4 ``lm``
    cols (`sl = s_idx * 4 + lm`) for its ``(k_idx, s_idx)`` pair. Pads
    columns in ``[n_sl_valid, N_sl_pad)`` and entire pad k-rows to
    zero.


    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx, s_idx)``; each thread processes one (k-vector, s_idx) work item.

    Parameters
    ----------
    potential : wp.array2d, shape (N_k, 2), dtype wp.float64
        Per-k reciprocal-space potential factor.
    receiver_phi_hat : wp.array4d, shape (N_k, N_σ, 4, 2), dtype wp.float64
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    k_factor_proj : wp.array, shape (N_k,), dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    n_k_valid : wp.int32
        Number of valid (non-padded) k-vectors.
    n_sl_valid : wp.int32, shape (before padding)
        = N_σ * 4.
    a_flat : wp.array2d, shape (N_k_pad, N_sl_pad), dtype wp.float64
        OUTPUT: Flattened left operand for the tiled matmul.
    b_flat : wp.array2d, shape (N_k_pad, N_sl_pad), dtype wp.float64
        OUTPUT: Flattened right operand for the tiled matmul.
    """
    k_idx, s_idx = wp.tid()

    if k_idx >= n_k_valid:
        # Pad k-row: clear all 4 cols this (k, σ) thread owns.
        base = s_idx * 4
        for lm in range(4):
            col = base + lm
            if col < a_flat.shape[1]:
                a_flat[k_idx, col] = wp.float64(0.0)
                b_flat[k_idx, col] = wp.float64(0.0)
        return

    v_r = potential[k_idx, 0]
    v_i = potential[k_idx, 1]
    kfp = k_factor_proj[k_idx]
    scale = _INV_TWO_PI_CUBED_TIMES_TWO * kfp

    base = s_idx * 4
    for lm in range(4):
        col = base + lm
        if col >= a_flat.shape[1]:
            continue
        if col >= n_sl_valid:
            a_flat[k_idx, col] = wp.float64(0.0)
            b_flat[k_idx, col] = wp.float64(0.0)
            continue
        phi_r = receiver_phi_hat[k_idx, s_idx, lm, 0]
        phi_i = receiver_phi_hat[k_idx, s_idx, lm, 1]
        a_flat[k_idx, col] = scale * (v_r * phi_r + v_i * phi_i)
        b_flat[k_idx, col] = scale * (v_r * phi_i - v_i * phi_r)


# _project_features_tiled_matmul_kernel → replaced by shared
# :func:`_cossin_native_matmul_kernel` (native layout, bounds-checked).


@wp.kernel
def _project_features_postprocess_kernel(
    contribs: wp.array2d(dtype=wp.float64),  # (N_atoms_pad, N_sl_pad)
    source_feats_lm: wp.array2d(dtype=wp.float64),  # (N_atoms, 4)
    overlap_constants: wp.array2d(dtype=wp.float64),  # (N_σ, 2)
    subtract_self: wp.int32,
    out_col_lut: wp.array2d(dtype=wp.int32),  # (N_σ, 4)
    n_atoms_valid: wp.int32,
    features: wp.array2d(dtype=wp.float64),  # (N_atoms, N_σ*4) OUTPUT
):
    r"""Apply the self-interaction subtract and the output-column LUT.

    Launch dim ``(N_atoms, N_σ, 4)`` — matches the original kernel's
    grid so the per-``(i, σ, lm)`` logic is identical (other than
    reading the k-sum from ``contribs`` instead of computing it).


    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx, s_idx, lm_idx)``; each thread processes one (i_idx, s_idx, lm_idx) work item.

    Parameters
    ----------
    contribs : wp.array2d, shape (N_atoms_pad, N_sl_pad), dtype wp.float64
        OUTPUT: per-tile partial contributions awaiting reduction.
    source_feats_lm : wp.array2d, shape (N_atoms, 4), dtype wp.float64
        Per-atom source features in the spherical (l, m) layout.
    overlap_constants : wp.array2d, shape (N_σ, 2), dtype wp.float64
        Per-channel overlap normalization constants.
    subtract_self : wp.int32
        Flag: subtract the self-interaction term when nonzero.
    out_col_lut : wp.array2d, shape (N_σ, 4), dtype wp.int32
        Lookup table mapping channels to output feature columns.
    n_atoms_valid : wp.int32
        Number of valid (non-padded) atoms.
    features : wp.array2d, shape (N_atoms, N_σ*4), dtype wp.float64
        OUTPUT: projected per-atom features.
    """
    i_idx, s_idx, lm_idx = wp.tid()
    if i_idx >= n_atoms_valid:
        return

    sl = s_idx * 4 + lm_idx
    value = contribs[i_idx, sl]

    if subtract_self != 0:
        if lm_idx == 0:
            l_idx = wp.int32(0)
        else:
            l_idx = wp.int32(1)
        value = value - overlap_constants[s_idx, l_idx] * source_feats_lm[i_idx, lm_idx]

    out_col = out_col_lut[s_idx, lm_idx]
    features[i_idx, out_col] = value


def _project_features_dipole_tiled_launch(
    potential: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    source_feats_lm: wp.array,
    overlap_constants: wp.array,
    subtract_self: bool,
    out_col_lut: wp.array,
    features: wp.array,
    device: str,
) -> None:
    r"""CUDA-only three-phase tile-matmul implementation of :func:`project_features_dipole`.

    Sub-kernels:

    1. :func:`_project_features_precompute_ab_kernel` → ``a_flat`` / ``b_flat``
       with ``2/(2π)³ · kfp`` already folded in.
    2. :func:`_project_features_tiled_matmul_kernel` → ``contribs``.
    3. :func:`_project_features_postprocess_kernel` → ``features`` with
       optional self-interaction subtract + output-column remap.
    """
    import torch

    n_k = cosines.shape[0]
    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]
    n_sl = n_sigma * 4

    # ``a_flat`` / ``b_flat``: per-(k, σ) precompute (no pad rows/cols). The
    # contraction is a cuBLAS GEMM over the transposed ``(N_k, N_atoms)`` cos/sin
    # views — no manual transpose+zero-pad copy, no int32 tile-offset chunking.
    cos_t = wp.to_torch(cosines)  # (N_k, N_atoms) — for .device only
    a_flat = torch.empty((n_k, n_sl), dtype=torch.float64, device=cos_t.device)
    b_flat = torch.empty_like(a_flat)

    # -- Launch 1: precompute a_flat / b_flat. -------------------------------
    wp.launch(
        _project_features_precompute_ab_kernel,
        dim=(n_k, n_sigma),
        inputs=[
            potential,
            receiver_phi_hat,
            k_factor_proj,
            wp.int32(n_k),
            wp.int32(n_sl),
            wp.from_torch(a_flat, dtype=wp.float64),
            wp.from_torch(b_flat, dtype=wp.float64),
        ],
        device=device,
    )

    # -- Step 2: contribs = cosᵀ @ a_flat + sinᵀ @ b_flat via the native-layout
    #    Warp tile matmul (in-kernel tile_transpose; no transpose+pad copy, no
    #    torch matmul).
    contribs = torch.empty((n_atoms, n_sl), dtype=torch.float64, device=cos_t.device)
    _launch_cossin_native_matmul(
        cosines,
        sines,
        wp.from_torch(a_flat, dtype=wp.float64),
        wp.from_torch(b_flat, dtype=wp.float64),
        contribs,
        device,
    )

    # -- Launch 3: post-process. ---------------------------------------------
    wp.launch(
        _project_features_postprocess_kernel,
        dim=(n_atoms, n_sigma, 4),
        inputs=[
            wp.from_torch(contribs, dtype=wp.float64),
            source_feats_lm,
            overlap_constants,
            wp.int32(1 if subtract_self else 0),
            out_col_lut,
            wp.int32(n_atoms),
            features,
        ],
        device=device,
    )


# =============================================================================
# Backward of project_features_dipole w.r.t. V(k)  (Phase 8c)
# =============================================================================
#
# Transpose of the forward projection: launch per-k, inner sum over atoms
# and (σ, lm). Produces ``∂L/∂V(k)`` given ``grad_raw[i, σ, lm]`` — the
# cotangent of the un-self-interaction-subtracted, natural-layout
# feature output.
#
#   Q_r(k, i) = Σ_{σ, lm} grad_raw(i, σ, lm) · φ̂_r(k, σ, lm)
#   Q_i(k, i) = Σ_{σ, lm} grad_raw(i, σ, lm) · φ̂_i(k, σ, lm)
#   ∂L/∂V_r(k) = (2/(2π)³) · k_factor_proj(k) · Σ_i [Q_r cos(k·r_i)
#                                                  + Q_i sin(k·r_i)]
#   ∂L/∂V_i(k) = (2/(2π)³) · k_factor_proj(k) · Σ_i [Q_i cos(k·r_i)
#                                                  − Q_r sin(k·r_i)]
#
# Derivation verified vs finite difference to float64 FD noise floor.


_INV_TWO_PI_CUBED_TIMES_TWO = wp.constant(wp.float64(2.0 / ((2.0 * math.pi) ** 3)))


@wp.kernel
def _v_gradient_from_feature_grad_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    grad_v: wp.array2d(dtype=wp.float64),
):
    r"""Analytical backward of :func:`project_features_dipole` w.r.t. ``V(k)``.

    One thread per k-vector; inner loop over atoms and ``(σ, lm)``. The
    per-atom ``(Q_r, Q_i)`` reduction over ``(σ, lm)`` is recomputed
    inside each thread rather than cached, to avoid materializing the
    ``(N_k, N_atoms, 2)`` intermediate.

    Launch Grid
    -----------
    dim = [N_k].

    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_atoms, N_σ, 4), dtype wp.float64
        Cotangent of the raw (un-self-interaction-subtracted,
        natural-layout) feature tensor.
    receiver_phi_hat : wp.array4d, shape (N_k, N_σ, 4, 2), dtype wp.float64
        Receiver-basis Fourier coefficients from
        :func:`eval_receiver_gto_fourier_dipole`.
    cosines, sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Structure-factor tables from the forward.
    k_factor_proj : wp.array, shape (N_k,), dtype wp.float64
        Per-k weight (``0.5`` at k = 0, ``1`` elsewhere in the direct k-space sum).
    grad_v : wp.array2d, shape (N_k, 2), dtype wp.float64
        OUTPUT: ``∂L/∂V(k)`` with ``[:, 0]`` / ``[:, 1]`` = (real, imag).
        Does not need pre-zeroing.
    """
    k_idx = wp.tid()

    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]

    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)

    for i_idx in range(n_atoms):
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        # Q_r(k, i), Q_i(k, i) = Σ_{σ, lm} grad_raw · φ̂.
        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(4):
                g = grad_raw[i_idx, s_idx, lm_idx]
                q_r += receiver_phi_hat[k_idx, s_idx, lm_idx, 0] * g
                q_i += receiver_phi_hat[k_idx, s_idx, lm_idx, 1] * g

        acc_r += q_r * cos_ki + q_i * sin_ki
        acc_i += q_i * cos_ki - q_r * sin_ki

    kfp = k_factor_proj[k_idx]
    grad_v[k_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_r
    grad_v[k_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_i


def v_gradient_from_feature_grad(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    grad_v: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_v_gradient_from_feature_grad_kernel`.

    The per-k kernel here is already well-served by a serial inner
    ``atoms × (σ * 4)`` loop — ``≤ N_atoms · 4 ≲ 3k`` iterations per
    thread at ``N_atoms ≲ 700``. A tile-matmul rewrite (see
    :func:`_v_gradient_from_feature_grad_tiled_launch` below) was
    tried and came out 3× slower because the matmul's ``N`` dimension
    (``N_σ * 4``) is too narrow for Tensor Cores to amortize, and
    materializing the ``GRC`` / ``GRS`` intermediates doubled the
    output bandwidth. We therefore dispatch the per-k kernel on both
    CPU and CUDA. Kept the tile implementation for reference /
    future reconsideration with larger ``N_σ``.


    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    grad_v : wp.array
        OUTPUT: gradient w.r.t. the input feature vector ``v``.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_k = receiver_phi_hat.shape[0]
    if device is None:
        device = str(receiver_phi_hat.device)

    wp.launch(
        _v_gradient_from_feature_grad_kernel,
        dim=n_k,
        inputs=[grad_raw, receiver_phi_hat, cosines, sines, k_factor_proj, grad_v],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched v_gradient_from_feature_grad
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_v_gradient_from_feature_grad_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),  # (N_total, N_σ, 4)
    receiver_phi_hat: wp.array4d(dtype=wp.vec2d),  # (B, K_max, N_σ, 4) vec2d
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_factor_proj: wp.array2d(dtype=wp.float64),  # (B, K_max)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    grad_v: wp.array3d(dtype=wp.float64),  # (B, K_max, 2) OUTPUT
):
    r"""Batched ``∂L/∂V(k)`` backward of feature projection.

    One thread per ``(b, k_idx)``. Inner sum over atoms in system
    ``b`` (from ``atom_start[b]`` to ``atom_end[b]``) and over the
    ``(σ, lm)`` feature axes. ``receiver_phi_hat`` stored as
    ``vec2d`` for the ``(real, imag)`` axis to fit within 4-D
    Warp arrays.

    Launch Grid
    -----------
    ``dim = (B, K_max)``.


    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_total, N_σ, 4), dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array4d, shape (B, K_max, N_σ, 4), dtype wp.vec2d
        Vec2d.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    atom_start : wp.array, shape (B,), dtype wp.int32
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array, shape (B,), dtype wp.int32
        Per-system end offset into the flat atom arrays.
    grad_v : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        OUTPUT: gradient w.r.t. the input feature vector ``v``.
    """
    b, k_idx = wp.tid()
    n_sigma = receiver_phi_hat.shape[2]

    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)

    i_lo = atom_start[b]
    i_hi = atom_end[b]
    for i_idx in range(i_lo, i_hi):
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(4):
                g = grad_raw[i_idx, s_idx, lm_idx]
                phi = receiver_phi_hat[b, k_idx, s_idx, lm_idx]
                q_r += phi[0] * g
                q_i += phi[1] * g

        acc_r += q_r * cos_ki + q_i * sin_ki
        acc_i += q_i * cos_ki - q_r * sin_ki

    kfp = k_factor_proj[b, k_idx]
    grad_v[b, k_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_r
    grad_v[b, k_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_i


def batch_v_gradient_from_feature_grad(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    grad_v: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_v_gradient_from_feature_grad_kernel`.

    Parameters
    ----------
    grad_raw : wp.array, shape (N_total, N_σ, 4), dtype wp.float64
    receiver_phi_hat : wp.array, shape (B, K_max, N_σ, 4), dtype wp.vec2d
    cosines, sines : wp.array, shape (K_max, N_total), dtype wp.float64
    k_factor_proj : wp.array, shape (B, K_max), dtype wp.float64
    atom_start, atom_end : wp.array, shape (B,), dtype wp.int32
    grad_v : wp.array, shape (B, K_max, 2), dtype wp.float64
    device : str, optional
    """
    batch_size = receiver_phi_hat.shape[0]
    k_max = receiver_phi_hat.shape[1]
    if device is None:
        device = str(cosines.device)

    wp.launch(
        _batch_v_gradient_from_feature_grad_kernel,
        dim=(batch_size, k_max),
        inputs=[
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            atom_start,
            atom_end,
            grad_v,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Tile-based GPU implementation of v_gradient_from_feature_grad
# -----------------------------------------------------------------------------
#
# Same three-phase refactor as position_gradient_from_rhok /
# project_features_dipole. Derivation:
#
# .. math::
#
#     \partial L / \partial V_r(k)
#         = \tfrac{2}{(2\pi)^3} \, w(k) \sum_{i, sl}
#             \mathrm{grad\_raw}(i, sl) \bigl[
#                 \hat\phi_r(k, sl) \cos(k r_i)
#               + \hat\phi_i(k, sl) \sin(k r_i) \bigr]
#
# where ``sl = σ * 4 + lm``. Factoring the atoms sum into an explicit
# matmul pass:
#
# .. math::
#
#     \mathrm{GRC}(k, sl) = \sum_i \mathrm{grad\_raw}(i, sl) \cos(k, i)
#                         = (\cos \cdot \mathrm{grad\_raw\_flat})(k, sl)
#     \mathrm{GRS}(k, sl) = (\sin \cdot \mathrm{grad\_raw\_flat})(k, sl)
#
# then
#
# .. math::
#
#     \partial L/\partial V_r(k)
#         = \tfrac{2}{(2\pi)^3} w(k) \sum_{sl} \bigl[
#             \hat\phi_r(k, sl) \mathrm{GRC}(k, sl) + \hat\phi_i(k, sl) \mathrm{GRS}(k, sl) \bigr]
#     \partial L/\partial V_i(k)
#         = \tfrac{2}{(2\pi)^3} w(k) \sum_{sl} \bigl[
#             \hat\phi_i(k, sl) \mathrm{GRC}(k, sl) - \hat\phi_r(k, sl) \mathrm{GRS}(k, sl) \bigr]
#
# Note that the matmul ``M`` axis here is ``N_k`` (large) and the ``K``
# axis is ``N_atoms`` (small–medium). No cos/sin transpose needed
# because they're already stored as ``(N_k, N_atoms)``.

_VG_TILE_M = wp.constant(8)  # k-vectors per block (matches pos_grad sweet spot)
_VG_TILE_K = wp.constant(32)  # atoms per inner iteration
_VG_TILE_N = wp.constant(4)  # N_σ*4 cols per block (4 = exact fit for N_σ=1)
_VG_BLOCK_DIM = 128


@wp.kernel
def _v_grad_flatten_grad_raw_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),  # (N_atoms, N_σ, 4)
    n_atoms_valid: wp.int32,
    n_sl_valid: wp.int32,
    grad_raw_flat: wp.array2d(dtype=wp.float64),  # (N_atoms_pad, N_sl_pad) OUTPUT
):
    r"""Flatten ``grad_raw`` from ``(N_atoms, N_σ, 4)`` to ``(N_atoms_pad, N_σ*4)``.

    Launch dim: ``(N_atoms_pad, N_sl_pad)``. Pad entries beyond the
    real ``(N_atoms, N_σ*4)`` get zero so the tile matmul doesn't pick
    up garbage.


    Launch Grid
    -----------
    ``dim`` indexes ``(i, sl)``; each thread processes one (output element, sl) work item.

    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_atoms, N_σ, 4), dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    n_atoms_valid : wp.int32
        Number of valid (non-padded) atoms.
    n_sl_valid : wp.int32
        Number of valid (l, m) feature slots.
    grad_raw_flat : wp.array2d, shape (N_atoms, N_sl), dtype wp.float64
        OUTPUT: flattened raw-feature gradient buffer.
    """
    i, sl = wp.tid()
    if i >= n_atoms_valid or sl >= n_sl_valid:
        grad_raw_flat[i, sl] = wp.float64(0.0)
        return
    s = sl // 4
    lm = sl % 4
    grad_raw_flat[i, sl] = grad_raw[i, s, lm]


@wp.kernel
def _v_grad_tiled_matmul_kernel(
    cosines: wp.array2d(dtype=wp.float64),  # (N_k_pad, N_atoms_pad)
    sines: wp.array2d(dtype=wp.float64),
    grad_raw_flat: wp.array2d(dtype=wp.float64),  # (N_atoms_pad, N_sl_pad)
    grc: wp.array2d(dtype=wp.float64),  # (N_k_pad, N_sl_pad) OUTPUT
    grs: wp.array2d(dtype=wp.float64),  # (N_k, N_sl) OUTPUT
):
    r"""Compute ``GRC = cos @ grad_raw_flat``, ``GRS = sin @ grad_raw_flat``.

    Single kernel producing both intermediates in parallel — the
    ``grad_raw_flat`` tile load is shared between the two matmul ops.


    Launch Grid
    -----------
    ``dim`` indexes ``(m_block, n_block)``; each thread processes one (m_block, n_block) work item.

    Parameters
    ----------
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    grad_raw_flat : wp.array2d, shape (N_atoms, N_sl), dtype wp.float64
        OUTPUT: flattened raw-feature gradient buffer.
    grc : wp.array2d, shape (N_k, N_sl), dtype wp.float64
        OUTPUT: cosine-channel reduced intermediate.
    grs : wp.array2d, shape (N_k_pad, N_sl_pad), dtype wp.float64
        OUTPUT: sine-channel reduced intermediate.
    """
    m_block, n_block = wp.tid()
    n_atoms = cosines.shape[1]

    acc_c = wp.tile_zeros(shape=(_VG_TILE_M, _VG_TILE_N), dtype=wp.float64)
    acc_s = wp.tile_zeros(shape=(_VG_TILE_M, _VG_TILE_N), dtype=wp.float64)

    m_off = m_block * _VG_TILE_M
    n_off = n_block * _VG_TILE_N

    for kk in range(0, n_atoms, _VG_TILE_K):
        c_tile = wp.tile_load(
            cosines, shape=(_VG_TILE_M, _VG_TILE_K), offset=(m_off, kk)
        )
        s_tile = wp.tile_load(sines, shape=(_VG_TILE_M, _VG_TILE_K), offset=(m_off, kk))
        gr_tile = wp.tile_load(
            grad_raw_flat, shape=(_VG_TILE_K, _VG_TILE_N), offset=(kk, n_off)
        )
        wp.tile_matmul(c_tile, gr_tile, acc_c)
        wp.tile_matmul(s_tile, gr_tile, acc_s)

    wp.tile_store(grc, acc_c, offset=(m_off, n_off))
    wp.tile_store(grs, acc_s, offset=(m_off, n_off))


@wp.kernel
def _v_grad_per_k_reduce_kernel(
    receiver_phi_hat: wp.array4d(dtype=wp.float64),  # (N_k, N_σ, 4, 2)
    grc: wp.array2d(dtype=wp.float64),  # (N_k, N_sl)
    grs: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    n_k_valid: wp.int32,
    grad_v: wp.array2d(dtype=wp.float64),  # (N_k, 2) OUTPUT
):
    r"""Per-k final reduction: combine ``GRC`` / ``GRS`` with receiver φ̂.

    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx)``; each thread processes one k-vector index.

    Parameters
    ----------
    receiver_phi_hat : wp.array4d, shape (N_k, N_σ, 4, 2), dtype wp.float64
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grc : wp.array2d, shape (N_k, N_sl), dtype wp.float64
        OUTPUT: cosine-channel reduced intermediate.
    grs : wp.array2d, dtype wp.float64
        OUTPUT: sine-channel reduced intermediate.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    n_k_valid : wp.int32
        Number of valid (non-padded) k-vectors.
    grad_v : wp.array2d, shape (N_k, 2), dtype wp.float64
        OUTPUT: gradient w.r.t. the input feature vector ``v``.
    """
    k_idx = wp.tid()
    if k_idx >= n_k_valid:
        return

    kfp = k_factor_proj[k_idx]
    n_sigma = receiver_phi_hat.shape[1]

    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)
    for s_idx in range(n_sigma):
        for lm_idx in range(4):
            sl = s_idx * 4 + lm_idx
            phi_r = receiver_phi_hat[k_idx, s_idx, lm_idx, 0]
            phi_i = receiver_phi_hat[k_idx, s_idx, lm_idx, 1]
            grc_val = grc[k_idx, sl]
            grs_val = grs[k_idx, sl]
            acc_r += phi_r * grc_val + phi_i * grs_val
            acc_i += phi_i * grc_val - phi_r * grs_val

    grad_v[k_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_r
    grad_v[k_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_i


def _v_gradient_from_feature_grad_tiled_launch(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    grad_v: wp.array,
    device: str,
) -> None:
    r"""CUDA tile-matmul implementation of :func:`v_gradient_from_feature_grad`.

    grad_raw is already contiguous in ``(N_atoms, N_σ, 4)`` memory
    layout, which is identical to ``(N_atoms, N_σ*4)`` — we just
    reinterpret as a 2D tensor via ``torch.reshape`` (a view, no copy).
    cos/sin/grad_raw are passed to the matmul unpadded; ``wp.tile_load`` /
    ``wp.tile_store`` bounds-check, so no transpose+pad copy is materialized.
    """
    import torch

    n_k = cosines.shape[0]
    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]
    n_sl = n_sigma * 4

    tile_m = int(_VG_TILE_M)
    tile_n = int(_VG_TILE_N)

    cos_t = wp.to_torch(cosines).contiguous()
    sin_t = wp.to_torch(sines).contiguous()
    grad_raw_t = wp.to_torch(grad_raw).reshape(n_atoms, n_sl).contiguous()

    grc = torch.empty((n_k, n_sl), dtype=torch.float64, device=cos_t.device)
    grs = torch.empty_like(grc)

    # -- Launch 1: native-layout tile matmul (two matmuls share the grad_raw
    #    tile; cos/sin read unpadded + bounds-checked — no transpose+pad copy).
    _launch_v_grad_tile_matmul_chunked(
        _v_grad_tiled_matmul_kernel,
        cosines=cos_t,
        sines=sin_t,
        grad_raw=grad_raw_t,
        grc=grc,
        grs=grs,
        tile_m=tile_m,
        tile_n=tile_n,
        block_dim=int(_VG_BLOCK_DIM),
        device=device,
    )

    # -- Launch 2: per-k reduction. ------------------------------------------
    wp.launch(
        _v_grad_per_k_reduce_kernel,
        dim=n_k,
        inputs=[
            receiver_phi_hat,
            wp.from_torch(grc, dtype=wp.float64),
            wp.from_torch(grs, dtype=wp.float64),
            k_factor_proj,
            wp.int32(n_k),
            grad_v,
        ],
        device=device,
    )


# =============================================================================
# Backward of project_features_dipole w.r.t. positions  (Phase 8c)
# =============================================================================
#
# Position gradient of the raw (un-self-int) feature projection:
#
#   C(k, i) = V_r(k) · Q_i(k, i) − V_i(k) · Q_r(k, i)
#   D(k, i) = V_r(k) · Q_r(k, i) + V_i(k) · Q_i(k, i)
#   ∂L/∂r_{i, α} = (2/(2π)³) · Σ_k k_factor_proj(k) · k_α
#                  · [C(k, i) cos(k·r_i) − D(k, i) sin(k·r_i)]
#
# where Q is the same reduction as in the V-gradient kernel. Structure
# mirrors position_gradient_from_rhok: one thread per atom, inner over
# k. Verified to float64 FD precision during design review.


@wp.kernel
def _position_gradient_from_feature_grad_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    potential: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    grad_positions: wp.array2d(dtype=wp.float64),
):
    r"""Analytical backward of :func:`project_features_dipole` w.r.t. atomic positions.

    One thread per atom. Inner loop over k performs the same ``(Q_r, Q_i)``
    reduction over ``(σ, lm)`` as the V-gradient kernel, then combines
    with ``V(k)`` to form ``(C, D)`` and accumulates the three Cartesian
    components of the position gradient.

    Launch Grid
    -----------
    dim = [N_atoms]; output shape ``(N_atoms, 3)``.

    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_atoms, N_σ, 4), dtype wp.float64
    receiver_phi_hat : wp.array4d, shape (N_k, N_σ, 4, 2), dtype wp.float64
    cosines, sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
    k_factor_proj : wp.array, shape (N_k,), dtype wp.float64
    potential : wp.array2d, shape (N_k, 2), dtype wp.float64
        ``V(k)`` from the forward. Must be the same ``V(k)`` that was
        fed to :func:`project_features_dipole` on this step.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
    grad_positions : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        OUTPUT.
    """
    i_idx = wp.tid()

    n_k = cosines.shape[0]
    n_sigma = receiver_phi_hat.shape[1]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(n_k):
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        v_r = potential[k_idx, 0]
        v_i = potential[k_idx, 1]
        kfp = k_factor_proj[k_idx]

        # Q_r, Q_i: inner reduction over (σ, lm).
        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(4):
                g = grad_raw[i_idx, s_idx, lm_idx]
                q_r += receiver_phi_hat[k_idx, s_idx, lm_idx, 0] * g
                q_i += receiver_phi_hat[k_idx, s_idx, lm_idx, 1] * g

        c_k = v_r * q_i - v_i * q_r
        d_k = v_r * q_r + v_i * q_i

        per_k = kfp * (c_k * cos_ki - d_k * sin_ki)
        k_vec = k_vectors[k_idx]
        gx += k_vec[0] * per_k
        gy += k_vec[1] * per_k
        gz += k_vec[2] * per_k

    grad_positions[i_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * gx
    grad_positions[i_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * gy
    grad_positions[i_idx, 2] = _INV_TWO_PI_CUBED_TIMES_TWO * gz


def position_gradient_from_feature_grad(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    k_vectors: wp.array,
    grad_positions: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_position_gradient_from_feature_grad_kernel`.

    Dispatches to a three-phase tile-matmul implementation on CUDA and
    to the serial per-atom kernel on CPU.


    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    grad_positions : wp.array
        OUTPUT: gradient w.r.t. atomic positions.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_atoms = cosines.shape[1]
    if device is None:
        device = str(cosines.device)

    if "cuda" in str(device):
        _position_gradient_from_feature_grad_tiled_launch(
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            potential,
            k_vectors,
            grad_positions,
            device,
        )
        return

    wp.launch(
        _position_gradient_from_feature_grad_kernel,
        dim=n_atoms,
        inputs=[
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            potential,
            k_vectors,
            grad_positions,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched position_gradient_from_feature_grad
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_position_gradient_from_feature_grad_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),  # (N_total, N_σ, 4)
    receiver_phi_hat: wp.array4d(dtype=wp.vec2d),  # (B, K_max, N_σ, 4) vec2d
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_factor_proj: wp.array2d(dtype=wp.float64),  # (B, K_max)
    potential: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    grad_positions: wp.array2d(dtype=wp.float64),  # (N_total, 3) OUTPUT
):
    r"""Batched analytical backward of feature projection w.r.t. atomic positions.

    One thread per atom (flat across the batch). ``batch_idx[i]``
    resolves the system; inner k-loop reads per-``b`` state from
    ``potential``, ``receiver_phi_hat``, ``k_factor_proj``,
    ``k_vectors``. Pad k-vectors have ``k_factor_proj = 0`` and
    ``receiver_phi_hat = 0`` in the batched cache so their
    contributions cancel.

    Launch Grid
    -----------
    ``dim = N_total``.


    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_total, N_σ, 4), dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array4d, shape (B, K_max, N_σ, 4), dtype wp.vec2d
        Vec2d.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Per-k reciprocal-space potential factor.
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    grad_positions : wp.array2d, shape (N_total, 3), dtype wp.float64
        OUTPUT: gradient w.r.t. atomic positions.
    """
    i_idx = wp.tid()
    b = batch_idx[i_idx]

    k_max = k_vectors.shape[1]
    n_sigma = receiver_phi_hat.shape[2]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(k_max):
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        v_r = potential[b, k_idx, 0]
        v_i = potential[b, k_idx, 1]
        kfp = k_factor_proj[b, k_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(4):
                g = grad_raw[i_idx, s_idx, lm_idx]
                phi = receiver_phi_hat[b, k_idx, s_idx, lm_idx]
                q_r += phi[0] * g
                q_i += phi[1] * g

        c_k = v_r * q_i - v_i * q_r
        d_k = v_r * q_r + v_i * q_i

        per_k = kfp * (c_k * cos_ki - d_k * sin_ki)
        k_vec = k_vectors[b, k_idx]
        gx += k_vec[0] * per_k
        gy += k_vec[1] * per_k
        gz += k_vec[2] * per_k

    grad_positions[i_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * gx
    grad_positions[i_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * gy
    grad_positions[i_idx, 2] = _INV_TWO_PI_CUBED_TIMES_TWO * gz


def batch_position_gradient_from_feature_grad(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    k_vectors: wp.array,
    batch_idx: wp.array,
    grad_positions: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_position_gradient_from_feature_grad_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    grad_positions : wp.array
        OUTPUT: gradient w.r.t. atomic positions.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_total = cosines.shape[1]
    if device is None:
        device = str(cosines.device)

    wp.launch(
        _batch_position_gradient_from_feature_grad_kernel,
        dim=n_total,
        inputs=[
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            potential,
            k_vectors,
            batch_idx,
            grad_positions,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Tile-based GPU implementation of position_gradient_from_feature_grad
# -----------------------------------------------------------------------------
#
# Same three-phase refactor as position_gradient_from_rhok. Derivation:
#
# .. math::
#
#     \partial L / \partial r_{i, \alpha} = \mathrm{scale} \sum_k w(k) \, k_\alpha
#         \bigl[ C(k, i) \cos(k r_i) - D(k, i) \sin(k r_i) \bigr]
#
#     C(k, i) = \sum_{sl} \mathrm{grad\_raw}(i, sl) \, \alpha_1(k, sl)
#     D(k, i) = \sum_{sl} \mathrm{grad\_raw}(i, sl) \, \alpha_2(k, sl)
#     \alpha_1(k, sl) = V_r \hat\phi_i - V_i \hat\phi_r
#     \alpha_2(k, sl) = V_r \hat\phi_r + V_i \hat\phi_i
#
# Substituting ``C`` and ``D`` and factoring the k-sum via
# ``β(k, α*N_sl+sl)``:
#
# .. math::
#
#     \beta_\cos(k, \alpha*N_{sl}+sl) &= w(k) \, k_\alpha \, \alpha_1(k, sl) \\
#     \beta_\sin(k, \alpha*N_{sl}+sl) &= -w(k) \, k_\alpha \, \alpha_2(k, sl)
#
# (sign on ``β_sin`` folds the ``-D sin`` term into a positive
# accumulation in the matmul), and
#
# .. math::
#
#     T(i, p) = \sum_k \bigl[
#         \cos(k, i) \beta_\cos(k, p) + \sin(k, i) \beta_\sin(k, p) \bigr]
#
# which is two ``(N_atoms, N_k) × (N_k, 3 N_{sl})`` matmuls. Final
# per-atom contraction closes the ``sl`` axis against ``grad_raw``:
#
# .. math::
#
#     \partial L / \partial r_{i, \alpha} = \tfrac{2}{(2\pi)^3}
#         \sum_{sl} \mathrm{grad\_raw}(i, sl) \, T(i, \alpha*N_{sl}+sl)
#
# For ``N_σ = 1`` the intermediate ``T`` is ``(N_atoms, 12)`` — same
# shape as the rho-position tiled output, so the same tile sizes apply.

_FPG_TILE_I = wp.constant(8)  # atoms per block
_FPG_TILE_K = wp.constant(32)  # k-vectors per inner iteration
_FPG_TILE_J = wp.constant(16)  # col dim per block (pad 3*N_σ*4 up to multiple)
_FPG_BLOCK_DIM = 128


@wp.kernel
def _feat_pos_grad_precompute_beta_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),  # (N_atoms, N_σ, 4) — unused (for symmetry)
    receiver_phi_hat: wp.array4d(dtype=wp.float64),  # (N_k, N_σ, 4, 2)
    k_factor_proj: wp.array(dtype=wp.float64),  # (N_k,)
    potential: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    n_k_valid: wp.int32,
    n_p_valid: wp.int32,  # = 3 * N_σ * 4
    beta_cos: wp.array2d(dtype=wp.float64),  # (N_k_pad, N_p_pad) OUTPUT
    beta_sin: wp.array2d(dtype=wp.float64),  # (N_k_pad, N_p_pad) OUTPUT
):
    r"""Precompute β matrices for the tile matmul.

    Launch dim ``(N_k_pad, N_σ)``. Each thread handles one ``(k, σ)``
    pair and writes 3·4 = 12 columns of β_cos / β_sin.


    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx, s_idx)``; each thread processes one (k-vector, s_idx) work item.

    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_atoms, N_σ, 4), dtype wp.float64
        — unused (for symmetry).
    receiver_phi_hat : wp.array4d, shape (N_k, N_σ, 4, 2), dtype wp.float64
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    k_factor_proj : wp.array, shape (N_k,), dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array2d, shape (N_k, 2), dtype wp.float64
        Per-k reciprocal-space potential factor.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-space k-vectors.
    n_k_valid : wp.int32
        Number of valid (non-padded) k-vectors.
    n_p_valid : wp.int32
        = 3 * N_σ * 4.
    beta_cos : wp.array2d, shape (N_k_pad, N_p_pad), dtype wp.float64
        OUTPUT: precomputed per-k cosine-weighted reduction coefficients.
    beta_sin : wp.array2d, shape (N_k_pad, N_p_pad), dtype wp.float64
        OUTPUT: precomputed per-k sine-weighted reduction coefficients.
    """
    k_idx, s_idx = wp.tid()

    if k_idx >= n_k_valid:
        # Pad rows — clear the 3*4 = 12 columns this (k, σ) thread owns.
        base = s_idx * 12
        for c in range(12):
            col = base + c
            if col < beta_cos.shape[1]:
                beta_cos[k_idx, col] = wp.float64(0.0)
                beta_sin[k_idx, col] = wp.float64(0.0)
        return

    v_r = potential[k_idx, 0]
    v_i = potential[k_idx, 1]
    kfp = k_factor_proj[k_idx]
    k_vec = k_vectors[k_idx]
    kx = k_vec[0]
    ky = k_vec[1]
    kz = k_vec[2]

    # Per-σ: iterate over the 4 lm cols.
    # Column layout: p = α * N_sl + sl, sl = s_idx * 4 + lm.
    # So within this (k, σ) thread we write columns
    #   [0 * N_sl + sl, 1 * N_sl + sl, 2 * N_sl + sl] for sl ∈ [s_idx*4, s_idx*4 + 4).
    n_sl = receiver_phi_hat.shape[1] * 4

    for lm in range(4):
        phi_r = receiver_phi_hat[k_idx, s_idx, lm, 0]
        phi_i = receiver_phi_hat[k_idx, s_idx, lm, 1]
        alpha1 = v_r * phi_i - v_i * phi_r
        alpha2 = v_r * phi_r + v_i * phi_i

        sl = s_idx * 4 + lm
        for alpha in range(3):
            col = alpha * n_sl + sl
            if col >= beta_cos.shape[1]:
                continue
            if alpha == 0:
                k_a = kx
            else:
                if alpha == 1:
                    k_a = ky
                else:
                    k_a = kz
            if col >= n_p_valid:
                beta_cos[k_idx, col] = wp.float64(0.0)
                beta_sin[k_idx, col] = wp.float64(0.0)
            else:
                beta_cos[k_idx, col] = kfp * k_a * alpha1
                beta_sin[k_idx, col] = -kfp * k_a * alpha2


# _feat_pos_grad_tiled_matmul_kernel → replaced by shared
# :func:`_cossin_native_matmul_kernel`.


@wp.kernel
def _feat_pos_grad_reduce_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),  # (N_atoms, N_σ, 4)
    contribs: wp.array2d(dtype=wp.float64),  # (N_atoms_pad, N_p_pad)
    n_atoms_valid: wp.int32,
    grad_positions: wp.array2d(dtype=wp.float64),  # (N_atoms, 3) OUTPUT
):
    r"""Final per-atom contraction: ``∂L/∂r_{i,α} = scale · Σ_{sl} grad_raw(i, sl) · T(i, α*N_sl+sl)``.

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_atoms, N_σ, 4), dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    contribs : wp.array2d, shape (N_atoms_pad, N_p_pad), dtype wp.float64
        OUTPUT: per-tile partial contributions awaiting reduction.
    n_atoms_valid : wp.int32
        Number of valid (non-padded) atoms.
    grad_positions : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        OUTPUT: gradient w.r.t. atomic positions.
    """
    i_idx = wp.tid()
    if i_idx >= n_atoms_valid:
        return

    n_sigma = grad_raw.shape[1]
    n_sl = n_sigma * 4

    for alpha in range(3):
        total = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm in range(4):
                sl = s_idx * 4 + lm
                total += grad_raw[i_idx, s_idx, lm] * contribs[i_idx, alpha * n_sl + sl]
        grad_positions[i_idx, alpha] = _INV_TWO_PI_CUBED_TIMES_TWO * total


def _position_gradient_from_feature_grad_tiled_launch(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    k_vectors: wp.array,
    grad_positions: wp.array,
    device: str,
) -> None:
    r"""CUDA tile-matmul orchestrator for :func:`position_gradient_from_feature_grad`."""
    import torch

    n_k = cosines.shape[0]
    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]
    n_sl = n_sigma * 4
    n_p = 3 * n_sl  # output col count in intermediate

    tile_i = int(_FPG_TILE_I)
    tile_k = int(_FPG_TILE_K)
    tile_j = int(_FPG_TILE_J)

    n_k_pad = ((n_k + tile_k - 1) // tile_k) * tile_k
    n_atoms_pad = ((n_atoms + tile_i - 1) // tile_i) * tile_i
    n_p_pad = ((n_p + tile_j - 1) // tile_j) * tile_j

    # -- Pad + transpose cos / sin to (N_atoms_pad, N_k_pad). ----------------
    # cos/sin stay in their native (N_k, N_atoms) layout — the native-layout
    # tile matmul transposes per-tile (wp.tile_transpose), so no (N_atoms, N_k)
    # transpose+zero-pad copy is materialized.
    cos_t = wp.to_torch(cosines)  # for .device only

    beta_cos = torch.empty((n_k_pad, n_p_pad), dtype=torch.float64, device=cos_t.device)
    beta_sin = torch.empty_like(beta_cos)
    contribs = torch.empty(
        (n_atoms_pad, n_p_pad), dtype=torch.float64, device=cos_t.device
    )

    # -- Launch 1: precompute β_cos / β_sin. --------------------------------
    wp.launch(
        _feat_pos_grad_precompute_beta_kernel,
        dim=(n_k_pad, n_sigma),
        inputs=[
            grad_raw,
            receiver_phi_hat,
            k_factor_proj,
            potential,
            k_vectors,
            wp.int32(n_k),
            wp.int32(n_p),
            wp.from_torch(beta_cos, dtype=wp.float64),
            wp.from_torch(beta_sin, dtype=wp.float64),
        ],
        device=device,
    )

    # -- Launch 2: tile matmul. ----------------------------------------------
    _launch_cossin_native_matmul(
        cosines,
        sines,
        wp.from_torch(beta_cos, dtype=wp.float64),
        wp.from_torch(beta_sin, dtype=wp.float64),
        contribs,
        device,
    )

    # -- Launch 3: per-atom reduction. --------------------------------------
    wp.launch(
        _feat_pos_grad_reduce_kernel,
        dim=n_atoms_pad,
        inputs=[
            grad_raw,
            wp.from_torch(contribs, dtype=wp.float64),
            wp.int32(n_atoms),
            grad_positions,
        ],
        device=device,
    )


# =============================================================================
# Phase 8d — second-order backward kernels (double-backward support)
# =============================================================================
#
# The following nine kernels implement the analytical backward of every
# Phase 8a-c backward kernel, so that the ``torch.autograd.Function``
# wrappers can compose under ``create_graph=True`` for MLIP training
# (force-loss and stress-loss gradients flowing back to model
# parameters). Each kernel was derived from the forward and verified to
# finite-difference float64 precision during design review.
#
# Naming convention: ``<forward-name>_backward_<input>`` — e.g.
# ``source_phi_hat_backward_k_vectors`` is the gradient of
# :func:`eval_gto_fourier_dipole` w.r.t. its ``k_vectors`` input. Where
# a kernel covers several inputs at once (K1 / K2 fuse ``k_vectors``
# and ``k_norm2``; K4 fuses ``charges`` and ``dipoles``) the output
# names disambiguate.


# -----------------------------------------------------------------------------
# K1: backward of eval_gto_fourier_dipole w.r.t. (k_vectors, k_norm2)
# -----------------------------------------------------------------------------


@wp.kernel
def _source_phi_hat_backward_dipole_kernel(
    grad_output: wp.array3d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    k_norm2: wp.array(dtype=wp.float64),
    sigma: wp.float64,
    inv_cl_l0: wp.float64,
    inv_cl_l1: wp.float64,
    grad_k_vectors: wp.array(dtype=wp.vec3d),
    grad_k_norm2: wp.array(dtype=wp.float64),
):
    r"""Analytical backward of :func:`eval_gto_fourier_dipole` w.r.t. ``(k_vectors, k_norm2)``.

    Forward at a glance:

    .. math::

        \hat\phi_{0,0}(k)      &= A_0 \cdot g(k) \\
        \hat\phi_{1,\pm 1/0}(k) &= c_1 \cdot g(k) \cdot k_\alpha

    where :math:`g(k) = \exp(-\tfrac{1}{2} k^2 \sigma^2)`,
    :math:`A_0 = \text{invcl}_0 \cdot 4\pi\sqrt{\pi/2} \cdot \sigma^3 \cdot Y_{00}`,
    and :math:`c_1 = -\text{invcl}_1 \cdot 4\pi\sqrt{\pi/2} \cdot \sigma^5 \cdot Y_1`.
    The l=1 block is purely imaginary so the three k-vector components
    are written to ``output[k, 1..3, 1]``.

    Backward formulas (per thread, per k-index):

    .. math::

        \frac{\partial L}{\partial k^2_k} &= -\tfrac{1}{2} \sigma^2 g(k) \bigl[
            g_0 A_0 + c_1 (g_1 k_y + g_2 k_z + g_3 k_x) \bigr] \\
        \frac{\partial L}{\partial k_{k,x}} &= c_1 \, g(k) \, g_3 \\
        \frac{\partial L}{\partial k_{k,y}} &= c_1 \, g(k) \, g_1 \\
        \frac{\partial L}{\partial k_{k,z}} &= c_1 \, g(k) \, g_2

    with :math:`g_{lm} = \text{grad\_output}[k, lm, 1]` for ``lm > 0``
    and :math:`g_0 = \text{grad\_output}[k, 0, 0]` (the l=0 block is
    purely real, so only the real slot contributes).

    Launch Grid
    -----------
    dim = [N_k].

    Parameters
    ----------
    grad_output : wp.array3d, shape (N_k, 4, 2), dtype wp.float64
        Upstream cotangent ``∂L/∂source_phi_hat``.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
    k_norm2 : wp.array, shape (N_k,), dtype wp.float64
    sigma, inv_cl_l0, inv_cl_l1 : wp.float64
        Same forward-side scalars as the kernel under backward.
    grad_k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        OUTPUT. Written unconditionally.
    grad_k_norm2 : wp.array, shape (N_k,), dtype wp.float64
        OUTPUT. Written unconditionally.
    """
    k_idx = wp.tid()

    k_vec = k_vectors[k_idx]
    k2 = k_norm2[k_idx]

    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    sigma5 = sigma3 * sigma2

    gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)
    common_radial = _FOUR_PI_SQRT_PI_OVER_2 * gauss

    a0_base = inv_cl_l0 * common_radial * sigma3 * Y00_COEFF
    coeff_l1 = -inv_cl_l1 * common_radial * sigma5 * Y1_COEFF

    # l=1 imag entries: stored in grad_output[k, 1, 1] (k_y), [k, 2, 1] (k_z),
    # [k, 3, 1] (k_x).
    g_l0_r = grad_output[k_idx, 0, 0]
    g_ky = grad_output[k_idx, 1, 1]
    g_kz = grad_output[k_idx, 2, 1]
    g_kx = grad_output[k_idx, 3, 1]

    # d g / d k2 = -0.5 * sigma^2 * g   — already baked into the a0_base /
    # coeff_l1 chain since both are proportional to g.
    dot_l1 = coeff_l1 * (g_ky * k_vec[1] + g_kz * k_vec[2] + g_kx * k_vec[0])
    grad_k_norm2[k_idx] = -wp.float64(0.5) * sigma2 * (g_l0_r * a0_base + dot_l1)

    # k_vec components only appear as explicit multipliers in the l=1 block.
    grad_k_vectors[k_idx] = wp.vec3d(
        coeff_l1 * g_kx,
        coeff_l1 * g_ky,
        coeff_l1 * g_kz,
    )


def source_phi_hat_backward_dipole(
    grad_output: wp.array,
    k_vectors: wp.array,
    k_norm2: wp.array,
    sigma: float,
    inv_cl_l0: float,
    inv_cl_l1: float,
    grad_k_vectors: wp.array,
    grad_k_norm2: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_source_phi_hat_backward_dipole_kernel`.

    Parameters
    ----------
    grad_output : wp.array
        Upstream gradient flowing into this backward kernel.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    k_norm2 : wp.array
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigma : float
        Gaussian (GTO) width parameter.
    inv_cl_l0 : float
        Inverse :math:`l=0` overlap normalization constant.
    inv_cl_l1 : float
        Inverse :math:`l=1` overlap normalization constant.
    grad_k_vectors : wp.array
        OUTPUT: gradient w.r.t. the k-vectors.
    grad_k_norm2 : wp.array
        OUTPUT: gradient w.r.t. :math:`|k|^2`.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_k = k_vectors.shape[0]
    if device is None:
        device = str(k_vectors.device)
    wp.launch(
        _source_phi_hat_backward_dipole_kernel,
        dim=n_k,
        inputs=[
            grad_output,
            k_vectors,
            k_norm2,
            wp.float64(sigma),
            wp.float64(inv_cl_l0),
            wp.float64(inv_cl_l1),
            grad_k_vectors,
            grad_k_norm2,
        ],
        device=device,
    )


@wp.kernel
def _source_phi_hat_double_backward_dipole_kernel(
    gg_k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,) cotangent on grad_k_vectors
    gg_k_norm2: wp.array(dtype=wp.float64),  # (N_k,) cotangent on grad_k_norm2
    grad_output: wp.array3d(dtype=wp.float64),  # (N_k, 4, 2)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    k_norm2: wp.array(dtype=wp.float64),  # (N_k,)
    sigma: wp.float64,
    inv_cl_l0: wp.float64,
    inv_cl_l1: wp.float64,
    grad_grad_output: wp.array3d(dtype=wp.float64),  # (N_k, 4, 2) OUTPUT
    grad_k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,) OUTPUT
    grad_k_norm2: wp.array(dtype=wp.float64),  # (N_k,) OUTPUT
):
    r"""Second-order backward of :func:`eval_gto_fourier_dipole`.

    The first-order backward (:func:`_source_phi_hat_backward_dipole_kernel`) is
    linear in ``grad_output`` and depends on ``(k_vectors, k_norm2)`` through the
    shared Gaussian ``g(k²) = C₀·e^{γ·k²}`` (``γ = -σ²/2``). With upstream
    cotangents ``(gg_kv, gg_k2)`` on ``(grad_k_vectors, grad_k_norm2)`` the
    closed-form second-order grads (per k) are, with ``a₀ = invcl₀·g·σ³·Y₀₀``,
    ``c₁ = -invcl₁·g·σ⁵·Y₁``, ``S = Σ_α G_{1,α}·k_α``:

    .. math::

        \partial L_2/\partial G_{0,0} &= gg_{k²}\,\gamma\,a_0 \\
        \partial L_2/\partial G_{1,\alpha} &= c_1(gg_{kv,\alpha} + gg_{k²}\,\gamma\,k_\alpha) \\
        \partial L_2/\partial k_\alpha &= gg_{k²}\,\gamma\,c_1\,G_{1,\alpha} \\
        \partial L_2/\partial k² &= \gamma\,c_1(gg_{kv}\!\cdot\!G_{1})
            + gg_{k²}\,\gamma^2(a_0 G_{0,0} + c_1 S)

    Launch Grid
    -----------
    dim = [N_k].
    """
    k_idx = wp.tid()
    k_vec = k_vectors[k_idx]
    k2 = k_norm2[k_idx]

    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    sigma5 = sigma3 * sigma2
    gamma = -wp.float64(0.5) * sigma2  # d(common)/dk2 = gamma * common

    gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)
    common_radial = _FOUR_PI_SQRT_PI_OVER_2 * gauss
    a0_base = inv_cl_l0 * common_radial * sigma3 * Y00_COEFF
    coeff_l1 = -inv_cl_l1 * common_radial * sigma5 * Y1_COEFF

    # grad_output relevant entries (l=0 real, l=1 imag at rows 1/2/3 -> k_y/k_z/k_x).
    g_l0_r = grad_output[k_idx, 0, 0]
    g_ky = grad_output[k_idx, 1, 1]
    g_kz = grad_output[k_idx, 2, 1]
    g_kx = grad_output[k_idx, 3, 1]

    ggkv = gg_k_vectors[k_idx]
    ggkv_x = ggkv[0]
    ggkv_y = ggkv[1]
    ggkv_z = ggkv[2]
    ggk2 = gg_k_norm2[k_idx]

    # S = Σ_α G_{1,α} k_α with the (m=-1,0,+1) -> (k_y, k_z, k_x) layout.
    s_dot = g_ky * k_vec[1] + g_kz * k_vec[2] + g_kx * k_vec[0]
    ggkv_dot_g = ggkv_x * g_kx + ggkv_y * g_ky + ggkv_z * g_kz

    # grad_grad_output: zero everywhere except the l=0 real + l=1 imag slots.
    for lm in range(4):
        grad_grad_output[k_idx, lm, 0] = wp.float64(0.0)
        grad_grad_output[k_idx, lm, 1] = wp.float64(0.0)
    grad_grad_output[k_idx, 0, 0] = ggk2 * gamma * a0_base
    grad_grad_output[k_idx, 3, 1] = coeff_l1 * (ggkv_x + ggk2 * gamma * k_vec[0])
    grad_grad_output[k_idx, 1, 1] = coeff_l1 * (ggkv_y + ggk2 * gamma * k_vec[1])
    grad_grad_output[k_idx, 2, 1] = coeff_l1 * (ggkv_z + ggk2 * gamma * k_vec[2])

    # grad w.r.t. k_vectors (k enters only the gg_k2 path via S).
    sc_kv = ggk2 * gamma * coeff_l1
    grad_k_vectors[k_idx] = wp.vec3d(sc_kv * g_kx, sc_kv * g_ky, sc_kv * g_kz)

    # grad w.r.t. k_norm2 (common's k2-dependence, factor gamma on a0/coeff_l1).
    grad_k_norm2[k_idx] = gamma * coeff_l1 * ggkv_dot_g + ggk2 * gamma * gamma * (
        a0_base * g_l0_r + coeff_l1 * s_dot
    )


def source_phi_hat_double_backward_dipole(
    gg_k_vectors: wp.array,
    gg_k_norm2: wp.array,
    grad_output: wp.array,
    k_vectors: wp.array,
    k_norm2: wp.array,
    sigma: float,
    inv_cl_l0: float,
    inv_cl_l1: float,
    grad_grad_output: wp.array,
    grad_k_vectors: wp.array,
    grad_k_norm2: wp.array,
    device: str | None = None,
) -> None:
    """Launcher for :func:`_source_phi_hat_double_backward_dipole_kernel`."""
    n_k = k_vectors.shape[0]
    if device is None:
        device = str(k_vectors.device)
    wp.launch(
        _source_phi_hat_double_backward_dipole_kernel,
        dim=n_k,
        inputs=[
            gg_k_vectors,
            gg_k_norm2,
            grad_output,
            k_vectors,
            k_norm2,
            wp.float64(sigma),
            wp.float64(inv_cl_l0),
            wp.float64(inv_cl_l1),
            grad_grad_output,
            grad_k_vectors,
            grad_k_norm2,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# K2: backward of eval_receiver_gto_fourier_dipole w.r.t. (k_vectors, k_norm2)
# -----------------------------------------------------------------------------


@wp.kernel
def _receiver_phi_hat_backward_dipole_kernel(
    grad_output: wp.array4d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    k_norm2: wp.array(dtype=wp.float64),
    sigmas: wp.array(dtype=wp.float64),
    inv_cl_table: wp.array2d(dtype=wp.float64),
    grad_k_vectors: wp.array(dtype=wp.vec3d),
    grad_k_norm2: wp.array(dtype=wp.float64),
):
    r"""Analytical backward of :func:`eval_receiver_gto_fourier_dipole` w.r.t. ``(k_vectors, k_norm2)``.

    Same structure as :func:`_source_phi_hat_backward_dipole_kernel` but
    with an inner ``σ`` loop — each (k, σ) block of ``grad_output``
    contributes to the same ``(grad_k_vectors[k], grad_k_norm2[k])``
    slot. Launched one thread per k-vector; the thread sweeps all σ to
    avoid a per-(k, σ) atomic add.

    Launch Grid
    -----------
    dim = [N_k].

    Parameters
    ----------
    grad_output : wp.array4d, shape (N_k, N_σ, 4, 2), dtype wp.float64
        Upstream cotangent ``∂L/∂receiver_phi_hat``.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
    k_norm2 : wp.array, shape (N_k,), dtype wp.float64
    sigmas : wp.array, shape (N_σ,), dtype wp.float64
    inv_cl_table : wp.array2d, shape (N_σ, 2), dtype wp.float64
    grad_k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        OUTPUT. Written unconditionally.
    grad_k_norm2 : wp.array, shape (N_k,), dtype wp.float64
        OUTPUT. Written unconditionally.
    """
    k_idx = wp.tid()
    n_sigma = sigmas.shape[0]

    k_vec = k_vectors[k_idx]
    k2 = k_norm2[k_idx]

    sum_k2 = wp.float64(0.0)
    sum_kx = wp.float64(0.0)
    sum_ky = wp.float64(0.0)
    sum_kz = wp.float64(0.0)

    for s_idx in range(n_sigma):
        sigma = sigmas[s_idx]
        sigma2 = sigma * sigma
        sigma3 = sigma2 * sigma
        sigma5 = sigma3 * sigma2

        gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)
        common_radial = _FOUR_PI_SQRT_PI_OVER_2 * gauss

        a0_base = inv_cl_table[s_idx, 0] * common_radial * sigma3 * Y00_COEFF
        coeff_l1 = -inv_cl_table[s_idx, 1] * common_radial * sigma5 * Y1_COEFF

        g_l0_r = grad_output[k_idx, s_idx, 0, 0]
        g_ky = grad_output[k_idx, s_idx, 1, 1]
        g_kz = grad_output[k_idx, s_idx, 2, 1]
        g_kx = grad_output[k_idx, s_idx, 3, 1]

        dot_l1 = coeff_l1 * (g_ky * k_vec[1] + g_kz * k_vec[2] + g_kx * k_vec[0])
        sum_k2 += -wp.float64(0.5) * sigma2 * (g_l0_r * a0_base + dot_l1)

        sum_kx += coeff_l1 * g_kx
        sum_ky += coeff_l1 * g_ky
        sum_kz += coeff_l1 * g_kz

    grad_k_norm2[k_idx] = sum_k2
    grad_k_vectors[k_idx] = wp.vec3d(sum_kx, sum_ky, sum_kz)


def receiver_phi_hat_backward_dipole(
    grad_output: wp.array,
    k_vectors: wp.array,
    k_norm2: wp.array,
    sigmas: wp.array,
    inv_cl_table: wp.array,
    grad_k_vectors: wp.array,
    grad_k_norm2: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_receiver_phi_hat_backward_dipole_kernel`.

    Parameters
    ----------
    grad_output : wp.array
        Upstream gradient flowing into this backward kernel.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    k_norm2 : wp.array
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigmas : wp.array
        Per-channel Gaussian (GTO) width parameters.
    inv_cl_table : wp.array
        Per-channel inverse overlap normalization constants.
    grad_k_vectors : wp.array
        OUTPUT: gradient w.r.t. the k-vectors.
    grad_k_norm2 : wp.array
        OUTPUT: gradient w.r.t. :math:`|k|^2`.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_k = k_vectors.shape[0]
    if device is None:
        device = str(k_vectors.device)
    wp.launch(
        _receiver_phi_hat_backward_dipole_kernel,
        dim=n_k,
        inputs=[
            grad_output,
            k_vectors,
            k_norm2,
            sigmas,
            inv_cl_table,
            grad_k_vectors,
            grad_k_norm2,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# K3: backward of position_gradient_from_rhok w.r.t. grad_rho
# -----------------------------------------------------------------------------


@wp.kernel
def _rhok_position_grad_backward_grad_rho_kernel(
    charges: wp.array(dtype=Any),
    dipoles: wp.array(dtype=Any),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    source_phi_hat: wp.array3d(dtype=wp.float64),
    gg_positions: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    scale: wp.float64,
    ggrad_grad_rho: wp.array2d(dtype=wp.float64),
):
    r"""Backward of :func:`position_gradient_from_rhok` w.r.t. ``grad_rho``.

    Given upstream cotangent ``gg_positions`` of shape ``(N_atoms, 3)``,
    produces ``ggrad_grad_rho`` of shape ``(N_k, 2)``:

    .. math::

        h(k, i) &= \sum_\alpha k_\alpha \cdot gg_{pos}(i, \alpha) \\
        \tilde g_r(k) &= \mathrm{scale} \sum_i h(k, i) \bigl[
            P_i(k, i) \cos(k \cdot r_i) - P_r(k, i) \sin(k \cdot r_i) \bigr] \\
        \tilde g_i(k) &= -\mathrm{scale} \sum_i h(k, i) \bigl[
            P_r(k, i) \cos(k \cdot r_i) + P_i(k, i) \sin(k \cdot r_i) \bigr]

    with :math:`P_{r/i}(k, i) = \sum_{lm} \hat\phi_{r/i}(k, lm)\,q(i, lm)`
    in the same e3nn lm layout as the forward.

    Launch Grid
    -----------
    dim = [N_k] — one thread per k-vector; inner loop over atoms.


    Parameters
    ----------
    charges : wp.array, dtype Any
        Per-atom monopole charges.
    dipoles : wp.array, dtype Any
        Per-atom Cartesian dipole moments.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array3d, dtype wp.float64
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    gg_positions : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    ggrad_grad_rho : wp.array2d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. ``grad_rho``.
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]

    k_vec = k_vectors[k_idx]

    pr0 = source_phi_hat[k_idx, 0, 0]
    pi0 = source_phi_hat[k_idx, 0, 1]
    pr1 = source_phi_hat[k_idx, 1, 0]
    pi1 = source_phi_hat[k_idx, 1, 1]
    pr2 = source_phi_hat[k_idx, 2, 0]
    pi2 = source_phi_hat[k_idx, 2, 1]
    pr3 = source_phi_hat[k_idx, 3, 0]
    pi3 = source_phi_hat[k_idx, 3, 1]

    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)

    for i_idx in range(n_atoms):
        q = wp.float64(charges[i_idx])
        mu = dipoles[i_idx]
        mu_x = wp.float64(mu[0])
        mu_y = wp.float64(mu[1])
        mu_z = wp.float64(mu[2])

        p_r = pr0 * q + pr1 * mu_y + pr2 * mu_z + pr3 * mu_x
        p_i = pi0 * q + pi1 * mu_y + pi2 * mu_z + pi3 * mu_x

        h = (
            k_vec[0] * gg_positions[i_idx, 0]
            + k_vec[1] * gg_positions[i_idx, 1]
            + k_vec[2] * gg_positions[i_idx, 2]
        )
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        acc_r += h * (p_i * cos_ki - p_r * sin_ki)
        acc_i -= h * (p_r * cos_ki + p_i * sin_ki)

    ggrad_grad_rho[k_idx, 0] = scale * acc_r
    ggrad_grad_rho[k_idx, 1] = scale * acc_i


def _rhok_pg_back_grad_rho_sig(v, t):
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array3d(dtype=wp.float64),  # source_phi_hat
        wp.array2d(dtype=wp.float64),  # gg_positions
        wp.array(dtype=wp.vec3d),  # k_vectors
        wp.float64,  # scale
        wp.array2d(dtype=wp.float64),  # ggrad_grad_rho
    ]


_rhok_position_grad_backward_grad_rho_overloads = register_overloads(
    _rhok_position_grad_backward_grad_rho_kernel, _rhok_pg_back_grad_rho_sig
)


def rhok_position_grad_backward_grad_rho(
    charges: wp.array,
    dipoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    scale: float,
    ggrad_grad_rho: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_rhok_position_grad_backward_grad_rho_kernel`.

    Parameters
    ----------
    charges : wp.array
        Per-atom monopole charges.
    dipoles : wp.array
        Per-atom Cartesian dipole moments.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    scale : float
        Scalar prefactor applied to the contribution.
    ggrad_grad_rho : wp.array
        OUTPUT: double-backward gradient w.r.t. ``grad_rho``.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    n_k = cosines.shape[0]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rhok_position_grad_backward_grad_rho_overloads[vec_dtype],
        dim=n_k,
        inputs=[
            charges,
            dipoles,
            cosines,
            sines,
            source_phi_hat,
            gg_positions,
            k_vectors,
            wp.float64(scale),
            ggrad_grad_rho,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# K4: backward of position_gradient_from_rhok w.r.t. (charges, dipoles)
# -----------------------------------------------------------------------------


@wp.kernel
def _rhok_position_grad_backward_moments_kernel(
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    source_phi_hat: wp.array3d(dtype=wp.float64),
    grad_rho: wp.array2d(dtype=wp.float64),
    gg_positions: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    scale: wp.float64,
    ggrad_moments: wp.array2d(dtype=wp.float64),
):
    r"""Backward of :func:`position_gradient_from_rhok` w.r.t. ``(charges, dipoles)``.

    Produces ``ggrad_moments`` of shape ``(N_atoms, 4)`` in e3nn layout
    (``lm = 0`` → charge, ``1`` → μ_y, ``2`` → μ_z, ``3`` → μ_x). The
    caller is responsible for splitting into charge / Cartesian-dipole
    pieces.

    .. math::

        \tilde g_q(i, lm) = \mathrm{scale} \sum_k h(k, i) \bigl\{
            \hat\phi_i(k, lm) (g^\rho_r \cos - g^\rho_i \sin)
          - \hat\phi_r(k, lm) (g^\rho_i \cos + g^\rho_r \sin) \bigr\}

    with :math:`h(k, i) = \sum_\alpha k_\alpha \, gg_{pos}(i, \alpha)`.

    Launch Grid
    -----------
    dim = [N_atoms, 4] — one thread per ``(atom i, lm)``.


    Parameters
    ----------
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array3d, dtype wp.float64
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array2d, dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    gg_positions : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    ggrad_moments : wp.array2d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. the multipole moments.
    """
    i_idx, lm_idx = wp.tid()

    n_k = cosines.shape[0]

    # Cache the per-atom h-vector coordinates (needed each k iteration).
    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    acc = wp.float64(0.0)
    for k_idx in range(n_k):
        k_vec = k_vectors[k_idx]
        h = k_vec[0] * gp_x + k_vec[1] * gp_y + k_vec[2] * gp_z
        gr = grad_rho[k_idx, 0]
        gi = grad_rho[k_idx, 1]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        phi_r = source_phi_hat[k_idx, lm_idx, 0]
        phi_i = source_phi_hat[k_idx, lm_idx, 1]

        # phi_i · (gr·cos - gi·sin) - phi_r · (gi·cos + gr·sin)
        term = phi_i * (gr * cos_ki - gi * sin_ki) - phi_r * (gi * cos_ki + gr * sin_ki)
        acc += h * term

    ggrad_moments[i_idx, lm_idx] = scale * acc


def rhok_position_grad_backward_moments(
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    grad_rho: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    scale: float,
    ggrad_moments: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_rhok_position_grad_backward_moments_kernel`.

    Dispatches to the tile-matmul implementation on CUDA and the
    per-(atom, lm) serial kernel on CPU.

    The tile implementation shares the inner-matmul intermediate
    ``T = cos.T @ β_cos + sin.T @ β_sin`` with
    :func:`position_gradient_from_rhok` — only the final per-atom
    reduction differs (contracts α instead of lm).


    Parameters
    ----------
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    scale : float
        Scalar prefactor applied to the contribution.
    ggrad_moments : wp.array
        OUTPUT: double-backward gradient w.r.t. the multipole moments.
    device : str
        Warp device string; defaults to the input array's device.
    """
    if device is None:
        device = str(cosines.device)

    if "cuda" in str(device):
        _rhok_pg_back_moments_tiled_launch(
            cosines,
            sines,
            source_phi_hat,
            grad_rho,
            gg_positions,
            k_vectors,
            scale,
            ggrad_moments,
            device,
        )
        return

    n_atoms = cosines.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rhok_position_grad_backward_moments_kernel,
        dim=(n_atoms, 4),
        inputs=[
            cosines,
            sines,
            source_phi_hat,
            grad_rho,
            gg_positions,
            k_vectors,
            wp.float64(scale),
            ggrad_moments,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Tile-based GPU implementation of K4
# -----------------------------------------------------------------------------
#
# Substituting the moments ``q(i, lm)`` dependence of
# :func:`_position_gradient_from_rhok_kernel` into K4 reveals that the
# ``(N_atoms, 12)`` intermediate ``T(i, α*4+lm) = cos.T @ β_cos + sin.T @ β_sin``
# is literally the same matrix the pos-grad forward builds.
#
# .. math::
#
#     gg_q(i, lm) = \mathrm{scale} \sum_\alpha gp_\alpha(i) \, T(i, \alpha*4+lm)
#
# So K4's tiled implementation reuses the pos-grad forward's
# precompute + matmul kernels verbatim and only swaps in a different
# per-atom reducer (contracts α against ``gp_α`` instead of contracting
# lm against ``q``).


@wp.kernel
def _rhok_pg_back_moments_reduce_kernel(
    gg_positions: wp.array2d(dtype=wp.float64),  # (N_atoms, 3)
    contribs: wp.array2d(dtype=wp.float64),  # (N_atoms_pad, 16)
    scale: wp.float64,
    n_atoms_valid: wp.int32,
    ggrad_moments: wp.array2d(dtype=wp.float64),  # (N_atoms, 4) OUTPUT
):
    r"""``gg_q(i, lm) = scale · Σ_α gp_α(i) · T(i, α*4+lm)``.

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    gg_positions : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    contribs : wp.array2d, shape (N_atoms_pad, 16), dtype wp.float64
        OUTPUT: per-tile partial contributions awaiting reduction.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    n_atoms_valid : wp.int32
        Number of valid (non-padded) atoms.
    ggrad_moments : wp.array2d, shape (N_atoms, 4), dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. the multipole moments.
    """
    i_idx = wp.tid()
    if i_idx >= n_atoms_valid:
        return

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    for lm in range(4):
        total = (
            gp_x * contribs[i_idx, 0 * 4 + lm]
            + gp_y * contribs[i_idx, 1 * 4 + lm]
            + gp_z * contribs[i_idx, 2 * 4 + lm]
        )
        ggrad_moments[i_idx, lm] = scale * total


def _rhok_pg_back_moments_tiled_launch(
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    grad_rho: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    scale: float,
    ggrad_moments: wp.array,
    device: str,
) -> None:
    r"""CUDA tile-matmul orchestrator for K4 — reuses pos-grad forward's matmul."""
    import torch

    n_k = cosines.shape[0]
    n_atoms = cosines.shape[1]

    tile_i = int(_RHOK_PG_TILE_I)
    tile_k = int(_RHOK_PG_TILE_K)
    tile_j = int(_RHOK_PG_TILE_J)

    n_k_pad = ((n_k + tile_k - 1) // tile_k) * tile_k
    n_atoms_pad = ((n_atoms + tile_i - 1) // tile_i) * tile_i
    n_cols_pad = tile_j  # 16

    # -- Pad + transpose cos / sin. ------------------------------------------
    # cos/sin stay in their native (N_k, N_atoms) layout — the native-layout
    # tile matmul transposes per-tile (wp.tile_transpose), so no (N_atoms, N_k)
    # transpose+zero-pad copy is materialized.
    cos_t = wp.to_torch(cosines)  # for .device only

    big_cos = torch.empty(
        (n_k_pad, n_cols_pad), dtype=torch.float64, device=cos_t.device
    )
    big_sin = torch.empty_like(big_cos)
    contribs = torch.empty(
        (n_atoms_pad, n_cols_pad), dtype=torch.float64, device=cos_t.device
    )

    # -- Launch 1: reuse pos-grad forward's precompute kernel. --------------
    wp.launch(
        _rhok_pos_grad_precompute_big_matrices_kernel,
        dim=n_k_pad,
        inputs=[
            grad_rho,
            source_phi_hat,
            k_vectors,
            wp.int32(n_k),
            wp.from_torch(big_cos, dtype=wp.float64),
            wp.from_torch(big_sin, dtype=wp.float64),
        ],
        device=device,
    )

    # -- Launch 2: reuse pos-grad forward's tile matmul. --------------------
    _launch_cossin_native_matmul(
        cosines,
        sines,
        wp.from_torch(big_cos, dtype=wp.float64),
        wp.from_torch(big_sin, dtype=wp.float64),
        contribs,
        device,
    )

    # -- Launch 3: K4-specific reduce (contracts α, writes per-lm output). --
    wp.launch(
        _rhok_pg_back_moments_reduce_kernel,
        dim=n_atoms_pad,
        inputs=[
            gg_positions,
            wp.from_torch(contribs, dtype=wp.float64),
            wp.float64(scale),
            wp.int32(n_atoms),
            ggrad_moments,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# K5: backward of position_gradient_from_rhok w.r.t. positions
# -----------------------------------------------------------------------------


@wp.kernel
def _rhok_position_grad_backward_positions_kernel(
    charges: wp.array(dtype=Any),
    dipoles: wp.array(dtype=Any),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    source_phi_hat: wp.array3d(dtype=wp.float64),
    grad_rho: wp.array2d(dtype=wp.float64),
    gg_positions: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    scale: wp.float64,
    ggrad_positions: wp.array2d(dtype=wp.float64),
):
    r"""Backward of :func:`position_gradient_from_rhok` w.r.t. atomic positions.

    Position Hessian diagonal-block contribution:

    .. math::

        \tilde g_r(i, \beta) = \mathrm{scale} \sum_k h(k, i) \, k_\beta
            \bigl[ B(k, i) \cos(k \cdot r_i) - A(k, i) \sin(k \cdot r_i) \bigr]

    with :math:`A`, :math:`B` the same combinations the forward kernel
    uses, and :math:`h(k, i) = \sum_\alpha k_\alpha \, gg_{pos}(i, \alpha)`.

    Launch Grid
    -----------
    dim = [N_atoms] — one thread per atom; inner loop over k.


    Parameters
    ----------
    charges : wp.array, dtype Any
        Per-atom monopole charges.
    dipoles : wp.array, dtype Any
        Per-atom Cartesian dipole moments.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array3d, dtype wp.float64
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array2d, dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    gg_positions : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    ggrad_positions : wp.array2d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. positions.
    """
    i_idx = wp.tid()
    n_k = cosines.shape[0]

    q = wp.float64(charges[i_idx])
    mu = dipoles[i_idx]
    mu_x = wp.float64(mu[0])
    mu_y = wp.float64(mu[1])
    mu_z = wp.float64(mu[2])

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(n_k):
        k_vec = k_vectors[k_idx]
        gr = grad_rho[k_idx, 0]
        gi = grad_rho[k_idx, 1]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        pr0 = source_phi_hat[k_idx, 0, 0]
        pi0 = source_phi_hat[k_idx, 0, 1]
        pr1 = source_phi_hat[k_idx, 1, 0]
        pi1 = source_phi_hat[k_idx, 1, 1]
        pr2 = source_phi_hat[k_idx, 2, 0]
        pi2 = source_phi_hat[k_idx, 2, 1]
        pr3 = source_phi_hat[k_idx, 3, 0]
        pi3 = source_phi_hat[k_idx, 3, 1]

        p_r = pr0 * q + pr1 * mu_y + pr2 * mu_z + pr3 * mu_x
        p_i = pi0 * q + pi1 * mu_y + pi2 * mu_z + pi3 * mu_x

        a_k = gr * p_i - gi * p_r
        b_k = -(gr * p_r + gi * p_i)

        h = k_vec[0] * gp_x + k_vec[1] * gp_y + k_vec[2] * gp_z
        weight = h * (b_k * cos_ki - a_k * sin_ki)
        gx += k_vec[0] * weight
        gy += k_vec[1] * weight
        gz += k_vec[2] * weight

    ggrad_positions[i_idx, 0] = scale * gx
    ggrad_positions[i_idx, 1] = scale * gy
    ggrad_positions[i_idx, 2] = scale * gz


def _rhok_pg_back_positions_sig(v, t):
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array3d(dtype=wp.float64),  # source_phi_hat
        wp.array2d(dtype=wp.float64),  # grad_rho
        wp.array2d(dtype=wp.float64),  # gg_positions
        wp.array(dtype=wp.vec3d),  # k_vectors
        wp.float64,  # scale
        wp.array2d(dtype=wp.float64),  # ggrad_positions
    ]


_rhok_position_grad_backward_positions_overloads = register_overloads(
    _rhok_position_grad_backward_positions_kernel, _rhok_pg_back_positions_sig
)


def rhok_position_grad_backward_positions(
    charges: wp.array,
    dipoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    grad_rho: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    scale: float,
    ggrad_positions: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_rhok_position_grad_backward_positions_kernel`.

    Dispatches to the tile-matmul implementation on CUDA and the
    per-atom serial kernel on CPU.


    Parameters
    ----------
    charges : wp.array
        Per-atom monopole charges.
    dipoles : wp.array
        Per-atom Cartesian dipole moments.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    scale : float
        Scalar prefactor applied to the contribution.
    ggrad_positions : wp.array
        OUTPUT: double-backward gradient w.r.t. positions.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    if device is None:
        device = str(cosines.device)

    if "cuda" in str(device):
        _rhok_pg_back_positions_tiled_launch(
            charges,
            dipoles,
            cosines,
            sines,
            source_phi_hat,
            grad_rho,
            gg_positions,
            k_vectors,
            scale,
            ggrad_positions,
            wp_dtype,
            device,
        )
        return

    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    n_atoms = cosines.shape[1]
    wp.launch(
        _rhok_position_grad_backward_positions_overloads[vec_dtype],
        dim=n_atoms,
        inputs=[
            charges,
            dipoles,
            cosines,
            sines,
            source_phi_hat,
            grad_rho,
            gg_positions,
            k_vectors,
            wp.float64(scale),
            ggrad_positions,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Tile-based GPU implementation of K5
# -----------------------------------------------------------------------------
#
# Derivation — factor ``gg_pos[i, β]`` so the inner ``k`` sum becomes a
# dense GEMM:
#
# .. math::
#
#     gg_{pos}[i, \beta] = \mathrm{scale} \sum_{\alpha, lm}
#         g_\alpha(i) \, q(i, lm) \sum_k \bigl[
#             P_\cos(k, \beta, \alpha, lm) \cos(k r_i)
#           - P_\sin(k, \beta, \alpha, lm) \sin(k r_i) \bigr]
#
# where ``g_α(i) = gg_positions[i, α]``, ``q(i, lm)`` is the per-atom
# moment in e3nn layout ``[q, μ_y, μ_z, μ_x]``, and
#
# .. math::
#
#     P_\cos(k, \beta, \alpha, lm) &= k_\alpha k_\beta d_1(k, lm) \\
#     P_\sin(k, \beta, \alpha, lm) &= k_\alpha k_\beta c_1(k, lm) \\
#     c_1(k, lm) &= g^\rho_r \hat\phi_i(k, lm) - g^\rho_i \hat\phi_r(k, lm) \\
#     d_1(k, lm) &= -(g^\rho_r \hat\phi_r + g^\rho_i \hat\phi_i)
#
# Stack the ``(β, α, lm)`` indices into a single packed column
# ``q = β * 12 + α * 4 + lm`` (36 total for l_max=1) and fold the
# ``-`` sign of the ``P_sin`` term into the matrix so a single
# accumulating matmul pair gives us the full intermediate:
#
# .. math::
#
#     T(i, q) = \sum_k \bigl[ \cos(k, i) P_\cos(k, q) + \sin(k, i) (-P_\sin(k, q)) \bigr]
#
# Final per-atom reduction closes the ``(α, lm)`` axis:
#
# .. math::
#
#     gg_{pos}[i, \beta] = \mathrm{scale} \sum_p G(i, p) \, T(i, \beta*12 + p)
#
# with ``G(i, p) = g_{α(p)}(i) · q(i, lm(p))``.

_RPGP_TILE_I = wp.constant(8)
_RPGP_TILE_K = wp.constant(32)
_RPGP_TILE_J = wp.constant(16)  # 36 packed cols pad → 48 (3 j-blocks)
_RPGP_BLOCK_DIM = 128


@wp.kernel
def _rhok_pg_back_positions_precompute_kernel(
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    source_phi_hat: wp.array3d(dtype=wp.float64),  # (N_k, 4, 2)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    n_k_valid: wp.int32,
    beta_cos: wp.array2d(dtype=wp.float64),  # (N_k_pad, 48) OUTPUT
    beta_sin: wp.array2d(dtype=wp.float64),  # (N_k_pad, 48) OUTPUT
):
    r"""Precompute ``P_cos`` / ``P_sin`` packed into 48-col rows (36 real, 12 padded).

    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx)``; each thread processes one k-vector index.

    Parameters
    ----------
    grad_rho : wp.array2d, shape (N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    source_phi_hat : wp.array3d, shape (N_k, 4, 2), dtype wp.float64
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-space k-vectors.
    n_k_valid : wp.int32
        Number of valid (non-padded) k-vectors.
    beta_cos : wp.array2d, shape (N_k_pad, 48), dtype wp.float64
        OUTPUT: precomputed per-k cosine-weighted reduction coefficients.
    beta_sin : wp.array2d, shape (N_k_pad, 48), dtype wp.float64
        OUTPUT: precomputed per-k sine-weighted reduction coefficients.
    """
    k_idx = wp.tid()

    if k_idx >= n_k_valid:
        for c in range(48):
            beta_cos[k_idx, c] = wp.float64(0.0)
            beta_sin[k_idx, c] = wp.float64(0.0)
        return

    gr = grad_rho[k_idx, 0]
    gi = grad_rho[k_idx, 1]
    k_vec = k_vectors[k_idx]

    # c_lm and d_lm for lm ∈ {0, 1, 2, 3}.
    for lm in range(4):
        phi_r = source_phi_hat[k_idx, lm, 0]
        phi_i = source_phi_hat[k_idx, lm, 1]
        c_lm = gr * phi_i - gi * phi_r
        d_lm = -(gr * phi_r + gi * phi_i)

        # For each β ∈ {0, 1, 2} and α ∈ {0, 1, 2}:
        for beta in range(3):
            k_b = k_vec[beta]
            for alpha in range(3):
                k_a = k_vec[alpha]
                # Packed column: q = β * 12 + α * 4 + lm.
                col = beta * 12 + alpha * 4 + lm
                beta_cos[k_idx, col] = k_a * k_b * d_lm
                beta_sin[k_idx, col] = -k_a * k_b * c_lm  # fold minus sign

    # Zero the 12 pad columns [36, 48).
    for c in range(36, 48):
        beta_cos[k_idx, c] = wp.float64(0.0)
        beta_sin[k_idx, c] = wp.float64(0.0)


# _rhok_pg_back_positions_tiled_matmul_kernel → replaced by shared
# :func:`_cossin_native_matmul_kernel`.


@wp.kernel
def _rhok_pg_back_positions_reduce_kernel(
    charges: wp.array(dtype=Any),
    dipoles: wp.array(dtype=Any),
    gg_positions: wp.array2d(dtype=wp.float64),  # (N_atoms, 3)
    contribs: wp.array2d(dtype=wp.float64),  # (N_atoms_pad, 48)
    scale: wp.float64,
    n_atoms_valid: wp.int32,
    ggrad_positions: wp.array2d(dtype=wp.float64),  # (N_atoms, 3) OUTPUT
):
    r"""Per-atom reduce: ``gg_pos[i, β] = scale · Σ_{α,lm} gp_α(i) · q(i, lm) · T(i, β*12 + α*4 + lm)``.

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    charges : wp.array, dtype Any
        Per-atom monopole charges.
    dipoles : wp.array, dtype Any
        Per-atom Cartesian dipole moments.
    gg_positions : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    contribs : wp.array2d, shape (N_atoms_pad, 48), dtype wp.float64
        OUTPUT: per-tile partial contributions awaiting reduction.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    n_atoms_valid : wp.int32
        Number of valid (non-padded) atoms.
    ggrad_positions : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. positions.
    """
    i_idx = wp.tid()
    if i_idx >= n_atoms_valid:
        return

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]
    q_val = wp.float64(charges[i_idx])
    mu = dipoles[i_idx]
    mu_x = wp.float64(mu[0])
    mu_y = wp.float64(mu[1])
    mu_z = wp.float64(mu[2])

    for beta in range(3):
        base = beta * 12
        total = wp.float64(0.0)
        # lm layout: 0 → charge, 1 → μ_y, 2 → μ_z, 3 → μ_x
        # α layout: 0 → x, 1 → y, 2 → z
        # col(β, α, lm) = β*12 + α*4 + lm
        for alpha in range(3):
            if alpha == 0:
                gp_a = gp_x
            else:
                if alpha == 1:
                    gp_a = gp_y
                else:
                    gp_a = gp_z
            col_base = base + alpha * 4
            total += gp_a * (
                q_val * contribs[i_idx, col_base + 0]
                + mu_y * contribs[i_idx, col_base + 1]
                + mu_z * contribs[i_idx, col_base + 2]
                + mu_x * contribs[i_idx, col_base + 3]
            )
        ggrad_positions[i_idx, beta] = scale * total


def _rhok_pg_back_positions_reduce_sig(v, t):
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array2d(dtype=wp.float64),  # gg_positions
        wp.array2d(dtype=wp.float64),  # contribs
        wp.float64,  # scale
        wp.int32,  # n_atoms_valid
        wp.array2d(dtype=wp.float64),  # ggrad_positions
    ]


_rhok_pg_back_positions_reduce_overloads = register_overloads(
    _rhok_pg_back_positions_reduce_kernel, _rhok_pg_back_positions_reduce_sig
)


def _rhok_pg_back_positions_tiled_launch(
    charges: wp.array,
    dipoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    grad_rho: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    scale: float,
    ggrad_positions: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    r"""CUDA tile-matmul orchestrator for K5."""
    import torch

    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f

    n_k = cosines.shape[0]
    n_atoms = cosines.shape[1]

    tile_i = int(_RPGP_TILE_I)
    tile_k = int(_RPGP_TILE_K)
    tile_j = int(_RPGP_TILE_J)

    n_k_pad = ((n_k + tile_k - 1) // tile_k) * tile_k
    n_atoms_pad = ((n_atoms + tile_i - 1) // tile_i) * tile_i
    # 36 real cols pad to multiple of tile_j=16 → 48.
    n_cols_pad = ((36 + tile_j - 1) // tile_j) * tile_j

    # -- Pad + transpose cos / sin. ------------------------------------------
    # cos/sin stay in their native (N_k, N_atoms) layout — the native-layout
    # tile matmul transposes per-tile (wp.tile_transpose), so no (N_atoms, N_k)
    # transpose+zero-pad copy is materialized.
    cos_t = wp.to_torch(cosines)  # for .device only

    beta_cos = torch.empty(
        (n_k_pad, n_cols_pad), dtype=torch.float64, device=cos_t.device
    )
    beta_sin = torch.empty_like(beta_cos)
    contribs = torch.empty(
        (n_atoms_pad, n_cols_pad), dtype=torch.float64, device=cos_t.device
    )

    # -- Launch 1: precompute β_cos / β_sin. --------------------------------
    wp.launch(
        _rhok_pg_back_positions_precompute_kernel,
        dim=n_k_pad,
        inputs=[
            grad_rho,
            source_phi_hat,
            k_vectors,
            wp.int32(n_k),
            wp.from_torch(beta_cos, dtype=wp.float64),
            wp.from_torch(beta_sin, dtype=wp.float64),
        ],
        device=device,
    )

    # -- Launch 2: tile matmul. ----------------------------------------------
    _launch_cossin_native_matmul(
        cosines,
        sines,
        wp.from_torch(beta_cos, dtype=wp.float64),
        wp.from_torch(beta_sin, dtype=wp.float64),
        contribs,
        device,
    )

    # -- Launch 3: per-atom reduce. ------------------------------------------
    wp.launch(
        _rhok_pg_back_positions_reduce_overloads[vec_dtype],
        dim=n_atoms_pad,
        inputs=[
            charges,
            dipoles,
            gg_positions,
            wp.from_torch(contribs, dtype=wp.float64),
            wp.float64(scale),
            wp.int32(n_atoms),
            ggrad_positions,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# K6: backward of position_gradient_from_feature_grad w.r.t. grad_raw
# -----------------------------------------------------------------------------


@wp.kernel
def _feat_position_grad_backward_grad_raw_kernel(
    receiver_phi_hat: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    potential: wp.array2d(dtype=wp.float64),
    gg_positions: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    ggrad_grad_raw: wp.array3d(dtype=wp.float64),
):
    r"""Backward of :func:`position_gradient_from_feature_grad` w.r.t. ``grad_raw``.

    .. math::

        \tilde g_{raw}(i, \sigma, lm) = \mathrm{scale} \sum_k
            w(k) h(k, i) \bigl[
                dC(k, \sigma, lm) \cos(k \cdot r_i)
              - dD(k, \sigma, lm) \sin(k \cdot r_i)
            \bigr]

    where :math:`dC = V_r \hat\phi_i - V_i \hat\phi_r`,
    :math:`dD = V_r \hat\phi_r + V_i \hat\phi_i`, :math:`w` =
    ``k_factor_proj``, and :math:`h(k, i) = \sum_\alpha k_\alpha \,
    gg_{pos}(i, \alpha)`. The ``scale`` prefactor ``2/(2π)³`` is baked
    in as the module-level constant :data:`_INV_TWO_PI_CUBED_TIMES_TWO`.

    Launch Grid
    -----------
    dim = [N_atoms, N_σ, 4].


    Parameters
    ----------
    receiver_phi_hat : wp.array4d, dtype wp.float64
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array2d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    ggrad_grad_raw : wp.array3d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. ``grad_raw``.
    """
    i_idx, s_idx, lm_idx = wp.tid()
    n_k = cosines.shape[0]

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    acc = wp.float64(0.0)
    for k_idx in range(n_k):
        k_vec = k_vectors[k_idx]
        h = k_vec[0] * gp_x + k_vec[1] * gp_y + k_vec[2] * gp_z
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        v_r = potential[k_idx, 0]
        v_i = potential[k_idx, 1]
        phi_r = receiver_phi_hat[k_idx, s_idx, lm_idx, 0]
        phi_i = receiver_phi_hat[k_idx, s_idx, lm_idx, 1]
        kfp = k_factor_proj[k_idx]

        dc = v_r * phi_i - v_i * phi_r
        dd = v_r * phi_r + v_i * phi_i
        acc += kfp * h * (dc * cos_ki - dd * sin_ki)

    ggrad_grad_raw[i_idx, s_idx, lm_idx] = _INV_TWO_PI_CUBED_TIMES_TWO * acc


def feat_position_grad_backward_grad_raw(
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    ggrad_grad_raw: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_feat_position_grad_backward_grad_raw_kernel`.

    Dispatches to the tile-matmul implementation on CUDA and the
    serial per-(i, σ, lm) kernel on CPU.


    Parameters
    ----------
    receiver_phi_hat : wp.array
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    ggrad_grad_raw : wp.array
        OUTPUT: double-backward gradient w.r.t. ``grad_raw``.
    device : str
        Warp device string; defaults to the input array's device.
    """
    if device is None:
        device = str(cosines.device)

    if "cuda" in str(device):
        _feat_pg_back_grad_raw_tiled_launch(
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            potential,
            gg_positions,
            k_vectors,
            ggrad_grad_raw,
            device,
        )
        return

    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]
    wp.launch(
        _feat_position_grad_backward_grad_raw_kernel,
        dim=(n_atoms, n_sigma, 4),
        inputs=[
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            potential,
            gg_positions,
            k_vectors,
            ggrad_grad_raw,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Tile-based GPU implementation of K6
# -----------------------------------------------------------------------------
#
# Math:
#
# .. math::
#
#     \mathrm{gg\_grad\_raw}(i, \sigma, lm)
#         = \mathrm{scale} \sum_\alpha gp_\alpha(i) \sum_k \bigl[
#             M_\cos(k, \alpha, \sigma, lm) \cos(k r_i)
#           + M_\sin(k, \alpha, \sigma, lm) \sin(k r_i) \bigr]
#
# with
#
# .. math::
#
#     M_\cos(k, \alpha, sl) &= w(k) k_\alpha [V_r \hat\phi_i - V_i \hat\phi_r](k, sl) \\
#     M_\sin(k, \alpha, sl) &= -w(k) k_\alpha [V_r \hat\phi_r + V_i \hat\phi_i](k, sl)
#
# Packed cols ``p = α * (N_σ · 4) + σ · 4 + lm`` — 12 for ``N_σ=1``.
# Final reduce closes the α axis against ``gp_α(i)``.

_FPGR_TILE_I = wp.constant(8)
_FPGR_TILE_K = wp.constant(32)
_FPGR_TILE_J = wp.constant(16)
_FPGR_BLOCK_DIM = 128


@wp.kernel
def _feat_pg_back_grad_raw_precompute_kernel(
    receiver_phi_hat: wp.array4d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    potential: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    n_k_valid: wp.int32,
    n_p_valid: wp.int32,  # 3 * N_σ * 4
    m_cos: wp.array2d(dtype=wp.float64),
    m_sin: wp.array2d(dtype=wp.float64),
):
    r"""Precompute M_cos / M_sin packed ``p = α * (N_σ * 4) + σ * 4 + lm``.

    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx, s_idx)``; each thread processes one (k-vector, s_idx) work item.

    Parameters
    ----------
    receiver_phi_hat : wp.array4d, dtype wp.float64
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array2d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    n_k_valid : wp.int32
        Number of valid (non-padded) k-vectors.
    n_p_valid : wp.int32
        3 * N_σ * 4.
    m_cos : wp.array2d, dtype wp.float64
        OUTPUT: precomputed per-k cosine-weighted intermediate matrix.
    m_sin : wp.array2d, dtype wp.float64
        OUTPUT: precomputed per-k sine-weighted intermediate matrix.
    """
    k_idx, s_idx = wp.tid()

    if k_idx >= n_k_valid:
        # Clear 3 * 4 = 12 cols this σ thread owns across β=α axis.
        n_sigma = receiver_phi_hat.shape[1]
        sl_per_alpha = n_sigma * 4
        for alpha in range(3):
            for lm in range(4):
                col = alpha * sl_per_alpha + s_idx * 4 + lm
                if col < m_cos.shape[1]:
                    m_cos[k_idx, col] = wp.float64(0.0)
                    m_sin[k_idx, col] = wp.float64(0.0)
        return

    v_r = potential[k_idx, 0]
    v_i = potential[k_idx, 1]
    kfp = k_factor_proj[k_idx]
    k_vec = k_vectors[k_idx]

    n_sigma = receiver_phi_hat.shape[1]
    sl_per_alpha = n_sigma * 4

    for lm in range(4):
        phi_r = receiver_phi_hat[k_idx, s_idx, lm, 0]
        phi_i = receiver_phi_hat[k_idx, s_idx, lm, 1]
        part_cos = v_r * phi_i - v_i * phi_r
        part_sin = v_r * phi_r + v_i * phi_i

        for alpha in range(3):
            k_a = k_vec[alpha]
            col = alpha * sl_per_alpha + s_idx * 4 + lm
            if col >= m_cos.shape[1]:
                continue
            if col >= n_p_valid:
                m_cos[k_idx, col] = wp.float64(0.0)
                m_sin[k_idx, col] = wp.float64(0.0)
            else:
                m_cos[k_idx, col] = kfp * k_a * part_cos
                m_sin[k_idx, col] = -kfp * k_a * part_sin


# _feat_pg_back_grad_raw_tiled_matmul_kernel → replaced by shared
# :func:`_cossin_native_matmul_kernel`.


@wp.kernel
def _feat_pg_back_grad_raw_reduce_kernel(
    gg_positions: wp.array2d(dtype=wp.float64),
    contribs: wp.array2d(dtype=wp.float64),
    n_atoms_valid: wp.int32,
    n_sigma: wp.int32,
    ggrad_grad_raw: wp.array3d(dtype=wp.float64),
):
    r"""``gg_grad_raw(i, σ, lm) = scale · Σ_α gp_α(i) · T(i, α * (N_σ*4) + σ*4 + lm)``.

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx, s_idx, lm_idx)``; each thread processes one (i_idx, s_idx, lm_idx) work item.

    Parameters
    ----------
    gg_positions : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    contribs : wp.array2d, dtype wp.float64
        OUTPUT: per-tile partial contributions awaiting reduction.
    n_atoms_valid : wp.int32
        Number of valid (non-padded) atoms.
    n_sigma : wp.int32
        Number of GTO width channels.
    ggrad_grad_raw : wp.array3d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. ``grad_raw``.
    """
    i_idx, s_idx, lm_idx = wp.tid()
    if i_idx >= n_atoms_valid:
        return

    sl_per_alpha = n_sigma * 4
    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    total = (
        gp_x * contribs[i_idx, 0 * sl_per_alpha + s_idx * 4 + lm_idx]
        + gp_y * contribs[i_idx, 1 * sl_per_alpha + s_idx * 4 + lm_idx]
        + gp_z * contribs[i_idx, 2 * sl_per_alpha + s_idx * 4 + lm_idx]
    )
    ggrad_grad_raw[i_idx, s_idx, lm_idx] = _INV_TWO_PI_CUBED_TIMES_TWO * total


def _feat_pg_back_grad_raw_tiled_launch(
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    ggrad_grad_raw: wp.array,
    device: str,
) -> None:
    r"""CUDA tile-matmul orchestrator for K6."""
    import torch

    n_k = cosines.shape[0]
    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]
    n_p = 3 * n_sigma * 4  # 12 for N_σ=1

    tile_i = int(_FPGR_TILE_I)
    tile_k = int(_FPGR_TILE_K)
    tile_j = int(_FPGR_TILE_J)

    n_k_pad = ((n_k + tile_k - 1) // tile_k) * tile_k
    n_atoms_pad = ((n_atoms + tile_i - 1) // tile_i) * tile_i
    n_p_pad = ((n_p + tile_j - 1) // tile_j) * tile_j

    # cos/sin stay in their native (N_k, N_atoms) layout — the native-layout
    # tile matmul transposes per-tile (wp.tile_transpose), so no (N_atoms, N_k)
    # transpose+zero-pad copy is materialized.
    cos_t = wp.to_torch(cosines)  # for .device only

    m_cos = torch.empty((n_k_pad, n_p_pad), dtype=torch.float64, device=cos_t.device)
    m_sin = torch.empty_like(m_cos)
    contribs = torch.empty(
        (n_atoms_pad, n_p_pad), dtype=torch.float64, device=cos_t.device
    )

    wp.launch(
        _feat_pg_back_grad_raw_precompute_kernel,
        dim=(n_k_pad, n_sigma),
        inputs=[
            receiver_phi_hat,
            k_factor_proj,
            potential,
            k_vectors,
            wp.int32(n_k),
            wp.int32(n_p),
            wp.from_torch(m_cos, dtype=wp.float64),
            wp.from_torch(m_sin, dtype=wp.float64),
        ],
        device=device,
    )

    _launch_cossin_native_matmul(
        cosines,
        sines,
        wp.from_torch(m_cos, dtype=wp.float64),
        wp.from_torch(m_sin, dtype=wp.float64),
        contribs,
        device,
    )

    wp.launch(
        _feat_pg_back_grad_raw_reduce_kernel,
        dim=(n_atoms, n_sigma, 4),
        inputs=[
            gg_positions,
            wp.from_torch(contribs, dtype=wp.float64),
            wp.int32(n_atoms),
            wp.int32(n_sigma),
            ggrad_grad_raw,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# K7: backward of position_gradient_from_feature_grad w.r.t. V(k)
# -----------------------------------------------------------------------------


@wp.kernel
def _feat_position_grad_backward_v_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    gg_positions: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    ggrad_v: wp.array2d(dtype=wp.float64),
):
    r"""Backward of :func:`position_gradient_from_feature_grad` w.r.t. ``V(k)``.

    Per-k thread; inner sum over atoms and (σ, lm) to rebuild the Q-tuple:

    .. math::

        \tilde g_{V_r}(k) &= \mathrm{scale} \, w(k) \sum_i h(k, i)
            [Q_i(k, i) \cos(k \cdot r_i) - Q_r(k, i) \sin(k \cdot r_i)] \\
        \tilde g_{V_i}(k) &= \mathrm{scale} \, w(k) \sum_i h(k, i)
            [-Q_r(k, i) \cos(k \cdot r_i) - Q_i(k, i) \sin(k \cdot r_i)]

    Launch Grid
    -----------
    dim = [N_k].


    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array4d, dtype wp.float64
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    gg_positions : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    ggrad_v : wp.array2d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. ``v``.
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]

    k_vec = k_vectors[k_idx]
    kfp = k_factor_proj[k_idx]

    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)

    for i_idx in range(n_atoms):
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(4):
                g = grad_raw[i_idx, s_idx, lm_idx]
                q_r += receiver_phi_hat[k_idx, s_idx, lm_idx, 0] * g
                q_i += receiver_phi_hat[k_idx, s_idx, lm_idx, 1] * g

        h = (
            k_vec[0] * gg_positions[i_idx, 0]
            + k_vec[1] * gg_positions[i_idx, 1]
            + k_vec[2] * gg_positions[i_idx, 2]
        )
        acc_r += h * (q_i * cos_ki - q_r * sin_ki)
        acc_i += h * (-q_r * cos_ki - q_i * sin_ki)

    ggrad_v[k_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_r
    ggrad_v[k_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_i


def feat_position_grad_backward_v(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    ggrad_v: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_feat_position_grad_backward_v_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    ggrad_v : wp.array
        OUTPUT: double-backward gradient w.r.t. ``v``.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_k = receiver_phi_hat.shape[0]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _feat_position_grad_backward_v_kernel,
        dim=n_k,
        inputs=[
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            gg_positions,
            k_vectors,
            ggrad_v,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# K8: backward of position_gradient_from_feature_grad w.r.t. positions
# -----------------------------------------------------------------------------


@wp.kernel
def _feat_position_grad_backward_positions_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    potential: wp.array2d(dtype=wp.float64),
    gg_positions: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    ggrad_positions: wp.array2d(dtype=wp.float64),
):
    r"""Backward of :func:`position_gradient_from_feature_grad` w.r.t. positions.

    .. math::

        \tilde g_{r_i, \beta} = -\mathrm{scale} \sum_k w(k) \, h(k, i)
            \, k_\beta \bigl[
                C(k, i) \sin(k \cdot r_i) + D(k, i) \cos(k \cdot r_i) \bigr]

    with the same :math:`C, D` as the forward
    :func:`position_gradient_from_feature_grad` (``C = V_r Q_i - V_i Q_r``,
    ``D = V_r Q_r + V_i Q_i``) and :math:`h(k, i) = \sum_\alpha k_\alpha \,
    gg_{pos}(i, \alpha)`.

    Launch Grid
    -----------
    dim = [N_atoms]; output shape (N_atoms, 3).


    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array4d, dtype wp.float64
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array2d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    ggrad_positions : wp.array2d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. positions.
    """
    i_idx = wp.tid()
    n_k = cosines.shape[0]
    n_sigma = receiver_phi_hat.shape[1]

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(n_k):
        k_vec = k_vectors[k_idx]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        v_r = potential[k_idx, 0]
        v_i = potential[k_idx, 1]
        kfp = k_factor_proj[k_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(4):
                g = grad_raw[i_idx, s_idx, lm_idx]
                q_r += receiver_phi_hat[k_idx, s_idx, lm_idx, 0] * g
                q_i += receiver_phi_hat[k_idx, s_idx, lm_idx, 1] * g

        c_k = v_r * q_i - v_i * q_r
        d_k = v_r * q_r + v_i * q_i

        h = k_vec[0] * gp_x + k_vec[1] * gp_y + k_vec[2] * gp_z
        weight = -kfp * h * (c_k * sin_ki + d_k * cos_ki)
        gx += k_vec[0] * weight
        gy += k_vec[1] * weight
        gz += k_vec[2] * weight

    ggrad_positions[i_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * gx
    ggrad_positions[i_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * gy
    ggrad_positions[i_idx, 2] = _INV_TWO_PI_CUBED_TIMES_TWO * gz


def feat_position_grad_backward_positions(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    ggrad_positions: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_feat_position_grad_backward_positions_kernel`.

    Dispatches to the tile-matmul implementation on CUDA and the
    per-atom serial kernel on CPU.


    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    ggrad_positions : wp.array
        OUTPUT: double-backward gradient w.r.t. positions.
    device : str
        Warp device string; defaults to the input array's device.
    """
    if device is None:
        device = str(cosines.device)

    if "cuda" in str(device):
        _feat_pg_back_positions_tiled_launch(
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            potential,
            gg_positions,
            k_vectors,
            ggrad_positions,
            device,
        )
        return

    n_atoms = cosines.shape[1]
    wp.launch(
        _feat_position_grad_backward_positions_kernel,
        dim=n_atoms,
        inputs=[
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            potential,
            gg_positions,
            k_vectors,
            ggrad_positions,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Tile-based GPU implementation of K8
# -----------------------------------------------------------------------------
#
# Same (β, α, sl) packing as K5, sign flip to fold the leading ``-`` into
# the precomputed matrices so the matmul accumulates. Math:
#
# .. math::
#
#     gg_{pos}[i, \beta] = \mathrm{scale} \sum_{\alpha, sl}
#         gp_\alpha(i) \, \mathrm{grad\_raw}(i, sl) \sum_k \bigl[
#             M_\cos(k, \beta, \alpha, sl) \cos(k r_i)
#           + M_\sin(k, \beta, \alpha, sl) \sin(k r_i) \bigr]
#
# with
#
# .. math::
#
#     M_\cos(k, \beta, \alpha, sl) &= -w(k) \, k_\alpha k_\beta \,
#         [V_r \hat\phi_r(k, sl) + V_i \hat\phi_i(k, sl)] \\
#     M_\sin(k, \beta, \alpha, sl) &= -w(k) \, k_\alpha k_\beta \,
#         [V_r \hat\phi_i(k, sl) - V_i \hat\phi_r(k, sl)]
#
# For ``N_σ = 1`` the intermediate is ``(N_atoms, 3·3·4 = 36)`` padded
# to ``(N_atoms, 48)``. For larger ``N_σ`` the col count grows as
# ``3·3·N_σ·4``; we assume ``N_σ ≤ 5`` so 5·12·3 = 180 cols ≤ the
# budget the tile matmul kernel can handle; if ``N_σ`` ever grows much
# past that we'd split along ``β`` too.

_FPGP_TILE_I = wp.constant(8)
_FPGP_TILE_K = wp.constant(32)
_FPGP_TILE_J = wp.constant(16)
_FPGP_BLOCK_DIM = 128


@wp.kernel
def _feat_pg_back_positions_precompute_kernel(
    receiver_phi_hat: wp.array4d(dtype=wp.float64),  # (N_k, N_σ, 4, 2)
    k_factor_proj: wp.array(dtype=wp.float64),  # (N_k,)
    potential: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    n_k_valid: wp.int32,
    n_p_valid: wp.int32,  # = 3 * 3 * N_σ * 4 = 36 for N_σ=1
    m_cos: wp.array2d(dtype=wp.float64),  # (N_k_pad, N_p_pad) OUTPUT
    m_sin: wp.array2d(dtype=wp.float64),
):
    r"""Precompute M_cos / M_sin for K8.

    Launch dim ``(N_k_pad, N_σ)`` — each thread handles one ``(k, σ)``
    pair and writes the ``3·3·4 = 36`` cols for that σ block.


    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx, s_idx)``; each thread processes one (k-vector, s_idx) work item.

    Parameters
    ----------
    receiver_phi_hat : wp.array4d, shape (N_k, N_σ, 4, 2), dtype wp.float64
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    k_factor_proj : wp.array, shape (N_k,), dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array2d, shape (N_k, 2), dtype wp.float64
        Per-k reciprocal-space potential factor.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-space k-vectors.
    n_k_valid : wp.int32
        Number of valid (non-padded) k-vectors.
    n_p_valid : wp.int32
        = 3 * 3 * N_σ * 4 = 36 for N_σ=1.
    m_cos : wp.array2d, shape (N_k_pad, N_p_pad), dtype wp.float64
        OUTPUT: precomputed per-k cosine-weighted intermediate matrix.
    m_sin : wp.array2d, dtype wp.float64
        OUTPUT: precomputed per-k sine-weighted intermediate matrix.
    """
    k_idx, s_idx = wp.tid()

    if k_idx >= n_k_valid:
        # Pad k-row: clear the σ's owned columns.
        base_s = s_idx * 36  # each σ owns 3·3·4 = 36 cols
        for c in range(36):
            col = base_s + c
            if col < m_cos.shape[1]:
                m_cos[k_idx, col] = wp.float64(0.0)
                m_sin[k_idx, col] = wp.float64(0.0)
        return

    v_r = potential[k_idx, 0]
    v_i = potential[k_idx, 1]
    kfp = k_factor_proj[k_idx]
    k_vec = k_vectors[k_idx]
    # Column layout within this σ block: q = β*12 + α*4 + lm.
    # Global col = s_idx * 36 + q (when packing σ blocks contiguously).
    base_s = s_idx * 36

    for lm in range(4):
        phi_r = receiver_phi_hat[k_idx, s_idx, lm, 0]
        phi_i = receiver_phi_hat[k_idx, s_idx, lm, 1]
        # α1, α2 from the V-dependent combinations:
        alpha1 = v_r * phi_i - v_i * phi_r
        alpha2 = v_r * phi_r + v_i * phi_i

        for beta in range(3):
            k_b = k_vec[beta]
            for alpha in range(3):
                k_a = k_vec[alpha]
                col = base_s + beta * 12 + alpha * 4 + lm
                if col >= m_cos.shape[1]:
                    continue
                if col >= n_p_valid:
                    m_cos[k_idx, col] = wp.float64(0.0)
                    m_sin[k_idx, col] = wp.float64(0.0)
                else:
                    factor = kfp * k_a * k_b
                    m_cos[k_idx, col] = -factor * alpha2
                    m_sin[k_idx, col] = -factor * alpha1


# _feat_pg_back_positions_tiled_matmul_kernel → replaced by shared
# :func:`_cossin_native_matmul_kernel`.


@wp.kernel
def _feat_pg_back_positions_reduce_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),  # (N_atoms, N_σ, 4)
    gg_positions: wp.array2d(dtype=wp.float64),  # (N_atoms, 3)
    contribs: wp.array2d(dtype=wp.float64),  # (N_atoms_pad, N_p_pad)
    n_atoms_valid: wp.int32,
    ggrad_positions: wp.array2d(dtype=wp.float64),  # (N_atoms, 3) OUTPUT
):
    r"""``gg_pos[i, β] = scale · Σ_{α,σ,lm} gp_α(i) · grad_raw(i,σ,lm) · T(i, σ*36 + β*12 + α*4 + lm)``.

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_atoms, N_σ, 4), dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    gg_positions : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    contribs : wp.array2d, shape (N_atoms_pad, N_p_pad), dtype wp.float64
        OUTPUT: per-tile partial contributions awaiting reduction.
    n_atoms_valid : wp.int32
        Number of valid (non-padded) atoms.
    ggrad_positions : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. positions.
    """
    i_idx = wp.tid()
    if i_idx >= n_atoms_valid:
        return

    n_sigma = grad_raw.shape[1]

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    for beta in range(3):
        total = wp.float64(0.0)
        for s_idx in range(n_sigma):
            base_s = s_idx * 36
            for alpha in range(3):
                if alpha == 0:
                    gp_a = gp_x
                else:
                    if alpha == 1:
                        gp_a = gp_y
                    else:
                        gp_a = gp_z
                col_base = base_s + beta * 12 + alpha * 4
                inner = wp.float64(0.0)
                for lm in range(4):
                    inner += grad_raw[i_idx, s_idx, lm] * contribs[i_idx, col_base + lm]
                total += gp_a * inner
        ggrad_positions[i_idx, beta] = _INV_TWO_PI_CUBED_TIMES_TWO * total


def _feat_pg_back_positions_tiled_launch(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    ggrad_positions: wp.array,
    device: str,
) -> None:
    r"""CUDA tile-matmul orchestrator for K8."""
    import torch

    n_k = cosines.shape[0]
    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]
    n_p = n_sigma * 36  # σ blocks × (3·3·4)

    tile_i = int(_FPGP_TILE_I)
    tile_k = int(_FPGP_TILE_K)
    tile_j = int(_FPGP_TILE_J)

    n_k_pad = ((n_k + tile_k - 1) // tile_k) * tile_k
    n_atoms_pad = ((n_atoms + tile_i - 1) // tile_i) * tile_i
    n_p_pad = ((n_p + tile_j - 1) // tile_j) * tile_j

    # cos/sin stay in their native (N_k, N_atoms) layout — the native-layout
    # tile matmul transposes per-tile (wp.tile_transpose), so no (N_atoms, N_k)
    # transpose+zero-pad copy is materialized.
    cos_t = wp.to_torch(cosines)  # for .device only

    m_cos = torch.empty((n_k_pad, n_p_pad), dtype=torch.float64, device=cos_t.device)
    m_sin = torch.empty_like(m_cos)
    contribs = torch.empty(
        (n_atoms_pad, n_p_pad), dtype=torch.float64, device=cos_t.device
    )

    wp.launch(
        _feat_pg_back_positions_precompute_kernel,
        dim=(n_k_pad, n_sigma),
        inputs=[
            receiver_phi_hat,
            k_factor_proj,
            potential,
            k_vectors,
            wp.int32(n_k),
            wp.int32(n_p),
            wp.from_torch(m_cos, dtype=wp.float64),
            wp.from_torch(m_sin, dtype=wp.float64),
        ],
        device=device,
    )

    _launch_cossin_native_matmul(
        cosines,
        sines,
        wp.from_torch(m_cos, dtype=wp.float64),
        wp.from_torch(m_sin, dtype=wp.float64),
        contribs,
        device,
    )

    wp.launch(
        _feat_pg_back_positions_reduce_kernel,
        dim=n_atoms_pad,
        inputs=[
            grad_raw,
            gg_positions,
            wp.from_torch(contribs, dtype=wp.float64),
            wp.int32(n_atoms),
            ggrad_positions,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# K9: backward of v_gradient_from_feature_grad w.r.t. positions
# -----------------------------------------------------------------------------


@wp.kernel
def _v_grad_from_feat_grad_backward_positions_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    gg_v: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    ggrad_positions: wp.array2d(dtype=wp.float64),
):
    r"""Backward of :func:`v_gradient_from_feature_grad` w.r.t. atomic positions.

    .. math::

        \tilde g_{r_i, \beta} = \mathrm{scale} \sum_k w(k) \, k_\beta \bigl\{
            \cos(k \cdot r_i) [gg_{V_r} Q_i - gg_{V_i} Q_r]
          - \sin(k \cdot r_i) [gg_{V_r} Q_r + gg_{V_i} Q_i] \bigr\}

    One thread per atom; inner loop over k, inner-inner over ``(σ, lm)``.

    Launch Grid
    -----------
    dim = [N_atoms]; output shape (N_atoms, 3).


    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array4d, dtype wp.float64
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    gg_v : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. ``v`` (HVP seed).
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    ggrad_positions : wp.array2d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. positions.
    """
    i_idx = wp.tid()
    n_k = cosines.shape[0]
    n_sigma = receiver_phi_hat.shape[1]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(n_k):
        k_vec = k_vectors[k_idx]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        kfp = k_factor_proj[k_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(4):
                g = grad_raw[i_idx, s_idx, lm_idx]
                q_r += receiver_phi_hat[k_idx, s_idx, lm_idx, 0] * g
                q_i += receiver_phi_hat[k_idx, s_idx, lm_idx, 1] * g

        gvr = gg_v[k_idx, 0]
        gvi = gg_v[k_idx, 1]

        cos_term = gvr * q_i - gvi * q_r
        sin_term = gvr * q_r + gvi * q_i

        weight = kfp * (cos_ki * cos_term - sin_ki * sin_term)
        gx += k_vec[0] * weight
        gy += k_vec[1] * weight
        gz += k_vec[2] * weight

    ggrad_positions[i_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * gx
    ggrad_positions[i_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * gy
    ggrad_positions[i_idx, 2] = _INV_TWO_PI_CUBED_TIMES_TWO * gz


def v_grad_from_feat_grad_backward_positions(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    gg_v: wp.array,
    k_vectors: wp.array,
    ggrad_positions: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_v_grad_from_feat_grad_backward_positions_kernel`.

    Dispatches to the tile-matmul implementation on CUDA and the
    per-atom serial kernel on CPU.


    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    gg_v : wp.array
        Second-order upstream gradient w.r.t. ``v`` (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    ggrad_positions : wp.array
        OUTPUT: double-backward gradient w.r.t. positions.
    device : str
        Warp device string; defaults to the input array's device.
    """
    if device is None:
        device = str(cosines.device)

    if "cuda" in str(device):
        _v_grad_back_positions_tiled_launch(
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            gg_v,
            k_vectors,
            ggrad_positions,
            device,
        )
        return

    n_atoms = cosines.shape[1]
    wp.launch(
        _v_grad_from_feat_grad_backward_positions_kernel,
        dim=n_atoms,
        inputs=[
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            gg_v,
            k_vectors,
            ggrad_positions,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Tile-based GPU implementation of K9
# -----------------------------------------------------------------------------
#
# Similar to K8 but simpler: no outer ``α`` sum, only ``(β, σ, lm)``
# packed cols (3 · N_σ · 4 = 12 for N_σ=1). Math:
#
# .. math::
#
#     gg_{pos}[i, \beta] = \mathrm{scale} \sum_{sl} \mathrm{grad\_raw}(i, sl)
#         \sum_k \bigl[
#             M_\cos(k, \beta, sl) \cos(k r_i)
#           + M_\sin(k, \beta, sl) \sin(k r_i) \bigr]
#
# with
#
# .. math::
#
#     M_\cos(k, \beta, sl) &= w(k) k_\beta [gg_{V_r} \hat\phi_i - gg_{V_i} \hat\phi_r](k, sl) \\
#     M_\sin(k, \beta, sl) &= -w(k) k_\beta [gg_{V_r} \hat\phi_r + gg_{V_i} \hat\phi_i](k, sl)
#
# Output cols: packed ``q = β · 4·N_σ + sl`` → 12 cols for N_σ=1,
# padded to tile_j=16 (one j-block). Same layout / tile sizes as
# ``position_gradient_from_feature_grad``.

_VGBP_TILE_I = wp.constant(8)
_VGBP_TILE_K = wp.constant(32)
_VGBP_TILE_J = wp.constant(16)
_VGBP_BLOCK_DIM = 128


@wp.kernel
def _v_grad_back_positions_precompute_kernel(
    receiver_phi_hat: wp.array4d(dtype=wp.float64),  # (N_k, N_σ, 4, 2)
    k_factor_proj: wp.array(dtype=wp.float64),  # (N_k,)
    gg_v: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    n_k_valid: wp.int32,
    n_p_valid: wp.int32,  # = 3 * N_σ * 4
    m_cos: wp.array2d(dtype=wp.float64),  # (N_k_pad, N_p_pad) OUTPUT
    m_sin: wp.array2d(dtype=wp.float64),
):
    r"""Precompute M_cos / M_sin for K9. Launch dim: ``(N_k_pad, N_σ)``.

    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx, s_idx)``; each thread processes one (k-vector, s_idx) work item.

    Parameters
    ----------
    receiver_phi_hat : wp.array4d, shape (N_k, N_σ, 4, 2), dtype wp.float64
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    k_factor_proj : wp.array, shape (N_k,), dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    gg_v : wp.array2d, shape (N_k, 2), dtype wp.float64
        Second-order upstream gradient w.r.t. ``v`` (HVP seed).
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-space k-vectors.
    n_k_valid : wp.int32
        Number of valid (non-padded) k-vectors.
    n_p_valid : wp.int32
        = 3 * N_σ * 4.
    m_cos : wp.array2d, shape (N_k_pad, N_p_pad), dtype wp.float64
        OUTPUT: precomputed per-k cosine-weighted intermediate matrix.
    m_sin : wp.array2d, dtype wp.float64
        OUTPUT: precomputed per-k sine-weighted intermediate matrix.
    """
    k_idx, s_idx = wp.tid()

    if k_idx >= n_k_valid:
        # Pad row: clear this σ's 3·4 = 12 cols.
        for c in range(12):
            # Global col = β * (N_σ * 4) + s_idx * 4 + lm; for each β we touch
            # a 4-col segment for this σ. So just zero via a loop over the
            # pad-specific locations.
            for beta in range(3):
                col = beta * (m_cos.shape[1] // 3) + s_idx * 4 + (c % 4)
                if col < m_cos.shape[1]:
                    m_cos[k_idx, col] = wp.float64(0.0)
                    m_sin[k_idx, col] = wp.float64(0.0)
        return

    gvr = gg_v[k_idx, 0]
    gvi = gg_v[k_idx, 1]
    kfp = k_factor_proj[k_idx]
    k_vec = k_vectors[k_idx]

    n_sigma = receiver_phi_hat.shape[1]
    sl_per_beta = n_sigma * 4  # stride of β in column-major packing

    for lm in range(4):
        phi_r = receiver_phi_hat[k_idx, s_idx, lm, 0]
        phi_i = receiver_phi_hat[k_idx, s_idx, lm, 1]
        # Column offset within a β block: sl = s_idx * 4 + lm.
        sl = s_idx * 4 + lm
        part_cos = gvr * phi_i - gvi * phi_r
        part_sin = gvr * phi_r + gvi * phi_i

        for beta in range(3):
            k_b = k_vec[beta]
            col = beta * sl_per_beta + sl
            if col >= m_cos.shape[1]:
                continue
            if col >= n_p_valid:
                m_cos[k_idx, col] = wp.float64(0.0)
                m_sin[k_idx, col] = wp.float64(0.0)
            else:
                m_cos[k_idx, col] = kfp * k_b * part_cos
                m_sin[k_idx, col] = -kfp * k_b * part_sin


# _v_grad_back_positions_tiled_matmul_kernel → replaced by shared
# :func:`_cossin_native_matmul_kernel`.


@wp.kernel
def _v_grad_back_positions_reduce_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    contribs: wp.array2d(dtype=wp.float64),
    n_atoms_valid: wp.int32,
    ggrad_positions: wp.array2d(dtype=wp.float64),
):
    r"""``gg_pos[i, β] = scale · Σ_{σ,lm} grad_raw(i, σ, lm) · T(i, β*(N_σ*4) + σ*4 + lm)``.

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    contribs : wp.array2d, dtype wp.float64
        OUTPUT: per-tile partial contributions awaiting reduction.
    n_atoms_valid : wp.int32
        Number of valid (non-padded) atoms.
    ggrad_positions : wp.array2d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. positions.
    """
    i_idx = wp.tid()
    if i_idx >= n_atoms_valid:
        return

    n_sigma = grad_raw.shape[1]
    sl_per_beta = n_sigma * 4

    for beta in range(3):
        base = beta * sl_per_beta
        total = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm in range(4):
                sl = s_idx * 4 + lm
                total += grad_raw[i_idx, s_idx, lm] * contribs[i_idx, base + sl]
        ggrad_positions[i_idx, beta] = _INV_TWO_PI_CUBED_TIMES_TWO * total


def _v_grad_back_positions_tiled_launch(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    gg_v: wp.array,
    k_vectors: wp.array,
    ggrad_positions: wp.array,
    device: str,
) -> None:
    r"""CUDA tile-matmul orchestrator for K9."""
    import torch

    n_k = cosines.shape[0]
    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]
    n_p = 3 * n_sigma * 4  # 12 for N_σ=1

    tile_i = int(_VGBP_TILE_I)
    tile_k = int(_VGBP_TILE_K)
    tile_j = int(_VGBP_TILE_J)

    n_k_pad = ((n_k + tile_k - 1) // tile_k) * tile_k
    n_atoms_pad = ((n_atoms + tile_i - 1) // tile_i) * tile_i
    n_p_pad = ((n_p + tile_j - 1) // tile_j) * tile_j

    # cos/sin stay in their native (N_k, N_atoms) layout — the native-layout
    # tile matmul transposes per-tile (wp.tile_transpose), so no (N_atoms, N_k)
    # transpose+zero-pad copy is materialized.
    cos_t = wp.to_torch(cosines)  # for .device only

    m_cos = torch.empty((n_k_pad, n_p_pad), dtype=torch.float64, device=cos_t.device)
    m_sin = torch.empty_like(m_cos)
    contribs = torch.empty(
        (n_atoms_pad, n_p_pad), dtype=torch.float64, device=cos_t.device
    )

    wp.launch(
        _v_grad_back_positions_precompute_kernel,
        dim=(n_k_pad, n_sigma),
        inputs=[
            receiver_phi_hat,
            k_factor_proj,
            gg_v,
            k_vectors,
            wp.int32(n_k),
            wp.int32(n_p),
            wp.from_torch(m_cos, dtype=wp.float64),
            wp.from_torch(m_sin, dtype=wp.float64),
        ],
        device=device,
    )

    _launch_cossin_native_matmul(
        cosines,
        sines,
        wp.from_torch(m_cos, dtype=wp.float64),
        wp.from_torch(m_sin, dtype=wp.float64),
        contribs,
        device,
    )

    wp.launch(
        _v_grad_back_positions_reduce_kernel,
        dim=n_atoms_pad,
        inputs=[
            grad_raw,
            wp.from_torch(contribs, dtype=wp.float64),
            wp.int32(n_atoms),
            ggrad_positions,
        ],
        device=device,
    )


# =============================================================================
# Batched K-family kernels (Phase 8g)
# =============================================================================
#
# Batched variants of K1..K9 — the second-order (double-backward) kernels.
# Every batched variant mirrors its single-system counterpart math-for-math
# and simply threads the extra batch axis:
#
# * per-k outputs live in ``(B, K_max, ...)`` tensors;
# * per-atom outputs are flat ``(N_total, ...)`` with ``batch_idx[i]`` or
#   ``atom_start / atom_end`` picking the owning system;
# * k-space state (``source_phi_hat``, ``receiver_phi_hat``, ``potential``,
#   ``grad_rho``, ``k_factor_proj``, ``k_vectors``, ...) is read with a
#   ``(b, k_idx)`` lookup;
# * pad k-rows in the cache are filled with zeros, so they contribute
#   nothing to any of the math below without explicit bounds checks.


# -----------------------------------------------------------------------------
# Batched K1: backward of eval_gto_fourier_dipole w.r.t. (k_vectors, k_norm2)
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_source_phi_hat_backward_dipole_kernel(
    grad_output: wp.array4d(dtype=wp.float64),  # (B, K_max, 4, 2)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    k_norm2: wp.array2d(dtype=wp.float64),  # (B, K_max)
    sigma: wp.float64,
    inv_cl_l0: wp.float64,
    inv_cl_l1: wp.float64,
    grad_k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max) OUTPUT
    grad_k_norm2: wp.array2d(dtype=wp.float64),  # (B, K_max) OUTPUT
):
    r"""Batched per-(b, k) backward of source GTO Fourier w.r.t. (k_vec, k²).

    Mirror of :func:`_source_phi_hat_backward_dipole_kernel`. Pad k-rows
    have ``grad_output = 0`` and ``k_vectors = 0`` so their outputs are
    zero without explicit bounds checks.

    Launch Grid
    -----------
    ``dim = (B, K_max)``.


    Parameters
    ----------
    grad_output : wp.array4d, shape (B, K_max, 4, 2), dtype wp.float64
        Upstream gradient flowing into this backward kernel.
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    k_norm2 : wp.array2d, shape (B, K_max), dtype wp.float64
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigma : wp.float64
        Gaussian (GTO) width parameter.
    inv_cl_l0 : wp.float64
        Inverse :math:`l=0` overlap normalization constant.
    inv_cl_l1 : wp.float64
        Inverse :math:`l=1` overlap normalization constant.
    grad_k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        OUTPUT: gradient w.r.t. the k-vectors.
    grad_k_norm2 : wp.array2d, shape (B, K_max), dtype wp.float64
        OUTPUT: gradient w.r.t. :math:`|k|^2`.
    """
    b, k_idx = wp.tid()

    k_vec = k_vectors[b, k_idx]
    k2 = k_norm2[b, k_idx]

    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    sigma5 = sigma3 * sigma2

    gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)
    common_radial = _FOUR_PI_SQRT_PI_OVER_2 * gauss

    a0_base = inv_cl_l0 * common_radial * sigma3 * Y00_COEFF
    coeff_l1 = -inv_cl_l1 * common_radial * sigma5 * Y1_COEFF

    g_l0_r = grad_output[b, k_idx, 0, 0]
    g_ky = grad_output[b, k_idx, 1, 1]
    g_kz = grad_output[b, k_idx, 2, 1]
    g_kx = grad_output[b, k_idx, 3, 1]

    dot_l1 = coeff_l1 * (g_ky * k_vec[1] + g_kz * k_vec[2] + g_kx * k_vec[0])
    grad_k_norm2[b, k_idx] = -wp.float64(0.5) * sigma2 * (g_l0_r * a0_base + dot_l1)
    grad_k_vectors[b, k_idx] = wp.vec3d(
        coeff_l1 * g_kx,
        coeff_l1 * g_ky,
        coeff_l1 * g_kz,
    )


def batch_source_phi_hat_backward_dipole(
    grad_output: wp.array,
    k_vectors: wp.array,
    k_norm2: wp.array,
    sigma: float,
    inv_cl_l0: float,
    inv_cl_l1: float,
    grad_k_vectors: wp.array,
    grad_k_norm2: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_source_phi_hat_backward_dipole_kernel`.

    Parameters
    ----------
    grad_output : wp.array
        Upstream gradient flowing into this backward kernel.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    k_norm2 : wp.array
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigma : float
        Gaussian (GTO) width parameter.
    inv_cl_l0 : float
        Inverse :math:`l=0` overlap normalization constant.
    inv_cl_l1 : float
        Inverse :math:`l=1` overlap normalization constant.
    grad_k_vectors : wp.array
        OUTPUT: gradient w.r.t. the k-vectors.
    grad_k_norm2 : wp.array
        OUTPUT: gradient w.r.t. :math:`|k|^2`.
    device : str
        Warp device string; defaults to the input array's device.
    """
    batch_size = k_vectors.shape[0]
    k_max = k_vectors.shape[1]
    if device is None:
        device = str(k_vectors.device)
    wp.launch(
        _batch_source_phi_hat_backward_dipole_kernel,
        dim=(batch_size, k_max),
        inputs=[
            grad_output,
            k_vectors,
            k_norm2,
            wp.float64(sigma),
            wp.float64(inv_cl_l0),
            wp.float64(inv_cl_l1),
            grad_k_vectors,
            grad_k_norm2,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched K2: backward of eval_receiver_gto_fourier_dipole w.r.t. (k_vectors, k_norm2)
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_receiver_phi_hat_backward_dipole_kernel(
    grad_output: wp.array4d(dtype=wp.vec2d),  # (B, K_max, N_σ, 4) vec2d
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    k_norm2: wp.array2d(dtype=wp.float64),  # (B, K_max)
    sigmas: wp.array(dtype=wp.float64),  # (N_σ,)
    inv_cl_table: wp.array2d(dtype=wp.float64),  # (N_σ, 2)
    grad_k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max) OUTPUT
    grad_k_norm2: wp.array2d(dtype=wp.float64),  # (B, K_max) OUTPUT
):
    r"""Batched per-(b, k) backward of receiver GTO Fourier w.r.t. (k_vec, k²).

    Mirror of :func:`_receiver_phi_hat_backward_dipole_kernel`. Inner σ
    loop at each ``(b, k)``. ``grad_output`` uses the ``vec2d`` storage
    trick (see :func:`_batch_eval_receiver_gto_fourier_dipole_kernel`)
    to keep the 4-D array cap.

    Launch Grid
    -----------
    ``dim = (B, K_max)``.


    Parameters
    ----------
    grad_output : wp.array4d, shape (B, K_max, N_σ, 4), dtype wp.vec2d
        Vec2d.
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    k_norm2 : wp.array2d, shape (B, K_max), dtype wp.float64
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigmas : wp.array, shape (N_σ,), dtype wp.float64
        Per-channel Gaussian (GTO) width parameters.
    inv_cl_table : wp.array2d, shape (N_σ, 2), dtype wp.float64
        Per-channel inverse overlap normalization constants.
    grad_k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        OUTPUT: gradient w.r.t. the k-vectors.
    grad_k_norm2 : wp.array2d, shape (B, K_max), dtype wp.float64
        OUTPUT: gradient w.r.t. :math:`|k|^2`.
    """
    b, k_idx = wp.tid()
    n_sigma = sigmas.shape[0]

    k_vec = k_vectors[b, k_idx]
    k2 = k_norm2[b, k_idx]

    sum_k2 = wp.float64(0.0)
    sum_kx = wp.float64(0.0)
    sum_ky = wp.float64(0.0)
    sum_kz = wp.float64(0.0)

    for s_idx in range(n_sigma):
        sigma = sigmas[s_idx]
        sigma2 = sigma * sigma
        sigma3 = sigma2 * sigma
        sigma5 = sigma3 * sigma2

        gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)
        common_radial = _FOUR_PI_SQRT_PI_OVER_2 * gauss

        a0_base = inv_cl_table[s_idx, 0] * common_radial * sigma3 * Y00_COEFF
        coeff_l1 = -inv_cl_table[s_idx, 1] * common_radial * sigma5 * Y1_COEFF

        g0 = grad_output[b, k_idx, s_idx, 0]
        g1 = grad_output[b, k_idx, s_idx, 1]
        g2 = grad_output[b, k_idx, s_idx, 2]
        g3 = grad_output[b, k_idx, s_idx, 3]
        g_l0_r = g0[0]
        g_ky = g1[1]
        g_kz = g2[1]
        g_kx = g3[1]

        dot_l1 = coeff_l1 * (g_ky * k_vec[1] + g_kz * k_vec[2] + g_kx * k_vec[0])
        sum_k2 += -wp.float64(0.5) * sigma2 * (g_l0_r * a0_base + dot_l1)

        sum_kx += coeff_l1 * g_kx
        sum_ky += coeff_l1 * g_ky
        sum_kz += coeff_l1 * g_kz

    grad_k_norm2[b, k_idx] = sum_k2
    grad_k_vectors[b, k_idx] = wp.vec3d(sum_kx, sum_ky, sum_kz)


def batch_receiver_phi_hat_backward_dipole(
    grad_output: wp.array,
    k_vectors: wp.array,
    k_norm2: wp.array,
    sigmas: wp.array,
    inv_cl_table: wp.array,
    grad_k_vectors: wp.array,
    grad_k_norm2: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_receiver_phi_hat_backward_dipole_kernel`.

    Parameters
    ----------
    grad_output : wp.array
        Upstream gradient flowing into this backward kernel.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    k_norm2 : wp.array
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigmas : wp.array
        Per-channel Gaussian (GTO) width parameters.
    inv_cl_table : wp.array
        Per-channel inverse overlap normalization constants.
    grad_k_vectors : wp.array
        OUTPUT: gradient w.r.t. the k-vectors.
    grad_k_norm2 : wp.array
        OUTPUT: gradient w.r.t. :math:`|k|^2`.
    device : str
        Warp device string; defaults to the input array's device.
    """
    batch_size = k_vectors.shape[0]
    k_max = k_vectors.shape[1]
    if device is None:
        device = str(k_vectors.device)
    wp.launch(
        _batch_receiver_phi_hat_backward_dipole_kernel,
        dim=(batch_size, k_max),
        inputs=[
            grad_output,
            k_vectors,
            k_norm2,
            sigmas,
            inv_cl_table,
            grad_k_vectors,
            grad_k_norm2,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched K3: backward of position_gradient_from_rhok w.r.t. grad_rho
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_rhok_position_grad_backward_grad_rho_kernel(
    charges: wp.array(dtype=Any),  # (N_total,)
    dipoles: wp.array(dtype=Any),  # (N_total,)
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    source_phi_hat: wp.array4d(dtype=wp.float64),  # (B, K_max, 4, 2)
    gg_positions: wp.array2d(dtype=wp.float64),  # (N_total, 3)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    scale: wp.array(dtype=wp.float64),  # (B,)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    ggrad_grad_rho: wp.array3d(dtype=wp.float64),  # (B, K_max, 2) OUTPUT
):
    r"""Batched K3 — ``∂²L / (∂grad_rho ∂…)`` contribution.

    One thread per ``(b, k_idx)``. Inner atom loop from ``atom_start[b]``
    to ``atom_end[b]``. Pad rows have ``source_phi_hat = 0`` so their
    outputs are zero without bounds checks.

    Launch Grid
    -----------
    ``dim = (B, K_max)``.


    Parameters
    ----------
    charges : wp.array, shape (N_total,), dtype Any
        Per-atom monopole charges.
    dipoles : wp.array, shape (N_total,), dtype Any
        Per-atom Cartesian dipole moments.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array4d, shape (B, K_max, 4, 2), dtype wp.float64
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    gg_positions : wp.array2d, shape (N_total, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    scale : wp.array, shape (B,), dtype wp.float64
        Scalar prefactor applied to the contribution.
    atom_start : wp.array, shape (B,), dtype wp.int32
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array, shape (B,), dtype wp.int32
        Per-system end offset into the flat atom arrays.
    ggrad_grad_rho : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. ``grad_rho``.
    """
    b, k_idx = wp.tid()

    k_vec = k_vectors[b, k_idx]

    pr0 = source_phi_hat[b, k_idx, 0, 0]
    pi0 = source_phi_hat[b, k_idx, 0, 1]
    pr1 = source_phi_hat[b, k_idx, 1, 0]
    pi1 = source_phi_hat[b, k_idx, 1, 1]
    pr2 = source_phi_hat[b, k_idx, 2, 0]
    pi2 = source_phi_hat[b, k_idx, 2, 1]
    pr3 = source_phi_hat[b, k_idx, 3, 0]
    pi3 = source_phi_hat[b, k_idx, 3, 1]

    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)

    i_lo = atom_start[b]
    i_hi = atom_end[b]
    for i_idx in range(i_lo, i_hi):
        q = wp.float64(charges[i_idx])
        mu = dipoles[i_idx]
        mu_x = wp.float64(mu[0])
        mu_y = wp.float64(mu[1])
        mu_z = wp.float64(mu[2])

        p_r = pr0 * q + pr1 * mu_y + pr2 * mu_z + pr3 * mu_x
        p_i = pi0 * q + pi1 * mu_y + pi2 * mu_z + pi3 * mu_x

        h = (
            k_vec[0] * gg_positions[i_idx, 0]
            + k_vec[1] * gg_positions[i_idx, 1]
            + k_vec[2] * gg_positions[i_idx, 2]
        )
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        acc_r += h * (p_i * cos_ki - p_r * sin_ki)
        acc_i -= h * (p_r * cos_ki + p_i * sin_ki)

    scale_b = scale[b]
    ggrad_grad_rho[b, k_idx, 0] = scale_b * acc_r
    ggrad_grad_rho[b, k_idx, 1] = scale_b * acc_i


def _batch_rhok_pg_back_grad_rho_sig(v, t):
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array4d(dtype=wp.float64),  # source_phi_hat
        wp.array2d(dtype=wp.float64),  # gg_positions
        wp.array2d(dtype=wp.vec3d),  # k_vectors
        wp.array(dtype=wp.float64),  # scale
        wp.array(dtype=wp.int32),  # atom_start
        wp.array(dtype=wp.int32),  # atom_end
        wp.array3d(dtype=wp.float64),  # ggrad_grad_rho
    ]


_batch_rhok_position_grad_backward_grad_rho_overloads = register_overloads(
    _batch_rhok_position_grad_backward_grad_rho_kernel,
    _batch_rhok_pg_back_grad_rho_sig,
)


def batch_rhok_position_grad_backward_grad_rho(
    charges: wp.array,
    dipoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    scale: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    ggrad_grad_rho: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_rhok_position_grad_backward_grad_rho_kernel`.

    Parameters
    ----------
    charges : wp.array
        Per-atom monopole charges.
    dipoles : wp.array
        Per-atom Cartesian dipole moments.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    scale : wp.array
        Scalar prefactor applied to the contribution.
    atom_start : wp.array
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array
        Per-system end offset into the flat atom arrays.
    ggrad_grad_rho : wp.array
        OUTPUT: double-backward gradient w.r.t. ``grad_rho``.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    batch_size = k_vectors.shape[0]
    k_max = k_vectors.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_rhok_position_grad_backward_grad_rho_overloads[vec_dtype],
        dim=(batch_size, k_max),
        inputs=[
            charges,
            dipoles,
            cosines,
            sines,
            source_phi_hat,
            gg_positions,
            k_vectors,
            scale,
            atom_start,
            atom_end,
            ggrad_grad_rho,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched K4: backward of position_gradient_from_rhok w.r.t. (charges, dipoles)
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_rhok_position_grad_backward_moments_kernel(
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    source_phi_hat: wp.array4d(dtype=wp.float64),  # (B, K_max, 4, 2)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    gg_positions: wp.array2d(dtype=wp.float64),  # (N_total, 3)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    scale: wp.array(dtype=wp.float64),  # (B,)
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    ggrad_moments: wp.array2d(dtype=wp.float64),  # (N_total, 4) OUTPUT
):
    r"""Batched K4 — ``∂²L / (∂moments ∂…)`` contribution.

    One thread per ``(i, lm)`` (flat over atoms). ``batch_idx[i]`` picks
    the system. Inner loop over the full ``K_max`` axis; pad k-rows
    have ``source_phi_hat = 0`` and ``grad_rho = 0`` so they contribute
    nothing.

    Launch Grid
    -----------
    ``dim = (N_total, 4)``.


    Parameters
    ----------
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array4d, shape (B, K_max, 4, 2), dtype wp.float64
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    gg_positions : wp.array2d, shape (N_total, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    scale : wp.array, shape (B,), dtype wp.float64
        Scalar prefactor applied to the contribution.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    ggrad_moments : wp.array2d, shape (N_total, 4), dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. the multipole moments.
    """
    i_idx, lm_idx = wp.tid()
    b = batch_idx[i_idx]
    k_max = k_vectors.shape[1]

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    acc = wp.float64(0.0)
    for k_idx in range(k_max):
        k_vec = k_vectors[b, k_idx]
        h = k_vec[0] * gp_x + k_vec[1] * gp_y + k_vec[2] * gp_z
        gr = grad_rho[b, k_idx, 0]
        gi = grad_rho[b, k_idx, 1]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        phi_r = source_phi_hat[b, k_idx, lm_idx, 0]
        phi_i = source_phi_hat[b, k_idx, lm_idx, 1]

        term = phi_i * (gr * cos_ki - gi * sin_ki) - phi_r * (gi * cos_ki + gr * sin_ki)
        acc += h * term

    ggrad_moments[i_idx, lm_idx] = scale[b] * acc


def batch_rhok_position_grad_backward_moments(
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    grad_rho: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    scale: wp.array,
    batch_idx: wp.array,
    ggrad_moments: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_rhok_position_grad_backward_moments_kernel`.

    Parameters
    ----------
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    scale : wp.array
        Scalar prefactor applied to the contribution.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    ggrad_moments : wp.array
        OUTPUT: double-backward gradient w.r.t. the multipole moments.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_total = cosines.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_rhok_position_grad_backward_moments_kernel,
        dim=(n_total, 4),
        inputs=[
            cosines,
            sines,
            source_phi_hat,
            grad_rho,
            gg_positions,
            k_vectors,
            scale,
            batch_idx,
            ggrad_moments,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched K5: backward of position_gradient_from_rhok w.r.t. positions
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_rhok_position_grad_backward_positions_kernel(
    charges: wp.array(dtype=Any),  # (N_total,)
    dipoles: wp.array(dtype=Any),  # (N_total,)
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    source_phi_hat: wp.array4d(dtype=wp.float64),  # (B, K_max, 4, 2)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    gg_positions: wp.array2d(dtype=wp.float64),  # (N_total, 3)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    scale: wp.array(dtype=wp.float64),  # (B,)
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    ggrad_positions: wp.array2d(dtype=wp.float64),  # (N_total, 3) OUTPUT
):
    r"""Batched K5 — ``∂²L / (∂r_i ∂…)`` position Hessian diagonal block.

    One thread per atom (flat across batch). ``batch_idx[i]`` picks the
    system; inner loop over the full ``K_max`` axis. Pad k-rows have
    ``source_phi_hat = 0`` (or ``k_vectors = 0``) so they contribute
    nothing.

    Launch Grid
    -----------
    ``dim = N_total``.


    Parameters
    ----------
    charges : wp.array, shape (N_total,), dtype Any
        Per-atom monopole charges.
    dipoles : wp.array, shape (N_total,), dtype Any
        Per-atom Cartesian dipole moments.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array4d, shape (B, K_max, 4, 2), dtype wp.float64
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    gg_positions : wp.array2d, shape (N_total, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    scale : wp.array, shape (B,), dtype wp.float64
        Scalar prefactor applied to the contribution.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    ggrad_positions : wp.array2d, shape (N_total, 3), dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. positions.
    """
    i_idx = wp.tid()
    b = batch_idx[i_idx]
    k_max = k_vectors.shape[1]

    q = wp.float64(charges[i_idx])
    mu = dipoles[i_idx]
    mu_x = wp.float64(mu[0])
    mu_y = wp.float64(mu[1])
    mu_z = wp.float64(mu[2])

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(k_max):
        k_vec = k_vectors[b, k_idx]
        gr = grad_rho[b, k_idx, 0]
        gi = grad_rho[b, k_idx, 1]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        pr0 = source_phi_hat[b, k_idx, 0, 0]
        pi0 = source_phi_hat[b, k_idx, 0, 1]
        pr1 = source_phi_hat[b, k_idx, 1, 0]
        pi1 = source_phi_hat[b, k_idx, 1, 1]
        pr2 = source_phi_hat[b, k_idx, 2, 0]
        pi2 = source_phi_hat[b, k_idx, 2, 1]
        pr3 = source_phi_hat[b, k_idx, 3, 0]
        pi3 = source_phi_hat[b, k_idx, 3, 1]

        p_r = pr0 * q + pr1 * mu_y + pr2 * mu_z + pr3 * mu_x
        p_i = pi0 * q + pi1 * mu_y + pi2 * mu_z + pi3 * mu_x

        a_k = gr * p_i - gi * p_r
        b_k = -(gr * p_r + gi * p_i)

        h = k_vec[0] * gp_x + k_vec[1] * gp_y + k_vec[2] * gp_z
        weight = h * (b_k * cos_ki - a_k * sin_ki)
        gx += k_vec[0] * weight
        gy += k_vec[1] * weight
        gz += k_vec[2] * weight

    scale_b = scale[b]
    ggrad_positions[i_idx, 0] = scale_b * gx
    ggrad_positions[i_idx, 1] = scale_b * gy
    ggrad_positions[i_idx, 2] = scale_b * gz


def _batch_rhok_pg_back_positions_sig(v, t):
    return [
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array4d(dtype=wp.float64),  # source_phi_hat
        wp.array3d(dtype=wp.float64),  # grad_rho
        wp.array2d(dtype=wp.float64),  # gg_positions
        wp.array2d(dtype=wp.vec3d),  # k_vectors
        wp.array(dtype=wp.float64),  # scale
        wp.array(dtype=wp.int32),  # batch_idx
        wp.array2d(dtype=wp.float64),  # ggrad_positions
    ]


_batch_rhok_position_grad_backward_positions_overloads = register_overloads(
    _batch_rhok_position_grad_backward_positions_kernel,
    _batch_rhok_pg_back_positions_sig,
)


def batch_rhok_position_grad_backward_positions(
    charges: wp.array,
    dipoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    grad_rho: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    scale: wp.array,
    batch_idx: wp.array,
    ggrad_positions: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_rhok_position_grad_backward_positions_kernel`.

    Parameters
    ----------
    charges : wp.array
        Per-atom monopole charges.
    dipoles : wp.array
        Per-atom Cartesian dipole moments.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    scale : wp.array
        Scalar prefactor applied to the contribution.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    ggrad_positions : wp.array
        OUTPUT: double-backward gradient w.r.t. positions.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    n_total = cosines.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_rhok_position_grad_backward_positions_overloads[vec_dtype],
        dim=n_total,
        inputs=[
            charges,
            dipoles,
            cosines,
            sines,
            source_phi_hat,
            grad_rho,
            gg_positions,
            k_vectors,
            scale,
            batch_idx,
            ggrad_positions,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched K6: backward of position_gradient_from_feature_grad w.r.t. grad_raw
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_feat_position_grad_backward_grad_raw_kernel(
    receiver_phi_hat: wp.array4d(dtype=wp.vec2d),  # (B, K_max, N_σ, 4) vec2d
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_factor_proj: wp.array2d(dtype=wp.float64),  # (B, K_max)
    potential: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    gg_positions: wp.array2d(dtype=wp.float64),  # (N_total, 3)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    ggrad_grad_raw: wp.array3d(dtype=wp.float64),  # (N_total, N_σ, 4) OUTPUT
):
    r"""Batched K6 — ``∂²L / (∂grad_raw ∂…)`` contribution.

    One thread per ``(i, σ, lm)``. ``batch_idx[i]`` picks the system.
    Pad rows have ``k_factor_proj = 0`` and ``receiver_phi_hat = 0``
    so they contribute nothing.

    Launch Grid
    -----------
    ``dim = (N_total, N_σ, 4)``.


    Parameters
    ----------
    receiver_phi_hat : wp.array4d, shape (B, K_max, N_σ, 4), dtype wp.vec2d
        Vec2d.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array2d, shape (N_total, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    ggrad_grad_raw : wp.array3d, shape (N_total, N_σ, 4), dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. ``grad_raw``.
    """
    i_idx, s_idx, lm_idx = wp.tid()
    b = batch_idx[i_idx]
    k_max = k_vectors.shape[1]

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    acc = wp.float64(0.0)
    for k_idx in range(k_max):
        k_vec = k_vectors[b, k_idx]
        h = k_vec[0] * gp_x + k_vec[1] * gp_y + k_vec[2] * gp_z
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        v_r = potential[b, k_idx, 0]
        v_i = potential[b, k_idx, 1]
        phi = receiver_phi_hat[b, k_idx, s_idx, lm_idx]
        phi_r = phi[0]
        phi_i = phi[1]
        kfp = k_factor_proj[b, k_idx]

        dc = v_r * phi_i - v_i * phi_r
        dd = v_r * phi_r + v_i * phi_i
        acc += kfp * h * (dc * cos_ki - dd * sin_ki)

    ggrad_grad_raw[i_idx, s_idx, lm_idx] = _INV_TWO_PI_CUBED_TIMES_TWO * acc


def batch_feat_position_grad_backward_grad_raw(
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    batch_idx: wp.array,
    ggrad_grad_raw: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_feat_position_grad_backward_grad_raw_kernel`.

    Parameters
    ----------
    receiver_phi_hat : wp.array
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    ggrad_grad_raw : wp.array
        OUTPUT: double-backward gradient w.r.t. ``grad_raw``.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_total = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[2]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_feat_position_grad_backward_grad_raw_kernel,
        dim=(n_total, n_sigma, 4),
        inputs=[
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            potential,
            gg_positions,
            k_vectors,
            batch_idx,
            ggrad_grad_raw,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched K7: backward of position_gradient_from_feature_grad w.r.t. V(k)
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_feat_position_grad_backward_v_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),  # (N_total, N_σ, 4)
    receiver_phi_hat: wp.array4d(dtype=wp.vec2d),  # (B, K_max, N_σ, 4) vec2d
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_factor_proj: wp.array2d(dtype=wp.float64),  # (B, K_max)
    gg_positions: wp.array2d(dtype=wp.float64),  # (N_total, 3)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    ggrad_v: wp.array3d(dtype=wp.float64),  # (B, K_max, 2) OUTPUT
):
    r"""Batched K7 — ``∂²L / (∂V(k) ∂…)`` contribution.

    One thread per ``(b, k_idx)``. Inner atom loop from
    ``atom_start[b]`` to ``atom_end[b]``. ``grad_v`` / ``ggrad_v``
    (output) pad rows are naturally 0 because the inner sum has no
    atoms when ``k_factor_proj = 0`` anyway — but we also factor
    ``kfp`` out so writing scale * kfp = 0 on pad is explicit.

    Launch Grid
    -----------
    ``dim = (B, K_max)``.


    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_total, N_σ, 4), dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array4d, shape (B, K_max, N_σ, 4), dtype wp.vec2d
        Vec2d.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    gg_positions : wp.array2d, shape (N_total, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    atom_start : wp.array, shape (B,), dtype wp.int32
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array, shape (B,), dtype wp.int32
        Per-system end offset into the flat atom arrays.
    ggrad_v : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. ``v``.
    """
    b, k_idx = wp.tid()
    n_sigma = receiver_phi_hat.shape[2]

    k_vec = k_vectors[b, k_idx]
    kfp = k_factor_proj[b, k_idx]

    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)

    i_lo = atom_start[b]
    i_hi = atom_end[b]
    for i_idx in range(i_lo, i_hi):
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(4):
                g = grad_raw[i_idx, s_idx, lm_idx]
                phi = receiver_phi_hat[b, k_idx, s_idx, lm_idx]
                q_r += phi[0] * g
                q_i += phi[1] * g

        h = (
            k_vec[0] * gg_positions[i_idx, 0]
            + k_vec[1] * gg_positions[i_idx, 1]
            + k_vec[2] * gg_positions[i_idx, 2]
        )
        acc_r += h * (q_i * cos_ki - q_r * sin_ki)
        acc_i += h * (-q_r * cos_ki - q_i * sin_ki)

    ggrad_v[b, k_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_r
    ggrad_v[b, k_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_i


def batch_feat_position_grad_backward_v(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    ggrad_v: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_feat_position_grad_backward_v_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    atom_start : wp.array
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array
        Per-system end offset into the flat atom arrays.
    ggrad_v : wp.array
        OUTPUT: double-backward gradient w.r.t. ``v``.
    device : str
        Warp device string; defaults to the input array's device.
    """
    batch_size = receiver_phi_hat.shape[0]
    k_max = receiver_phi_hat.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_feat_position_grad_backward_v_kernel,
        dim=(batch_size, k_max),
        inputs=[
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            gg_positions,
            k_vectors,
            atom_start,
            atom_end,
            ggrad_v,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched K8: backward of position_gradient_from_feature_grad w.r.t. positions
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_feat_position_grad_backward_positions_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),  # (N_total, N_σ, 4)
    receiver_phi_hat: wp.array4d(dtype=wp.vec2d),  # (B, K_max, N_σ, 4) vec2d
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_factor_proj: wp.array2d(dtype=wp.float64),  # (B, K_max)
    potential: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    gg_positions: wp.array2d(dtype=wp.float64),  # (N_total, 3)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    ggrad_positions: wp.array2d(dtype=wp.float64),  # (N_total, 3) OUTPUT
):
    r"""Batched K8 — ``∂²L / (∂r_i ∂…)`` position Hessian diagonal block (features).

    One thread per atom (flat across batch). ``batch_idx[i]`` picks the
    system; inner ``K_max`` loop. Pad rows have ``k_factor_proj = 0``
    and ``receiver_phi_hat = 0`` so they contribute nothing.

    Launch Grid
    -----------
    ``dim = N_total``.


    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_total, N_σ, 4), dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array4d, shape (B, K_max, N_σ, 4), dtype wp.vec2d
        Vec2d.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array2d, shape (N_total, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    ggrad_positions : wp.array2d, shape (N_total, 3), dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. positions.
    """
    i_idx = wp.tid()
    b = batch_idx[i_idx]
    k_max = k_vectors.shape[1]
    n_sigma = receiver_phi_hat.shape[2]

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(k_max):
        k_vec = k_vectors[b, k_idx]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        v_r = potential[b, k_idx, 0]
        v_i = potential[b, k_idx, 1]
        kfp = k_factor_proj[b, k_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(4):
                g = grad_raw[i_idx, s_idx, lm_idx]
                phi = receiver_phi_hat[b, k_idx, s_idx, lm_idx]
                q_r += phi[0] * g
                q_i += phi[1] * g

        c_k = v_r * q_i - v_i * q_r
        d_k = v_r * q_r + v_i * q_i

        h = k_vec[0] * gp_x + k_vec[1] * gp_y + k_vec[2] * gp_z
        weight = -kfp * h * (c_k * sin_ki + d_k * cos_ki)
        gx += k_vec[0] * weight
        gy += k_vec[1] * weight
        gz += k_vec[2] * weight

    ggrad_positions[i_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * gx
    ggrad_positions[i_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * gy
    ggrad_positions[i_idx, 2] = _INV_TWO_PI_CUBED_TIMES_TWO * gz


def batch_feat_position_grad_backward_positions(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    batch_idx: wp.array,
    ggrad_positions: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_feat_position_grad_backward_positions_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    ggrad_positions : wp.array
        OUTPUT: double-backward gradient w.r.t. positions.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_total = cosines.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_feat_position_grad_backward_positions_kernel,
        dim=n_total,
        inputs=[
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            potential,
            gg_positions,
            k_vectors,
            batch_idx,
            ggrad_positions,
        ],
        device=device,
    )


# -----------------------------------------------------------------------------
# Batched K9: backward of v_gradient_from_feature_grad w.r.t. positions
# -----------------------------------------------------------------------------


@wp.kernel
def _batch_v_grad_from_feat_grad_backward_positions_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),  # (N_total, N_σ, 4)
    receiver_phi_hat: wp.array4d(dtype=wp.vec2d),  # (B, K_max, N_σ, 4) vec2d
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_factor_proj: wp.array2d(dtype=wp.float64),  # (B, K_max)
    gg_v: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    ggrad_positions: wp.array2d(dtype=wp.float64),  # (N_total, 3) OUTPUT
):
    r"""Batched K9 — ``∂²L / (∂r_i ∂…)`` via v_gradient_from_feature_grad.

    One thread per atom (flat across batch). ``batch_idx[i]`` picks the
    system; inner ``K_max`` loop. Pad rows have ``k_factor_proj = 0``
    and ``receiver_phi_hat = 0`` so they contribute nothing.

    Launch Grid
    -----------
    ``dim = N_total``.


    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_total, N_σ, 4), dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array4d, shape (B, K_max, N_σ, 4), dtype wp.vec2d
        Vec2d.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    gg_v : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Second-order upstream gradient w.r.t. ``v`` (HVP seed).
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    ggrad_positions : wp.array2d, shape (N_total, 3), dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. positions.
    """
    i_idx = wp.tid()
    b = batch_idx[i_idx]
    k_max = k_vectors.shape[1]
    n_sigma = receiver_phi_hat.shape[2]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(k_max):
        k_vec = k_vectors[b, k_idx]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        kfp = k_factor_proj[b, k_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(4):
                g = grad_raw[i_idx, s_idx, lm_idx]
                phi = receiver_phi_hat[b, k_idx, s_idx, lm_idx]
                q_r += phi[0] * g
                q_i += phi[1] * g

        gvr = gg_v[b, k_idx, 0]
        gvi = gg_v[b, k_idx, 1]

        cos_term = gvr * q_i - gvi * q_r
        sin_term = gvr * q_r + gvi * q_i

        weight = kfp * (cos_ki * cos_term - sin_ki * sin_term)
        gx += k_vec[0] * weight
        gy += k_vec[1] * weight
        gz += k_vec[2] * weight

    ggrad_positions[i_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * gx
    ggrad_positions[i_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * gy
    ggrad_positions[i_idx, 2] = _INV_TWO_PI_CUBED_TIMES_TWO * gz


def batch_v_grad_from_feat_grad_backward_positions(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    gg_v: wp.array,
    k_vectors: wp.array,
    batch_idx: wp.array,
    ggrad_positions: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_v_grad_from_feat_grad_backward_positions_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    gg_v : wp.array
        Second-order upstream gradient w.r.t. ``v`` (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    ggrad_positions : wp.array
        OUTPUT: double-backward gradient w.r.t. positions.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_total = cosines.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_v_grad_from_feat_grad_backward_positions_kernel,
        dim=n_total,
        inputs=[
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            gg_v,
            k_vectors,
            batch_idx,
            ggrad_positions,
        ],
        device=device,
    )


# ==== recovered quadrupole / reciprocal-grad functions ====


@wp.kernel
def _assemble_rho_q_kernel(
    quadrupoles: wp.array(dtype=Any),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    coeff2: wp.array(dtype=wp.float64),
    volume: wp.float64,
    rho: wp.array2d(dtype=wp.float64),
):
    r"""Cartesian-quadrupole contribution to rho(k) (REAL channel).

    Per k: ``acc_c = Σ_i (k·Q_i·k) cos(k·r_i)``,
    ``acc_s = Σ_i (k·Q_i·k) sin(k·r_i)``, then

    .. math::

        \rho_Q(\mathbf{k})_{\mathrm{real}} &= \mathrm{scale}\cdot
            \mathrm{coeff2}(\mathbf{k})\cdot \mathrm{acc}_c, \\
        \rho_Q(\mathbf{k})_{\mathrm{imag}} &= -\mathrm{scale}\cdot
            \mathrm{coeff2}(\mathbf{k})\cdot \mathrm{acc}_s,

    with ``scale = (2π)³ / V``.

    Launch Grid
    -----------
    dim = [N_k] — one thread per k-vector; each thread sweeps all atoms.

    Parameters
    ----------
    quadrupoles : wp.array, shape (N_atoms,), dtype wp.mat33f or wp.mat33d
        Cartesian quadrupole tensors :math:`Q_i`.
    cosines, sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Pre-computed structure-factor tables.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-lattice k-vectors.
    coeff2 : wp.array, shape (N_k,), dtype wp.float64
        Per-k Cartesian-quadrupole coefficient from
        :func:`eval_gto_fourier_q`.
    volume : wp.float64
        Periodic-cell volume.
    rho : wp.array2d, shape (N_k, 2), dtype wp.float64
        OUTPUT (additive Q channel). ``[..., 0]`` real, ``[..., 1]`` imag.
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]
    k_vec = k_vectors[k_idx]
    acc_c = wp.float64(0.0)
    acc_s = wp.float64(0.0)
    for i in range(n_atoms):
        Q = quadrupoles[i]
        kQ0 = (
            wp.float64(Q[0, 0]) * k_vec[0]
            + wp.float64(Q[0, 1]) * k_vec[1]
            + wp.float64(Q[0, 2]) * k_vec[2]
        )
        kQ1 = (
            wp.float64(Q[1, 0]) * k_vec[0]
            + wp.float64(Q[1, 1]) * k_vec[1]
            + wp.float64(Q[1, 2]) * k_vec[2]
        )
        kQ2 = (
            wp.float64(Q[2, 0]) * k_vec[0]
            + wp.float64(Q[2, 1]) * k_vec[1]
            + wp.float64(Q[2, 2]) * k_vec[2]
        )
        kQk = kQ0 * k_vec[0] + kQ1 * k_vec[1] + kQ2 * k_vec[2]
        acc_c += cosines[k_idx, i] * kQk
        acc_s += sines[k_idx, i] * kQk
    scale = _TWO_PI_CUBED / volume
    c2 = coeff2[k_idx]
    rho[k_idx, 0] = scale * c2 * acc_c
    rho[k_idx, 1] = -scale * c2 * acc_s


def _assemble_rho_q_sig(v, t):
    """Signature builder: quadrupoles takes mat33 ``m`` (matches dtype t)."""
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=m),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array(dtype=wp.vec3d),
        wp.array(dtype=wp.float64),
        wp.float64,
        wp.array2d(dtype=wp.float64),
    ]


@wp.kernel
def _batch_assemble_rho_q_kernel(
    quadrupoles: wp.array(dtype=Any),  # (N_total,)
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    coeff2: wp.array2d(dtype=wp.float64),  # (B, K_max)
    volume: wp.array(dtype=wp.float64),  # (B,)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    rho: wp.array3d(dtype=wp.float64),  # (B, K_max, 2) OUTPUT
):
    r"""Batched Cartesian-quadrupole ρ_Q(k) (REAL channel).

    Each thread sums over ``[atom_start[b], atom_end[b])`` and writes the
    additive Cartesian-quadrupole contribution to ``rho[b, k_idx, :]``.

    Launch Grid
    -----------
    ``dim = (B, K_max)``.


    Parameters
    ----------
    quadrupoles : wp.array, shape (N_total,), dtype Any
        Per-atom Cartesian quadrupole moments.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    coeff2 : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    volume : wp.array, shape (B,), dtype wp.float64
        Unit-cell volume.
    atom_start : wp.array, shape (B,), dtype wp.int32
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array, shape (B,), dtype wp.int32
        Per-system end offset into the flat atom arrays.
    rho : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        OUTPUT: complex reciprocal-space density :math:`\hat\rho(k)` (re, im).
    """
    b, k_idx = wp.tid()
    k_vec = k_vectors[b, k_idx]
    acc_c = wp.float64(0.0)
    acc_s = wp.float64(0.0)
    i_lo = atom_start[b]
    i_hi = atom_end[b]
    for i in range(i_lo, i_hi):
        Q = quadrupoles[i]
        kQ0 = (
            wp.float64(Q[0, 0]) * k_vec[0]
            + wp.float64(Q[0, 1]) * k_vec[1]
            + wp.float64(Q[0, 2]) * k_vec[2]
        )
        kQ1 = (
            wp.float64(Q[1, 0]) * k_vec[0]
            + wp.float64(Q[1, 1]) * k_vec[1]
            + wp.float64(Q[1, 2]) * k_vec[2]
        )
        kQ2 = (
            wp.float64(Q[2, 0]) * k_vec[0]
            + wp.float64(Q[2, 1]) * k_vec[1]
            + wp.float64(Q[2, 2]) * k_vec[2]
        )
        kQk = kQ0 * k_vec[0] + kQ1 * k_vec[1] + kQ2 * k_vec[2]
        acc_c += cosines[k_idx, i] * kQk
        acc_s += sines[k_idx, i] * kQk
    scale = _TWO_PI_CUBED / volume[b]
    c2 = coeff2[b, k_idx]
    rho[b, k_idx, 0] = scale * c2 * acc_c
    rho[b, k_idx, 1] = -scale * c2 * acc_s


def _batch_assemble_rho_q_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=m),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.vec3d),
        wp.array2d(dtype=wp.float64),
        wp.array(dtype=wp.float64),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array3d(dtype=wp.float64),
    ]


@wp.kernel
def _batch_eval_receiver_gto_fourier_quadrupole_kernel(
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    k_norm2: wp.array2d(dtype=wp.float64),  # (B, K_max)
    sigmas: wp.array(dtype=wp.float64),  # (N_σ,)
    inv_cl_l2: wp.array(dtype=wp.float64),  # (N_σ,)
    output: wp.array4d(dtype=wp.vec2d),  # (B, K_max, N_σ, 5) — vec2d = (real, imag)
):
    r"""Batched receiver-basis φ̂ at l = 2 across ``B`` systems.

    Launch Grid
    -----------
    ``dim`` indexes ``(b, k_idx, s_idx)``; each thread processes one (system, k-vector, s_idx) work item.

    Parameters
    ----------
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    k_norm2 : wp.array2d, shape (B, K_max), dtype wp.float64
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigmas : wp.array, shape (N_σ,), dtype wp.float64
        Per-channel Gaussian (GTO) width parameters.
    inv_cl_l2 : wp.array, shape (N_σ,), dtype wp.float64
        Inverse :math:`l=2` overlap normalization constant.
    output : wp.array4d, shape (B, K_max, N_σ, 5), dtype wp.vec2d
        — vec2d = (real, imag).
    """
    b, k_idx, s_idx = wp.tid()

    k_vec = k_vectors[b, k_idx]
    k2 = k_norm2[b, k_idx]
    kx = k_vec[0]
    ky = k_vec[1]
    kz = k_vec[2]

    sigma = sigmas[s_idx]
    sigma2 = sigma * sigma
    sigma7 = sigma2 * sigma2 * sigma2 * sigma

    gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)

    coeff_l2 = -inv_cl_l2[s_idx] * _FOUR_PI_SQRT_PI_OVER_2 * gauss * sigma7

    zero = wp.float64(0.0)

    output[b, k_idx, s_idx, 0] = wp.vec2d(coeff_l2 * Y2_M2_COEFF * kx * ky, zero)
    output[b, k_idx, s_idx, 1] = wp.vec2d(coeff_l2 * Y2_M1_COEFF * ky * kz, zero)
    output[b, k_idx, s_idx, 2] = wp.vec2d(
        coeff_l2 * Y2_0_COEFF * (wp.float64(3.0) * kz * kz - k2), zero
    )
    output[b, k_idx, s_idx, 3] = wp.vec2d(coeff_l2 * Y2_P1_COEFF * kx * kz, zero)
    output[b, k_idx, s_idx, 4] = wp.vec2d(
        coeff_l2 * Y2_P2_COEFF * (kx * kx - ky * ky), zero
    )


@wp.kernel
def _batch_feat_position_grad_backward_grad_raw_quadrupole_kernel(
    receiver_phi_hat_l2: wp.array4d(dtype=wp.vec2d),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array2d(dtype=wp.float64),
    potential: wp.array3d(dtype=wp.float64),
    gg_positions: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array2d(dtype=wp.vec3d),
    batch_idx: wp.array(dtype=wp.int32),
    ggrad_grad_raw: wp.array3d(dtype=wp.float64),
):
    r"""Batched l=2 K6.

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx, s_idx, lm_idx)``; each thread processes one (i_idx, s_idx, lm_idx) work item.

    Parameters
    ----------
    receiver_phi_hat_l2 : wp.array4d, dtype wp.vec2d
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array3d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array2d, dtype wp.vec3d
        Reciprocal-space k-vectors.
    batch_idx : wp.array, dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    ggrad_grad_raw : wp.array3d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. ``grad_raw``.
    """
    i_idx, s_idx, lm_idx = wp.tid()
    b = batch_idx[i_idx]
    k_max = k_vectors.shape[1]

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    acc = wp.float64(0.0)
    for k_idx in range(k_max):
        k_vec = k_vectors[b, k_idx]
        h = k_vec[0] * gp_x + k_vec[1] * gp_y + k_vec[2] * gp_z
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        v_r = potential[b, k_idx, 0]
        v_i = potential[b, k_idx, 1]
        phi = receiver_phi_hat_l2[b, k_idx, s_idx, lm_idx]

        dc = v_r * phi[1] - v_i * phi[0]
        dd = v_r * phi[0] + v_i * phi[1]
        kfp = k_factor_proj[b, k_idx]
        acc += kfp * h * (dc * cos_ki - dd * sin_ki)

    ggrad_grad_raw[i_idx, s_idx, lm_idx] = _INV_TWO_PI_CUBED_TIMES_TWO * acc


@wp.kernel
def _batch_feat_position_grad_backward_positions_quadrupole_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat_l2: wp.array4d(dtype=wp.vec2d),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array2d(dtype=wp.float64),
    potential: wp.array3d(dtype=wp.float64),
    gg_positions: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array2d(dtype=wp.vec3d),
    batch_idx: wp.array(dtype=wp.int32),
    ggrad_positions: wp.array2d(dtype=wp.float64),
):
    r"""Batched l=2 K8.

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array4d, dtype wp.vec2d
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array3d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array2d, dtype wp.vec3d
        Reciprocal-space k-vectors.
    batch_idx : wp.array, dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    ggrad_positions : wp.array2d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. positions.
    """
    i_idx = wp.tid()
    b = batch_idx[i_idx]
    k_max = k_vectors.shape[1]
    n_sigma = receiver_phi_hat_l2.shape[2]

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(k_max):
        k_vec = k_vectors[b, k_idx]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        v_r = potential[b, k_idx, 0]
        v_i = potential[b, k_idx, 1]
        kfp = k_factor_proj[b, k_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(5):
                g = grad_raw[i_idx, s_idx, lm_idx]
                phi = receiver_phi_hat_l2[b, k_idx, s_idx, lm_idx]
                q_r += phi[0] * g
                q_i += phi[1] * g

        c_k = v_r * q_i - v_i * q_r
        d_k = v_r * q_r + v_i * q_i

        h = k_vec[0] * gp_x + k_vec[1] * gp_y + k_vec[2] * gp_z
        weight = -kfp * h * (c_k * sin_ki + d_k * cos_ki)
        gx += k_vec[0] * weight
        gy += k_vec[1] * weight
        gz += k_vec[2] * weight

    ggrad_positions[i_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * gx
    ggrad_positions[i_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * gy
    ggrad_positions[i_idx, 2] = _INV_TWO_PI_CUBED_TIMES_TWO * gz


@wp.kernel
def _batch_feat_position_grad_backward_v_quadrupole_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat_l2: wp.array4d(dtype=wp.vec2d),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array2d(dtype=wp.float64),
    gg_positions: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array2d(dtype=wp.vec3d),
    atom_start: wp.array(dtype=wp.int32),
    atom_end: wp.array(dtype=wp.int32),
    ggrad_v: wp.array3d(dtype=wp.float64),
):
    r"""Batched l=2 K7.

    Launch Grid
    -----------
    ``dim`` indexes ``(b, k_idx)``; each thread processes one (system, k-vector) work item.

    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array4d, dtype wp.vec2d
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    gg_positions : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array2d, dtype wp.vec3d
        Reciprocal-space k-vectors.
    atom_start : wp.array, dtype wp.int32
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array, dtype wp.int32
        Per-system end offset into the flat atom arrays.
    ggrad_v : wp.array3d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. ``v``.
    """
    b, k_idx = wp.tid()
    n_sigma = receiver_phi_hat_l2.shape[2]

    k_vec = k_vectors[b, k_idx]
    kfp = k_factor_proj[b, k_idx]

    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)

    i_lo = atom_start[b]
    i_hi = atom_end[b]
    for i_idx in range(i_lo, i_hi):
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(5):
                g = grad_raw[i_idx, s_idx, lm_idx]
                phi = receiver_phi_hat_l2[b, k_idx, s_idx, lm_idx]
                q_r += phi[0] * g
                q_i += phi[1] * g

        h = (
            k_vec[0] * gg_positions[i_idx, 0]
            + k_vec[1] * gg_positions[i_idx, 1]
            + k_vec[2] * gg_positions[i_idx, 2]
        )
        acc_r += h * (q_i * cos_ki - q_r * sin_ki)
        acc_i += h * (-q_r * cos_ki - q_i * sin_ki)

    ggrad_v[b, k_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_r
    ggrad_v[b, k_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_i


@wp.kernel
def _batch_position_gradient_from_feature_grad_quadrupole_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat_l2: wp.array4d(dtype=wp.vec2d),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array2d(dtype=wp.float64),
    potential: wp.array3d(dtype=wp.float64),
    k_vectors: wp.array2d(dtype=wp.vec3d),
    batch_idx: wp.array(dtype=wp.int32),
    grad_positions: wp.array2d(dtype=wp.float64),
):
    r"""Batched ``∂L/∂r_i`` backward of the l=2 projection. ``dim = N_total``.

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array4d, dtype wp.vec2d
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array3d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    k_vectors : wp.array2d, dtype wp.vec3d
        Reciprocal-space k-vectors.
    batch_idx : wp.array, dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    grad_positions : wp.array2d, dtype wp.float64
        OUTPUT: gradient w.r.t. atomic positions.
    """
    i_idx = wp.tid()
    b = batch_idx[i_idx]

    k_max = k_vectors.shape[1]
    n_sigma = receiver_phi_hat_l2.shape[2]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(k_max):
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        v_r = potential[b, k_idx, 0]
        v_i = potential[b, k_idx, 1]
        kfp = k_factor_proj[b, k_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(5):
                g = grad_raw[i_idx, s_idx, lm_idx]
                phi = receiver_phi_hat_l2[b, k_idx, s_idx, lm_idx]
                q_r += phi[0] * g
                q_i += phi[1] * g

        c_k = v_r * q_i - v_i * q_r
        d_k = v_r * q_r + v_i * q_i

        per_k = kfp * (c_k * cos_ki - d_k * sin_ki)
        k_vec = k_vectors[b, k_idx]
        gx += k_vec[0] * per_k
        gy += k_vec[1] * per_k
        gz += k_vec[2] * per_k

    grad_positions[i_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * gx
    grad_positions[i_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * gy
    grad_positions[i_idx, 2] = _INV_TWO_PI_CUBED_TIMES_TWO * gz


@wp.kernel
def _batch_position_gradient_from_rhoq_kernel(
    quadrupoles: wp.array(dtype=Any),  # (N_total,) mat33
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    coeff2: wp.array2d(dtype=wp.float64),  # (B, K_max)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    scale: wp.array(dtype=wp.float64),  # (B,)
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    grad_positions: wp.array2d(dtype=wp.float64),  # (N_total, 3) OUTPUT
):
    r"""Batched Q-channel ∂L/∂r_i (one thread per atom).

    Launch Grid
    -----------
    dim = [N_total] — one thread per atom; sweeps all k-vectors of its system.


    Parameters
    ----------
    quadrupoles : wp.array, shape (N_total,), dtype Any
        Mat33.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    coeff2 : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    grad_rho : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : wp.array, shape (B,), dtype wp.float64
        Scalar prefactor applied to the contribution.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    grad_positions : wp.array2d, shape (N_total, 3), dtype wp.float64
        OUTPUT: gradient w.r.t. atomic positions.
    """
    i_idx = wp.tid()
    b = batch_idx[i_idx]
    k_max = k_vectors.shape[1]
    Q = quadrupoles[i_idx]
    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)
    for k_idx in range(k_max):
        k_vec = k_vectors[b, k_idx]
        kQ0 = (
            wp.float64(Q[0, 0]) * k_vec[0]
            + wp.float64(Q[0, 1]) * k_vec[1]
            + wp.float64(Q[0, 2]) * k_vec[2]
        )
        kQ1 = (
            wp.float64(Q[1, 0]) * k_vec[0]
            + wp.float64(Q[1, 1]) * k_vec[1]
            + wp.float64(Q[1, 2]) * k_vec[2]
        )
        kQ2 = (
            wp.float64(Q[2, 0]) * k_vec[0]
            + wp.float64(Q[2, 1]) * k_vec[1]
            + wp.float64(Q[2, 2]) * k_vec[2]
        )
        kQk = kQ0 * k_vec[0] + kQ1 * k_vec[1] + kQ2 * k_vec[2]
        p_r = coeff2[b, k_idx] * kQk
        gr = grad_rho[b, k_idx, 0]
        gi = grad_rho[b, k_idx, 1]
        a_k = -gi * p_r
        b_k = -gr * p_r
        contrib = a_k * cosines[k_idx, i_idx] + b_k * sines[k_idx, i_idx]
        gx += k_vec[0] * contrib
        gy += k_vec[1] * contrib
        gz += k_vec[2] * contrib
    scale_b = scale[b]
    grad_positions[i_idx, 0] = scale_b * gx
    grad_positions[i_idx, 1] = scale_b * gy
    grad_positions[i_idx, 2] = scale_b * gz


def _batch_position_gradient_rhoq_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=m),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.vec3d),
        wp.array2d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
        wp.array(dtype=wp.float64),
        wp.array(dtype=wp.int32),
        wp.array2d(dtype=wp.float64),
    ]


@wp.kernel
def _batch_project_features_quadrupole_kernel(
    potential: wp.array3d(dtype=wp.float64),
    receiver_phi_hat_l2: wp.array4d(dtype=wp.vec2d),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array2d(dtype=wp.float64),
    batch_idx: wp.array(dtype=wp.int32),
    features: wp.array3d(dtype=wp.float64),
):
    r"""Batched projection of :math:`V(\mathbf{k})` onto the l = 2 receiver basis.

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx, s_idx, lm_idx)``; each thread processes one (i_idx, s_idx, lm_idx) work item.

    Parameters
    ----------
    potential : wp.array3d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    receiver_phi_hat_l2 : wp.array4d, dtype wp.vec2d
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    batch_idx : wp.array, dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    features : wp.array3d, dtype wp.float64
        OUTPUT: projected per-atom features.
    """
    i_idx, s_idx, lm_idx = wp.tid()

    b = batch_idx[i_idx]

    n_k_max = potential.shape[1]
    acc = wp.float64(0.0)

    for k_idx in range(n_k_max):
        v_r = potential[b, k_idx, 0]
        v_i = potential[b, k_idx, 1]
        phi = receiver_phi_hat_l2[b, k_idx, s_idx, lm_idx]
        phi_r = phi[0]
        phi_i = phi[1]

        a = v_r * phi_r + v_i * phi_i
        b_term = v_r * phi_i - v_i * phi_r

        w = k_factor_proj[b, k_idx]
        c = cosines[k_idx, i_idx]
        s = sines[k_idx, i_idx]
        acc += w * (a * c + b_term * s)

    features[i_idx, s_idx, lm_idx] = wp.float64(2.0) * acc * _INV_TWO_PI_CUBED


@wp.kernel
def _batch_project_kphase_grad_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat: wp.array4d(dtype=wp.vec2d),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array2d(dtype=wp.float64),
    potential: wp.array3d(dtype=wp.float64),
    positions: wp.array(dtype=wp.vec3d),
    batch_idx: wp.array(dtype=wp.int32),
    grad_k_vectors: wp.array2d(dtype=wp.vec3d),
):
    r"""Batched backward of the l≤1 feature projection w.r.t. the k-phase ``k·r`` (stress path).

    Launch Grid
    -----------
    ``dim`` indexes ``(b, k_idx)``; each thread processes one (system, k-vector) work item.

    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array4d, dtype wp.vec2d
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array3d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    positions : wp.array, dtype wp.vec3d
        Atomic Cartesian positions.
    batch_idx : wp.array, dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    grad_k_vectors : wp.array2d, dtype wp.vec3d
        OUTPUT: gradient w.r.t. the k-vectors.
    """
    b, k_idx = wp.tid()

    n_total = grad_raw.shape[0]
    n_sigma = receiver_phi_hat.shape[2]
    n_lm = receiver_phi_hat.shape[3]

    v_r = potential[b, k_idx, 0]
    v_i = potential[b, k_idx, 1]
    w = k_factor_proj[b, k_idx]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for i in range(n_total):
        if batch_idx[i] == b:
            c = cosines[k_idx, i]
            s = sines[k_idx, i]

            q_r = wp.float64(0.0)
            q_i = wp.float64(0.0)
            for s_idx in range(n_sigma):
                for lm_idx in range(n_lm):
                    g = grad_raw[i, s_idx, lm_idx]
                    phi = receiver_phi_hat[b, k_idx, s_idx, lm_idx]
                    q_r += phi[0] * g
                    q_i += phi[1] * g

            c_k = v_r * q_i - v_i * q_r
            d_k = v_r * q_r + v_i * q_i
            per = w * (c_k * c - d_k * s)

            r_i = positions[i]
            gx += r_i[0] * per
            gy += r_i[1] * per
            gz += r_i[2] * per

    grad_k_vectors[b, k_idx] = wp.vec3d(
        _INV_TWO_PI_CUBED_TIMES_TWO * gx,
        _INV_TWO_PI_CUBED_TIMES_TWO * gy,
        _INV_TWO_PI_CUBED_TIMES_TWO * gz,
    )


@wp.kernel
def _batch_project_phihat_grad_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array2d(dtype=wp.float64),
    potential: wp.array3d(dtype=wp.float64),
    batch_idx: wp.array(dtype=wp.int32),
    grad_phi: wp.array4d(dtype=wp.vec2d),
):
    r"""Batched backward of the l≤1 feature projection w.r.t. ``receiver_phi_hat`` (stress path).

    Launch Grid
    -----------
    ``dim`` indexes ``(b, k_idx, s_idx, lm_idx)``; each thread processes one (system, k-vector, s_idx, lm_idx) work item.

    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array3d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    batch_idx : wp.array, dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    grad_phi : wp.array4d, dtype wp.vec2d
        OUTPUT: gradient w.r.t. the GTO Fourier coefficients :math:`\hat\phi(k)`.
    """
    b, k_idx, s_idx, lm_idx = wp.tid()

    n_total = grad_raw.shape[0]

    v_r = potential[b, k_idx, 0]
    v_i = potential[b, k_idx, 1]
    w = k_factor_proj[b, k_idx]

    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)

    for i in range(n_total):
        if batch_idx[i] == b:
            g = grad_raw[i, s_idx, lm_idx]
            c = cosines[k_idx, i]
            s = sines[k_idx, i]
            acc_r += g * (v_r * c - v_i * s)
            acc_i += g * (v_i * c + v_r * s)

    grad_phi[b, k_idx, s_idx, lm_idx] = wp.vec2d(
        _INV_TWO_PI_CUBED_TIMES_TWO * w * acc_r,
        _INV_TWO_PI_CUBED_TIMES_TWO * w * acc_i,
    )


@wp.kernel
def _batch_receiver_phi_hat_backward_quadrupole_kernel(
    grad_output: wp.array4d(dtype=wp.vec2d),  # (B, K_max, N_σ, 5) vec2d
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    k_norm2: wp.array2d(dtype=wp.float64),  # (B, K_max)
    sigmas: wp.array(dtype=wp.float64),  # (N_σ,)
    inv_cl_l2: wp.array(dtype=wp.float64),  # (N_σ,)
    grad_k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max) OUTPUT
    grad_k_norm2: wp.array2d(dtype=wp.float64),  # (B, K_max) OUTPUT
):
    r"""Batched per-(b, k) backward of receiver GTO Fourier (l = 2) w.r.t. (k_vec, k²).

    Launch Grid
    -----------
    ``dim`` indexes ``(b, k_idx)``; each thread processes one (system, k-vector) work item.

    Parameters
    ----------
    grad_output : wp.array4d, shape (B, K_max, N_σ, 5), dtype wp.vec2d
        Vec2d.
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    k_norm2 : wp.array2d, shape (B, K_max), dtype wp.float64
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigmas : wp.array, shape (N_σ,), dtype wp.float64
        Per-channel Gaussian (GTO) width parameters.
    inv_cl_l2 : wp.array, shape (N_σ,), dtype wp.float64
        Inverse :math:`l=2` overlap normalization constant.
    grad_k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        OUTPUT: gradient w.r.t. the k-vectors.
    grad_k_norm2 : wp.array2d, shape (B, K_max), dtype wp.float64
        OUTPUT: gradient w.r.t. :math:`|k|^2`.
    """
    b, k_idx = wp.tid()
    n_sigma = sigmas.shape[0]

    k_vec = k_vectors[b, k_idx]
    k2 = k_norm2[b, k_idx]
    kx = k_vec[0]
    ky = k_vec[1]
    kz = k_vec[2]

    sum_kx = wp.float64(0.0)
    sum_ky = wp.float64(0.0)
    sum_kz = wp.float64(0.0)
    sum_k2 = wp.float64(0.0)

    for s_idx in range(n_sigma):
        sigma = sigmas[s_idx]
        sigma2 = sigma * sigma
        sigma7 = sigma2 * sigma2 * sigma2 * sigma

        gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)
        coeff_l2 = -inv_cl_l2[s_idx] * _FOUR_PI_SQRT_PI_OVER_2 * gauss * sigma7

        gv0 = grad_output[b, k_idx, s_idx, 0]
        gv1 = grad_output[b, k_idx, s_idx, 1]
        gv2 = grad_output[b, k_idx, s_idx, 2]
        gv3 = grad_output[b, k_idx, s_idx, 3]
        gv4 = grad_output[b, k_idx, s_idx, 4]

        grads = _recv_l2_grad_kspace(
            gv0[0],
            gv1[0],
            gv2[0],
            gv3[0],
            gv4[0],
            coeff_l2,
            sigma2,
            kx,
            ky,
            kz,
            k2,
        )

        sum_kx += grads[0]
        sum_ky += grads[1]
        sum_kz += grads[2]
        sum_k2 += grads[3]

    grad_k_vectors[b, k_idx] = wp.vec3d(sum_kx, sum_ky, sum_kz)
    grad_k_norm2[b, k_idx] = sum_k2


@wp.kernel
def _batch_rho_kphase_grad_kernel(
    charges: wp.array(dtype=Any),  # (N_atoms,)
    dipoles: wp.array(dtype=Any),  # (N_atoms,) vec3
    positions: wp.array(dtype=Any),  # (N_atoms,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    source_phi_hat: wp.array4d(dtype=wp.float64),  # (B, N_k, 4, 2)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, N_k, 2)
    volume: wp.array(dtype=wp.float64),  # (B,)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    grad_k: wp.array3d(dtype=wp.float64),  # (B, N_k, 3) OUTPUT
):
    r"""Batched backward of :math:`\hat\rho(k)` w.r.t. the k-vector phase.

    Accumulates, per system ``b`` and k-index, the gradient of the
    reciprocal density w.r.t. the k-vector through the
    :math:`\cos(k\cdot r)` / :math:`\sin(k\cdot r)` phase factors.

    Launch Grid
    -----------
    ``dim`` indexes ``(b, k_idx)``; each thread processes one
    (system, k-vector) work item.

    Parameters
    ----------
    charges : wp.array, shape (N_atoms,), dtype Any
        Per-atom monopole charges.
    dipoles : wp.array, shape (N_atoms,), dtype Any (vec3)
        Per-atom Cartesian dipole moments.
    positions : wp.array, shape (N_atoms,), dtype Any (vec3)
        Atomic Cartesian positions.
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array4d, shape (B, N_k, 4, 2), dtype wp.float64
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array3d, shape (B, N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    volume : wp.array, shape (B,), dtype wp.float64
        Per-system unit-cell volume.
    atom_start : wp.array, shape (B,), dtype wp.int32
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array, shape (B,), dtype wp.int32
        Per-system end offset into the flat atom arrays.
    grad_k : wp.array3d, shape (B, N_k, 3), dtype wp.float64
        OUTPUT: gradient w.r.t. the k-vectors.
    """
    b, k_idx = wp.tid()
    gr = grad_rho[b, k_idx, 0]
    gi = grad_rho[b, k_idx, 1]
    pr0 = source_phi_hat[b, k_idx, 0, 0]
    pi0 = source_phi_hat[b, k_idx, 0, 1]
    pr1 = source_phi_hat[b, k_idx, 1, 0]
    pi1 = source_phi_hat[b, k_idx, 1, 1]
    pr2 = source_phi_hat[b, k_idx, 2, 0]
    pi2 = source_phi_hat[b, k_idx, 2, 1]
    pr3 = source_phi_hat[b, k_idx, 3, 0]
    pi3 = source_phi_hat[b, k_idx, 3, 1]
    gkx = wp.float64(0.0)
    gky = wp.float64(0.0)
    gkz = wp.float64(0.0)
    for i in range(atom_start[b], atom_end[b]):
        q = wp.float64(charges[i])
        mu = dipoles[i]
        mu_x = wp.float64(mu[0])
        mu_y = wp.float64(mu[1])
        mu_z = wp.float64(mu[2])
        p_r = pr0 * q + pr1 * mu_y + pr2 * mu_z + pr3 * mu_x
        p_i = pi0 * q + pi1 * mu_y + pi2 * mu_z + pi3 * mu_x
        a_k = gr * p_i - gi * p_r
        b_k = -(gr * p_r + gi * p_i)
        contrib = a_k * cosines[k_idx, i] + b_k * sines[k_idx, i]
        pos_i = positions[i]
        gkx += wp.float64(pos_i[0]) * contrib
        gky += wp.float64(pos_i[1]) * contrib
        gkz += wp.float64(pos_i[2]) * contrib
    scale = _TWO_PI_CUBED / volume[b]
    grad_k[b, k_idx, 0] = scale * gkx
    grad_k[b, k_idx, 1] = scale * gky
    grad_k[b, k_idx, 2] = scale * gkz


def _batch_rho_kphase_grad_sig(v, t):
    return [
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array4d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
        wp.array(dtype=wp.float64),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array3d(dtype=wp.float64),
    ]


@wp.kernel
def _batch_rho_phihat_grad_kernel(
    charges: wp.array(dtype=Any),  # (N_atoms,)
    dipoles: wp.array(dtype=Any),  # (N_atoms,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, N_k, 2)
    volume: wp.array(dtype=wp.float64),  # (B,)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    grad_phi: wp.array4d(dtype=wp.float64),  # (B, N_k, 4, 2) OUTPUT
):
    r"""Batched backward of :math:`\hat\rho(k)` w.r.t. :math:`\hat\phi(k)`.

    Accumulates, per system ``b`` and k-index, the gradient of the
    reciprocal density w.r.t. the GTO Fourier coefficients
    :math:`\hat\phi(k)` by summing the (charge, dipole) weighted phase
    factors over the system's atoms.

    Launch Grid
    -----------
    ``dim`` indexes ``(b, k_idx)``; each thread processes one
    (system, k-vector) work item.

    Parameters
    ----------
    charges : wp.array, shape (N_atoms,), dtype Any
        Per-atom monopole charges.
    dipoles : wp.array, shape (N_atoms,), dtype Any (vec3)
        Per-atom Cartesian dipole moments.
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    grad_rho : wp.array3d, shape (B, N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    volume : wp.array, shape (B,), dtype wp.float64
        Per-system unit-cell volume.
    atom_start : wp.array, shape (B,), dtype wp.int32
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array, shape (B,), dtype wp.int32
        Per-system end offset into the flat atom arrays.
    grad_phi : wp.array4d, shape (B, N_k, 4, 2), dtype wp.float64
        OUTPUT: gradient w.r.t. the GTO Fourier coefficients :math:`\hat\phi(k)`.
    """
    b, k_idx = wp.tid()
    c0 = wp.float64(0.0)
    c1 = wp.float64(0.0)
    c2 = wp.float64(0.0)
    c3 = wp.float64(0.0)
    s0 = wp.float64(0.0)
    s1 = wp.float64(0.0)
    s2 = wp.float64(0.0)
    s3 = wp.float64(0.0)
    for i in range(atom_start[b], atom_end[b]):
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        q = wp.float64(charges[i])
        mu = dipoles[i]
        mu_x = wp.float64(mu[0])
        mu_y = wp.float64(mu[1])
        mu_z = wp.float64(mu[2])
        c0 += cos_ki * q
        s0 += sin_ki * q
        c1 += cos_ki * mu_y
        s1 += sin_ki * mu_y
        c2 += cos_ki * mu_z
        s2 += sin_ki * mu_z
        c3 += cos_ki * mu_x
        s3 += sin_ki * mu_x
    scale = _TWO_PI_CUBED / volume[b]
    gr = grad_rho[b, k_idx, 0]
    gi = grad_rho[b, k_idx, 1]
    grad_phi[b, k_idx, 0, 0] = scale * (gr * c0 - gi * s0)
    grad_phi[b, k_idx, 0, 1] = scale * (gr * s0 + gi * c0)
    grad_phi[b, k_idx, 1, 0] = scale * (gr * c1 - gi * s1)
    grad_phi[b, k_idx, 1, 1] = scale * (gr * s1 + gi * c1)
    grad_phi[b, k_idx, 2, 0] = scale * (gr * c2 - gi * s2)
    grad_phi[b, k_idx, 2, 1] = scale * (gr * s2 + gi * c2)
    grad_phi[b, k_idx, 3, 0] = scale * (gr * c3 - gi * s3)
    grad_phi[b, k_idx, 3, 1] = scale * (gr * s3 + gi * c3)


def _batch_rho_phihat_grad_sig(v, t):
    return [
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
        wp.array(dtype=wp.float64),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array4d(dtype=wp.float64),
    ]


@wp.kernel
def _batch_rho_q_coeff2_grad_kernel(
    quadrupoles: wp.array(dtype=Any),  # (N_total,) mat33
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    volume: wp.array(dtype=wp.float64),  # (B,)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    grad_coeff2: wp.array2d(dtype=wp.float64),  # (B, K_max) OUTPUT
):
    r"""Batched backward of the :math:`l=2` :math:`\hat\rho(k)` w.r.t. ``coeff2``.

    Accumulates, per system ``b`` and k-index, the gradient of the
    quadrupole (:math:`l=2`) reciprocal density w.r.t. the per-k
    coefficient ``coeff2`` by summing the :math:`(k\cdot Q\cdot k)`
    weighted phase factors over the system's atoms.

    Launch Grid
    -----------
    ``dim`` indexes ``(b, k_idx)``; each thread processes one
    (system, k-vector) work item.

    Parameters
    ----------
    quadrupoles : wp.array, shape (N_total,), dtype Any (mat33)
        Per-atom Cartesian quadrupole moments.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Per-system reciprocal-space k-vectors.
    grad_rho : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    volume : wp.array, shape (B,), dtype wp.float64
        Per-system unit-cell volume.
    atom_start : wp.array, shape (B,), dtype wp.int32
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array, shape (B,), dtype wp.int32
        Per-system end offset into the flat atom arrays.
    grad_coeff2 : wp.array2d, shape (B, K_max), dtype wp.float64
        OUTPUT: gradient w.r.t. the :math:`l=2` coefficient ``coeff2``.
    """
    b, k_idx = wp.tid()
    k_vec = k_vectors[b, k_idx]
    C_Q = wp.float64(0.0)
    S_Q = wp.float64(0.0)
    for i in range(atom_start[b], atom_end[b]):
        Q = quadrupoles[i]
        kQ0 = (
            wp.float64(Q[0, 0]) * k_vec[0]
            + wp.float64(Q[0, 1]) * k_vec[1]
            + wp.float64(Q[0, 2]) * k_vec[2]
        )
        kQ1 = (
            wp.float64(Q[1, 0]) * k_vec[0]
            + wp.float64(Q[1, 1]) * k_vec[1]
            + wp.float64(Q[1, 2]) * k_vec[2]
        )
        kQ2 = (
            wp.float64(Q[2, 0]) * k_vec[0]
            + wp.float64(Q[2, 1]) * k_vec[1]
            + wp.float64(Q[2, 2]) * k_vec[2]
        )
        kQk = kQ0 * k_vec[0] + kQ1 * k_vec[1] + kQ2 * k_vec[2]
        C_Q += kQk * cosines[k_idx, i]
        S_Q += kQk * sines[k_idx, i]
    scale = _TWO_PI_CUBED / volume[b]
    gr = grad_rho[b, k_idx, 0]
    gi = grad_rho[b, k_idx, 1]
    grad_coeff2[b, k_idx] = scale * (gr * C_Q - gi * S_Q)


def _batch_rho_q_coeff2_grad_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=m),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.vec3d),
        wp.array3d(dtype=wp.float64),
        wp.array(dtype=wp.float64),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array2d(dtype=wp.float64),
    ]


@wp.kernel
def _batch_rho_q_kvec_grad_kernel(
    quadrupoles: wp.array(dtype=Any),  # (N_total,) mat33
    positions: wp.array(dtype=Any),  # (N_total,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    coeff2: wp.array2d(dtype=wp.float64),  # (B, K_max)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    volume: wp.array(dtype=wp.float64),  # (B,)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    grad_k: wp.array3d(dtype=wp.float64),  # (B, K_max, 3) OUTPUT
):
    r"""Batched backward of the :math:`l=2` :math:`\hat\rho(k)` w.r.t. the k-vector.

    Accumulates, per system ``b`` and k-index, the gradient of the
    quadrupole (:math:`l=2`) reciprocal density w.r.t. the k-vector,
    combining the explicit :math:`(k\cdot Q\cdot k)` dependence with the
    :math:`\cos(k\cdot r)` / :math:`\sin(k\cdot r)` phase dependence.

    Launch Grid
    -----------
    ``dim`` indexes ``(b, k_idx)``; each thread processes one
    (system, k-vector) work item.

    Parameters
    ----------
    quadrupoles : wp.array, shape (N_total,), dtype Any (mat33)
        Per-atom Cartesian quadrupole moments.
    positions : wp.array, shape (N_total,), dtype Any (vec3)
        Atomic Cartesian positions.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Per-system reciprocal-space k-vectors.
    coeff2 : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    grad_rho : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    volume : wp.array, shape (B,), dtype wp.float64
        Per-system unit-cell volume.
    atom_start : wp.array, shape (B,), dtype wp.int32
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array, shape (B,), dtype wp.int32
        Per-system end offset into the flat atom arrays.
    grad_k : wp.array3d, shape (B, K_max, 3), dtype wp.float64
        OUTPUT: gradient w.r.t. the k-vectors.
    """
    b, k_idx = wp.tid()
    k_vec = k_vectors[b, k_idx]
    gr = grad_rho[b, k_idx, 0]
    gi = grad_rho[b, k_idx, 1]
    c2 = coeff2[b, k_idx]
    gkx = wp.float64(0.0)
    gky = wp.float64(0.0)
    gkz = wp.float64(0.0)
    for i in range(atom_start[b], atom_end[b]):
        Q = quadrupoles[i]
        qk0 = (
            wp.float64(Q[0, 0]) * k_vec[0]
            + wp.float64(Q[0, 1]) * k_vec[1]
            + wp.float64(Q[0, 2]) * k_vec[2]
        )
        qk1 = (
            wp.float64(Q[1, 0]) * k_vec[0]
            + wp.float64(Q[1, 1]) * k_vec[1]
            + wp.float64(Q[1, 2]) * k_vec[2]
        )
        qk2 = (
            wp.float64(Q[2, 0]) * k_vec[0]
            + wp.float64(Q[2, 1]) * k_vec[1]
            + wp.float64(Q[2, 2]) * k_vec[2]
        )
        kQk = qk0 * k_vec[0] + qk1 * k_vec[1] + qk2 * k_vec[2]
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        w1 = wp.float64(2.0) * (gr * cos_ki - gi * sin_ki)
        w2 = -kQk * (gr * sin_ki + gi * cos_ki)
        pos_i = positions[i]
        gkx += w1 * qk0 + w2 * wp.float64(pos_i[0])
        gky += w1 * qk1 + w2 * wp.float64(pos_i[1])
        gkz += w1 * qk2 + w2 * wp.float64(pos_i[2])
    sc = _TWO_PI_CUBED / volume[b] * c2
    grad_k[b, k_idx, 0] = sc * gkx
    grad_k[b, k_idx, 1] = sc * gky
    grad_k[b, k_idx, 2] = sc * gkz


def _batch_rho_q_kvec_grad_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=m),
        wp.array(dtype=v),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.vec3d),
        wp.array2d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
        wp.array(dtype=wp.float64),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array3d(dtype=wp.float64),
    ]


@wp.kernel
def _batch_rho_q_moment_grad_kernel(
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    coeff2: wp.array2d(dtype=wp.float64),  # (B, K_max)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    scale: wp.array(dtype=wp.float64),  # (B,)
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    grad_quadrupoles: wp.array2d(dtype=wp.float64),  # (N_total, 9) OUTPUT
):
    r"""Batched Q-channel ∂L/∂Q_i (one thread per atom; symmetric 3x3).

    Launch Grid
    -----------
    dim = [N_total] — one thread per atom; sweeps all k-vectors of its system.


    Parameters
    ----------
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    coeff2 : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    grad_rho : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : wp.array, shape (B,), dtype wp.float64
        Scalar prefactor applied to the contribution.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    grad_quadrupoles : wp.array2d, shape (N_total, 9), dtype wp.float64
        OUTPUT: gradient w.r.t. the quadrupole moments.
    """
    i_idx = wp.tid()
    b = batch_idx[i_idx]
    k_max = k_vectors.shape[1]
    g00 = wp.float64(0.0)
    g01 = wp.float64(0.0)
    g02 = wp.float64(0.0)
    g11 = wp.float64(0.0)
    g12 = wp.float64(0.0)
    g22 = wp.float64(0.0)
    for k_idx in range(k_max):
        k_vec = k_vectors[b, k_idx]
        w = coeff2[b, k_idx] * (
            grad_rho[b, k_idx, 0] * cosines[k_idx, i_idx]
            - grad_rho[b, k_idx, 1] * sines[k_idx, i_idx]
        )
        g00 += w * k_vec[0] * k_vec[0]
        g01 += w * k_vec[0] * k_vec[1]
        g02 += w * k_vec[0] * k_vec[2]
        g11 += w * k_vec[1] * k_vec[1]
        g12 += w * k_vec[1] * k_vec[2]
        g22 += w * k_vec[2] * k_vec[2]
    scale_b = scale[b]
    grad_quadrupoles[i_idx, 0] = scale_b * g00
    grad_quadrupoles[i_idx, 1] = scale_b * g01
    grad_quadrupoles[i_idx, 2] = scale_b * g02
    grad_quadrupoles[i_idx, 3] = scale_b * g01
    grad_quadrupoles[i_idx, 4] = scale_b * g11
    grad_quadrupoles[i_idx, 5] = scale_b * g12
    grad_quadrupoles[i_idx, 6] = scale_b * g02
    grad_quadrupoles[i_idx, 7] = scale_b * g12
    grad_quadrupoles[i_idx, 8] = scale_b * g22


@wp.kernel
def _batch_rhoq_posgrad_backward_grad_rho_kernel(
    quadrupoles: wp.array(dtype=Any),  # (N_atoms,) mat33
    gg_pos: wp.array2d(dtype=wp.float64),  # (N_atoms, 3)
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, N_k)
    coeff2: wp.array2d(dtype=wp.float64),  # (B, N_k)
    scale: wp.array(dtype=wp.float64),  # (B,)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    grad_rho_grad: wp.array3d(dtype=wp.float64),  # (B, N_k, 2) OUTPUT
):
    r"""Batched K_a (one thread per (b, k); sweeps that system's atoms).

    Launch Grid
    -----------
    ``dim`` indexes ``(b, k_idx)``; each thread processes one (system, k-vector) work item.

    Parameters
    ----------
    quadrupoles : wp.array, shape (N_atoms,), dtype Any
        Mat33.
    gg_pos : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array2d, shape (B, N_k), dtype wp.vec3d
        Reciprocal-space k-vectors.
    coeff2 : wp.array2d, shape (B, N_k), dtype wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    scale : wp.array, shape (B,), dtype wp.float64
        Scalar prefactor applied to the contribution.
    atom_start : wp.array, shape (B,), dtype wp.int32
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array, shape (B,), dtype wp.int32
        Per-system end offset into the flat atom arrays.
    grad_rho_grad : wp.array3d, shape (B, N_k, 2), dtype wp.float64
        OUTPUT: gradient w.r.t. the incoming ``grad_rho``.
    """
    b, k_idx = wp.tid()
    k_vec = k_vectors[b, k_idx]
    acc_s = wp.float64(0.0)
    acc_c = wp.float64(0.0)
    i_lo = atom_start[b]
    i_hi = atom_end[b]
    for i in range(i_lo, i_hi):
        Q = quadrupoles[i]
        kQ0 = (
            wp.float64(Q[0, 0]) * k_vec[0]
            + wp.float64(Q[0, 1]) * k_vec[1]
            + wp.float64(Q[0, 2]) * k_vec[2]
        )
        kQ1 = (
            wp.float64(Q[1, 0]) * k_vec[0]
            + wp.float64(Q[1, 1]) * k_vec[1]
            + wp.float64(Q[1, 2]) * k_vec[2]
        )
        kQ2 = (
            wp.float64(Q[2, 0]) * k_vec[0]
            + wp.float64(Q[2, 1]) * k_vec[1]
            + wp.float64(Q[2, 2]) * k_vec[2]
        )
        kQk = kQ0 * k_vec[0] + kQ1 * k_vec[1] + kQ2 * k_vec[2]
        kdotgg = (
            k_vec[0] * gg_pos[i, 0] + k_vec[1] * gg_pos[i, 1] + k_vec[2] * gg_pos[i, 2]
        )
        u = kQk * kdotgg
        acc_s += u * sines[k_idx, i]
        acc_c += u * cosines[k_idx, i]
    nsc = -scale[b] * coeff2[b, k_idx]
    grad_rho_grad[b, k_idx, 0] = nsc * acc_s
    grad_rho_grad[b, k_idx, 1] = nsc * acc_c


@wp.kernel
def _batch_rhoq_posgrad_backward_positions_kernel(
    quadrupoles: wp.array(dtype=Any),  # (N_atoms,) mat33
    gg_pos: wp.array2d(dtype=wp.float64),  # (N_atoms, 3)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, N_k, 2)
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, N_k)
    coeff2: wp.array2d(dtype=wp.float64),  # (B, N_k)
    scale: wp.array(dtype=wp.float64),  # (B,)
    batch_idx: wp.array(dtype=wp.int32),  # (N_atoms,)
    grad_positions: wp.array2d(dtype=wp.float64),  # (N_atoms, 3) OUTPUT
):
    r"""Batched K_c (one thread per atom; position-Hessian diagonal).

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    quadrupoles : wp.array, shape (N_atoms,), dtype Any
        Mat33.
    gg_pos : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    grad_rho : wp.array3d, shape (B, N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array2d, shape (B, N_k), dtype wp.vec3d
        Reciprocal-space k-vectors.
    coeff2 : wp.array2d, shape (B, N_k), dtype wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    scale : wp.array, shape (B,), dtype wp.float64
        Scalar prefactor applied to the contribution.
    batch_idx : wp.array, shape (N_atoms,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    grad_positions : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        OUTPUT: gradient w.r.t. atomic positions.
    """
    i_idx = wp.tid()
    b = batch_idx[i_idx]
    k_max = k_vectors.shape[1]
    Q = quadrupoles[i_idx]
    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)
    for k_idx in range(k_max):
        k_vec = k_vectors[b, k_idx]
        kQ0 = (
            wp.float64(Q[0, 0]) * k_vec[0]
            + wp.float64(Q[0, 1]) * k_vec[1]
            + wp.float64(Q[0, 2]) * k_vec[2]
        )
        kQ1 = (
            wp.float64(Q[1, 0]) * k_vec[0]
            + wp.float64(Q[1, 1]) * k_vec[1]
            + wp.float64(Q[1, 2]) * k_vec[2]
        )
        kQ2 = (
            wp.float64(Q[2, 0]) * k_vec[0]
            + wp.float64(Q[2, 1]) * k_vec[1]
            + wp.float64(Q[2, 2]) * k_vec[2]
        )
        kQk = kQ0 * k_vec[0] + kQ1 * k_vec[1] + kQ2 * k_vec[2]
        kdotgg = (
            k_vec[0] * gg_pos[i_idx, 0]
            + k_vec[1] * gg_pos[i_idx, 1]
            + k_vec[2] * gg_pos[i_idx, 2]
        )
        gr = grad_rho[b, k_idx, 0]
        gi = grad_rho[b, k_idx, 1]
        v = (
            coeff2[b, k_idx]
            * kQk
            * kdotgg
            * (gr * cosines[k_idx, i_idx] - gi * sines[k_idx, i_idx])
        )
        gx += v * k_vec[0]
        gy += v * k_vec[1]
        gz += v * k_vec[2]
    nscale = -scale[b]
    grad_positions[i_idx, 0] = nscale * gx
    grad_positions[i_idx, 1] = nscale * gy
    grad_positions[i_idx, 2] = nscale * gz


@wp.kernel
def _batch_rhoq_posgrad_backward_quad_kernel(
    gg_pos: wp.array2d(dtype=wp.float64),  # (N_atoms, 3)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, N_k, 2)
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, N_k)
    coeff2: wp.array2d(dtype=wp.float64),  # (B, N_k)
    scale: wp.array(dtype=wp.float64),  # (B,)
    batch_idx: wp.array(dtype=wp.int32),  # (N_atoms,)
    grad_quadrupoles: wp.array2d(dtype=wp.float64),  # (N_atoms, 9) OUTPUT
):
    r"""Batched K_b (one thread per atom; independent of Q; symmetric).

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    gg_pos : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    grad_rho : wp.array3d, shape (B, N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array2d, shape (B, N_k), dtype wp.vec3d
        Reciprocal-space k-vectors.
    coeff2 : wp.array2d, shape (B, N_k), dtype wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    scale : wp.array, shape (B,), dtype wp.float64
        Scalar prefactor applied to the contribution.
    batch_idx : wp.array, shape (N_atoms,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    grad_quadrupoles : wp.array2d, shape (N_atoms, 9), dtype wp.float64
        OUTPUT: gradient w.r.t. the quadrupole moments.
    """
    i_idx = wp.tid()
    b = batch_idx[i_idx]
    k_max = k_vectors.shape[1]
    g00 = wp.float64(0.0)
    g01 = wp.float64(0.0)
    g02 = wp.float64(0.0)
    g11 = wp.float64(0.0)
    g12 = wp.float64(0.0)
    g22 = wp.float64(0.0)
    for k_idx in range(k_max):
        k_vec = k_vectors[b, k_idx]
        kdotgg = (
            k_vec[0] * gg_pos[i_idx, 0]
            + k_vec[1] * gg_pos[i_idx, 1]
            + k_vec[2] * gg_pos[i_idx, 2]
        )
        gr = grad_rho[b, k_idx, 0]
        gi = grad_rho[b, k_idx, 1]
        w = (
            coeff2[b, k_idx]
            * kdotgg
            * (gr * sines[k_idx, i_idx] + gi * cosines[k_idx, i_idx])
        )
        g00 += w * k_vec[0] * k_vec[0]
        g01 += w * k_vec[0] * k_vec[1]
        g02 += w * k_vec[0] * k_vec[2]
        g11 += w * k_vec[1] * k_vec[1]
        g12 += w * k_vec[1] * k_vec[2]
        g22 += w * k_vec[2] * k_vec[2]
    nscale = -scale[b]
    grad_quadrupoles[i_idx, 0] = nscale * g00
    grad_quadrupoles[i_idx, 1] = nscale * g01
    grad_quadrupoles[i_idx, 2] = nscale * g02
    grad_quadrupoles[i_idx, 3] = nscale * g01
    grad_quadrupoles[i_idx, 4] = nscale * g11
    grad_quadrupoles[i_idx, 5] = nscale * g12
    grad_quadrupoles[i_idx, 6] = nscale * g02
    grad_quadrupoles[i_idx, 7] = nscale * g12
    grad_quadrupoles[i_idx, 8] = nscale * g22


def _batch_rhoq_posgrad_bw_grad_rho_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=m),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.vec3d),
        wp.array2d(dtype=wp.float64),
        wp.array(dtype=wp.float64),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array3d(dtype=wp.float64),
    ]


def _batch_rhoq_posgrad_bw_positions_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=m),
        wp.array2d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.vec3d),
        wp.array2d(dtype=wp.float64),
        wp.array(dtype=wp.float64),
        wp.array(dtype=wp.int32),
        wp.array2d(dtype=wp.float64),
    ]


@wp.kernel
def _batch_v_grad_from_feat_grad_backward_positions_quadrupole_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),  # (N_total, N_σ, 5)
    receiver_phi_hat_l2: wp.array4d(dtype=wp.vec2d),  # (B, K_max, N_σ, 5) vec2d
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_factor_proj: wp.array2d(dtype=wp.float64),  # (B, K_max)
    gg_v: wp.array3d(dtype=wp.float64),  # (B, K_max, 2)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, K_max)
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,)
    ggrad_positions: wp.array2d(dtype=wp.float64),  # (N_total, 3) OUTPUT
):
    r"""Batched K9 — ``l = 2`` ``∂²L / (∂r_i ∂…)`` via v_gradient_from_feature_grad.

    One thread per atom (flat across batch). ``batch_idx[i]`` picks the
    system; inner ``K_max`` loop with the five ``l = 2`` angular channels.
    Pad rows have ``k_factor_proj = 0`` and ``receiver_phi_hat_l2 = 0`` so
    they contribute nothing.

    Launch Grid
    -----------
    ``dim = N_total``.


    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_total, N_σ, 5), dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array4d, shape (B, K_max, N_σ, 5), dtype wp.vec2d
        Vec2d.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    gg_v : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        Second-order upstream gradient w.r.t. ``v`` (HVP seed).
    k_vectors : wp.array2d, shape (B, K_max), dtype wp.vec3d
        Reciprocal-space k-vectors.
    batch_idx : wp.array, shape (N_total,), dtype wp.int32
        Per-atom system index into the batch (or scalar system id).
    ggrad_positions : wp.array2d, shape (N_total, 3), dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. positions.
    """
    i_idx = wp.tid()
    b = batch_idx[i_idx]
    k_max = k_vectors.shape[1]
    n_sigma = receiver_phi_hat_l2.shape[2]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(k_max):
        k_vec = k_vectors[b, k_idx]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        kfp = k_factor_proj[b, k_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(5):
                g = grad_raw[i_idx, s_idx, lm_idx]
                phi = receiver_phi_hat_l2[b, k_idx, s_idx, lm_idx]
                q_r += phi[0] * g
                q_i += phi[1] * g

        gvr = gg_v[b, k_idx, 0]
        gvi = gg_v[b, k_idx, 1]

        cos_term = gvr * q_i - gvi * q_r
        sin_term = gvr * q_r + gvi * q_i

        weight = kfp * (cos_ki * cos_term - sin_ki * sin_term)
        gx += k_vec[0] * weight
        gy += k_vec[1] * weight
        gz += k_vec[2] * weight

    ggrad_positions[i_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * gx
    ggrad_positions[i_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * gy
    ggrad_positions[i_idx, 2] = _INV_TWO_PI_CUBED_TIMES_TWO * gz


@wp.kernel
def _batch_v_gradient_from_feature_grad_quadrupole_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),  # (N_total, N_σ, 5)
    receiver_phi_hat_l2: wp.array4d(dtype=wp.vec2d),  # (B, K_max, N_σ, 5) vec2d
    cosines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (K_max, N_total)
    k_factor_proj: wp.array2d(dtype=wp.float64),  # (B, K_max)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    grad_v: wp.array3d(dtype=wp.float64),  # (B, K_max, 2) OUTPUT
):
    r"""Batched ``∂L/∂V(k)`` backward of ``l = 2`` feature projection.

    One thread per ``(b, k_idx)``. Inner sum over atoms in system
    ``b`` (from ``atom_start[b]`` to ``atom_end[b]``) and over the
    ``(σ, lm)`` feature axes with the five ``l = 2`` angular channels.
    ``receiver_phi_hat_l2`` stored as ``vec2d`` for the ``(real, imag)``
    axis to fit within 4-D Warp arrays.

    Launch Grid
    -----------
    ``dim = (B, K_max)``.


    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_total, N_σ, 5), dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array4d, shape (B, K_max, N_σ, 5), dtype wp.vec2d
        Vec2d.
    cosines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (K_max, N_total), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array2d, shape (B, K_max), dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    atom_start : wp.array, shape (B,), dtype wp.int32
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array, shape (B,), dtype wp.int32
        Per-system end offset into the flat atom arrays.
    grad_v : wp.array3d, shape (B, K_max, 2), dtype wp.float64
        OUTPUT: gradient w.r.t. the input feature vector ``v``.
    """
    b, k_idx = wp.tid()
    n_sigma = receiver_phi_hat_l2.shape[2]

    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)

    i_lo = atom_start[b]
    i_hi = atom_end[b]
    for i_idx in range(i_lo, i_hi):
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(5):
                g = grad_raw[i_idx, s_idx, lm_idx]
                phi = receiver_phi_hat_l2[b, k_idx, s_idx, lm_idx]
                q_r += phi[0] * g
                q_i += phi[1] * g

        acc_r += q_r * cos_ki + q_i * sin_ki
        acc_i += q_i * cos_ki - q_r * sin_ki

    kfp = k_factor_proj[b, k_idx]
    grad_v[b, k_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_r
    grad_v[b, k_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_i


@wp.kernel
def _eval_gto_fourier_q_kernel(
    k_norm2: wp.array(dtype=wp.float64),
    sigma: wp.float64,
    inv_cl_l0: wp.float64,
    coeff2: wp.array(dtype=wp.float64),
):
    r"""Per-k Cartesian-quadrupole coefficient ``coeff2(k) = -0.5 * phi0(k)``.

    Reuses the l=0 radial form (``inv_cl_l0``) for the Cartesian-quadrupole
    reciprocal channel. For each k-vector,

    .. math::

        \mathrm{coeff2}(\mathbf{k}) \;=\; -\tfrac{1}{2}\,\phi_0(\mathbf{k}),
        \qquad
        \phi_0(\mathbf{k}) \;=\;
            \frac{1}{C_0(\sigma)}\,4\pi\sqrt{\tfrac{\pi}{2}}\,\sigma^3\,
            e^{-k^2 \sigma^2 / 2}\, Y_0^0.

    Launch Grid
    -----------
    dim = [N_k] — one thread per k-vector.

    Parameters
    ----------
    k_norm2 : wp.array, shape (N_k,), dtype wp.float64
        Pre-computed :math:`|\mathbf{k}|^2`.
    sigma : wp.float64
        Single-σ density-basis width.
    inv_cl_l0 : wp.float64
        Host-computed :math:`1/C_0(\sigma)` factor.
    coeff2 : wp.array, shape (N_k,), dtype wp.float64
        OUTPUT. Per-k Cartesian-quadrupole coefficient.
    """
    idx = wp.tid()
    k2 = k_norm2[idx]
    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)
    phi0 = inv_cl_l0 * _FOUR_PI_SQRT_PI_OVER_2 * sigma3 * gauss * Y00_COEFF
    coeff2[idx] = -wp.float64(0.5) * phi0


@wp.kernel
def _eval_receiver_gto_fourier_quadrupole_kernel(
    k_vectors: wp.array(dtype=wp.vec3d),
    k_norm2: wp.array(dtype=wp.float64),
    sigmas: wp.array(dtype=wp.float64),
    inv_cl_l2: wp.array(dtype=wp.float64),
    output: wp.array4d(dtype=wp.float64),
):
    r"""Evaluate receiver-basis :math:`\hat\phi_{2,m}^{\sigma_r}(\mathbf{k})` across multi-σ at l = 2.

    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx, s_idx)``; each thread processes one (k-vector, s_idx) work item.

    Parameters
    ----------
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    k_norm2 : wp.array, dtype wp.float64
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigmas : wp.array, dtype wp.float64
        Per-channel Gaussian (GTO) width parameters.
    inv_cl_l2 : wp.array, dtype wp.float64
        Inverse :math:`l=2` overlap normalization constant.
    output : wp.array4d, dtype wp.float64
        OUTPUT: GTO Fourier coefficients :math:`\hat\phi_{l,m}^{\sigma}(k)`.
    """
    k_idx, s_idx = wp.tid()

    k_vec = k_vectors[k_idx]
    k2 = k_norm2[k_idx]
    kx = k_vec[0]
    ky = k_vec[1]
    kz = k_vec[2]

    sigma = sigmas[s_idx]
    sigma2 = sigma * sigma
    sigma7 = sigma2 * sigma2 * sigma2 * sigma

    gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)

    coeff_l2 = -inv_cl_l2[s_idx] * _FOUR_PI_SQRT_PI_OVER_2 * gauss * sigma7

    zero = wp.float64(0.0)

    output[k_idx, s_idx, 0, 0] = coeff_l2 * Y2_M2_COEFF * kx * ky
    output[k_idx, s_idx, 0, 1] = zero

    output[k_idx, s_idx, 1, 0] = coeff_l2 * Y2_M1_COEFF * ky * kz
    output[k_idx, s_idx, 1, 1] = zero

    output[k_idx, s_idx, 2, 0] = (
        coeff_l2 * Y2_0_COEFF * (wp.float64(3.0) * kz * kz - k2)
    )
    output[k_idx, s_idx, 2, 1] = zero

    output[k_idx, s_idx, 3, 0] = coeff_l2 * Y2_P1_COEFF * kx * kz
    output[k_idx, s_idx, 3, 1] = zero

    output[k_idx, s_idx, 4, 0] = coeff_l2 * Y2_P2_COEFF * (kx * kx - ky * ky)
    output[k_idx, s_idx, 4, 1] = zero


@wp.kernel
def _feat_position_grad_backward_grad_raw_quadrupole_kernel(
    receiver_phi_hat_l2: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    potential: wp.array2d(dtype=wp.float64),
    gg_positions: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    ggrad_grad_raw: wp.array3d(dtype=wp.float64),
):
    r"""l=2 K6 — backward of :func:`position_gradient_from_feature_grad_quadrupole` w.r.t. grad_raw.

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx, s_idx, lm_idx)``; each thread processes one (i_idx, s_idx, lm_idx) work item.

    Parameters
    ----------
    receiver_phi_hat_l2 : wp.array4d, dtype wp.float64
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array2d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    ggrad_grad_raw : wp.array3d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. ``grad_raw``.
    """
    i_idx, s_idx, lm_idx = wp.tid()
    n_k = cosines.shape[0]

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    acc = wp.float64(0.0)
    for k_idx in range(n_k):
        k_vec = k_vectors[k_idx]
        h = k_vec[0] * gp_x + k_vec[1] * gp_y + k_vec[2] * gp_z
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        v_r = potential[k_idx, 0]
        v_i = potential[k_idx, 1]
        phi_r = receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 0]
        phi_i = receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 1]
        kfp = k_factor_proj[k_idx]

        dc = v_r * phi_i - v_i * phi_r
        dd = v_r * phi_r + v_i * phi_i
        acc += kfp * h * (dc * cos_ki - dd * sin_ki)

    ggrad_grad_raw[i_idx, s_idx, lm_idx] = _INV_TWO_PI_CUBED_TIMES_TWO * acc


@wp.kernel
def _feat_position_grad_backward_positions_quadrupole_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat_l2: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    potential: wp.array2d(dtype=wp.float64),
    gg_positions: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    ggrad_positions: wp.array2d(dtype=wp.float64),
):
    r"""l=2 K8 — backward of :func:`position_gradient_from_feature_grad_quadrupole` w.r.t. positions.

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array4d, dtype wp.float64
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array2d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    ggrad_positions : wp.array2d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. positions.
    """
    i_idx = wp.tid()
    n_k = cosines.shape[0]
    n_sigma = receiver_phi_hat_l2.shape[1]

    gp_x = gg_positions[i_idx, 0]
    gp_y = gg_positions[i_idx, 1]
    gp_z = gg_positions[i_idx, 2]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(n_k):
        k_vec = k_vectors[k_idx]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        v_r = potential[k_idx, 0]
        v_i = potential[k_idx, 1]
        kfp = k_factor_proj[k_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(5):
                g = grad_raw[i_idx, s_idx, lm_idx]
                q_r += receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 0] * g
                q_i += receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 1] * g

        c_k = v_r * q_i - v_i * q_r
        d_k = v_r * q_r + v_i * q_i

        h = k_vec[0] * gp_x + k_vec[1] * gp_y + k_vec[2] * gp_z
        weight = -kfp * h * (c_k * sin_ki + d_k * cos_ki)
        gx += k_vec[0] * weight
        gy += k_vec[1] * weight
        gz += k_vec[2] * weight

    ggrad_positions[i_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * gx
    ggrad_positions[i_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * gy
    ggrad_positions[i_idx, 2] = _INV_TWO_PI_CUBED_TIMES_TWO * gz


@wp.kernel
def _feat_position_grad_backward_v_quadrupole_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat_l2: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    gg_positions: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    ggrad_v: wp.array2d(dtype=wp.float64),
):
    r"""l=2 K7 — backward of :func:`position_gradient_from_feature_grad_quadrupole` w.r.t. V(k).

    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx)``; each thread processes one k-vector index.

    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array4d, dtype wp.float64
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    gg_positions : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    ggrad_v : wp.array2d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. ``v``.
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat_l2.shape[1]

    k_vec = k_vectors[k_idx]
    kfp = k_factor_proj[k_idx]

    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)

    for i_idx in range(n_atoms):
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(5):
                g = grad_raw[i_idx, s_idx, lm_idx]
                q_r += receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 0] * g
                q_i += receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 1] * g

        h = (
            k_vec[0] * gg_positions[i_idx, 0]
            + k_vec[1] * gg_positions[i_idx, 1]
            + k_vec[2] * gg_positions[i_idx, 2]
        )
        acc_r += h * (q_i * cos_ki - q_r * sin_ki)
        acc_i += h * (-q_r * cos_ki - q_i * sin_ki)

    ggrad_v[k_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_r
    ggrad_v[k_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_i


@wp.kernel
def _position_gradient_from_feature_grad_quadrupole_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat_l2: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    potential: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    grad_positions: wp.array2d(dtype=wp.float64),
):
    r"""Analytical ``∂L/∂r_i`` backward of :func:`project_features_quadrupole`.

    One thread per atom; inner k-loop with the same ``(C, D)`` combination as
    the l≤1 :func:`_position_gradient_from_feature_grad_kernel`, but the Q
    reduction runs over the 5 l=2 channels. ``dim = N_atoms``.


    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array4d, dtype wp.float64
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array2d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    grad_positions : wp.array2d, dtype wp.float64
        OUTPUT: gradient w.r.t. atomic positions.
    """
    i_idx = wp.tid()

    n_k = cosines.shape[0]
    n_sigma = receiver_phi_hat_l2.shape[1]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(n_k):
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        v_r = potential[k_idx, 0]
        v_i = potential[k_idx, 1]
        kfp = k_factor_proj[k_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(5):
                g = grad_raw[i_idx, s_idx, lm_idx]
                q_r += receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 0] * g
                q_i += receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 1] * g

        c_k = v_r * q_i - v_i * q_r
        d_k = v_r * q_r + v_i * q_i

        per_k = kfp * (c_k * cos_ki - d_k * sin_ki)
        k_vec = k_vectors[k_idx]
        gx += k_vec[0] * per_k
        gy += k_vec[1] * per_k
        gz += k_vec[2] * per_k

    grad_positions[i_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * gx
    grad_positions[i_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * gy
    grad_positions[i_idx, 2] = _INV_TWO_PI_CUBED_TIMES_TWO * gz


@wp.kernel
def _position_gradient_from_rhoq_kernel(
    quadrupoles: wp.array(dtype=Any),  # (N_atoms,) mat33
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    coeff2: wp.array(dtype=wp.float64),  # (N_k,)
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    scale: wp.float64,
    grad_positions: wp.array2d(dtype=wp.float64),  # (N_atoms, 3) OUTPUT
):
    r"""Position gradient of the Q channel (backward of ``_assemble_rho_q``).

    Mirrors the rho-k position-gradient: per atom, accumulates over all
    k-vectors ``k · contrib`` where ``contrib = a_k·cos + b_k·sin`` with
    ``a_k = -gi·p_r``, ``b_k = -gr·p_r``, ``p_r = coeff2[k]·(k·Q·k)``.

    Launch Grid
    -----------
    dim = [N_atoms] — one thread per atom; sweeps all k-vectors.


    Parameters
    ----------
    quadrupoles : wp.array, shape (N_atoms,), dtype Any
        Mat33.
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-space k-vectors.
    coeff2 : wp.array, shape (N_k,), dtype wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    grad_rho : wp.array2d, shape (N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    grad_positions : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        OUTPUT: gradient w.r.t. atomic positions.
    """
    i_idx = wp.tid()
    n_k = cosines.shape[0]
    Q = quadrupoles[i_idx]
    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)
    for k_idx in range(n_k):
        k_vec = k_vectors[k_idx]
        kQ0 = (
            wp.float64(Q[0, 0]) * k_vec[0]
            + wp.float64(Q[0, 1]) * k_vec[1]
            + wp.float64(Q[0, 2]) * k_vec[2]
        )
        kQ1 = (
            wp.float64(Q[1, 0]) * k_vec[0]
            + wp.float64(Q[1, 1]) * k_vec[1]
            + wp.float64(Q[1, 2]) * k_vec[2]
        )
        kQ2 = (
            wp.float64(Q[2, 0]) * k_vec[0]
            + wp.float64(Q[2, 1]) * k_vec[1]
            + wp.float64(Q[2, 2]) * k_vec[2]
        )
        kQk = kQ0 * k_vec[0] + kQ1 * k_vec[1] + kQ2 * k_vec[2]
        p_r = coeff2[k_idx] * kQk
        gr = grad_rho[k_idx, 0]
        gi = grad_rho[k_idx, 1]
        a_k = -gi * p_r
        b_k = -gr * p_r
        contrib = a_k * cosines[k_idx, i_idx] + b_k * sines[k_idx, i_idx]
        gx += k_vec[0] * contrib
        gy += k_vec[1] * contrib
        gz += k_vec[2] * contrib
    grad_positions[i_idx, 0] = scale * gx
    grad_positions[i_idx, 1] = scale * gy
    grad_positions[i_idx, 2] = scale * gz


def _position_gradient_rhoq_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=m),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array(dtype=wp.vec3d),
        wp.array(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.float64,
        wp.array2d(dtype=wp.float64),
    ]


@wp.kernel
def _project_features_quadrupole_kernel(
    potential: wp.array2d(dtype=wp.float64),
    receiver_phi_hat_l2: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    features: wp.array3d(dtype=wp.float64),
):
    r"""Project :math:`V(\mathbf{k})` onto the l = 2 receiver basis at every atom.

    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx, s_idx, lm_idx)``; each thread processes one (i_idx, s_idx, lm_idx) work item.

    Parameters
    ----------
    potential : wp.array2d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    receiver_phi_hat_l2 : wp.array4d, dtype wp.float64
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    features : wp.array3d, dtype wp.float64
        OUTPUT: projected per-atom features.
    """
    i_idx, s_idx, lm_idx = wp.tid()

    n_k = potential.shape[0]
    acc = wp.float64(0.0)

    for k_idx in range(n_k):
        v_r = potential[k_idx, 0]
        v_i = potential[k_idx, 1]
        phi_r = receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 0]
        phi_i = receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 1]

        a = v_r * phi_r + v_i * phi_i
        b = v_r * phi_i - v_i * phi_r

        w = k_factor_proj[k_idx]
        c = cosines[k_idx, i_idx]
        s = sines[k_idx, i_idx]
        acc += w * (a * c + b * s)

    features[i_idx, s_idx, lm_idx] = wp.float64(2.0) * acc * _INV_TWO_PI_CUBED


@wp.kernel
def _project_kphase_grad_dipole_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    potential: wp.array2d(dtype=wp.float64),
    positions: wp.array(dtype=wp.vec3d),
    grad_k_vectors: wp.array2d(dtype=wp.float64),
):
    r"""Backward of the l≤1 feature projection w.r.t. the k-phase ``k·r`` (stress path).

    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx)``; each thread processes one k-vector index.

    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array4d, dtype wp.float64
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array2d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    positions : wp.array, dtype wp.vec3d
        Atomic Cartesian positions.
    grad_k_vectors : wp.array2d, dtype wp.float64
        OUTPUT: gradient w.r.t. the k-vectors.
    """
    k_idx = wp.tid()

    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]
    n_lm = receiver_phi_hat.shape[2]

    v_r = potential[k_idx, 0]
    v_i = potential[k_idx, 1]
    w = k_factor_proj[k_idx]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for i in range(n_atoms):
        c = cosines[k_idx, i]
        s = sines[k_idx, i]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(n_lm):
                g = grad_raw[i, s_idx, lm_idx]
                q_r += receiver_phi_hat[k_idx, s_idx, lm_idx, 0] * g
                q_i += receiver_phi_hat[k_idx, s_idx, lm_idx, 1] * g

        c_k = v_r * q_i - v_i * q_r
        d_k = v_r * q_r + v_i * q_i
        per = w * (c_k * c - d_k * s)

        r_i = positions[i]
        gx += r_i[0] * per
        gy += r_i[1] * per
        gz += r_i[2] * per

    grad_k_vectors[k_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * gx
    grad_k_vectors[k_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * gy
    grad_k_vectors[k_idx, 2] = _INV_TWO_PI_CUBED_TIMES_TWO * gz


@wp.kernel
def _project_phihat_grad_dipole_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    potential: wp.array2d(dtype=wp.float64),
    grad_phi: wp.array4d(dtype=wp.float64),
):
    r"""Backward of the l≤1 feature projection w.r.t. ``receiver_phi_hat`` (stress path).

    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx, s_idx, lm_idx)``; each thread processes one (k-vector, s_idx, lm_idx) work item.

    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array2d, dtype wp.float64
        Per-k reciprocal-space potential factor.
    grad_phi : wp.array4d, dtype wp.float64
        OUTPUT: gradient w.r.t. the GTO Fourier coefficients :math:`\hat\phi(k)`.
    """
    k_idx, s_idx, lm_idx = wp.tid()

    n_atoms = cosines.shape[1]

    v_r = potential[k_idx, 0]
    v_i = potential[k_idx, 1]
    w = k_factor_proj[k_idx]

    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)

    for i in range(n_atoms):
        g = grad_raw[i, s_idx, lm_idx]
        c = cosines[k_idx, i]
        s = sines[k_idx, i]
        acc_r += g * (v_r * c - v_i * s)
        acc_i += g * (v_i * c + v_r * s)

    grad_phi[k_idx, s_idx, lm_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * w * acc_r
    grad_phi[k_idx, s_idx, lm_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * w * acc_i


@wp.kernel
def _receiver_phi_hat_backward_quadrupole_kernel(
    grad_output: wp.array4d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    k_norm2: wp.array(dtype=wp.float64),
    sigmas: wp.array(dtype=wp.float64),
    inv_cl_l2: wp.array(dtype=wp.float64),
    grad_k_vectors: wp.array(dtype=wp.vec3d),
    grad_k_norm2: wp.array(dtype=wp.float64),
):
    r"""Analytical backward of :func:`eval_receiver_gto_fourier_quadrupole` w.r.t. ``(k_vectors, k_norm2)``.

    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx)``; each thread processes one k-vector index.

    Parameters
    ----------
    grad_output : wp.array4d, dtype wp.float64
        Upstream gradient flowing into this backward kernel.
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    k_norm2 : wp.array, dtype wp.float64
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigmas : wp.array, dtype wp.float64
        Per-channel Gaussian (GTO) width parameters.
    inv_cl_l2 : wp.array, dtype wp.float64
        Inverse :math:`l=2` overlap normalization constant.
    grad_k_vectors : wp.array, dtype wp.vec3d
        OUTPUT: gradient w.r.t. the k-vectors.
    grad_k_norm2 : wp.array, dtype wp.float64
        OUTPUT: gradient w.r.t. :math:`|k|^2`.
    """
    k_idx = wp.tid()
    n_sigma = sigmas.shape[0]

    k_vec = k_vectors[k_idx]
    k2 = k_norm2[k_idx]
    kx = k_vec[0]
    ky = k_vec[1]
    kz = k_vec[2]

    sum_kx = wp.float64(0.0)
    sum_ky = wp.float64(0.0)
    sum_kz = wp.float64(0.0)
    sum_k2 = wp.float64(0.0)

    for s_idx in range(n_sigma):
        sigma = sigmas[s_idx]
        sigma2 = sigma * sigma
        sigma7 = sigma2 * sigma2 * sigma2 * sigma

        gauss = wp.exp(-wp.float64(0.5) * k2 * sigma2)
        coeff_l2 = -inv_cl_l2[s_idx] * _FOUR_PI_SQRT_PI_OVER_2 * gauss * sigma7

        grads = _recv_l2_grad_kspace(
            grad_output[k_idx, s_idx, 0, 0],
            grad_output[k_idx, s_idx, 1, 0],
            grad_output[k_idx, s_idx, 2, 0],
            grad_output[k_idx, s_idx, 3, 0],
            grad_output[k_idx, s_idx, 4, 0],
            coeff_l2,
            sigma2,
            kx,
            ky,
            kz,
            k2,
        )

        sum_kx += grads[0]
        sum_ky += grads[1]
        sum_kz += grads[2]
        sum_k2 += grads[3]

    grad_k_vectors[k_idx] = wp.vec3d(sum_kx, sum_ky, sum_kz)
    grad_k_norm2[k_idx] = sum_k2


@wp.func
def _recv_l2_grad_kspace(
    g_m2: wp.float64,
    g_m1: wp.float64,
    g_0: wp.float64,
    g_p1: wp.float64,
    g_p2: wp.float64,
    coeff_l2: wp.float64,
    sigma2: wp.float64,
    kx: wp.float64,
    ky: wp.float64,
    kz: wp.float64,
    k2: wp.float64,
) -> wp.vec4d:
    r"""Per-(k, σ) cotangent → (grad_kx, grad_ky, grad_kz, grad_k2) for the
    l=2 receiver block. ``g_*`` are the real cotangents on the 5 columns;
    ``coeff_l2 = -inv_cl_l2 · 4π√(π/2) · σ⁷ · e^{-k²σ²/2}``.

    Parameters
    ----------
    g_m2 : wp.float64
        Per-k :math:`m=-2` spherical component.
    g_m1 : wp.float64
        Per-k :math:`m=-1` spherical component.
    g_0 : wp.float64
        Per-k :math:`m=0` spherical component.
    g_p1 : wp.float64
        Per-k :math:`m=+1` spherical component.
    g_p2 : wp.float64
        Per-k :math:`m=+2` spherical component.
    coeff_l2 : wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    sigma2 : wp.float64
        Squared Gaussian width :math:`\sigma^2`.
    kx : wp.float64
        x-component of the k-vector.
    ky : wp.float64
        y-component of the k-vector.
    kz : wp.float64
        z-component of the k-vector.
    k2 : wp.float64
        Squared k-vector magnitude :math:`|k|^2`.

    Returns
    -------
    wp.vec4d
        Return value (see summary).
    """
    p_m2 = Y2_M2_COEFF * kx * ky
    p_m1 = Y2_M1_COEFF * ky * kz
    p_0 = Y2_0_COEFF * (wp.float64(3.0) * kz * kz - k2)
    p_p1 = Y2_P1_COEFF * kx * kz
    p_p2 = Y2_P2_COEFF * (kx * kx - ky * ky)

    s_poly = g_m2 * p_m2 + g_m1 * p_m1 + g_0 * p_0 + g_p1 * p_p1 + g_p2 * p_p2

    two = wp.float64(2.0)

    grad_kx = coeff_l2 * (
        g_m2 * Y2_M2_COEFF * ky
        + g_p1 * Y2_P1_COEFF * kz
        + g_p2 * Y2_P2_COEFF * two * kx
    )

    grad_ky = coeff_l2 * (
        g_m2 * Y2_M2_COEFF * kx
        + g_m1 * Y2_M1_COEFF * kz
        - g_p2 * Y2_P2_COEFF * two * ky
    )

    grad_kz = coeff_l2 * (
        g_m1 * Y2_M1_COEFF * ky
        + g_0 * Y2_0_COEFF * wp.float64(6.0) * kz
        + g_p1 * Y2_P1_COEFF * kx
    )

    grad_k2 = (
        -wp.float64(0.5) * sigma2 * coeff_l2 * s_poly - coeff_l2 * Y2_0_COEFF * g_0
    )

    return wp.vec4d(grad_kx, grad_ky, grad_kz, grad_k2)


@wp.kernel
def _rho_kphase_grad_kernel(
    charges: wp.array(dtype=Any),  # (N_atoms,)
    dipoles: wp.array(dtype=Any),  # (N_atoms,) vec3
    positions: wp.array(dtype=Any),  # (N_atoms,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    source_phi_hat: wp.array3d(dtype=wp.float64),  # (N_k, 4, 2)
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    scale: wp.float64,
    grad_k: wp.array2d(dtype=wp.float64),  # (N_k, 3) OUTPUT
):
    r"""∂L/∂k_vectors through the phase (transpose of position_gradient_from_rhok).

    Per (k, i): ``contrib = A·cos + B·sin`` with
    ``A = gr·P_i − gi·P_r``, ``B = −(gr·P_r + gi·P_i)``,
    ``P_{r/i} = Σ_lm φ̂_{r/i}·Q_{i,lm}``. The position grad sums
    ``scale·Σ_k k·contrib``; the k-phase grad sums ``scale·Σ_i r_i·contrib``.
    Captures ONLY the phase k-dependence — φ̂(k) is threaded separately via
    ``source_phi_hat`` + SourcePhiHatFunction.


    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx)``; each thread processes one k-vector index.

    Parameters
    ----------
    charges : wp.array, shape (N_atoms,), dtype Any
        Per-atom monopole charges.
    dipoles : wp.array, shape (N_atoms,), dtype Any
        Vec3.
    positions : wp.array, shape (N_atoms,), dtype Any
        Vec3.
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array3d, shape (N_k, 4, 2), dtype wp.float64
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array2d, shape (N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    grad_k : wp.array2d, shape (N_k, 3), dtype wp.float64
        OUTPUT: gradient w.r.t. the k-vectors.
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]
    gr = grad_rho[k_idx, 0]
    gi = grad_rho[k_idx, 1]
    pr0 = source_phi_hat[k_idx, 0, 0]
    pi0 = source_phi_hat[k_idx, 0, 1]
    pr1 = source_phi_hat[k_idx, 1, 0]
    pi1 = source_phi_hat[k_idx, 1, 1]
    pr2 = source_phi_hat[k_idx, 2, 0]
    pi2 = source_phi_hat[k_idx, 2, 1]
    pr3 = source_phi_hat[k_idx, 3, 0]
    pi3 = source_phi_hat[k_idx, 3, 1]
    gkx = wp.float64(0.0)
    gky = wp.float64(0.0)
    gkz = wp.float64(0.0)
    for i in range(n_atoms):
        q = wp.float64(charges[i])
        mu = dipoles[i]
        mu_x = wp.float64(mu[0])
        mu_y = wp.float64(mu[1])
        mu_z = wp.float64(mu[2])
        p_r = pr0 * q + pr1 * mu_y + pr2 * mu_z + pr3 * mu_x
        p_i = pi0 * q + pi1 * mu_y + pi2 * mu_z + pi3 * mu_x
        a_k = gr * p_i - gi * p_r
        b_k = -(gr * p_r + gi * p_i)
        contrib = a_k * cosines[k_idx, i] + b_k * sines[k_idx, i]
        pos_i = positions[i]
        gkx += wp.float64(pos_i[0]) * contrib
        gky += wp.float64(pos_i[1]) * contrib
        gkz += wp.float64(pos_i[2]) * contrib
    grad_k[k_idx, 0] = scale * gkx
    grad_k[k_idx, 1] = scale * gky
    grad_k[k_idx, 2] = scale * gkz


def _rho_kphase_grad_sig(v, t):
    return [
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.float64,
        wp.array2d(dtype=wp.float64),
    ]


@wp.kernel
def _rho_kphase_grad_double_backward_kernel(
    g_k: wp.array2d(dtype=wp.float64),  # (N_k, 3) cotangent on grad_k
    charges: wp.array(dtype=Any),  # (N_atoms,)
    dipoles: wp.array(dtype=Any),  # (N_atoms,) vec3
    positions: wp.array(dtype=Any),  # (N_atoms,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    source_phi_hat: wp.array3d(dtype=wp.float64),  # (N_k, 4, 2)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    scale: wp.float64,
    ggrad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2) OUTPUT (per-k)
    ggrad_moments: wp.array2d(dtype=wp.float64),  # (N_atoms, 4) OUTPUT (atomic)
    ggrad_positions: wp.array(dtype=wp.vec3d),  # (N_atoms,) OUTPUT (atomic)
    ggrad_phi: wp.array3d(dtype=wp.float64),  # (N_k, 4, 2) OUTPUT (per-k)
    ggrad_kvec: wp.array2d(dtype=wp.float64),  # (N_k, 3) OUTPUT (per-k)
):
    r"""Second-order backward of :func:`_rho_kphase_grad_kernel`.

    First backward: ``grad_k[k] = scale·Σ_i r_i·e_{ki}``,
    ``e_{ki} = a_{ki}\cos + b_{ki}\sin``, ``a = gr·p_i − gi·p_r``,
    ``b = −(gr·p_r + gi·p_i)``, ``p_{r/i} = Σ_lm φ̂_{r/i}[lm]·Q_{i,lm}``. With
    cotangent ``G_k = ∂L/∂grad_k`` and ``d_{ki} = G_k·r_i``:

    grads (× ``scale``) — ``∂/∂gr = Σ_i d(p_i\cos − p_r\sin)``,
    ``∂/∂gi = Σ_i d(−p_r\cos − p_i\sin)``;
    ``∂/∂φ_r[lm] = Σ_i d·(−Q_{i,lm})(gi\cos + gr\sin)``,
    ``∂/∂φ_i[lm] = Σ_i d·Q_{i,lm}(gr\cos − gi\sin)``;
    ``∂/∂Q_{i,lm} = Σ_k d[(gr φ_i−gi φ_r)\cos − (gr φ_r+gi φ_i)\sin]``;
    ``∂/∂r_i = Σ_k [G_k e_{ki} + d·w_{ki}·k]``,
    ``∂/∂k = Σ_i d·w_{ki}·r_i`` with ``w_{ki} = −a\sin + b\cos``.
    ``ggrad_moments`` / ``ggrad_positions`` accumulate over k (pre-zero).

    Launch Grid
    -----------
    dim = [N_k].
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]
    gr = grad_rho[k_idx, 0]
    gi = grad_rho[k_idx, 1]
    kv = k_vectors[k_idx]
    gk0 = g_k[k_idx, 0]
    gk1 = g_k[k_idx, 1]
    gk2 = g_k[k_idx, 2]
    pr0 = source_phi_hat[k_idx, 0, 0]
    pi0 = source_phi_hat[k_idx, 0, 1]
    pr1 = source_phi_hat[k_idx, 1, 0]
    pi1 = source_phi_hat[k_idx, 1, 1]
    pr2 = source_phi_hat[k_idx, 2, 0]
    pi2 = source_phi_hat[k_idx, 2, 1]
    pr3 = source_phi_hat[k_idx, 3, 0]
    pi3 = source_phi_hat[k_idx, 3, 1]

    ggr_gr = wp.float64(0.0)
    ggr_gi = wp.float64(0.0)
    gphir0 = wp.float64(0.0)
    gphir1 = wp.float64(0.0)
    gphir2 = wp.float64(0.0)
    gphir3 = wp.float64(0.0)
    gphii0 = wp.float64(0.0)
    gphii1 = wp.float64(0.0)
    gphii2 = wp.float64(0.0)
    gphii3 = wp.float64(0.0)
    gkx = wp.float64(0.0)
    gky = wp.float64(0.0)
    gkz = wp.float64(0.0)
    for i in range(n_atoms):
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        q = wp.float64(charges[i])
        mu = dipoles[i]
        mu_x = wp.float64(mu[0])
        mu_y = wp.float64(mu[1])
        mu_z = wp.float64(mu[2])
        # p_{r/i} = Σ_lm φ̂_{r/i}[lm]·Q_{i,lm}, Q layout (q, mu_y, mu_z, mu_x).
        p_r = pr0 * q + pr1 * mu_y + pr2 * mu_z + pr3 * mu_x
        p_i = pi0 * q + pi1 * mu_y + pi2 * mu_z + pi3 * mu_x
        a = gr * p_i - gi * p_r
        b = -(gr * p_r + gi * p_i)
        e = a * cos_ki + b * sin_ki
        w = -a * sin_ki + b * cos_ki
        pos_i = positions[i]
        d = (
            gk0 * wp.float64(pos_i[0])
            + gk1 * wp.float64(pos_i[1])
            + gk2 * wp.float64(pos_i[2])
        )

        ggr_gr += d * (p_i * cos_ki - p_r * sin_ki)
        ggr_gi += d * (-p_r * cos_ki - p_i * sin_ki)

        fr = -d * (gi * cos_ki + gr * sin_ki)  # × Q_{i,lm} -> ∂/∂φ_r[lm]
        fi = d * (gr * cos_ki - gi * sin_ki)  # × Q_{i,lm} -> ∂/∂φ_i[lm]
        gphir0 += fr * q
        gphir1 += fr * mu_y
        gphir2 += fr * mu_z
        gphir3 += fr * mu_x
        gphii0 += fi * q
        gphii1 += fi * mu_y
        gphii2 += fi * mu_z
        gphii3 += fi * mu_x

        # ∂/∂Q_{i,lm} = d·[(gr φ_i[lm] − gi φ_r[lm])cos − (gr φ_r[lm] + gi φ_i[lm])sin]
        m0 = d * ((gr * pi0 - gi * pr0) * cos_ki - (gr * pr0 + gi * pi0) * sin_ki)
        m1 = d * ((gr * pi1 - gi * pr1) * cos_ki - (gr * pr1 + gi * pi1) * sin_ki)
        m2 = d * ((gr * pi2 - gi * pr2) * cos_ki - (gr * pr2 + gi * pi2) * sin_ki)
        m3 = d * ((gr * pi3 - gi * pr3) * cos_ki - (gr * pr3 + gi * pi3) * sin_ki)
        wp.atomic_add(ggrad_moments, i, 0, scale * m0)
        wp.atomic_add(ggrad_moments, i, 1, scale * m1)
        wp.atomic_add(ggrad_moments, i, 2, scale * m2)
        wp.atomic_add(ggrad_moments, i, 3, scale * m3)

        # ∂/∂r_i = G_k·e + d·w·k
        wp.atomic_add(
            ggrad_positions,
            i,
            wp.vec3d(
                scale * (gk0 * e + d * w * kv[0]),
                scale * (gk1 * e + d * w * kv[1]),
                scale * (gk2 * e + d * w * kv[2]),
            ),
        )
        # ∂/∂k = Σ_i d·w·r_i
        dw = d * w
        gkx += dw * wp.float64(pos_i[0])
        gky += dw * wp.float64(pos_i[1])
        gkz += dw * wp.float64(pos_i[2])

    ggrad_rho[k_idx, 0] = scale * ggr_gr
    ggrad_rho[k_idx, 1] = scale * ggr_gi
    ggrad_phi[k_idx, 0, 0] = scale * gphir0
    ggrad_phi[k_idx, 1, 0] = scale * gphir1
    ggrad_phi[k_idx, 2, 0] = scale * gphir2
    ggrad_phi[k_idx, 3, 0] = scale * gphir3
    ggrad_phi[k_idx, 0, 1] = scale * gphii0
    ggrad_phi[k_idx, 1, 1] = scale * gphii1
    ggrad_phi[k_idx, 2, 1] = scale * gphii2
    ggrad_phi[k_idx, 3, 1] = scale * gphii3
    ggrad_kvec[k_idx, 0] = scale * gkx
    ggrad_kvec[k_idx, 1] = scale * gky
    ggrad_kvec[k_idx, 2] = scale * gkz


def _rho_kphase_grad_double_backward_sig(v, t):
    return [
        wp.array2d(dtype=wp.float64),  # g_k
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=v),  # positions
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array3d(dtype=wp.float64),  # source_phi_hat
        wp.array(dtype=wp.vec3d),  # k_vectors
        wp.array2d(dtype=wp.float64),  # grad_rho
        wp.float64,  # scale
        wp.array2d(dtype=wp.float64),  # ggrad_rho
        wp.array2d(dtype=wp.float64),  # ggrad_moments
        wp.array(dtype=wp.vec3d),  # ggrad_positions
        wp.array3d(dtype=wp.float64),  # ggrad_phi
        wp.array2d(dtype=wp.float64),  # ggrad_kvec
    ]


@wp.kernel
def _batch_rho_kphase_grad_double_backward_kernel(
    g_k: wp.array3d(dtype=wp.float64),  # (B, N_k, 3) cotangent on grad_k
    charges: wp.array(dtype=Any),  # (N_total,)
    dipoles: wp.array(dtype=Any),  # (N_total,) vec3
    positions: wp.array(dtype=Any),  # (N_total,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_total)
    source_phi_hat: wp.array4d(dtype=wp.float64),  # (B, N_k, 4, 2)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, N_k)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, N_k, 2)
    volume: wp.array(dtype=wp.float64),  # (B,)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    ggrad_rho: wp.array3d(dtype=wp.float64),  # (B, N_k, 2) OUTPUT (per-k)
    ggrad_moments: wp.array2d(dtype=wp.float64),  # (N_total, 4) OUTPUT (atomic)
    ggrad_positions: wp.array(dtype=wp.vec3d),  # (N_total,) OUTPUT (atomic)
    ggrad_phi: wp.array4d(dtype=wp.float64),  # (B, N_k, 4, 2) OUTPUT (per-k)
    ggrad_kvec: wp.array3d(dtype=wp.float64),  # (B, N_k, 3) OUTPUT (per-k)
):
    r"""Batched second-order backward of :func:`_batch_rho_kphase_grad_kernel`.

    Per-system mirror of :func:`_rho_kphase_grad_double_backward_kernel`: the
    ``(b, k_idx)`` grid sweeps system ``b``'s atoms ``[atom_start[b],
    atom_end[b])`` and applies ``scale = (2π)³/volume[b]``. ``ggrad_moments`` /
    ``ggrad_positions`` accumulate over k and must be pre-zeroed.

    Launch Grid
    -----------
    dim = [B, N_k].
    """
    b, k_idx = wp.tid()
    scale = _TWO_PI_CUBED / volume[b]
    gr = grad_rho[b, k_idx, 0]
    gi = grad_rho[b, k_idx, 1]
    kv = k_vectors[b, k_idx]
    gk0 = g_k[b, k_idx, 0]
    gk1 = g_k[b, k_idx, 1]
    gk2 = g_k[b, k_idx, 2]
    pr0 = source_phi_hat[b, k_idx, 0, 0]
    pi0 = source_phi_hat[b, k_idx, 0, 1]
    pr1 = source_phi_hat[b, k_idx, 1, 0]
    pi1 = source_phi_hat[b, k_idx, 1, 1]
    pr2 = source_phi_hat[b, k_idx, 2, 0]
    pi2 = source_phi_hat[b, k_idx, 2, 1]
    pr3 = source_phi_hat[b, k_idx, 3, 0]
    pi3 = source_phi_hat[b, k_idx, 3, 1]

    ggr_gr = wp.float64(0.0)
    ggr_gi = wp.float64(0.0)
    gphir0 = wp.float64(0.0)
    gphir1 = wp.float64(0.0)
    gphir2 = wp.float64(0.0)
    gphir3 = wp.float64(0.0)
    gphii0 = wp.float64(0.0)
    gphii1 = wp.float64(0.0)
    gphii2 = wp.float64(0.0)
    gphii3 = wp.float64(0.0)
    gkx = wp.float64(0.0)
    gky = wp.float64(0.0)
    gkz = wp.float64(0.0)
    for i in range(atom_start[b], atom_end[b]):
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        q = wp.float64(charges[i])
        mu = dipoles[i]
        mu_x = wp.float64(mu[0])
        mu_y = wp.float64(mu[1])
        mu_z = wp.float64(mu[2])
        p_r = pr0 * q + pr1 * mu_y + pr2 * mu_z + pr3 * mu_x
        p_i = pi0 * q + pi1 * mu_y + pi2 * mu_z + pi3 * mu_x
        a = gr * p_i - gi * p_r
        b_co = -(gr * p_r + gi * p_i)
        e = a * cos_ki + b_co * sin_ki
        w = -a * sin_ki + b_co * cos_ki
        pos_i = positions[i]
        d = (
            gk0 * wp.float64(pos_i[0])
            + gk1 * wp.float64(pos_i[1])
            + gk2 * wp.float64(pos_i[2])
        )

        ggr_gr += d * (p_i * cos_ki - p_r * sin_ki)
        ggr_gi += d * (-p_r * cos_ki - p_i * sin_ki)

        fr = -d * (gi * cos_ki + gr * sin_ki)  # × Q_{i,lm} -> ∂/∂φ_r[lm]
        fi = d * (gr * cos_ki - gi * sin_ki)  # × Q_{i,lm} -> ∂/∂φ_i[lm]
        gphir0 += fr * q
        gphir1 += fr * mu_y
        gphir2 += fr * mu_z
        gphir3 += fr * mu_x
        gphii0 += fi * q
        gphii1 += fi * mu_y
        gphii2 += fi * mu_z
        gphii3 += fi * mu_x

        m0 = d * ((gr * pi0 - gi * pr0) * cos_ki - (gr * pr0 + gi * pi0) * sin_ki)
        m1 = d * ((gr * pi1 - gi * pr1) * cos_ki - (gr * pr1 + gi * pi1) * sin_ki)
        m2 = d * ((gr * pi2 - gi * pr2) * cos_ki - (gr * pr2 + gi * pi2) * sin_ki)
        m3 = d * ((gr * pi3 - gi * pr3) * cos_ki - (gr * pr3 + gi * pi3) * sin_ki)
        wp.atomic_add(ggrad_moments, i, 0, scale * m0)
        wp.atomic_add(ggrad_moments, i, 1, scale * m1)
        wp.atomic_add(ggrad_moments, i, 2, scale * m2)
        wp.atomic_add(ggrad_moments, i, 3, scale * m3)

        wp.atomic_add(
            ggrad_positions,
            i,
            wp.vec3d(
                scale * (gk0 * e + d * w * kv[0]),
                scale * (gk1 * e + d * w * kv[1]),
                scale * (gk2 * e + d * w * kv[2]),
            ),
        )
        dw = d * w
        gkx += dw * wp.float64(pos_i[0])
        gky += dw * wp.float64(pos_i[1])
        gkz += dw * wp.float64(pos_i[2])

    ggrad_rho[b, k_idx, 0] = scale * ggr_gr
    ggrad_rho[b, k_idx, 1] = scale * ggr_gi
    ggrad_phi[b, k_idx, 0, 0] = scale * gphir0
    ggrad_phi[b, k_idx, 1, 0] = scale * gphir1
    ggrad_phi[b, k_idx, 2, 0] = scale * gphir2
    ggrad_phi[b, k_idx, 3, 0] = scale * gphir3
    ggrad_phi[b, k_idx, 0, 1] = scale * gphii0
    ggrad_phi[b, k_idx, 1, 1] = scale * gphii1
    ggrad_phi[b, k_idx, 2, 1] = scale * gphii2
    ggrad_phi[b, k_idx, 3, 1] = scale * gphii3
    ggrad_kvec[b, k_idx, 0] = scale * gkx
    ggrad_kvec[b, k_idx, 1] = scale * gky
    ggrad_kvec[b, k_idx, 2] = scale * gkz


def _batch_rho_kphase_grad_double_backward_sig(v, t):
    return [
        wp.array3d(dtype=wp.float64),  # g_k
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=v),  # positions
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array4d(dtype=wp.float64),  # source_phi_hat
        wp.array2d(dtype=wp.vec3d),  # k_vectors
        wp.array3d(dtype=wp.float64),  # grad_rho
        wp.array(dtype=wp.float64),  # volume
        wp.array(dtype=wp.int32),  # atom_start
        wp.array(dtype=wp.int32),  # atom_end
        wp.array3d(dtype=wp.float64),  # ggrad_rho
        wp.array2d(dtype=wp.float64),  # ggrad_moments
        wp.array(dtype=wp.vec3d),  # ggrad_positions
        wp.array4d(dtype=wp.float64),  # ggrad_phi
        wp.array3d(dtype=wp.float64),  # ggrad_kvec
    ]


@wp.kernel
def _rho_phihat_grad_kernel(
    charges: wp.array(dtype=Any),  # (N_atoms,)
    dipoles: wp.array(dtype=Any),  # (N_atoms,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    scale: wp.float64,
    grad_phi: wp.array3d(dtype=wp.float64),  # (N_k, 4, 2) OUTPUT
):
    r"""∂L/∂source_phi_hat from grad_ρ + per-atom moments.

    ρ_r = scale·Σ_lm(φ_r·c_lm + φ_i·s_lm), ρ_i = scale·Σ_lm(φ_i·c_lm − φ_r·s_lm)
    with c_lm = Σ_i Q_{i,lm}·cos(k·r_i), s_lm = Σ_i Q_{i,lm}·sin(k·r_i). So
        ∂L/∂φ_r[lm] = scale·(gr·c_lm − gi·s_lm)
        ∂L/∂φ_i[lm] = scale·(gr·s_lm + gi·c_lm).


    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx)``; each thread processes one k-vector index.

    Parameters
    ----------
    charges : wp.array, shape (N_atoms,), dtype Any
        Per-atom monopole charges.
    dipoles : wp.array, shape (N_atoms,), dtype Any
        Vec3.
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    grad_rho : wp.array2d, shape (N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    grad_phi : wp.array3d, shape (N_k, 4, 2), dtype wp.float64
        OUTPUT: gradient w.r.t. the GTO Fourier coefficients :math:`\hat\phi(k)`.
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]
    c0 = wp.float64(0.0)
    c1 = wp.float64(0.0)
    c2 = wp.float64(0.0)
    c3 = wp.float64(0.0)
    s0 = wp.float64(0.0)
    s1 = wp.float64(0.0)
    s2 = wp.float64(0.0)
    s3 = wp.float64(0.0)
    for i in range(n_atoms):
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        q = wp.float64(charges[i])
        mu = dipoles[i]
        mu_x = wp.float64(mu[0])
        mu_y = wp.float64(mu[1])
        mu_z = wp.float64(mu[2])
        c0 += cos_ki * q
        s0 += sin_ki * q
        c1 += cos_ki * mu_y
        s1 += sin_ki * mu_y
        c2 += cos_ki * mu_z
        s2 += sin_ki * mu_z
        c3 += cos_ki * mu_x
        s3 += sin_ki * mu_x
    gr = grad_rho[k_idx, 0]
    gi = grad_rho[k_idx, 1]
    grad_phi[k_idx, 0, 0] = scale * (gr * c0 - gi * s0)
    grad_phi[k_idx, 0, 1] = scale * (gr * s0 + gi * c0)
    grad_phi[k_idx, 1, 0] = scale * (gr * c1 - gi * s1)
    grad_phi[k_idx, 1, 1] = scale * (gr * s1 + gi * c1)
    grad_phi[k_idx, 2, 0] = scale * (gr * c2 - gi * s2)
    grad_phi[k_idx, 2, 1] = scale * (gr * s2 + gi * c2)
    grad_phi[k_idx, 3, 0] = scale * (gr * c3 - gi * s3)
    grad_phi[k_idx, 3, 1] = scale * (gr * s3 + gi * c3)


def _rho_phihat_grad_sig(v, t):
    return [
        wp.array(dtype=t),
        wp.array(dtype=v),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.float64,
        wp.array3d(dtype=wp.float64),
    ]


@wp.kernel
def _rho_phihat_grad_double_backward_kernel(
    g_phi: wp.array3d(dtype=wp.float64),  # (N_k, 4, 2) cotangent on grad_phi
    charges: wp.array(dtype=Any),  # (N_atoms,)
    dipoles: wp.array(dtype=Any),  # (N_atoms,) vec3
    positions: wp.array(dtype=Any),  # (N_atoms,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    scale: wp.float64,
    ggrad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2) OUTPUT (per-k)
    ggrad_moments: wp.array2d(dtype=wp.float64),  # (N_atoms, 4) OUTPUT (atomic)
    ggrad_positions: wp.array(dtype=wp.vec3d),  # (N_atoms,) OUTPUT (atomic)
    ggrad_kvec: wp.array2d(dtype=wp.float64),  # (N_k, 3) OUTPUT (per-k)
):
    r"""Second-order backward of :func:`_rho_phihat_grad_kernel`.

    The first backward is ``grad_phi[lm] = scale·(gr·c_lm − gi·s_lm,
    gr·s_lm + gi·c_lm)`` with ``c_lm = Σ_i Q_{i,lm}·cos(k·r_i)``. With cotangent
    ``G = ∂L/∂grad_phi`` and ``dLdc[lm] = scale·(gr·G_{lm,r} + gi·G_{lm,i})``,
    ``dLds[lm] = scale·(−gi·G_{lm,r} + gr·G_{lm,i})``:

    .. math::

        \partial L/\partial gr &= scale·Σ_{lm}(G_{lm,r} c_{lm} + G_{lm,i} s_{lm}) \\
        \partial L/\partial gi &= scale·Σ_{lm}(−G_{lm,r} s_{lm} + G_{lm,i} c_{lm}) \\
        \partial L/\partial Q_{i,lm} &= Σ_k(dLdc[lm]\cos_{ki} + dLds[lm]\sin_{ki}) \\
        \partial L/\partial r_i &= Σ_k w_{ki}\,k, \quad
        \partial L/\partial k &= Σ_i w_{ki}\,r_i

    with ``w_{ki} = Σ_{lm} Q_{i,lm}(−dLdc[lm]\sin_{ki} + dLds[lm]\cos_{ki})``.
    ``Q_{i,lm}`` layout: ``(q, μ_y, μ_z, μ_x)``. ``ggrad_moments`` /
    ``ggrad_positions`` accumulate over k (per-k threads) so must be pre-zeroed.

    Launch Grid
    -----------
    dim = [N_k] — one thread per k-vector; sweeps all atoms.
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]
    gr = grad_rho[k_idx, 0]
    gi = grad_rho[k_idx, 1]
    kv = k_vectors[k_idx]

    dLdc0 = scale * (gr * g_phi[k_idx, 0, 0] + gi * g_phi[k_idx, 0, 1])
    dLds0 = scale * (-gi * g_phi[k_idx, 0, 0] + gr * g_phi[k_idx, 0, 1])
    dLdc1 = scale * (gr * g_phi[k_idx, 1, 0] + gi * g_phi[k_idx, 1, 1])
    dLds1 = scale * (-gi * g_phi[k_idx, 1, 0] + gr * g_phi[k_idx, 1, 1])
    dLdc2 = scale * (gr * g_phi[k_idx, 2, 0] + gi * g_phi[k_idx, 2, 1])
    dLds2 = scale * (-gi * g_phi[k_idx, 2, 0] + gr * g_phi[k_idx, 2, 1])
    dLdc3 = scale * (gr * g_phi[k_idx, 3, 0] + gi * g_phi[k_idx, 3, 1])
    dLds3 = scale * (-gi * g_phi[k_idx, 3, 0] + gr * g_phi[k_idx, 3, 1])

    c0 = wp.float64(0.0)
    c1 = wp.float64(0.0)
    c2 = wp.float64(0.0)
    c3 = wp.float64(0.0)
    s0 = wp.float64(0.0)
    s1 = wp.float64(0.0)
    s2 = wp.float64(0.0)
    s3 = wp.float64(0.0)
    gkx = wp.float64(0.0)
    gky = wp.float64(0.0)
    gkz = wp.float64(0.0)
    for i in range(n_atoms):
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        q = wp.float64(charges[i])
        mu = dipoles[i]
        mu_x = wp.float64(mu[0])
        mu_y = wp.float64(mu[1])
        mu_z = wp.float64(mu[2])
        c0 += cos_ki * q
        s0 += sin_ki * q
        c1 += cos_ki * mu_y
        s1 += sin_ki * mu_y
        c2 += cos_ki * mu_z
        s2 += sin_ki * mu_z
        c3 += cos_ki * mu_x
        s3 += sin_ki * mu_x

        wp.atomic_add(ggrad_moments, i, 0, dLdc0 * cos_ki + dLds0 * sin_ki)
        wp.atomic_add(ggrad_moments, i, 1, dLdc1 * cos_ki + dLds1 * sin_ki)
        wp.atomic_add(ggrad_moments, i, 2, dLdc2 * cos_ki + dLds2 * sin_ki)
        wp.atomic_add(ggrad_moments, i, 3, dLdc3 * cos_ki + dLds3 * sin_ki)

        a0 = -dLdc0 * sin_ki + dLds0 * cos_ki
        a1 = -dLdc1 * sin_ki + dLds1 * cos_ki
        a2 = -dLdc2 * sin_ki + dLds2 * cos_ki
        a3 = -dLdc3 * sin_ki + dLds3 * cos_ki
        w_ki = q * a0 + mu_y * a1 + mu_z * a2 + mu_x * a3
        wp.atomic_add(
            ggrad_positions, i, wp.vec3d(w_ki * kv[0], w_ki * kv[1], w_ki * kv[2])
        )
        pos_i = positions[i]
        gkx += w_ki * wp.float64(pos_i[0])
        gky += w_ki * wp.float64(pos_i[1])
        gkz += w_ki * wp.float64(pos_i[2])

    ggrad_rho[k_idx, 0] = scale * (
        g_phi[k_idx, 0, 0] * c0
        + g_phi[k_idx, 1, 0] * c1
        + g_phi[k_idx, 2, 0] * c2
        + g_phi[k_idx, 3, 0] * c3
        + g_phi[k_idx, 0, 1] * s0
        + g_phi[k_idx, 1, 1] * s1
        + g_phi[k_idx, 2, 1] * s2
        + g_phi[k_idx, 3, 1] * s3
    )
    ggrad_rho[k_idx, 1] = scale * (
        -g_phi[k_idx, 0, 0] * s0
        - g_phi[k_idx, 1, 0] * s1
        - g_phi[k_idx, 2, 0] * s2
        - g_phi[k_idx, 3, 0] * s3
        + g_phi[k_idx, 0, 1] * c0
        + g_phi[k_idx, 1, 1] * c1
        + g_phi[k_idx, 2, 1] * c2
        + g_phi[k_idx, 3, 1] * c3
    )
    ggrad_kvec[k_idx, 0] = gkx
    ggrad_kvec[k_idx, 1] = gky
    ggrad_kvec[k_idx, 2] = gkz


def _rho_phihat_grad_double_backward_sig(v, t):
    return [
        wp.array3d(dtype=wp.float64),  # g_phi
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=v),  # positions
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array(dtype=wp.vec3d),  # k_vectors
        wp.array2d(dtype=wp.float64),  # grad_rho
        wp.float64,  # scale
        wp.array2d(dtype=wp.float64),  # ggrad_rho
        wp.array2d(dtype=wp.float64),  # ggrad_moments
        wp.array(dtype=wp.vec3d),  # ggrad_positions
        wp.array2d(dtype=wp.float64),  # ggrad_kvec
    ]


@wp.kernel
def _batch_rho_phihat_grad_double_backward_kernel(
    g_phi: wp.array4d(dtype=wp.float64),  # (B, N_k, 4, 2) cotangent on grad_phi
    charges: wp.array(dtype=Any),  # (N_total,)
    dipoles: wp.array(dtype=Any),  # (N_total,) vec3
    positions: wp.array(dtype=Any),  # (N_total,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_total)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, N_k)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, N_k, 2)
    volume: wp.array(dtype=wp.float64),  # (B,)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    ggrad_rho: wp.array3d(dtype=wp.float64),  # (B, N_k, 2) OUTPUT (per-k)
    ggrad_moments: wp.array2d(dtype=wp.float64),  # (N_total, 4) OUTPUT (atomic)
    ggrad_positions: wp.array(dtype=wp.vec3d),  # (N_total,) OUTPUT (atomic)
    ggrad_kvec: wp.array3d(dtype=wp.float64),  # (B, N_k, 3) OUTPUT (per-k)
):
    r"""Batched second-order backward of :func:`_batch_rho_phihat_grad_kernel`.

    Per-system mirror of :func:`_rho_phihat_grad_double_backward_kernel`: the
    ``(b, k_idx)`` grid sweeps system ``b``'s atoms ``[atom_start[b],
    atom_end[b])`` and applies ``scale = (2π)³/volume[b]``. ``ggrad_moments`` /
    ``ggrad_positions`` accumulate over k and must be pre-zeroed.

    Launch Grid
    -----------
    dim = [B, N_k] — one thread per (system, k-vector); sweeps the system's atoms.
    """
    b, k_idx = wp.tid()
    scale = _TWO_PI_CUBED / volume[b]
    gr = grad_rho[b, k_idx, 0]
    gi = grad_rho[b, k_idx, 1]
    kv = k_vectors[b, k_idx]

    dLdc0 = scale * (gr * g_phi[b, k_idx, 0, 0] + gi * g_phi[b, k_idx, 0, 1])
    dLds0 = scale * (-gi * g_phi[b, k_idx, 0, 0] + gr * g_phi[b, k_idx, 0, 1])
    dLdc1 = scale * (gr * g_phi[b, k_idx, 1, 0] + gi * g_phi[b, k_idx, 1, 1])
    dLds1 = scale * (-gi * g_phi[b, k_idx, 1, 0] + gr * g_phi[b, k_idx, 1, 1])
    dLdc2 = scale * (gr * g_phi[b, k_idx, 2, 0] + gi * g_phi[b, k_idx, 2, 1])
    dLds2 = scale * (-gi * g_phi[b, k_idx, 2, 0] + gr * g_phi[b, k_idx, 2, 1])
    dLdc3 = scale * (gr * g_phi[b, k_idx, 3, 0] + gi * g_phi[b, k_idx, 3, 1])
    dLds3 = scale * (-gi * g_phi[b, k_idx, 3, 0] + gr * g_phi[b, k_idx, 3, 1])

    c0 = wp.float64(0.0)
    c1 = wp.float64(0.0)
    c2 = wp.float64(0.0)
    c3 = wp.float64(0.0)
    s0 = wp.float64(0.0)
    s1 = wp.float64(0.0)
    s2 = wp.float64(0.0)
    s3 = wp.float64(0.0)
    gkx = wp.float64(0.0)
    gky = wp.float64(0.0)
    gkz = wp.float64(0.0)
    for i in range(atom_start[b], atom_end[b]):
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        q = wp.float64(charges[i])
        mu = dipoles[i]
        mu_x = wp.float64(mu[0])
        mu_y = wp.float64(mu[1])
        mu_z = wp.float64(mu[2])
        c0 += cos_ki * q
        s0 += sin_ki * q
        c1 += cos_ki * mu_y
        s1 += sin_ki * mu_y
        c2 += cos_ki * mu_z
        s2 += sin_ki * mu_z
        c3 += cos_ki * mu_x
        s3 += sin_ki * mu_x

        wp.atomic_add(ggrad_moments, i, 0, dLdc0 * cos_ki + dLds0 * sin_ki)
        wp.atomic_add(ggrad_moments, i, 1, dLdc1 * cos_ki + dLds1 * sin_ki)
        wp.atomic_add(ggrad_moments, i, 2, dLdc2 * cos_ki + dLds2 * sin_ki)
        wp.atomic_add(ggrad_moments, i, 3, dLdc3 * cos_ki + dLds3 * sin_ki)

        a0 = -dLdc0 * sin_ki + dLds0 * cos_ki
        a1 = -dLdc1 * sin_ki + dLds1 * cos_ki
        a2 = -dLdc2 * sin_ki + dLds2 * cos_ki
        a3 = -dLdc3 * sin_ki + dLds3 * cos_ki
        w_ki = q * a0 + mu_y * a1 + mu_z * a2 + mu_x * a3
        wp.atomic_add(
            ggrad_positions, i, wp.vec3d(w_ki * kv[0], w_ki * kv[1], w_ki * kv[2])
        )
        pos_i = positions[i]
        gkx += w_ki * wp.float64(pos_i[0])
        gky += w_ki * wp.float64(pos_i[1])
        gkz += w_ki * wp.float64(pos_i[2])

    ggrad_rho[b, k_idx, 0] = scale * (
        g_phi[b, k_idx, 0, 0] * c0
        + g_phi[b, k_idx, 1, 0] * c1
        + g_phi[b, k_idx, 2, 0] * c2
        + g_phi[b, k_idx, 3, 0] * c3
        + g_phi[b, k_idx, 0, 1] * s0
        + g_phi[b, k_idx, 1, 1] * s1
        + g_phi[b, k_idx, 2, 1] * s2
        + g_phi[b, k_idx, 3, 1] * s3
    )
    ggrad_rho[b, k_idx, 1] = scale * (
        -g_phi[b, k_idx, 0, 0] * s0
        - g_phi[b, k_idx, 1, 0] * s1
        - g_phi[b, k_idx, 2, 0] * s2
        - g_phi[b, k_idx, 3, 0] * s3
        + g_phi[b, k_idx, 0, 1] * c0
        + g_phi[b, k_idx, 1, 1] * c1
        + g_phi[b, k_idx, 2, 1] * c2
        + g_phi[b, k_idx, 3, 1] * c3
    )
    ggrad_kvec[b, k_idx, 0] = gkx
    ggrad_kvec[b, k_idx, 1] = gky
    ggrad_kvec[b, k_idx, 2] = gkz


def _batch_rho_phihat_grad_double_backward_sig(v, t):
    return [
        wp.array4d(dtype=wp.float64),  # g_phi
        wp.array(dtype=t),  # charges
        wp.array(dtype=v),  # dipoles
        wp.array(dtype=v),  # positions
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array2d(dtype=wp.vec3d),  # k_vectors
        wp.array3d(dtype=wp.float64),  # grad_rho
        wp.array(dtype=wp.float64),  # volume
        wp.array(dtype=wp.int32),  # atom_start
        wp.array(dtype=wp.int32),  # atom_end
        wp.array3d(dtype=wp.float64),  # ggrad_rho
        wp.array2d(dtype=wp.float64),  # ggrad_moments
        wp.array(dtype=wp.vec3d),  # ggrad_positions
        wp.array3d(dtype=wp.float64),  # ggrad_kvec
    ]


@wp.kernel
def _rho_q_coeff2_grad_kernel(
    quadrupoles: wp.array(dtype=Any),  # (N_atoms,) mat33
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    scale: wp.float64,
    grad_coeff2: wp.array(dtype=wp.float64),  # (N_k,) OUTPUT
):
    r"""∂L/∂coeff2[k] = scale·(gr·C_Q − gi·S_Q), C_Q=Σ(kQk)cos, S_Q=Σ(kQk)sin.

    Launch Grid
    -----------
    dim = [N_k] — one thread per k-vector; sweeps all atoms.


    Parameters
    ----------
    quadrupoles : wp.array, shape (N_atoms,), dtype Any
        Mat33.
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-space k-vectors.
    grad_rho : wp.array2d, shape (N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    grad_coeff2 : wp.array, shape (N_k,), dtype wp.float64
        OUTPUT: gradient w.r.t. the :math:`l=2` coefficient ``coeff2``.
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]
    k_vec = k_vectors[k_idx]
    C_Q = wp.float64(0.0)
    S_Q = wp.float64(0.0)
    for i in range(n_atoms):
        Q = quadrupoles[i]
        kQ0 = (
            wp.float64(Q[0, 0]) * k_vec[0]
            + wp.float64(Q[0, 1]) * k_vec[1]
            + wp.float64(Q[0, 2]) * k_vec[2]
        )
        kQ1 = (
            wp.float64(Q[1, 0]) * k_vec[0]
            + wp.float64(Q[1, 1]) * k_vec[1]
            + wp.float64(Q[1, 2]) * k_vec[2]
        )
        kQ2 = (
            wp.float64(Q[2, 0]) * k_vec[0]
            + wp.float64(Q[2, 1]) * k_vec[1]
            + wp.float64(Q[2, 2]) * k_vec[2]
        )
        kQk = kQ0 * k_vec[0] + kQ1 * k_vec[1] + kQ2 * k_vec[2]
        C_Q += kQk * cosines[k_idx, i]
        S_Q += kQk * sines[k_idx, i]
    gr = grad_rho[k_idx, 0]
    gi = grad_rho[k_idx, 1]
    grad_coeff2[k_idx] = scale * (gr * C_Q - gi * S_Q)


def _rho_q_coeff2_grad_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=m),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array(dtype=wp.vec3d),
        wp.array2d(dtype=wp.float64),
        wp.float64,
        wp.array(dtype=wp.float64),
    ]


@wp.kernel
def _rho_q_coeff2_grad_double_backward_kernel(
    g_c: wp.array(dtype=wp.float64),  # (N_k,) cotangent on grad_coeff2
    quadrupoles: wp.array(dtype=Any),  # (N_atoms,) mat33
    positions: wp.array(dtype=Any),  # (N_atoms,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    scale: wp.float64,
    ggrad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2) OUTPUT (per-k)
    ggrad_quad: wp.array2d(dtype=wp.float64),  # (N_atoms, 9) OUTPUT (atomic)
    ggrad_positions: wp.array(dtype=wp.vec3d),  # (N_atoms,) OUTPUT (atomic)
    ggrad_kvec: wp.array2d(dtype=wp.float64),  # (N_k, 3) OUTPUT (per-k)
):
    r"""Second-order backward of :func:`_rho_q_coeff2_grad_kernel`.

    First backward: ``grad_coeff2[k] = scale·(gr·C_Q − gi·S_Q)``,
    ``C_Q = Σ_i (k·Q_i·k)\cos``, ``S_Q = Σ_i (k·Q_i·k)\sin``. With cotangent
    ``G_c = ∂L/∂grad_coeff2`` and ``f_{ki} = gr\cos_{ki} − gi\sin_{ki}``:

    grads (× ``scale·G_c[k]``) — ``∂/∂gr = Σ_i kQk\cos``,
    ``∂/∂gi = −Σ_i kQk\sin``;
    ``∂/∂Q_{i,ab} = f_{ki}\,k_a k_b``;
    ``∂/∂r_i = kQk(−gr\sin − gi\cos)\,k``;
    ``∂/∂k = Σ_i [2(Q_i k) f_{ki} + kQk(−gr\sin − gi\cos) r_i]``.
    ``ggrad_quad`` / ``ggrad_positions`` accumulate over k (pre-zero).

    Launch Grid
    -----------
    dim = [N_k].
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]
    gc = g_c[k_idx]
    gr = grad_rho[k_idx, 0]
    gi = grad_rho[k_idx, 1]
    kv = k_vectors[k_idx]
    sgc = scale * gc

    ggr_c = wp.float64(0.0)
    ggr_s = wp.float64(0.0)
    gkx = wp.float64(0.0)
    gky = wp.float64(0.0)
    gkz = wp.float64(0.0)
    for i in range(n_atoms):
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        q_mat = quadrupoles[i]
        qk0 = (
            wp.float64(q_mat[0, 0]) * kv[0]
            + wp.float64(q_mat[0, 1]) * kv[1]
            + wp.float64(q_mat[0, 2]) * kv[2]
        )
        qk1 = (
            wp.float64(q_mat[1, 0]) * kv[0]
            + wp.float64(q_mat[1, 1]) * kv[1]
            + wp.float64(q_mat[1, 2]) * kv[2]
        )
        qk2 = (
            wp.float64(q_mat[2, 0]) * kv[0]
            + wp.float64(q_mat[2, 1]) * kv[1]
            + wp.float64(q_mat[2, 2]) * kv[2]
        )
        kqk = qk0 * kv[0] + qk1 * kv[1] + qk2 * kv[2]
        f_ki = gr * cos_ki - gi * sin_ki
        h_ki = -gr * sin_ki - gi * cos_ki

        ggr_c += kqk * cos_ki
        ggr_s += kqk * sin_ki

        # ∂/∂Q_{i,ab} = scale·gc·f_ki·k_a k_b (symmetric).
        cf = sgc * f_ki
        wp.atomic_add(ggrad_quad, i, 0, cf * kv[0] * kv[0])
        wp.atomic_add(ggrad_quad, i, 1, cf * kv[0] * kv[1])
        wp.atomic_add(ggrad_quad, i, 2, cf * kv[0] * kv[2])
        wp.atomic_add(ggrad_quad, i, 3, cf * kv[1] * kv[0])
        wp.atomic_add(ggrad_quad, i, 4, cf * kv[1] * kv[1])
        wp.atomic_add(ggrad_quad, i, 5, cf * kv[1] * kv[2])
        wp.atomic_add(ggrad_quad, i, 6, cf * kv[2] * kv[0])
        wp.atomic_add(ggrad_quad, i, 7, cf * kv[2] * kv[1])
        wp.atomic_add(ggrad_quad, i, 8, cf * kv[2] * kv[2])

        # ∂/∂r_i = scale·gc·kQk·h_ki·k.
        pf = sgc * kqk * h_ki
        wp.atomic_add(ggrad_positions, i, wp.vec3d(pf * kv[0], pf * kv[1], pf * kv[2]))

        # ∂/∂k = scale·gc·[2(Q k) f_ki + kQk·h_ki·r_i].
        w1 = sgc * f_ki * wp.float64(2.0)
        w2 = sgc * kqk * h_ki
        pos_i = positions[i]
        gkx += w1 * qk0 + w2 * wp.float64(pos_i[0])
        gky += w1 * qk1 + w2 * wp.float64(pos_i[1])
        gkz += w1 * qk2 + w2 * wp.float64(pos_i[2])

    ggrad_rho[k_idx, 0] = sgc * ggr_c
    ggrad_rho[k_idx, 1] = -sgc * ggr_s
    ggrad_kvec[k_idx, 0] = gkx
    ggrad_kvec[k_idx, 1] = gky
    ggrad_kvec[k_idx, 2] = gkz


def _rho_q_coeff2_grad_double_backward_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=wp.float64),  # g_c
        wp.array(dtype=m),  # quadrupoles
        wp.array(dtype=v),  # positions
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array(dtype=wp.vec3d),  # k_vectors
        wp.array2d(dtype=wp.float64),  # grad_rho
        wp.float64,  # scale
        wp.array2d(dtype=wp.float64),  # ggrad_rho
        wp.array2d(dtype=wp.float64),  # ggrad_quad
        wp.array(dtype=wp.vec3d),  # ggrad_positions
        wp.array2d(dtype=wp.float64),  # ggrad_kvec
    ]


@wp.kernel
def _batch_rho_q_coeff2_grad_double_backward_kernel(
    g_c: wp.array2d(dtype=wp.float64),  # (B, N_k) cotangent on grad_coeff2
    quadrupoles: wp.array(dtype=Any),  # (N_total,) mat33
    positions: wp.array(dtype=Any),  # (N_total,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_total)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, N_k)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, N_k, 2)
    volume: wp.array(dtype=wp.float64),  # (B,)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    ggrad_rho: wp.array3d(dtype=wp.float64),  # (B, N_k, 2) OUTPUT (per-k)
    ggrad_quad: wp.array2d(dtype=wp.float64),  # (N_total, 9) OUTPUT (atomic)
    ggrad_positions: wp.array(dtype=wp.vec3d),  # (N_total,) OUTPUT (atomic)
    ggrad_kvec: wp.array3d(dtype=wp.float64),  # (B, N_k, 3) OUTPUT (per-k)
):
    r"""Batched second-order backward of :func:`_batch_rho_q_coeff2_grad_kernel`.

    Per-system mirror of :func:`_rho_q_coeff2_grad_double_backward_kernel`: the
    ``(b, k_idx)`` grid sweeps system ``b``'s atoms and applies ``scale =
    (2π)³/volume[b]``. ``ggrad_quad`` / ``ggrad_positions`` accumulate over k
    and must be pre-zeroed.

    Launch Grid
    -----------
    dim = [B, N_k].
    """
    b, k_idx = wp.tid()
    scale = _TWO_PI_CUBED / volume[b]
    gc = g_c[b, k_idx]
    gr = grad_rho[b, k_idx, 0]
    gi = grad_rho[b, k_idx, 1]
    kv = k_vectors[b, k_idx]
    sgc = scale * gc

    ggr_c = wp.float64(0.0)
    ggr_s = wp.float64(0.0)
    gkx = wp.float64(0.0)
    gky = wp.float64(0.0)
    gkz = wp.float64(0.0)
    for i in range(atom_start[b], atom_end[b]):
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        q_mat = quadrupoles[i]
        qk0 = (
            wp.float64(q_mat[0, 0]) * kv[0]
            + wp.float64(q_mat[0, 1]) * kv[1]
            + wp.float64(q_mat[0, 2]) * kv[2]
        )
        qk1 = (
            wp.float64(q_mat[1, 0]) * kv[0]
            + wp.float64(q_mat[1, 1]) * kv[1]
            + wp.float64(q_mat[1, 2]) * kv[2]
        )
        qk2 = (
            wp.float64(q_mat[2, 0]) * kv[0]
            + wp.float64(q_mat[2, 1]) * kv[1]
            + wp.float64(q_mat[2, 2]) * kv[2]
        )
        kqk = qk0 * kv[0] + qk1 * kv[1] + qk2 * kv[2]
        f_ki = gr * cos_ki - gi * sin_ki
        h_ki = -gr * sin_ki - gi * cos_ki

        ggr_c += kqk * cos_ki
        ggr_s += kqk * sin_ki

        cf = sgc * f_ki
        wp.atomic_add(ggrad_quad, i, 0, cf * kv[0] * kv[0])
        wp.atomic_add(ggrad_quad, i, 1, cf * kv[0] * kv[1])
        wp.atomic_add(ggrad_quad, i, 2, cf * kv[0] * kv[2])
        wp.atomic_add(ggrad_quad, i, 3, cf * kv[1] * kv[0])
        wp.atomic_add(ggrad_quad, i, 4, cf * kv[1] * kv[1])
        wp.atomic_add(ggrad_quad, i, 5, cf * kv[1] * kv[2])
        wp.atomic_add(ggrad_quad, i, 6, cf * kv[2] * kv[0])
        wp.atomic_add(ggrad_quad, i, 7, cf * kv[2] * kv[1])
        wp.atomic_add(ggrad_quad, i, 8, cf * kv[2] * kv[2])

        pf = sgc * kqk * h_ki
        wp.atomic_add(ggrad_positions, i, wp.vec3d(pf * kv[0], pf * kv[1], pf * kv[2]))

        w1 = sgc * f_ki * wp.float64(2.0)
        w2 = sgc * kqk * h_ki
        pos_i = positions[i]
        gkx += w1 * qk0 + w2 * wp.float64(pos_i[0])
        gky += w1 * qk1 + w2 * wp.float64(pos_i[1])
        gkz += w1 * qk2 + w2 * wp.float64(pos_i[2])

    ggrad_rho[b, k_idx, 0] = sgc * ggr_c
    ggrad_rho[b, k_idx, 1] = -sgc * ggr_s
    ggrad_kvec[b, k_idx, 0] = gkx
    ggrad_kvec[b, k_idx, 1] = gky
    ggrad_kvec[b, k_idx, 2] = gkz


def _batch_rho_q_coeff2_grad_double_backward_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array2d(dtype=wp.float64),  # g_c
        wp.array(dtype=m),  # quadrupoles
        wp.array(dtype=v),  # positions
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array2d(dtype=wp.vec3d),  # k_vectors
        wp.array3d(dtype=wp.float64),  # grad_rho
        wp.array(dtype=wp.float64),  # volume
        wp.array(dtype=wp.int32),  # atom_start
        wp.array(dtype=wp.int32),  # atom_end
        wp.array3d(dtype=wp.float64),  # ggrad_rho
        wp.array2d(dtype=wp.float64),  # ggrad_quad
        wp.array(dtype=wp.vec3d),  # ggrad_positions
        wp.array3d(dtype=wp.float64),  # ggrad_kvec
    ]


@wp.kernel
def _rho_q_kvec_grad_kernel(
    quadrupoles: wp.array(dtype=Any),  # (N_atoms,) mat33
    positions: wp.array(dtype=Any),  # (N_atoms,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    coeff2: wp.array(dtype=wp.float64),  # (N_k,)
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    scale: wp.float64,
    grad_k: wp.array2d(dtype=wp.float64),  # (N_k, 3) OUTPUT
):
    r"""∂L/∂k via the (k·Q·k) form AND the phase (coeff2 held fixed).

    ∂L/∂k = scale·c2·Σ_i [ 2(gr·cos − gi·sin)·(Q·k)
              − (k·Q·k)(gr·sin + gi·cos)·r_i ].

    Launch Grid
    -----------
    dim = [N_k] — one thread per k-vector; sweeps all atoms.


    Parameters
    ----------
    quadrupoles : wp.array, shape (N_atoms,), dtype Any
        Mat33.
    positions : wp.array, shape (N_atoms,), dtype Any
        Vec3.
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-space k-vectors.
    coeff2 : wp.array, shape (N_k,), dtype wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    grad_rho : wp.array2d, shape (N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    grad_k : wp.array2d, shape (N_k, 3), dtype wp.float64
        OUTPUT: gradient w.r.t. the k-vectors.
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]
    k_vec = k_vectors[k_idx]
    gr = grad_rho[k_idx, 0]
    gi = grad_rho[k_idx, 1]
    c2 = coeff2[k_idx]
    gkx = wp.float64(0.0)
    gky = wp.float64(0.0)
    gkz = wp.float64(0.0)
    for i in range(n_atoms):
        Q = quadrupoles[i]
        qk0 = (
            wp.float64(Q[0, 0]) * k_vec[0]
            + wp.float64(Q[0, 1]) * k_vec[1]
            + wp.float64(Q[0, 2]) * k_vec[2]
        )
        qk1 = (
            wp.float64(Q[1, 0]) * k_vec[0]
            + wp.float64(Q[1, 1]) * k_vec[1]
            + wp.float64(Q[1, 2]) * k_vec[2]
        )
        qk2 = (
            wp.float64(Q[2, 0]) * k_vec[0]
            + wp.float64(Q[2, 1]) * k_vec[1]
            + wp.float64(Q[2, 2]) * k_vec[2]
        )
        kQk = qk0 * k_vec[0] + qk1 * k_vec[1] + qk2 * k_vec[2]
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        w1 = wp.float64(2.0) * (gr * cos_ki - gi * sin_ki)
        w2 = -kQk * (gr * sin_ki + gi * cos_ki)
        pos_i = positions[i]
        gkx += w1 * qk0 + w2 * wp.float64(pos_i[0])
        gky += w1 * qk1 + w2 * wp.float64(pos_i[1])
        gkz += w1 * qk2 + w2 * wp.float64(pos_i[2])
    sc = scale * c2
    grad_k[k_idx, 0] = sc * gkx
    grad_k[k_idx, 1] = sc * gky
    grad_k[k_idx, 2] = sc * gkz


def _rho_q_kvec_grad_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=m),
        wp.array(dtype=v),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array(dtype=wp.vec3d),
        wp.array(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.float64,
        wp.array2d(dtype=wp.float64),
    ]


@wp.kernel
def _rho_q_kvec_grad_double_backward_kernel(
    g_k: wp.array2d(dtype=wp.float64),  # (N_k, 3) cotangent on grad_k
    quadrupoles: wp.array(dtype=Any),  # (N_atoms,) mat33
    positions: wp.array(dtype=Any),  # (N_atoms,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    source_coeff2: wp.array(dtype=wp.float64),  # (N_k,)
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    scale: wp.float64,
    ggrad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2) OUTPUT (per-k)
    ggrad_quad: wp.array2d(dtype=wp.float64),  # (N_atoms, 9) OUTPUT (atomic)
    ggrad_positions: wp.array(dtype=wp.vec3d),  # (N_atoms,) OUTPUT (atomic)
    ggrad_coeff2: wp.array(dtype=wp.float64),  # (N_k,) OUTPUT (per-k)
    ggrad_kvec: wp.array2d(dtype=wp.float64),  # (N_k, 3) OUTPUT (per-k)
):
    r"""Second-order backward of :func:`_rho_q_kvec_grad_kernel`.

    First backward: ``grad_k[k] = scale·c2·Σ_i [w1·(Q_i k) + w2·r_i]``,
    ``w1 = 2(gr\cos − gi\sin)``, ``w2 = −kQk(gr\sin + gi\cos)``. With cotangent
    ``G_k``, ``GQk = G_k·(Q_i k)``, ``d = G_k·r_i``, ``s = gr\sin+gi\cos``,
    ``h = gr\cos−gi\sin``, ``h1 = −gr\sin−gi\cos``: grads (× ``scale·c2`` except
    coeff2 which is ``× scale``) —
    ``∂/∂c2 = Σ_i(w1 GQk + w2 d)`` (×scale);
    ``∂/∂gr = Σ_i[2\cos GQk − kQk\sin d]``, ``∂/∂gi = Σ_i[−2\sin GQk − kQk\cos d]``;
    ``∂/∂Q_{i,ab} = w1 G_k[a] k_b − d·s·k_a k_b``;
    ``∂/∂r_i = w2 G_k + (2 GQk h1 − d kQk h) k``;
    ``∂/∂k = Σ_i[2 h1 GQk r_i + w1(Q G_k) + (−2(Q k)s − kQk h r_i) d]``.
    ``ggrad_quad`` / ``ggrad_positions`` accumulate over k (pre-zero).

    Launch Grid
    -----------
    dim = [N_k].
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]
    gr = grad_rho[k_idx, 0]
    gi = grad_rho[k_idx, 1]
    kv = k_vectors[k_idx]
    gk0 = g_k[k_idx, 0]
    gk1 = g_k[k_idx, 1]
    gk2 = g_k[k_idx, 2]
    c2 = source_coeff2[k_idx]
    sgc = scale * c2

    ggc = wp.float64(0.0)
    ggr_gr = wp.float64(0.0)
    ggr_gi = wp.float64(0.0)
    gkx = wp.float64(0.0)
    gky = wp.float64(0.0)
    gkz = wp.float64(0.0)
    for i in range(n_atoms):
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        q_mat = quadrupoles[i]
        qk0 = (
            wp.float64(q_mat[0, 0]) * kv[0]
            + wp.float64(q_mat[0, 1]) * kv[1]
            + wp.float64(q_mat[0, 2]) * kv[2]
        )
        qk1 = (
            wp.float64(q_mat[1, 0]) * kv[0]
            + wp.float64(q_mat[1, 1]) * kv[1]
            + wp.float64(q_mat[1, 2]) * kv[2]
        )
        qk2 = (
            wp.float64(q_mat[2, 0]) * kv[0]
            + wp.float64(q_mat[2, 1]) * kv[1]
            + wp.float64(q_mat[2, 2]) * kv[2]
        )
        kqk = qk0 * kv[0] + qk1 * kv[1] + qk2 * kv[2]
        w1 = wp.float64(2.0) * (gr * cos_ki - gi * sin_ki)
        w2 = -kqk * (gr * sin_ki + gi * cos_ki)
        s_t = gr * sin_ki + gi * cos_ki
        h_t = gr * cos_ki - gi * sin_ki
        h1 = -gr * sin_ki - gi * cos_ki
        pos_i = positions[i]
        px = wp.float64(pos_i[0])
        py = wp.float64(pos_i[1])
        pz = wp.float64(pos_i[2])
        gqk = gk0 * qk0 + gk1 * qk1 + gk2 * qk2
        d = gk0 * px + gk1 * py + gk2 * pz

        ggc += w1 * gqk + w2 * d
        ggr_gr += wp.float64(2.0) * cos_ki * gqk - kqk * sin_ki * d
        ggr_gi += -wp.float64(2.0) * sin_ki * gqk - kqk * cos_ki * d

        # ∂/∂Q_{i,ab} = sgc·(w1·G_k[a]·k_b − d·s·k_a k_b).
        gka = wp.vec3d(gk0, gk1, gk2)
        for a in range(3):
            for bb in range(3):
                wp.atomic_add(
                    ggrad_quad,
                    i,
                    3 * a + bb,
                    sgc * (w1 * gka[a] * kv[bb] - d * s_t * kv[a] * kv[bb]),
                )

        # ∂/∂r_i = sgc·(w2·G_k + (2 GQk h1 − d kQk h) k).
        pc = wp.float64(2.0) * gqk * h1 - d * kqk * h_t
        wp.atomic_add(
            ggrad_positions,
            i,
            wp.vec3d(
                sgc * (w2 * gk0 + pc * kv[0]),
                sgc * (w2 * gk1 + pc * kv[1]),
                sgc * (w2 * gk2 + pc * kv[2]),
            ),
        )

        # ∂/∂k = sgc·Σ_i [2 h1 GQk r_i + w1 (Q G_k) − (2(Q k)s + kQk h r_i) d].
        qg0 = (
            wp.float64(q_mat[0, 0]) * gk0
            + wp.float64(q_mat[0, 1]) * gk1
            + wp.float64(q_mat[0, 2]) * gk2
        )
        qg1 = (
            wp.float64(q_mat[1, 0]) * gk0
            + wp.float64(q_mat[1, 1]) * gk1
            + wp.float64(q_mat[1, 2]) * gk2
        )
        qg2 = (
            wp.float64(q_mat[2, 0]) * gk0
            + wp.float64(q_mat[2, 1]) * gk1
            + wp.float64(q_mat[2, 2]) * gk2
        )
        base = wp.float64(2.0) * h1 * gqk
        kqk_h_d = kqk * h_t * d
        two_s_d = wp.float64(2.0) * s_t * d
        gkx += base * px + w1 * qg0 - (two_s_d * qk0 + kqk_h_d * px)
        gky += base * py + w1 * qg1 - (two_s_d * qk1 + kqk_h_d * py)
        gkz += base * pz + w1 * qg2 - (two_s_d * qk2 + kqk_h_d * pz)

    ggrad_rho[k_idx, 0] = sgc * ggr_gr
    ggrad_rho[k_idx, 1] = sgc * ggr_gi
    ggrad_coeff2[k_idx] = scale * ggc
    ggrad_kvec[k_idx, 0] = sgc * gkx
    ggrad_kvec[k_idx, 1] = sgc * gky
    ggrad_kvec[k_idx, 2] = sgc * gkz


def _rho_q_kvec_grad_double_backward_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array2d(dtype=wp.float64),  # g_k
        wp.array(dtype=m),  # quadrupoles
        wp.array(dtype=v),  # positions
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array(dtype=wp.vec3d),  # k_vectors
        wp.array(dtype=wp.float64),  # source_coeff2
        wp.array2d(dtype=wp.float64),  # grad_rho
        wp.float64,  # scale
        wp.array2d(dtype=wp.float64),  # ggrad_rho
        wp.array2d(dtype=wp.float64),  # ggrad_quad
        wp.array(dtype=wp.vec3d),  # ggrad_positions
        wp.array(dtype=wp.float64),  # ggrad_coeff2
        wp.array2d(dtype=wp.float64),  # ggrad_kvec
    ]


@wp.kernel
def _batch_rho_q_kvec_grad_double_backward_kernel(
    g_k: wp.array3d(dtype=wp.float64),  # (B, N_k, 3) cotangent on grad_k
    quadrupoles: wp.array(dtype=Any),  # (N_total,) mat33
    positions: wp.array(dtype=Any),  # (N_total,) vec3
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_total)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_total)
    k_vectors: wp.array2d(dtype=wp.vec3d),  # (B, N_k)
    source_coeff2: wp.array2d(dtype=wp.float64),  # (B, N_k)
    grad_rho: wp.array3d(dtype=wp.float64),  # (B, N_k, 2)
    volume: wp.array(dtype=wp.float64),  # (B,)
    atom_start: wp.array(dtype=wp.int32),  # (B,)
    atom_end: wp.array(dtype=wp.int32),  # (B,)
    ggrad_rho: wp.array3d(dtype=wp.float64),  # (B, N_k, 2) OUTPUT (per-k)
    ggrad_quad: wp.array2d(dtype=wp.float64),  # (N_total, 9) OUTPUT (atomic)
    ggrad_positions: wp.array(dtype=wp.vec3d),  # (N_total,) OUTPUT (atomic)
    ggrad_coeff2: wp.array2d(dtype=wp.float64),  # (B, N_k) OUTPUT (per-k)
    ggrad_kvec: wp.array3d(dtype=wp.float64),  # (B, N_k, 3) OUTPUT (per-k)
):
    r"""Batched second-order backward of :func:`_batch_rho_q_kvec_grad_kernel`.

    Per-system mirror of :func:`_rho_q_kvec_grad_double_backward_kernel`: the
    ``(b, k_idx)`` grid sweeps system ``b``'s atoms and applies ``scale =
    (2π)³/volume[b]``. ``ggrad_quad`` / ``ggrad_positions`` accumulate over k
    and must be pre-zeroed.

    Launch Grid
    -----------
    dim = [B, N_k].
    """
    b, k_idx = wp.tid()
    scale = _TWO_PI_CUBED / volume[b]
    gr = grad_rho[b, k_idx, 0]
    gi = grad_rho[b, k_idx, 1]
    kv = k_vectors[b, k_idx]
    gk0 = g_k[b, k_idx, 0]
    gk1 = g_k[b, k_idx, 1]
    gk2 = g_k[b, k_idx, 2]
    c2 = source_coeff2[b, k_idx]
    sgc = scale * c2

    ggc = wp.float64(0.0)
    ggr_gr = wp.float64(0.0)
    ggr_gi = wp.float64(0.0)
    gkx = wp.float64(0.0)
    gky = wp.float64(0.0)
    gkz = wp.float64(0.0)
    for i in range(atom_start[b], atom_end[b]):
        cos_ki = cosines[k_idx, i]
        sin_ki = sines[k_idx, i]
        q_mat = quadrupoles[i]
        qk0 = (
            wp.float64(q_mat[0, 0]) * kv[0]
            + wp.float64(q_mat[0, 1]) * kv[1]
            + wp.float64(q_mat[0, 2]) * kv[2]
        )
        qk1 = (
            wp.float64(q_mat[1, 0]) * kv[0]
            + wp.float64(q_mat[1, 1]) * kv[1]
            + wp.float64(q_mat[1, 2]) * kv[2]
        )
        qk2 = (
            wp.float64(q_mat[2, 0]) * kv[0]
            + wp.float64(q_mat[2, 1]) * kv[1]
            + wp.float64(q_mat[2, 2]) * kv[2]
        )
        kqk = qk0 * kv[0] + qk1 * kv[1] + qk2 * kv[2]
        w1 = wp.float64(2.0) * (gr * cos_ki - gi * sin_ki)
        w2 = -kqk * (gr * sin_ki + gi * cos_ki)
        s_t = gr * sin_ki + gi * cos_ki
        h_t = gr * cos_ki - gi * sin_ki
        h1 = -gr * sin_ki - gi * cos_ki
        pos_i = positions[i]
        px = wp.float64(pos_i[0])
        py = wp.float64(pos_i[1])
        pz = wp.float64(pos_i[2])
        gqk = gk0 * qk0 + gk1 * qk1 + gk2 * qk2
        d = gk0 * px + gk1 * py + gk2 * pz

        ggc += w1 * gqk + w2 * d
        ggr_gr += wp.float64(2.0) * cos_ki * gqk - kqk * sin_ki * d
        ggr_gi += -wp.float64(2.0) * sin_ki * gqk - kqk * cos_ki * d

        gka = wp.vec3d(gk0, gk1, gk2)
        for a in range(3):
            for bb in range(3):
                wp.atomic_add(
                    ggrad_quad,
                    i,
                    3 * a + bb,
                    sgc * (w1 * gka[a] * kv[bb] - d * s_t * kv[a] * kv[bb]),
                )

        pc = wp.float64(2.0) * gqk * h1 - d * kqk * h_t
        wp.atomic_add(
            ggrad_positions,
            i,
            wp.vec3d(
                sgc * (w2 * gk0 + pc * kv[0]),
                sgc * (w2 * gk1 + pc * kv[1]),
                sgc * (w2 * gk2 + pc * kv[2]),
            ),
        )

        qg0 = (
            wp.float64(q_mat[0, 0]) * gk0
            + wp.float64(q_mat[0, 1]) * gk1
            + wp.float64(q_mat[0, 2]) * gk2
        )
        qg1 = (
            wp.float64(q_mat[1, 0]) * gk0
            + wp.float64(q_mat[1, 1]) * gk1
            + wp.float64(q_mat[1, 2]) * gk2
        )
        qg2 = (
            wp.float64(q_mat[2, 0]) * gk0
            + wp.float64(q_mat[2, 1]) * gk1
            + wp.float64(q_mat[2, 2]) * gk2
        )
        base = wp.float64(2.0) * h1 * gqk
        kqk_h_d = kqk * h_t * d
        two_s_d = wp.float64(2.0) * s_t * d
        gkx += base * px + w1 * qg0 - (two_s_d * qk0 + kqk_h_d * px)
        gky += base * py + w1 * qg1 - (two_s_d * qk1 + kqk_h_d * py)
        gkz += base * pz + w1 * qg2 - (two_s_d * qk2 + kqk_h_d * pz)

    ggrad_rho[b, k_idx, 0] = sgc * ggr_gr
    ggrad_rho[b, k_idx, 1] = sgc * ggr_gi
    ggrad_coeff2[b, k_idx] = scale * ggc
    ggrad_kvec[b, k_idx, 0] = sgc * gkx
    ggrad_kvec[b, k_idx, 1] = sgc * gky
    ggrad_kvec[b, k_idx, 2] = sgc * gkz


def _batch_rho_q_kvec_grad_double_backward_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array3d(dtype=wp.float64),  # g_k
        wp.array(dtype=m),  # quadrupoles
        wp.array(dtype=v),  # positions
        wp.array2d(dtype=wp.float64),  # cosines
        wp.array2d(dtype=wp.float64),  # sines
        wp.array2d(dtype=wp.vec3d),  # k_vectors
        wp.array2d(dtype=wp.float64),  # source_coeff2
        wp.array3d(dtype=wp.float64),  # grad_rho
        wp.array(dtype=wp.float64),  # volume
        wp.array(dtype=wp.int32),  # atom_start
        wp.array(dtype=wp.int32),  # atom_end
        wp.array3d(dtype=wp.float64),  # ggrad_rho
        wp.array2d(dtype=wp.float64),  # ggrad_quad
        wp.array(dtype=wp.vec3d),  # ggrad_positions
        wp.array2d(dtype=wp.float64),  # ggrad_coeff2
        wp.array3d(dtype=wp.float64),  # ggrad_kvec
    ]


@wp.kernel
def _rho_q_moment_grad_kernel(
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    coeff2: wp.array(dtype=wp.float64),  # (N_k,)
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    scale: wp.float64,
    grad_quadrupoles: wp.array2d(dtype=wp.float64),  # (N_atoms, 9) OUTPUT
):
    r"""Moment gradient ``dL/dQ_i`` (backward of ``_assemble_rho_q`` w.r.t. Q).

    Per atom ``i``, accumulates over all k-vectors the symmetric outer
    product ``w · k ⊗ k`` where
    ``w = coeff2[k]·(grad_rho[k,0]·cos[k,i] − grad_rho[k,1]·sin[k,i])``,
    then writes ``scale·`` the full (symmetric) 3×3 into the flat 9-slot
    ``grad_quadrupoles`` row.

    Launch Grid
    -----------
    dim = [N_atoms] — one thread per atom; each thread sweeps all k-vectors.


    Parameters
    ----------
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-space k-vectors.
    coeff2 : wp.array, shape (N_k,), dtype wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    grad_rho : wp.array2d, shape (N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    grad_quadrupoles : wp.array2d, shape (N_atoms, 9), dtype wp.float64
        OUTPUT: gradient w.r.t. the quadrupole moments.
    """
    i_idx = wp.tid()
    n_k = cosines.shape[0]
    g00 = wp.float64(0.0)
    g01 = wp.float64(0.0)
    g02 = wp.float64(0.0)
    g11 = wp.float64(0.0)
    g12 = wp.float64(0.0)
    g22 = wp.float64(0.0)
    for k_idx in range(n_k):
        k_vec = k_vectors[k_idx]
        w = coeff2[k_idx] * (
            grad_rho[k_idx, 0] * cosines[k_idx, i_idx]
            - grad_rho[k_idx, 1] * sines[k_idx, i_idx]
        )
        g00 += w * k_vec[0] * k_vec[0]
        g01 += w * k_vec[0] * k_vec[1]
        g02 += w * k_vec[0] * k_vec[2]
        g11 += w * k_vec[1] * k_vec[1]
        g12 += w * k_vec[1] * k_vec[2]
        g22 += w * k_vec[2] * k_vec[2]
    grad_quadrupoles[i_idx, 0] = scale * g00
    grad_quadrupoles[i_idx, 1] = scale * g01
    grad_quadrupoles[i_idx, 2] = scale * g02
    grad_quadrupoles[i_idx, 3] = scale * g01
    grad_quadrupoles[i_idx, 4] = scale * g11
    grad_quadrupoles[i_idx, 5] = scale * g12
    grad_quadrupoles[i_idx, 6] = scale * g02
    grad_quadrupoles[i_idx, 7] = scale * g12
    grad_quadrupoles[i_idx, 8] = scale * g22


@wp.kernel
def _rhoq_posgrad_backward_grad_rho_kernel(
    quadrupoles: wp.array(dtype=Any),  # (N_atoms,) mat33
    gg_pos: wp.array2d(dtype=wp.float64),  # (N_atoms, 3)
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    coeff2: wp.array(dtype=wp.float64),  # (N_k,)
    scale: wp.float64,
    grad_rho_grad: wp.array2d(dtype=wp.float64),  # (N_k, 2) OUTPUT
):
    r"""K_a: ``∂L/∂grad_rho`` of the Q-channel position gradient.

    Per k (one thread per k, sweeps atoms):
        u_i = (k.Q_i.k) * (k . gg_pos_i)
        grad_rho_grad[k,0] = -scale * coeff2(k) * Sum_i u_i * sin_ki
        grad_rho_grad[k,1] = -scale * coeff2(k) * Sum_i u_i * cos_ki


    Launch Grid
    -----------
    ``dim`` indexes ``(k_idx)``; each thread processes one k-vector index.

    Parameters
    ----------
    quadrupoles : wp.array, shape (N_atoms,), dtype Any
        Mat33.
    gg_pos : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-space k-vectors.
    coeff2 : wp.array, shape (N_k,), dtype wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    grad_rho_grad : wp.array2d, shape (N_k, 2), dtype wp.float64
        OUTPUT: gradient w.r.t. the incoming ``grad_rho``.
    """
    k_idx = wp.tid()
    n_atoms = cosines.shape[1]
    k_vec = k_vectors[k_idx]
    acc_s = wp.float64(0.0)
    acc_c = wp.float64(0.0)
    for i in range(n_atoms):
        Q = quadrupoles[i]
        kQ0 = (
            wp.float64(Q[0, 0]) * k_vec[0]
            + wp.float64(Q[0, 1]) * k_vec[1]
            + wp.float64(Q[0, 2]) * k_vec[2]
        )
        kQ1 = (
            wp.float64(Q[1, 0]) * k_vec[0]
            + wp.float64(Q[1, 1]) * k_vec[1]
            + wp.float64(Q[1, 2]) * k_vec[2]
        )
        kQ2 = (
            wp.float64(Q[2, 0]) * k_vec[0]
            + wp.float64(Q[2, 1]) * k_vec[1]
            + wp.float64(Q[2, 2]) * k_vec[2]
        )
        kQk = kQ0 * k_vec[0] + kQ1 * k_vec[1] + kQ2 * k_vec[2]
        kdotgg = (
            k_vec[0] * gg_pos[i, 0] + k_vec[1] * gg_pos[i, 1] + k_vec[2] * gg_pos[i, 2]
        )
        u = kQk * kdotgg
        acc_s += u * sines[k_idx, i]
        acc_c += u * cosines[k_idx, i]
    c2 = coeff2[k_idx]
    grad_rho_grad[k_idx, 0] = -scale * c2 * acc_s
    grad_rho_grad[k_idx, 1] = -scale * c2 * acc_c


@wp.kernel
def _rhoq_posgrad_backward_positions_kernel(
    quadrupoles: wp.array(dtype=Any),  # (N_atoms,) mat33
    gg_pos: wp.array2d(dtype=wp.float64),  # (N_atoms, 3)
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    coeff2: wp.array(dtype=wp.float64),  # (N_k,)
    scale: wp.float64,
    grad_positions: wp.array2d(dtype=wp.float64),  # (N_atoms, 3) OUTPUT
):
    r"""K_c: ``∂L/∂positions`` of the Q-channel position gradient (Hessian diag).

    Per atom (one thread):
        v_k = coeff2(k) * (k.Q_i.k) * (k . gg_pos_i) * (gr*cos_ki - gi*sin_ki)
        grad_pos2[i,b] = -scale * Sum_k v_k * k_b.


    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    quadrupoles : wp.array, shape (N_atoms,), dtype Any
        Mat33.
    gg_pos : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    grad_rho : wp.array2d, shape (N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-space k-vectors.
    coeff2 : wp.array, shape (N_k,), dtype wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    grad_positions : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        OUTPUT: gradient w.r.t. atomic positions.
    """
    i_idx = wp.tid()
    n_k = cosines.shape[0]
    Q = quadrupoles[i_idx]
    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)
    for k_idx in range(n_k):
        k_vec = k_vectors[k_idx]
        kQ0 = (
            wp.float64(Q[0, 0]) * k_vec[0]
            + wp.float64(Q[0, 1]) * k_vec[1]
            + wp.float64(Q[0, 2]) * k_vec[2]
        )
        kQ1 = (
            wp.float64(Q[1, 0]) * k_vec[0]
            + wp.float64(Q[1, 1]) * k_vec[1]
            + wp.float64(Q[1, 2]) * k_vec[2]
        )
        kQ2 = (
            wp.float64(Q[2, 0]) * k_vec[0]
            + wp.float64(Q[2, 1]) * k_vec[1]
            + wp.float64(Q[2, 2]) * k_vec[2]
        )
        kQk = kQ0 * k_vec[0] + kQ1 * k_vec[1] + kQ2 * k_vec[2]
        kdotgg = (
            k_vec[0] * gg_pos[i_idx, 0]
            + k_vec[1] * gg_pos[i_idx, 1]
            + k_vec[2] * gg_pos[i_idx, 2]
        )
        gr = grad_rho[k_idx, 0]
        gi = grad_rho[k_idx, 1]
        v = (
            coeff2[k_idx]
            * kQk
            * kdotgg
            * (gr * cosines[k_idx, i_idx] - gi * sines[k_idx, i_idx])
        )
        gx += v * k_vec[0]
        gy += v * k_vec[1]
        gz += v * k_vec[2]
    grad_positions[i_idx, 0] = -scale * gx
    grad_positions[i_idx, 1] = -scale * gy
    grad_positions[i_idx, 2] = -scale * gz


@wp.kernel
def _rhoq_posgrad_backward_quad_kernel(
    gg_pos: wp.array2d(dtype=wp.float64),  # (N_atoms, 3)
    grad_rho: wp.array2d(dtype=wp.float64),  # (N_k, 2)
    cosines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    sines: wp.array2d(dtype=wp.float64),  # (N_k, N_atoms)
    k_vectors: wp.array(dtype=wp.vec3d),  # (N_k,)
    coeff2: wp.array(dtype=wp.float64),  # (N_k,)
    scale: wp.float64,
    grad_quadrupoles: wp.array2d(dtype=wp.float64),  # (N_atoms, 9) OUTPUT
):
    r"""K_b: ``∂L/∂Q_i`` of the Q-channel position gradient (mixed ∂r∂Q).

    Independent of Q (grad_pos is linear in Q). Per atom (one thread):
        w_k = coeff2(k) * (k . gg_pos_i) * (gr*sin_ki + gi*cos_ki)
        grad_Q[i,b,c] = -scale * Sum_k w_k * k_b * k_c  (symmetric).


    Launch Grid
    -----------
    ``dim`` indexes ``(i_idx)``; each thread processes one i_idx index.

    Parameters
    ----------
    gg_pos : wp.array2d, shape (N_atoms, 3), dtype wp.float64
        Second-order upstream gradient w.r.t. positions (HVP seed).
    grad_rho : wp.array2d, shape (N_k, 2), dtype wp.float64
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    cosines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
        Reciprocal-space k-vectors.
    coeff2 : wp.array, shape (N_k,), dtype wp.float64
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    scale : wp.float64
        Scalar prefactor applied to the contribution.
    grad_quadrupoles : wp.array2d, shape (N_atoms, 9), dtype wp.float64
        OUTPUT: gradient w.r.t. the quadrupole moments.
    """
    i_idx = wp.tid()
    n_k = cosines.shape[0]
    g00 = wp.float64(0.0)
    g01 = wp.float64(0.0)
    g02 = wp.float64(0.0)
    g11 = wp.float64(0.0)
    g12 = wp.float64(0.0)
    g22 = wp.float64(0.0)
    for k_idx in range(n_k):
        k_vec = k_vectors[k_idx]
        kdotgg = (
            k_vec[0] * gg_pos[i_idx, 0]
            + k_vec[1] * gg_pos[i_idx, 1]
            + k_vec[2] * gg_pos[i_idx, 2]
        )
        gr = grad_rho[k_idx, 0]
        gi = grad_rho[k_idx, 1]
        w = (
            coeff2[k_idx]
            * kdotgg
            * (gr * sines[k_idx, i_idx] + gi * cosines[k_idx, i_idx])
        )
        g00 += w * k_vec[0] * k_vec[0]
        g01 += w * k_vec[0] * k_vec[1]
        g02 += w * k_vec[0] * k_vec[2]
        g11 += w * k_vec[1] * k_vec[1]
        g12 += w * k_vec[1] * k_vec[2]
        g22 += w * k_vec[2] * k_vec[2]
    nscale = -scale
    grad_quadrupoles[i_idx, 0] = nscale * g00
    grad_quadrupoles[i_idx, 1] = nscale * g01
    grad_quadrupoles[i_idx, 2] = nscale * g02
    grad_quadrupoles[i_idx, 3] = nscale * g01
    grad_quadrupoles[i_idx, 4] = nscale * g11
    grad_quadrupoles[i_idx, 5] = nscale * g12
    grad_quadrupoles[i_idx, 6] = nscale * g02
    grad_quadrupoles[i_idx, 7] = nscale * g12
    grad_quadrupoles[i_idx, 8] = nscale * g22


def _rhoq_posgrad_bw_grad_rho_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=m),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array(dtype=wp.vec3d),
        wp.array(dtype=wp.float64),
        wp.float64,
        wp.array2d(dtype=wp.float64),
    ]


def _rhoq_posgrad_bw_positions_sig(v, t):
    m = wp.mat33d if t == wp.float64 else wp.mat33f
    return [
        wp.array(dtype=m),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array2d(dtype=wp.float64),
        wp.array(dtype=wp.vec3d),
        wp.array(dtype=wp.float64),
        wp.float64,
        wp.array2d(dtype=wp.float64),
    ]


@wp.kernel
def _v_grad_from_feat_grad_backward_positions_quadrupole_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat_l2: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    gg_v: wp.array2d(dtype=wp.float64),
    k_vectors: wp.array(dtype=wp.vec3d),
    ggrad_positions: wp.array2d(dtype=wp.float64),
):
    r"""Backward of ``l = 2`` ``v_gradient_from_feature_grad`` w.r.t. positions.

    .. math::

        \tilde g_{r_i, \beta} = \mathrm{scale} \sum_k w(k) \, k_\beta \bigl\{
            \cos(k \cdot r_i) [gg_{V_r} Q_i - gg_{V_i} Q_r]
          - \sin(k \cdot r_i) [gg_{V_r} Q_r + gg_{V_i} Q_i] \bigr\}

    One thread per atom; inner loop over k, inner-inner over ``(σ, lm)``
    with the five ``l = 2`` angular channels.

    Launch Grid
    -----------
    dim = [N_atoms]; output shape (N_atoms, 3).


    Parameters
    ----------
    grad_raw : wp.array3d, dtype wp.float64
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array4d, dtype wp.float64
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array2d, dtype wp.float64
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array2d, dtype wp.float64
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array, dtype wp.float64
        Per-k projection factor for the feature/direct-k-space projection.
    gg_v : wp.array2d, dtype wp.float64
        Second-order upstream gradient w.r.t. ``v`` (HVP seed).
    k_vectors : wp.array, dtype wp.vec3d
        Reciprocal-space k-vectors.
    ggrad_positions : wp.array2d, dtype wp.float64
        OUTPUT: double-backward gradient w.r.t. positions.
    """
    i_idx = wp.tid()
    n_k = cosines.shape[0]
    n_sigma = receiver_phi_hat_l2.shape[1]

    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)

    for k_idx in range(n_k):
        k_vec = k_vectors[k_idx]
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]
        kfp = k_factor_proj[k_idx]

        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(5):
                g = grad_raw[i_idx, s_idx, lm_idx]
                q_r += receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 0] * g
                q_i += receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 1] * g

        gvr = gg_v[k_idx, 0]
        gvi = gg_v[k_idx, 1]

        cos_term = gvr * q_i - gvi * q_r
        sin_term = gvr * q_r + gvi * q_i

        weight = kfp * (cos_ki * cos_term - sin_ki * sin_term)
        gx += k_vec[0] * weight
        gy += k_vec[1] * weight
        gz += k_vec[2] * weight

    ggrad_positions[i_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * gx
    ggrad_positions[i_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * gy
    ggrad_positions[i_idx, 2] = _INV_TWO_PI_CUBED_TIMES_TWO * gz


@wp.kernel
def _v_gradient_from_feature_grad_quadrupole_kernel(
    grad_raw: wp.array3d(dtype=wp.float64),
    receiver_phi_hat_l2: wp.array4d(dtype=wp.float64),
    cosines: wp.array2d(dtype=wp.float64),
    sines: wp.array2d(dtype=wp.float64),
    k_factor_proj: wp.array(dtype=wp.float64),
    grad_v: wp.array2d(dtype=wp.float64),
):
    r"""Analytical backward of :func:`project_features_lmax2` w.r.t. ``V(k)``.

    One thread per k-vector; inner loop over atoms and ``(σ, lm)`` with the
    five ``l = 2`` angular channels. The per-atom ``(Q_r, Q_i)`` reduction
    over ``(σ, lm)`` is recomputed inside each thread rather than cached, to
    avoid materializing the ``(N_k, N_atoms, 2)`` intermediate.

    Launch Grid
    -----------
    dim = [N_k].

    Parameters
    ----------
    grad_raw : wp.array3d, shape (N_atoms, N_σ, 5), dtype wp.float64
        Cotangent of the raw (un-self-interaction-subtracted,
        natural-layout) ``l = 2`` feature tensor.
    receiver_phi_hat_l2 : wp.array4d, shape (N_k, N_σ, 5, 2), dtype wp.float64
        Receiver-basis ``l = 2`` Fourier coefficients from
        :func:`eval_receiver_gto_fourier_quadrupole`.
    cosines, sines : wp.array2d, shape (N_k, N_atoms), dtype wp.float64
        Structure-factor tables from the forward.
    k_factor_proj : wp.array, shape (N_k,), dtype wp.float64
        Per-k weight (``0.5`` at k = 0, ``1`` elsewhere in the direct k-space sum).
    grad_v : wp.array2d, shape (N_k, 2), dtype wp.float64
        OUTPUT: ``∂L/∂V(k)`` with ``[:, 0]`` / ``[:, 1]`` = (real, imag).
        Does not need pre-zeroing.
    """
    k_idx = wp.tid()

    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat_l2.shape[1]

    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)

    for i_idx in range(n_atoms):
        cos_ki = cosines[k_idx, i_idx]
        sin_ki = sines[k_idx, i_idx]

        # Q_r(k, i), Q_i(k, i) = Σ_{σ, lm} grad_raw · φ̂.
        q_r = wp.float64(0.0)
        q_i = wp.float64(0.0)
        for s_idx in range(n_sigma):
            for lm_idx in range(5):
                g = grad_raw[i_idx, s_idx, lm_idx]
                q_r += receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 0] * g
                q_i += receiver_phi_hat_l2[k_idx, s_idx, lm_idx, 1] * g

        acc_r += q_r * cos_ki + q_i * sin_ki
        acc_i += q_i * cos_ki - q_r * sin_ki

    kfp = k_factor_proj[k_idx]
    grad_v[k_idx, 0] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_r
    grad_v[k_idx, 1] = _INV_TWO_PI_CUBED_TIMES_TWO * kfp * acc_i


def assemble_rho_q(
    quadrupoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    volume: float,
    rho: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_assemble_rho_q_kernel`.

    Parameters
    ----------
    quadrupoles : wp.array, shape (N_atoms,), dtype wp.mat33f or wp.mat33d
        Cartesian symmetric quadrupole tensor (trace included).
    cosines, sines : wp.array, shape (N_k, N_atoms), dtype wp.float64
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
    coeff2 : wp.array, shape (N_k,), dtype wp.float64
        From :func:`eval_gto_fourier_q`.
    volume : float
    rho : wp.array, shape (N_k, 2), dtype wp.float64
        Pre-allocated output (the Q-only contribution; caller adds it to the
        l<=1 rho).
    wp_dtype : type
        ``wp.float32`` or ``wp.float64``; selects the mat33 overload.
    device : str, optional
        Defaults to ``cosines.device``.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _assemble_rho_q_overloads[vec_dtype],
        dim=cosines.shape[0],
        inputs=[
            quadrupoles,
            cosines,
            sines,
            k_vectors,
            coeff2,
            wp.float64(volume),
            rho,
        ],
        device=device,
    )


def batch_assemble_rho_q(
    quadrupoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    volume: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    rho: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_assemble_rho_q_kernel`.

    Parameters
    ----------
    quadrupoles : wp.array
        Per-atom Cartesian quadrupole moments.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    coeff2 : wp.array
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    volume : wp.array
        Unit-cell volume.
    atom_start : wp.array
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array
        Per-system end offset into the flat atom arrays.
    rho : wp.array
        OUTPUT: complex reciprocal-space density :math:`\hat\rho(k)` (re, im).
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    batch_size = k_vectors.shape[0]
    k_max = k_vectors.shape[1]
    wp.launch(
        _batch_assemble_rho_q_overloads[vec_dtype],
        dim=(batch_size, k_max),
        inputs=[
            quadrupoles,
            cosines,
            sines,
            k_vectors,
            coeff2,
            volume,
            atom_start,
            atom_end,
            rho,
        ],
        device=device,
    )


def batch_eval_receiver_gto_fourier_quadrupole(
    k_vectors: wp.array,
    k_norm2: wp.array,
    sigmas: wp.array,
    inv_cl_l2: wp.array,
    output: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_eval_receiver_gto_fourier_quadrupole_kernel`.

    ``output`` is ``(B, K_max, N_σ, 5)`` dtype ``wp.vec2d`` (the underlying
    ``(B, K_max, N_σ, 5, 2)`` float64 buffer reinterpreted via
    ``wp.from_torch(t, dtype=wp.vec2d)``).


    Parameters
    ----------
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    k_norm2 : wp.array
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigmas : wp.array
        Per-channel Gaussian (GTO) width parameters.
    inv_cl_l2 : wp.array
        Inverse :math:`l=2` overlap normalization constant.
    output : wp.array
        OUTPUT: GTO Fourier coefficients :math:`\hat\phi_{l,m}^{\sigma}(k)`.
    device : str
        Warp device string; defaults to the input array's device.
    """
    batch_size = k_vectors.shape[0]
    k_max = k_vectors.shape[1]
    n_sigma = sigmas.shape[0]
    if device is None:
        device = str(k_vectors.device)
    wp.launch(
        _batch_eval_receiver_gto_fourier_quadrupole_kernel,
        dim=(batch_size, k_max, n_sigma),
        inputs=[
            k_vectors,
            k_norm2,
            sigmas,
            inv_cl_l2,
            output,
        ],
        device=device,
    )


def batch_feat_position_grad_backward_grad_raw_quadrupole(
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    batch_idx: wp.array,
    ggrad_grad_raw: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_feat_position_grad_backward_grad_raw_quadrupole_kernel`.

    Parameters
    ----------
    receiver_phi_hat_l2 : wp.array
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    ggrad_grad_raw : wp.array
        OUTPUT: double-backward gradient w.r.t. ``grad_raw``.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_total = cosines.shape[1]
    n_sigma = receiver_phi_hat_l2.shape[2]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_feat_position_grad_backward_grad_raw_quadrupole_kernel,
        dim=(n_total, n_sigma, 5),
        inputs=[
            receiver_phi_hat_l2,
            cosines,
            sines,
            k_factor_proj,
            potential,
            gg_positions,
            k_vectors,
            batch_idx,
            ggrad_grad_raw,
        ],
        device=device,
    )


def batch_feat_position_grad_backward_positions_quadrupole(
    grad_raw: wp.array,
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    batch_idx: wp.array,
    ggrad_positions: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_feat_position_grad_backward_positions_quadrupole_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    ggrad_positions : wp.array
        OUTPUT: double-backward gradient w.r.t. positions.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_total = cosines.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_feat_position_grad_backward_positions_quadrupole_kernel,
        dim=n_total,
        inputs=[
            grad_raw,
            receiver_phi_hat_l2,
            cosines,
            sines,
            k_factor_proj,
            potential,
            gg_positions,
            k_vectors,
            batch_idx,
            ggrad_positions,
        ],
        device=device,
    )


def batch_feat_position_grad_backward_v_quadrupole(
    grad_raw: wp.array,
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    ggrad_v: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_feat_position_grad_backward_v_quadrupole_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    atom_start : wp.array
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array
        Per-system end offset into the flat atom arrays.
    ggrad_v : wp.array
        OUTPUT: double-backward gradient w.r.t. ``v``.
    device : str
        Warp device string; defaults to the input array's device.
    """
    batch_size = receiver_phi_hat_l2.shape[0]
    k_max = receiver_phi_hat_l2.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_feat_position_grad_backward_v_quadrupole_kernel,
        dim=(batch_size, k_max),
        inputs=[
            grad_raw,
            receiver_phi_hat_l2,
            cosines,
            sines,
            k_factor_proj,
            gg_positions,
            k_vectors,
            atom_start,
            atom_end,
            ggrad_v,
        ],
        device=device,
    )


def batch_position_gradient_from_feature_grad_quadrupole(
    grad_raw: wp.array,
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    k_vectors: wp.array,
    batch_idx: wp.array,
    grad_positions: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_position_gradient_from_feature_grad_quadrupole_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    grad_positions : wp.array
        OUTPUT: gradient w.r.t. atomic positions.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_total = cosines.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_position_gradient_from_feature_grad_quadrupole_kernel,
        dim=n_total,
        inputs=[
            grad_raw,
            receiver_phi_hat_l2,
            cosines,
            sines,
            k_factor_proj,
            potential,
            k_vectors,
            batch_idx,
            grad_positions,
        ],
        device=device,
    )


def batch_position_gradient_from_rhoq(
    quadrupoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    grad_rho: wp.array,
    scale: float,
    batch_idx: wp.array,
    grad_positions: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_position_gradient_from_rhoq_kernel`.

    Parameters
    ----------
    quadrupoles : wp.array
        Per-atom Cartesian quadrupole moments.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    coeff2 : wp.array
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : float
        Scalar prefactor applied to the contribution.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    grad_positions : wp.array
        OUTPUT: gradient w.r.t. atomic positions.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_position_gradient_from_rhoq_overloads[vec_dtype],
        dim=cosines.shape[1],
        inputs=[
            quadrupoles,
            cosines,
            sines,
            k_vectors,
            coeff2,
            grad_rho,
            scale,
            batch_idx,
            grad_positions,
        ],
        device=device,
    )


def batch_project_features_quadrupole(
    potential: wp.array,
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    batch_idx: wp.array,
    features: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_project_features_quadrupole_kernel`.

    ``receiver_phi_hat_l2`` is ``(B, K_max, N_σ, 5)`` ``vec2d`` (caller passes
    the underlying ``(B, K_max, N_σ, 5, 2)`` torch tensor via
    ``wp.from_torch(t, dtype=wp.vec2d)``). ``features`` is
    ``(N_total, N_σ, 5)`` float64, pre-allocated.


    Parameters
    ----------
    potential : wp.array
        Per-k reciprocal-space potential factor.
    receiver_phi_hat_l2 : wp.array
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    features : wp.array
        OUTPUT: projected per-atom features.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_sigma = receiver_phi_hat_l2.shape[2]
    n_total = cosines.shape[1]
    if device is None:
        device = str(cosines.device)
    if receiver_phi_hat_l2.shape[3] != 5:
        raise ValueError(
            f"receiver_phi_hat_l2 must have 5 l=2 columns, got "
            f"{tuple(receiver_phi_hat_l2.shape)}"
        )
    if features.shape != (n_total, n_sigma, 5):
        raise ValueError(
            f"features must be (N_total={n_total}, N_σ={n_sigma}"
            f", 5), got {tuple(features.shape)}"
        )
    wp.launch(
        _batch_project_features_quadrupole_kernel,
        dim=(n_total, n_sigma, 5),
        inputs=[
            potential,
            receiver_phi_hat_l2,
            cosines,
            sines,
            k_factor_proj,
            batch_idx,
            features,
        ],
        device=device,
    )


def batch_project_kphase_grad(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    positions: wp.array,
    batch_idx: wp.array,
    grad_k_vectors: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_project_kphase_grad_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    positions : wp.array
        Atomic Cartesian positions.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    grad_k_vectors : wp.array
        OUTPUT: gradient w.r.t. the k-vectors.
    device : str
        Warp device string; defaults to the input array's device.
    """
    b_dim = k_factor_proj.shape[0]
    k_max = k_factor_proj.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_project_kphase_grad_kernel,
        dim=(b_dim, k_max),
        inputs=[
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            potential,
            positions,
            batch_idx,
            grad_k_vectors,
        ],
        device=device,
    )


def batch_project_phihat_grad(
    grad_raw: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    batch_idx: wp.array,
    grad_phi: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_project_phihat_grad_kernel` (channel-generic).

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    grad_phi : wp.array
        OUTPUT: gradient w.r.t. the GTO Fourier coefficients :math:`\hat\phi(k)`.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_sigma = grad_raw.shape[1]
    n_lm = grad_raw.shape[2]
    b_dim = k_factor_proj.shape[0]
    k_max = k_factor_proj.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_project_phihat_grad_kernel,
        dim=(b_dim, k_max, n_sigma, n_lm),
        inputs=[
            grad_raw,
            cosines,
            sines,
            k_factor_proj,
            potential,
            batch_idx,
            grad_phi,
        ],
        device=device,
    )


def batch_receiver_phi_hat_backward_quadrupole(
    grad_output: wp.array,
    k_vectors: wp.array,
    k_norm2: wp.array,
    sigmas: wp.array,
    inv_cl_l2: wp.array,
    grad_k_vectors: wp.array,
    grad_k_norm2: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_receiver_phi_hat_backward_quadrupole_kernel`.

    Parameters
    ----------
    grad_output : wp.array
        Upstream gradient flowing into this backward kernel.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    k_norm2 : wp.array
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigmas : wp.array
        Per-channel Gaussian (GTO) width parameters.
    inv_cl_l2 : wp.array
        Inverse :math:`l=2` overlap normalization constant.
    grad_k_vectors : wp.array
        OUTPUT: gradient w.r.t. the k-vectors.
    grad_k_norm2 : wp.array
        OUTPUT: gradient w.r.t. :math:`|k|^2`.
    device : str
        Warp device string; defaults to the input array's device.
    """
    batch_size = k_vectors.shape[0]
    k_max = k_vectors.shape[1]
    if device is None:
        device = str(k_vectors.device)
    wp.launch(
        _batch_receiver_phi_hat_backward_quadrupole_kernel,
        dim=(batch_size, k_max),
        inputs=[
            grad_output,
            k_vectors,
            k_norm2,
            sigmas,
            inv_cl_l2,
        ],
        outputs=[
            grad_k_vectors,
            grad_k_norm2,
        ],
        device=device,
    )


def batch_rho_kphase_grad(
    charges: wp.array,
    dipoles: wp.array,
    positions: wp.array,
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    grad_rho: wp.array,
    volume: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    grad_k: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_rho_kphase_grad_kernel`.

    Parameters
    ----------
    charges : wp.array
        Per-atom monopole charges.
    dipoles : wp.array
        Per-atom Cartesian dipole moments.
    positions : wp.array
        Atomic Cartesian positions.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    volume : wp.array
        Unit-cell volume.
    atom_start : wp.array
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array
        Per-system end offset into the flat atom arrays.
    grad_k : wp.array
        OUTPUT: gradient w.r.t. the k-vectors.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    B = grad_rho.shape[0]
    k_max = grad_rho.shape[1]
    wp.launch(
        _batch_rho_kphase_grad_overloads[vec_dtype],
        dim=(B, k_max),
        inputs=[
            charges,
            dipoles,
            positions,
            cosines,
            sines,
            source_phi_hat,
            grad_rho,
            volume,
            atom_start,
            atom_end,
            grad_k,
        ],
        device=device,
    )


def batch_rho_kphase_grad_double_backward(
    g_k: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    positions: wp.array,
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    k_vectors: wp.array,
    grad_rho: wp.array,
    volume: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    ggrad_rho: wp.array,
    ggrad_moments: wp.array,
    ggrad_positions: wp.array,
    ggrad_phi: wp.array,
    ggrad_kvec: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launcher for :func:`_batch_rho_kphase_grad_double_backward_kernel`.

    ``ggrad_moments`` / ``ggrad_positions`` accumulate over k via atomics and
    must be pre-zeroed by the caller. ``scale = (2π)³/volume[b]`` is applied
    per-system inside the kernel.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    B = grad_rho.shape[0]
    k_max = grad_rho.shape[1]
    wp.launch(
        _batch_rho_kphase_grad_double_backward_overloads[vec_dtype],
        dim=(B, k_max),
        inputs=[
            g_k,
            charges,
            dipoles,
            positions,
            cosines,
            sines,
            source_phi_hat,
            k_vectors,
            grad_rho,
            volume,
            atom_start,
            atom_end,
            ggrad_rho,
            ggrad_moments,
            ggrad_positions,
            ggrad_phi,
            ggrad_kvec,
        ],
        device=device,
    )


def batch_rho_phihat_grad(
    charges: wp.array,
    dipoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    grad_rho: wp.array,
    volume: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    grad_phi: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_rho_phihat_grad_kernel`.

    Parameters
    ----------
    charges : wp.array
        Per-atom monopole charges.
    dipoles : wp.array
        Per-atom Cartesian dipole moments.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    volume : wp.array
        Unit-cell volume.
    atom_start : wp.array
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array
        Per-system end offset into the flat atom arrays.
    grad_phi : wp.array
        OUTPUT: gradient w.r.t. the GTO Fourier coefficients :math:`\hat\phi(k)`.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    B = grad_rho.shape[0]
    k_max = grad_rho.shape[1]
    wp.launch(
        _batch_rho_phihat_grad_overloads[vec_dtype],
        dim=(B, k_max),
        inputs=[
            charges,
            dipoles,
            cosines,
            sines,
            grad_rho,
            volume,
            atom_start,
            atom_end,
            grad_phi,
        ],
        device=device,
    )


def batch_rho_phihat_grad_double_backward(
    g_phi: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    positions: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    grad_rho: wp.array,
    volume: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    ggrad_rho: wp.array,
    ggrad_moments: wp.array,
    ggrad_positions: wp.array,
    ggrad_kvec: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launcher for :func:`_batch_rho_phihat_grad_double_backward_kernel`.

    ``ggrad_moments`` / ``ggrad_positions`` accumulate over k via atomics and
    must be pre-zeroed by the caller. ``scale = (2π)³/volume[b]`` is applied
    per-system inside the kernel.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    B = grad_rho.shape[0]
    k_max = grad_rho.shape[1]
    wp.launch(
        _batch_rho_phihat_grad_double_backward_overloads[vec_dtype],
        dim=(B, k_max),
        inputs=[
            g_phi,
            charges,
            dipoles,
            positions,
            cosines,
            sines,
            k_vectors,
            grad_rho,
            volume,
            atom_start,
            atom_end,
            ggrad_rho,
            ggrad_moments,
            ggrad_positions,
            ggrad_kvec,
        ],
        device=device,
    )


def batch_rho_q_coeff2_grad(
    quadrupoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    grad_rho: wp.array,
    volume: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    grad_coeff2: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_rho_q_coeff2_grad_kernel`.

    Parameters
    ----------
    quadrupoles : wp.array
        Per-atom Cartesian quadrupole moments.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    volume : wp.array
        Unit-cell volume.
    atom_start : wp.array
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array
        Per-system end offset into the flat atom arrays.
    grad_coeff2 : wp.array
        OUTPUT: gradient w.r.t. the :math:`l=2` coefficient ``coeff2``.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    B = grad_rho.shape[0]
    k_max = grad_rho.shape[1]
    wp.launch(
        _batch_rho_q_coeff2_grad_overloads[vec_dtype],
        dim=(B, k_max),
        inputs=[
            quadrupoles,
            cosines,
            sines,
            k_vectors,
            grad_rho,
            volume,
            atom_start,
            atom_end,
            grad_coeff2,
        ],
        device=device,
    )


def batch_rho_q_kvec_grad(
    quadrupoles: wp.array,
    positions: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    grad_rho: wp.array,
    volume: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    grad_k: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_rho_q_kvec_grad_kernel`.

    Parameters
    ----------
    quadrupoles : wp.array
        Per-atom Cartesian quadrupole moments.
    positions : wp.array
        Atomic Cartesian positions.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    coeff2 : wp.array
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    volume : wp.array
        Unit-cell volume.
    atom_start : wp.array
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array
        Per-system end offset into the flat atom arrays.
    grad_k : wp.array
        OUTPUT: gradient w.r.t. the k-vectors.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    B = grad_rho.shape[0]
    k_max = grad_rho.shape[1]
    wp.launch(
        _batch_rho_q_kvec_grad_overloads[vec_dtype],
        dim=(B, k_max),
        inputs=[
            quadrupoles,
            positions,
            cosines,
            sines,
            k_vectors,
            coeff2,
            grad_rho,
            volume,
            atom_start,
            atom_end,
            grad_k,
        ],
        device=device,
    )


def batch_rho_q_coeff2_grad_double_backward(
    g_c: wp.array,
    quadrupoles: wp.array,
    positions: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    grad_rho: wp.array,
    volume: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    ggrad_rho: wp.array,
    ggrad_quad: wp.array,
    ggrad_positions: wp.array,
    ggrad_kvec: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launcher for :func:`_batch_rho_q_coeff2_grad_double_backward_kernel`.

    ``ggrad_quad`` / ``ggrad_positions`` accumulate over k via atomics and must
    be pre-zeroed. ``scale = (2π)³/volume[b]`` is applied per-system inside the
    kernel.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    B = grad_rho.shape[0]
    k_max = grad_rho.shape[1]
    wp.launch(
        _batch_rho_q_coeff2_grad_double_backward_overloads[vec_dtype],
        dim=(B, k_max),
        inputs=[
            g_c,
            quadrupoles,
            positions,
            cosines,
            sines,
            k_vectors,
            grad_rho,
            volume,
            atom_start,
            atom_end,
            ggrad_rho,
            ggrad_quad,
            ggrad_positions,
            ggrad_kvec,
        ],
        device=device,
    )


def batch_rho_q_kvec_grad_double_backward(
    g_k: wp.array,
    quadrupoles: wp.array,
    positions: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    source_coeff2: wp.array,
    grad_rho: wp.array,
    volume: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    ggrad_rho: wp.array,
    ggrad_quad: wp.array,
    ggrad_positions: wp.array,
    ggrad_coeff2: wp.array,
    ggrad_kvec: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launcher for :func:`_batch_rho_q_kvec_grad_double_backward_kernel`.

    ``ggrad_quad`` / ``ggrad_positions`` accumulate over k via atomics and must
    be pre-zeroed. ``scale = (2π)³/volume[b]`` is applied per-system inside the
    kernel.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    B = grad_rho.shape[0]
    k_max = grad_rho.shape[1]
    wp.launch(
        _batch_rho_q_kvec_grad_double_backward_overloads[vec_dtype],
        dim=(B, k_max),
        inputs=[
            g_k,
            quadrupoles,
            positions,
            cosines,
            sines,
            k_vectors,
            source_coeff2,
            grad_rho,
            volume,
            atom_start,
            atom_end,
            ggrad_rho,
            ggrad_quad,
            ggrad_positions,
            ggrad_coeff2,
            ggrad_kvec,
        ],
        device=device,
    )


def batch_rho_q_moment_grad(
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    grad_rho: wp.array,
    scale: float,
    batch_idx: wp.array,
    grad_quadrupoles: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_rho_q_moment_grad_kernel` (float64-only).

    Parameters
    ----------
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    coeff2 : wp.array
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : float
        Scalar prefactor applied to the contribution.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    grad_quadrupoles : wp.array
        OUTPUT: gradient w.r.t. the quadrupole moments.
    device : str
        Warp device string; defaults to the input array's device.
    """
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_rho_q_moment_grad_kernel,
        dim=cosines.shape[1],
        inputs=[
            cosines,
            sines,
            k_vectors,
            coeff2,
            grad_rho,
            scale,
            batch_idx,
            grad_quadrupoles,
        ],
        device=device,
    )


def batch_rhoq_posgrad_backward_grad_rho(
    quadrupoles: wp.array,
    gg_pos: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    scale: float,
    atom_start: wp.array,
    atom_end: wp.array,
    grad_rho_grad: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for batched K_a.

    Parameters
    ----------
    quadrupoles : wp.array
        Per-atom Cartesian quadrupole moments.
    gg_pos : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    coeff2 : wp.array
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    scale : float
        Scalar prefactor applied to the contribution.
    atom_start : wp.array
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array
        Per-system end offset into the flat atom arrays.
    grad_rho_grad : wp.array
        OUTPUT: gradient w.r.t. the incoming ``grad_rho``.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    batch_size = k_vectors.shape[0]
    k_max = k_vectors.shape[1]
    wp.launch(
        _batch_rhoq_posgrad_bw_grad_rho_overloads[vec_dtype],
        dim=(batch_size, k_max),
        inputs=[
            quadrupoles,
            gg_pos,
            cosines,
            sines,
            k_vectors,
            coeff2,
            scale,
            atom_start,
            atom_end,
            grad_rho_grad,
        ],
        device=device,
    )


def batch_rhoq_posgrad_backward_positions(
    quadrupoles: wp.array,
    gg_pos: wp.array,
    grad_rho: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    scale: float,
    batch_idx: wp.array,
    grad_positions: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for batched K_c.

    Parameters
    ----------
    quadrupoles : wp.array
        Per-atom Cartesian quadrupole moments.
    gg_pos : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    coeff2 : wp.array
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    scale : float
        Scalar prefactor applied to the contribution.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    grad_positions : wp.array
        OUTPUT: gradient w.r.t. atomic positions.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_rhoq_posgrad_bw_positions_overloads[vec_dtype],
        dim=cosines.shape[1],
        inputs=[
            quadrupoles,
            gg_pos,
            grad_rho,
            cosines,
            sines,
            k_vectors,
            coeff2,
            scale,
            batch_idx,
            grad_positions,
        ],
        device=device,
    )


def batch_rhoq_posgrad_backward_quad(
    gg_pos: wp.array,
    grad_rho: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    scale: float,
    batch_idx: wp.array,
    grad_quadrupoles: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for batched K_b (float64-only).

    Parameters
    ----------
    gg_pos : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    coeff2 : wp.array
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    scale : float
        Scalar prefactor applied to the contribution.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    grad_quadrupoles : wp.array
        OUTPUT: gradient w.r.t. the quadrupole moments.
    device : str
        Warp device string; defaults to the input array's device.
    """
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_rhoq_posgrad_backward_quad_kernel,
        dim=cosines.shape[1],
        inputs=[
            gg_pos,
            grad_rho,
            cosines,
            sines,
            k_vectors,
            coeff2,
            scale,
            batch_idx,
            grad_quadrupoles,
        ],
        device=device,
    )


def batch_v_grad_from_feat_grad_backward_positions_quadrupole(
    grad_raw: wp.array,
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    gg_v: wp.array,
    k_vectors: wp.array,
    batch_idx: wp.array,
    ggrad_positions: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_v_grad_from_feat_grad_backward_positions_quadrupole_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    gg_v : wp.array
        Second-order upstream gradient w.r.t. ``v`` (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    batch_idx : wp.array
        Per-atom system index into the batch (or scalar system id).
    ggrad_positions : wp.array
        OUTPUT: double-backward gradient w.r.t. positions.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_total = cosines.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_v_grad_from_feat_grad_backward_positions_quadrupole_kernel,
        dim=n_total,
        inputs=[
            grad_raw,
            receiver_phi_hat_l2,
            cosines,
            sines,
            k_factor_proj,
            gg_v,
            k_vectors,
            batch_idx,
            ggrad_positions,
        ],
        device=device,
    )


def batch_v_gradient_from_feature_grad_quadrupole(
    grad_raw: wp.array,
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    grad_v: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_batch_v_gradient_from_feature_grad_quadrupole_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    atom_start : wp.array
        Per-system start offset into the flat atom arrays.
    atom_end : wp.array
        Per-system end offset into the flat atom arrays.
    grad_v : wp.array
        OUTPUT: gradient w.r.t. the input feature vector ``v``.
    device : str
        Warp device string; defaults to the input array's device.
    """
    batch_size = receiver_phi_hat_l2.shape[0]
    k_max = receiver_phi_hat_l2.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _batch_v_gradient_from_feature_grad_quadrupole_kernel,
        dim=(batch_size, k_max),
        inputs=[
            grad_raw,
            receiver_phi_hat_l2,
            cosines,
            sines,
            k_factor_proj,
            atom_start,
            atom_end,
            grad_v,
        ],
        device=device,
    )


def eval_gto_fourier_q(
    k_norm2: wp.array,
    sigma: float,
    inv_cl_l0: float,
    coeff2: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_eval_gto_fourier_q_kernel`.

    Parameters
    ----------
    k_norm2 : wp.array, shape (N_k,), dtype wp.float64
        Pre-computed ``|k|^2``.
    sigma : float
        GTO density-basis width.
    inv_cl_l0 : float
        ``1/C_0(sigma, MULTIPOLES)`` (same constant the l=0 charge channel
        uses).
    coeff2 : wp.array, shape (N_k,), dtype wp.float64
        Pre-allocated output.
    device : str, optional
        Defaults to ``k_norm2.device``.
    """
    if device is None:
        device = str(k_norm2.device)
    wp.launch(
        _eval_gto_fourier_q_kernel,
        dim=k_norm2.shape[0],
        inputs=[
            k_norm2,
            wp.float64(sigma),
            wp.float64(inv_cl_l0),
            coeff2,
        ],
        device=device,
    )


def eval_receiver_gto_fourier_quadrupole(
    k_vectors: wp.array,
    k_norm2: wp.array,
    sigmas: wp.array,
    inv_cl_l2: wp.array,
    output: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_eval_receiver_gto_fourier_quadrupole_kernel`.

    Parameters
    ----------
    k_vectors : wp.array, shape (N_k,), dtype wp.vec3d
    k_norm2 : wp.array, shape (N_k,), dtype wp.float64
    sigmas : wp.array, shape (N_σ,), dtype wp.float64
    inv_cl_l2 : wp.array, shape (N_σ,), dtype wp.float64
        ``1 / C_2(σ, mode)`` per receiver σ.
    output : wp.array4d, shape (N_k, N_σ, 5, 2), dtype wp.float64
        Pre-allocated. The 5 l=2 columns only (m = -2..+2), purely real.
    device : str, optional
    """
    n_k = k_vectors.shape[0]
    n_sigma = sigmas.shape[0]
    if device is None:
        device = str(k_vectors.device)
    wp.launch(
        _eval_receiver_gto_fourier_quadrupole_kernel,
        dim=(n_k, n_sigma),
        inputs=[
            k_vectors,
            k_norm2,
            sigmas,
            inv_cl_l2,
            output,
        ],
        device=device,
    )


def feat_position_grad_backward_grad_raw_quadrupole(
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    ggrad_grad_raw: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_feat_position_grad_backward_grad_raw_quadrupole_kernel` (CPU + CUDA).

    Parameters
    ----------
    receiver_phi_hat_l2 : wp.array
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    ggrad_grad_raw : wp.array
        OUTPUT: double-backward gradient w.r.t. ``grad_raw``.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat_l2.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _feat_position_grad_backward_grad_raw_quadrupole_kernel,
        dim=(n_atoms, n_sigma, 5),
        inputs=[
            receiver_phi_hat_l2,
            cosines,
            sines,
            k_factor_proj,
            potential,
            gg_positions,
            k_vectors,
            ggrad_grad_raw,
        ],
        device=device,
    )


def feat_position_grad_backward_positions_quadrupole(
    grad_raw: wp.array,
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    ggrad_positions: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_feat_position_grad_backward_positions_quadrupole_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    ggrad_positions : wp.array
        OUTPUT: double-backward gradient w.r.t. positions.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_atoms = cosines.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _feat_position_grad_backward_positions_quadrupole_kernel,
        dim=n_atoms,
        inputs=[
            grad_raw,
            receiver_phi_hat_l2,
            cosines,
            sines,
            k_factor_proj,
            potential,
            gg_positions,
            k_vectors,
            ggrad_positions,
        ],
        device=device,
    )


def feat_position_grad_backward_v_quadrupole(
    grad_raw: wp.array,
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    gg_positions: wp.array,
    k_vectors: wp.array,
    ggrad_v: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_feat_position_grad_backward_v_quadrupole_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    gg_positions : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    ggrad_v : wp.array
        OUTPUT: double-backward gradient w.r.t. ``v``.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_k = receiver_phi_hat_l2.shape[0]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _feat_position_grad_backward_v_quadrupole_kernel,
        dim=n_k,
        inputs=[
            grad_raw,
            receiver_phi_hat_l2,
            cosines,
            sines,
            k_factor_proj,
            gg_positions,
            k_vectors,
            ggrad_v,
        ],
        device=device,
    )


def position_gradient_from_feature_grad_quadrupole(
    grad_raw: wp.array,
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    k_vectors: wp.array,
    grad_positions: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_position_gradient_from_feature_grad_quadrupole_kernel` (CPU + CUDA).

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    grad_positions : wp.array
        OUTPUT: gradient w.r.t. atomic positions.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_atoms = cosines.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _position_gradient_from_feature_grad_quadrupole_kernel,
        dim=n_atoms,
        inputs=[
            grad_raw,
            receiver_phi_hat_l2,
            cosines,
            sines,
            k_factor_proj,
            potential,
            k_vectors,
            grad_positions,
        ],
        device=device,
    )


def position_gradient_from_rhoq(
    quadrupoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    grad_rho: wp.array,
    scale: float,
    grad_positions: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_position_gradient_from_rhoq_kernel`.

    ``scale`` is the forward prefactor ``(2*pi)^3 / V``. ``grad_positions``
    (N_atoms, 3, float64) is written unconditionally (Q-channel contribution
    only).


    Parameters
    ----------
    quadrupoles : wp.array
        Per-atom Cartesian quadrupole moments.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    coeff2 : wp.array
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : float
        Scalar prefactor applied to the contribution.
    grad_positions : wp.array
        OUTPUT: gradient w.r.t. atomic positions.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _position_gradient_from_rhoq_overloads[vec_dtype],
        dim=cosines.shape[1],
        inputs=[
            quadrupoles,
            cosines,
            sines,
            k_vectors,
            coeff2,
            grad_rho,
            wp.float64(scale),
            grad_positions,
        ],
        device=device,
    )


def project_features_quadrupole(
    potential: wp.array,
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    features: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_project_features_quadrupole_kernel`.

    Parameters
    ----------
    potential : wp.array, shape (N_k, 2), dtype wp.float64
    receiver_phi_hat_l2 : wp.array, shape (N_k, N_σ, 5, 2), dtype wp.float64
        The l=2 sub-block of the cached 9-column ``receiver_phi_hat``
        (columns ``4:9``) — produced by
        :func:`eval_receiver_gto_fourier_quadrupole`.
    cosines, sines : wp.array, shape (N_k, N_atoms), dtype wp.float64
    k_factor_proj : wp.array, shape (N_k,), dtype wp.float64
    features : wp.array, shape (N_atoms, N_σ, 5), dtype wp.float64
        Pre-allocated raw-feature output (no self-subtract).
    device : str, optional

    Notes
    -----
    The serial per-``(i, σ, m)`` kernel is dispatched on **both** CPU and
    CUDA. As documented for :func:`v_gradient_from_feature_grad`, the
    ``N_σ * 5`` matmul N-dimension is too narrow for a tile-matmul rewrite to
    pay off; the serial kernel is the production path here.
    """
    n_k = potential.shape[0]
    n_sigma = receiver_phi_hat_l2.shape[1]
    n_atoms = cosines.shape[1]
    if device is None:
        device = str(potential.device)
    if cosines.shape[0] != n_k or sines.shape[0] != n_k:
        raise ValueError(
            f"cosines/sines must have N_k={n_k} rows, got "
            f"cosines.shape={tuple(cosines.shape)}, sines.shape={tuple(sines.shape)}"
        )
    if receiver_phi_hat_l2.shape[2] != 5:
        raise ValueError(
            f"receiver_phi_hat_l2 must have 5 l=2 columns, got "
            f"{tuple(receiver_phi_hat_l2.shape)}"
        )
    if features.shape != (n_atoms, n_sigma, 5):
        raise ValueError(
            f"features must have shape (N_atoms={n_atoms}, N_σ={n_sigma}"
            f", 5), got {tuple(features.shape)}"
        )
    wp.launch(
        _project_features_quadrupole_kernel,
        dim=(n_atoms, n_sigma, 5),
        inputs=[
            potential,
            receiver_phi_hat_l2,
            cosines,
            sines,
            k_factor_proj,
            features,
        ],
        device=device,
    )


def project_kphase_grad_dipole(
    grad_raw: wp.array,
    receiver_phi_hat: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    positions: wp.array,
    grad_k_vectors: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_project_kphase_grad_dipole_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat : wp.array
        Receiver-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    positions : wp.array
        Atomic Cartesian positions.
    grad_k_vectors : wp.array
        OUTPUT: gradient w.r.t. the k-vectors.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_k = cosines.shape[0]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _project_kphase_grad_dipole_kernel,
        dim=n_k,
        inputs=[
            grad_raw,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            potential,
            positions,
            grad_k_vectors,
        ],
        device=device,
    )


def project_phihat_grad_dipole(
    grad_raw: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    potential: wp.array,
    grad_phi: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_project_phihat_grad_dipole_kernel`.

    Channel-generic: the lm count comes from ``grad_raw.shape[2]`` (4 for the
    l≤1 block, 5 for the l=2 block), so the same kernel serves both.


    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    potential : wp.array
        Per-k reciprocal-space potential factor.
    grad_phi : wp.array
        OUTPUT: gradient w.r.t. the GTO Fourier coefficients :math:`\hat\phi(k)`.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_k = cosines.shape[0]
    n_sigma = grad_raw.shape[1]
    n_lm = grad_raw.shape[2]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _project_phihat_grad_dipole_kernel,
        dim=(n_k, n_sigma, n_lm),
        inputs=[
            grad_raw,
            cosines,
            sines,
            k_factor_proj,
            potential,
            grad_phi,
        ],
        device=device,
    )


def receiver_phi_hat_backward_quadrupole(
    grad_output: wp.array,
    k_vectors: wp.array,
    k_norm2: wp.array,
    sigmas: wp.array,
    inv_cl_l2: wp.array,
    grad_k_vectors: wp.array,
    grad_k_norm2: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_receiver_phi_hat_backward_quadrupole_kernel`.

    Parameters
    ----------
    grad_output : wp.array
        Upstream gradient flowing into this backward kernel.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    k_norm2 : wp.array
        Squared magnitudes :math:`|k|^2` of the k-vectors.
    sigmas : wp.array
        Per-channel Gaussian (GTO) width parameters.
    inv_cl_l2 : wp.array
        Inverse :math:`l=2` overlap normalization constant.
    grad_k_vectors : wp.array
        OUTPUT: gradient w.r.t. the k-vectors.
    grad_k_norm2 : wp.array
        OUTPUT: gradient w.r.t. :math:`|k|^2`.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_k = k_vectors.shape[0]
    if device is None:
        device = str(k_vectors.device)
    wp.launch(
        _receiver_phi_hat_backward_quadrupole_kernel,
        dim=n_k,
        inputs=[
            grad_output,
            k_vectors,
            k_norm2,
            sigmas,
            inv_cl_l2,
        ],
        outputs=[
            grad_k_vectors,
            grad_k_norm2,
        ],
        device=device,
    )


def rho_kphase_grad(
    charges: wp.array,
    dipoles: wp.array,
    positions: wp.array,
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    grad_rho: wp.array,
    scale: float,
    grad_k: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_rho_kphase_grad_kernel`.

    Parameters
    ----------
    charges : wp.array
        Per-atom monopole charges.
    dipoles : wp.array
        Per-atom Cartesian dipole moments.
    positions : wp.array
        Atomic Cartesian positions.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    source_phi_hat : wp.array
        Source-side GTO Fourier coefficients :math:`\hat\phi(k)`.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : float
        Scalar prefactor applied to the contribution.
    grad_k : wp.array
        OUTPUT: gradient w.r.t. the k-vectors.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rho_kphase_grad_overloads[vec_dtype],
        dim=cosines.shape[0],
        inputs=[
            charges,
            dipoles,
            positions,
            cosines,
            sines,
            source_phi_hat,
            grad_rho,
            wp.float64(scale),
            grad_k,
        ],
        device=device,
    )


def rho_kphase_grad_double_backward(
    g_k: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    positions: wp.array,
    cosines: wp.array,
    sines: wp.array,
    source_phi_hat: wp.array,
    k_vectors: wp.array,
    grad_rho: wp.array,
    scale: float,
    ggrad_rho: wp.array,
    ggrad_moments: wp.array,
    ggrad_positions: wp.array,
    ggrad_phi: wp.array,
    ggrad_kvec: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launcher for :func:`_rho_kphase_grad_double_backward_kernel`.

    ``ggrad_moments`` / ``ggrad_positions`` accumulate over k via atomics and
    must be pre-zeroed by the caller.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rho_kphase_grad_double_backward_overloads[vec_dtype],
        dim=cosines.shape[0],
        inputs=[
            g_k,
            charges,
            dipoles,
            positions,
            cosines,
            sines,
            source_phi_hat,
            k_vectors,
            grad_rho,
            wp.float64(scale),
            ggrad_rho,
            ggrad_moments,
            ggrad_positions,
            ggrad_phi,
            ggrad_kvec,
        ],
        device=device,
    )


def rho_phihat_grad(
    charges: wp.array,
    dipoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    grad_rho: wp.array,
    scale: float,
    grad_phi: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_rho_phihat_grad_kernel`.

    Parameters
    ----------
    charges : wp.array
        Per-atom monopole charges.
    dipoles : wp.array
        Per-atom Cartesian dipole moments.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : float
        Scalar prefactor applied to the contribution.
    grad_phi : wp.array
        OUTPUT: gradient w.r.t. the GTO Fourier coefficients :math:`\hat\phi(k)`.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rho_phihat_grad_overloads[vec_dtype],
        dim=cosines.shape[0],
        inputs=[
            charges,
            dipoles,
            cosines,
            sines,
            grad_rho,
            wp.float64(scale),
            grad_phi,
        ],
        device=device,
    )


def rho_phihat_grad_double_backward(
    g_phi: wp.array,
    charges: wp.array,
    dipoles: wp.array,
    positions: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    grad_rho: wp.array,
    scale: float,
    ggrad_rho: wp.array,
    ggrad_moments: wp.array,
    ggrad_positions: wp.array,
    ggrad_kvec: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launcher for :func:`_rho_phihat_grad_double_backward_kernel`.

    ``ggrad_moments`` / ``ggrad_positions`` accumulate over k via atomics and
    must be pre-zeroed by the caller.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rho_phihat_grad_double_backward_overloads[vec_dtype],
        dim=cosines.shape[0],
        inputs=[
            g_phi,
            charges,
            dipoles,
            positions,
            cosines,
            sines,
            k_vectors,
            grad_rho,
            wp.float64(scale),
            ggrad_rho,
            ggrad_moments,
            ggrad_positions,
            ggrad_kvec,
        ],
        device=device,
    )


def rho_q_coeff2_grad(
    quadrupoles: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    grad_rho: wp.array,
    scale: float,
    grad_coeff2: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_rho_q_coeff2_grad_kernel`.

    Parameters
    ----------
    quadrupoles : wp.array
        Per-atom Cartesian quadrupole moments.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : float
        Scalar prefactor applied to the contribution.
    grad_coeff2 : wp.array
        OUTPUT: gradient w.r.t. the :math:`l=2` coefficient ``coeff2``.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rho_q_coeff2_grad_overloads[vec_dtype],
        dim=cosines.shape[0],
        inputs=[
            quadrupoles,
            cosines,
            sines,
            k_vectors,
            grad_rho,
            wp.float64(scale),
            grad_coeff2,
        ],
        device=device,
    )


def rho_q_coeff2_grad_double_backward(
    g_c: wp.array,
    quadrupoles: wp.array,
    positions: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    grad_rho: wp.array,
    scale: float,
    ggrad_rho: wp.array,
    ggrad_quad: wp.array,
    ggrad_positions: wp.array,
    ggrad_kvec: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launcher for :func:`_rho_q_coeff2_grad_double_backward_kernel`.

    ``ggrad_quad`` / ``ggrad_positions`` accumulate over k via atomics and must
    be pre-zeroed by the caller.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rho_q_coeff2_grad_double_backward_overloads[vec_dtype],
        dim=cosines.shape[0],
        inputs=[
            g_c,
            quadrupoles,
            positions,
            cosines,
            sines,
            k_vectors,
            grad_rho,
            wp.float64(scale),
            ggrad_rho,
            ggrad_quad,
            ggrad_positions,
            ggrad_kvec,
        ],
        device=device,
    )


def rho_q_kvec_grad(
    quadrupoles: wp.array,
    positions: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    grad_rho: wp.array,
    scale: float,
    grad_k: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_rho_q_kvec_grad_kernel`.

    Parameters
    ----------
    quadrupoles : wp.array
        Per-atom Cartesian quadrupole moments.
    positions : wp.array
        Atomic Cartesian positions.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    coeff2 : wp.array
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : float
        Scalar prefactor applied to the contribution.
    grad_k : wp.array
        OUTPUT: gradient w.r.t. the k-vectors.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rho_q_kvec_grad_overloads[vec_dtype],
        dim=cosines.shape[0],
        inputs=[
            quadrupoles,
            positions,
            cosines,
            sines,
            k_vectors,
            coeff2,
            grad_rho,
            wp.float64(scale),
            grad_k,
        ],
        device=device,
    )


def rho_q_kvec_grad_double_backward(
    g_k: wp.array,
    quadrupoles: wp.array,
    positions: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    source_coeff2: wp.array,
    grad_rho: wp.array,
    scale: float,
    ggrad_rho: wp.array,
    ggrad_quad: wp.array,
    ggrad_positions: wp.array,
    ggrad_coeff2: wp.array,
    ggrad_kvec: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launcher for :func:`_rho_q_kvec_grad_double_backward_kernel`.

    ``ggrad_quad`` / ``ggrad_positions`` accumulate over k via atomics and must
    be pre-zeroed by the caller.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rho_q_kvec_grad_double_backward_overloads[vec_dtype],
        dim=cosines.shape[0],
        inputs=[
            g_k,
            quadrupoles,
            positions,
            cosines,
            sines,
            k_vectors,
            source_coeff2,
            grad_rho,
            wp.float64(scale),
            ggrad_rho,
            ggrad_quad,
            ggrad_positions,
            ggrad_coeff2,
            ggrad_kvec,
        ],
        device=device,
    )


def rho_q_moment_grad(
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    grad_rho: wp.array,
    scale: float,
    grad_quadrupoles: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_rho_q_moment_grad_kernel`.

    ``grad_quadrupoles`` is ``(N_atoms, 9)`` float64 (row-major flattened 3x3,
    symmetric); the torch layer reshapes to ``(N, 3, 3)``. No dtype overload —
    the kernel is float64-only (k-space accumulation), independent of the
    input Q dtype, which only matters at assembly time.


    Parameters
    ----------
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    coeff2 : wp.array
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    scale : float
        Scalar prefactor applied to the contribution.
    grad_quadrupoles : wp.array
        OUTPUT: gradient w.r.t. the quadrupole moments.
    device : str
        Warp device string; defaults to the input array's device.
    """
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rho_q_moment_grad_kernel,
        dim=cosines.shape[1],
        inputs=[
            cosines,
            sines,
            k_vectors,
            coeff2,
            grad_rho,
            wp.float64(scale),
            grad_quadrupoles,
        ],
        device=device,
    )


def rhoq_posgrad_backward_grad_rho(
    quadrupoles: wp.array,
    gg_pos: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    scale: float,
    grad_rho_grad: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for K_a (``∂L/∂grad_rho`` of the Q-channel position grad).

    Parameters
    ----------
    quadrupoles : wp.array
        Per-atom Cartesian quadrupole moments.
    gg_pos : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    coeff2 : wp.array
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    scale : float
        Scalar prefactor applied to the contribution.
    grad_rho_grad : wp.array
        OUTPUT: gradient w.r.t. the incoming ``grad_rho``.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rhoq_posgrad_bw_grad_rho_overloads[vec_dtype],
        dim=cosines.shape[0],
        inputs=[
            quadrupoles,
            gg_pos,
            cosines,
            sines,
            k_vectors,
            coeff2,
            wp.float64(scale),
            grad_rho_grad,
        ],
        device=device,
    )


def rhoq_posgrad_backward_positions(
    quadrupoles: wp.array,
    gg_pos: wp.array,
    grad_rho: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    scale: float,
    grad_positions: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    r"""Launcher for K_c (position-Hessian diagonal of the Q-channel pos grad).

    Parameters
    ----------
    quadrupoles : wp.array
        Per-atom Cartesian quadrupole moments.
    gg_pos : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    coeff2 : wp.array
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    scale : float
        Scalar prefactor applied to the contribution.
    grad_positions : wp.array
        OUTPUT: gradient w.r.t. atomic positions.
    wp_dtype : type
        Warp floating dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string; defaults to the input array's device.
    """
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rhoq_posgrad_bw_positions_overloads[vec_dtype],
        dim=cosines.shape[1],
        inputs=[
            quadrupoles,
            gg_pos,
            grad_rho,
            cosines,
            sines,
            k_vectors,
            coeff2,
            wp.float64(scale),
            grad_positions,
        ],
        device=device,
    )


def rhoq_posgrad_backward_quad(
    gg_pos: wp.array,
    grad_rho: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_vectors: wp.array,
    coeff2: wp.array,
    scale: float,
    grad_quadrupoles: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for K_b (float64-only; no Q dependence).

    Parameters
    ----------
    gg_pos : wp.array
        Second-order upstream gradient w.r.t. positions (HVP seed).
    grad_rho : wp.array
        Upstream gradient w.r.t. the reciprocal density :math:`\hat\rho(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    coeff2 : wp.array
        Per-k :math:`l=2` quadrupole reciprocal coefficient.
    scale : float
        Scalar prefactor applied to the contribution.
    grad_quadrupoles : wp.array
        OUTPUT: gradient w.r.t. the quadrupole moments.
    device : str
        Warp device string; defaults to the input array's device.
    """
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _rhoq_posgrad_backward_quad_kernel,
        dim=cosines.shape[1],
        inputs=[
            gg_pos,
            grad_rho,
            cosines,
            sines,
            k_vectors,
            coeff2,
            wp.float64(scale),
            grad_quadrupoles,
        ],
        device=device,
    )


def v_grad_from_feat_grad_backward_positions_quadrupole(
    grad_raw: wp.array,
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    gg_v: wp.array,
    k_vectors: wp.array,
    ggrad_positions: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_v_grad_from_feat_grad_backward_positions_quadrupole_kernel`.

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    gg_v : wp.array
        Second-order upstream gradient w.r.t. ``v`` (HVP seed).
    k_vectors : wp.array
        Reciprocal-space k-vectors.
    ggrad_positions : wp.array
        OUTPUT: double-backward gradient w.r.t. positions.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_atoms = cosines.shape[1]
    if device is None:
        device = str(cosines.device)
    wp.launch(
        _v_grad_from_feat_grad_backward_positions_quadrupole_kernel,
        dim=n_atoms,
        inputs=[
            grad_raw,
            receiver_phi_hat_l2,
            cosines,
            sines,
            k_factor_proj,
            gg_v,
            k_vectors,
            ggrad_positions,
        ],
        device=device,
    )


def v_gradient_from_feature_grad_quadrupole(
    grad_raw: wp.array,
    receiver_phi_hat_l2: wp.array,
    cosines: wp.array,
    sines: wp.array,
    k_factor_proj: wp.array,
    grad_v: wp.array,
    device: str | None = None,
) -> None:
    r"""Launcher for :func:`_v_gradient_from_feature_grad_quadrupole_kernel` (CPU + CUDA).

    Parameters
    ----------
    grad_raw : wp.array
        Upstream gradient w.r.t. the raw (pre-projection) features.
    receiver_phi_hat_l2 : wp.array
        Receiver-side :math:`l=2` GTO Fourier coefficients :math:`\hat\phi(k)`.
    cosines : wp.array
        Per-(k, atom) cosine table :math:`\cos(k\cdot r)`.
    sines : wp.array
        Per-(k, atom) sine table :math:`\sin(k\cdot r)`.
    k_factor_proj : wp.array
        Per-k projection factor for the feature/direct-k-space projection.
    grad_v : wp.array
        OUTPUT: gradient w.r.t. the input feature vector ``v``.
    device : str
        Warp device string; defaults to the input array's device.
    """
    n_k = receiver_phi_hat_l2.shape[0]
    if device is None:
        device = str(receiver_phi_hat_l2.device)
    wp.launch(
        _v_gradient_from_feature_grad_quadrupole_kernel,
        dim=n_k,
        inputs=[grad_raw, receiver_phi_hat_l2, cosines, sines, k_factor_proj, grad_v],
        device=device,
    )


# ---- overload registrations for recovered families ----
_assemble_rho_q_overloads = register_overloads(
    _assemble_rho_q_kernel, _assemble_rho_q_sig
)
_position_gradient_from_rhoq_overloads = register_overloads(
    _position_gradient_from_rhoq_kernel, _position_gradient_rhoq_sig
)
_rhoq_posgrad_bw_grad_rho_overloads = register_overloads(
    _rhoq_posgrad_backward_grad_rho_kernel, _rhoq_posgrad_bw_grad_rho_sig
)
_rhoq_posgrad_bw_positions_overloads = register_overloads(
    _rhoq_posgrad_backward_positions_kernel, _rhoq_posgrad_bw_positions_sig
)
_batch_assemble_rho_q_overloads = register_overloads(
    _batch_assemble_rho_q_kernel, _batch_assemble_rho_q_sig
)
_batch_position_gradient_from_rhoq_overloads = register_overloads(
    _batch_position_gradient_from_rhoq_kernel, _batch_position_gradient_rhoq_sig
)
_batch_rhoq_posgrad_bw_grad_rho_overloads = register_overloads(
    _batch_rhoq_posgrad_backward_grad_rho_kernel, _batch_rhoq_posgrad_bw_grad_rho_sig
)
_batch_rhoq_posgrad_bw_positions_overloads = register_overloads(
    _batch_rhoq_posgrad_backward_positions_kernel, _batch_rhoq_posgrad_bw_positions_sig
)
_rho_phihat_grad_overloads = register_overloads(
    _rho_phihat_grad_kernel, _rho_phihat_grad_sig
)
_rho_phihat_grad_double_backward_overloads = register_overloads(
    _rho_phihat_grad_double_backward_kernel, _rho_phihat_grad_double_backward_sig
)
_rho_kphase_grad_overloads = register_overloads(
    _rho_kphase_grad_kernel, _rho_kphase_grad_sig
)
_rho_kphase_grad_double_backward_overloads = register_overloads(
    _rho_kphase_grad_double_backward_kernel, _rho_kphase_grad_double_backward_sig
)
_rho_q_coeff2_grad_overloads = register_overloads(
    _rho_q_coeff2_grad_kernel, _rho_q_coeff2_grad_sig
)
_rho_q_coeff2_grad_double_backward_overloads = register_overloads(
    _rho_q_coeff2_grad_double_backward_kernel, _rho_q_coeff2_grad_double_backward_sig
)
_rho_q_kvec_grad_overloads = register_overloads(
    _rho_q_kvec_grad_kernel, _rho_q_kvec_grad_sig
)
_rho_q_kvec_grad_double_backward_overloads = register_overloads(
    _rho_q_kvec_grad_double_backward_kernel, _rho_q_kvec_grad_double_backward_sig
)
_batch_rho_phihat_grad_overloads = register_overloads(
    _batch_rho_phihat_grad_kernel, _batch_rho_phihat_grad_sig
)
_batch_rho_phihat_grad_double_backward_overloads = register_overloads(
    _batch_rho_phihat_grad_double_backward_kernel,
    _batch_rho_phihat_grad_double_backward_sig,
)
_batch_rho_kphase_grad_double_backward_overloads = register_overloads(
    _batch_rho_kphase_grad_double_backward_kernel,
    _batch_rho_kphase_grad_double_backward_sig,
)
_batch_rho_q_coeff2_grad_double_backward_overloads = register_overloads(
    _batch_rho_q_coeff2_grad_double_backward_kernel,
    _batch_rho_q_coeff2_grad_double_backward_sig,
)
_batch_rho_q_kvec_grad_double_backward_overloads = register_overloads(
    _batch_rho_q_kvec_grad_double_backward_kernel,
    _batch_rho_q_kvec_grad_double_backward_sig,
)
_batch_rho_kphase_grad_overloads = register_overloads(
    _batch_rho_kphase_grad_kernel, _batch_rho_kphase_grad_sig
)
_batch_rho_q_coeff2_grad_overloads = register_overloads(
    _batch_rho_q_coeff2_grad_kernel, _batch_rho_q_coeff2_grad_sig
)
_batch_rho_q_kvec_grad_overloads = register_overloads(
    _batch_rho_q_kvec_grad_kernel, _batch_rho_q_kvec_grad_sig
)
