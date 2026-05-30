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
Dispersion PME (LJ-PME) reciprocal-space Warp kernels
=====================================================

Warp kernels for the reciprocal-space half of dispersion (:math:`r^{-6}`) PME,
mirroring ``electrostatics/pme_kernels.py`` but with the dispersion influence
function in place of the Coulomb Green's function. The mesh spread/gather, FFT,
and B-spline deconvolution machinery are identical to electrostatics and reused
unchanged; **only the per-k convolution factor and the energy corrections
differ.**

Geometric combination rule gives a single mesh channel: the per-atom dispersion
charge is :math:`b_i = \sqrt{C_{6,i}}` and the structure factor is
:math:`S(k) = \sum_i b_i e^{i k \cdot r_i}`.

Convolution factor (replaces Coulomb's ``2π exp(-k²/4α²)/(V k²)``)

.. math::

    G_{\text{disp}}(k) = -\frac{\pi^{3/2} \alpha^3}{6 V}\, \frac{f(b)}{B^2(k)},
    \qquad b = \frac{k}{2\alpha},

.. math::

    f(b) = (1 - 2 b^2) e^{-b^2} + 2 \sqrt{\pi}\, b^3 \operatorname{erfc}(b) .

Two notable differences from Coulomb PME:

1. **No** :math:`1/k^2` pole — :math:`f` is finite at :math:`k=0`.
2. The :math:`k=0` term is **not** zeroed: for dispersion it is the (non-zero)
   background term and is produced automatically by the mesh DC component
   :math:`S(0) = \sum_i b_i`.

Per-atom energy correction (no background term; it is in the mesh):

.. math::

    E_i = b_i \varphi_i + \frac{\alpha^6}{12} b_i^2 ,

where the second term removes the spurious :math:`i=j` self interaction
(:math:`C_{6,i} = b_i^2`).

The energy total reconstructs

.. math::

    E_{\text{recip}} = -\frac{\pi^{3/2}\alpha^3}{6V}\sum_k f(b)\,|S(k)|^2
      + \frac{\alpha^6}{12}\sum_i C_{6,i} .

References
----------
- in 't Veld, Ismail & Grest, J. Chem. Phys. 127, 144711 (2007).
- Essmann et al., J. Chem. Phys. 103, 8577 (1995).
"""

from __future__ import annotations

import math
from typing import Any

import warp as wp

from nvalchemiops.math import wp_erfc

__all__ = [
    "lj_pme_convolve",
    "batch_lj_pme_convolve",
    "lj_pme_convolve_backward",
    "batch_lj_pme_convolve_backward",
    "lj_pme_energy_corrections",
    "batch_lj_pme_energy_corrections",
    "lj_pme_energy_corrections_backward",
    "batch_lj_pme_energy_corrections_backward",
]

PI = math.pi
# π^{3/2} / 6 : prefactor of the dispersion influence function.
DISP_PREFACTOR = math.pi**1.5 / 6.0
SQRT_PI = math.sqrt(math.pi)


# ==============================================================================
# Shared influence-function helpers
# ==============================================================================


@wp.func
def _disp_f(b2: Any) -> Any:
    """Dispersion reciprocal influence function ``f(b)`` as a function of ``b² ``.

    ``f(b) = (1 - 2 b²) e^{-b²} + 2 √π b³ erfc(b)``, ``b = √(b²)``.
    Finite and equal to 1 at ``b = 0``.
    """
    one = type(b2)(1.0)
    two = type(b2)(2.0)
    sqrtpi = type(b2)(SQRT_PI)
    b = wp.sqrt(b2)
    e = wp.exp(-b2)
    erfcb = wp_erfc(b)
    return (one - two * b2) * e + two * sqrtpi * b2 * b * erfcb


@wp.func
def _disp_g1(b2: Any) -> Any:
    """``df/d(b²) = 3 (√π b erfc(b) - e^{-b²})``, ``b = √(b²)``."""
    three = type(b2)(3.0)
    sqrtpi = type(b2)(SQRT_PI)
    b = wp.sqrt(b2)
    e = wp.exp(-b2)
    erfcb = wp_erfc(b)
    return three * (sqrtpi * b * erfcb - e)


# ==============================================================================
# Fused convolution (forward)
# ==============================================================================


@wp.kernel
def _lj_pme_convolve_kernel(
    mesh_fft: wp.array3d(dtype=Any),  # complex as vec2 (nx, ny, nz_r)
    k_squared: wp.array3d(dtype=Any),
    moduli_x: wp.array(dtype=Any),
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    convolved_mesh: wp.array3d(dtype=Any),
):
    """Single-system dispersion convolve: ``convolved = mesh_fft * G_disp(k)``.

    Unlike Coulomb, k=0 is kept (finite, gives the background term).
    """
    i, j, k = wp.tid()

    k_sq = k_squared[i, j, k]
    alpha_ = alpha[0]
    volume_ = volume[0]

    clamp_threshold = type(k_sq)(1e-10)
    four = type(k_sq)(4.0)
    prefac = type(k_sq)(DISP_PREFACTOR)

    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    b2 = k_sq / (four * alpha_ * alpha_)
    f = _disp_f(b2)
    # G_disp/B²(k) = -(π^{3/2} α³ / (6 V)) f(b) / sf²
    factor = -prefac * alpha_ * alpha_ * alpha_ * f / (volume_ * sf_sq)

    c = mesh_fft[i, j, k]
    convolved_mesh[i, j, k] = type(c)(c[0] * factor, c[1] * factor)


@wp.kernel
def _batch_lj_pme_convolve_kernel(
    mesh_fft: wp.array4d(dtype=Any),  # (B, nx, ny, nz_r), complex as vec2
    k_squared: wp.array4d(dtype=Any),
    moduli_x: wp.array(dtype=Any),
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),  # (B,)
    volumes: wp.array(dtype=Any),  # (B,)
    convolved_mesh: wp.array4d(dtype=Any),
):
    """Batched dispersion convolve. alpha/volume per-system; moduli shared."""
    batch_idx, i, j, k = wp.tid()

    k_sq = k_squared[batch_idx, i, j, k]
    alpha_ = alpha[batch_idx]
    volume_ = volumes[batch_idx]

    clamp_threshold = type(k_sq)(1e-10)
    four = type(k_sq)(4.0)
    prefac = type(k_sq)(DISP_PREFACTOR)

    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    b2 = k_sq / (four * alpha_ * alpha_)
    f = _disp_f(b2)
    factor = -prefac * alpha_ * alpha_ * alpha_ * f / (volume_ * sf_sq)

    c = mesh_fft[batch_idx, i, j, k]
    convolved_mesh[batch_idx, i, j, k] = type(c)(c[0] * factor, c[1] * factor)


# ==============================================================================
# Fused convolution (backward)
# ==============================================================================


@wp.kernel
def _lj_pme_convolve_backward_kernel(
    mesh_fft: wp.array3d(dtype=Any),  # saved forward input (vec2)
    grad_convolved: wp.array3d(dtype=Any),  # cotangent (vec2)
    k_squared: wp.array3d(dtype=Any),
    moduli_x: wp.array(dtype=Any),
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    grad_mesh_fft: wp.array3d(dtype=Any),  # output (vec2)
    grad_alpha: wp.array(dtype=Any),  # output (1,), atomic
    grad_volume: wp.array(dtype=Any),  # output (1,), atomic
    grad_k_squared: wp.array3d(dtype=Any),  # output
):
    """Single-system backward for the dispersion convolve.

    ``factor = -(π^{3/2} α³/(6 V)) f(b)/sf²``.  Analytic derivatives:
    ``dfactor/dα = (factor/(α·f))·(3f - 2b²·g1)``, ``dfactor/dV = -factor/V``,
    ``dfactor/d(k²) = (factor/f)·g1/(4α²)`` with ``g1 = df/db²``. To avoid a
    1/f division the ``factor/f`` ratio is recomputed as ``K`` directly.
    """
    i, j, k = wp.tid()

    k_sq = k_squared[i, j, k]
    alpha_ = alpha[0]
    volume_ = volume[0]

    clamp_threshold = type(k_sq)(1e-10)
    four = type(k_sq)(4.0)
    three = type(k_sq)(3.0)
    two = type(k_sq)(2.0)
    prefac = type(k_sq)(DISP_PREFACTOR)

    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    inv4a2 = type(k_sq)(1.0) / (four * alpha_ * alpha_)
    b2 = k_sq * inv4a2
    f = _disp_f(b2)
    g1 = _disp_g1(b2)
    # K = factor / f  (= -(π^{3/2} α³/(6 V)) / sf²); factor = K * f.
    big_k = -prefac * alpha_ * alpha_ * alpha_ / (volume_ * sf_sq)
    factor = big_k * f

    g = grad_convolved[i, j, k]
    m = mesh_fft[i, j, k]
    grad_mesh_fft[i, j, k] = type(g)(g[0] * factor, g[1] * factor)

    # No Wirtinger 2x (see electrostatics note): rfftn autograd already folds
    # the conjugate-pair contribution into grad_convolved.
    re_inner = g[0] * m[0] + g[1] * m[1]

    # dfactor/dα = (K/α)(3f - 2 b² g1)
    d_alpha = re_inner * (big_k / alpha_) * (three * f - two * b2 * g1)
    wp.atomic_add(grad_alpha, 0, d_alpha)
    # dfactor/dV = -factor/V
    d_vol = -re_inner * factor / volume_
    wp.atomic_add(grad_volume, 0, d_vol)
    # dfactor/dk² = K g1 / (4 α²)
    grad_k_squared[i, j, k] = re_inner * big_k * g1 * inv4a2


@wp.kernel
def _batch_lj_pme_convolve_backward_kernel(
    mesh_fft: wp.array4d(dtype=Any),
    grad_convolved: wp.array4d(dtype=Any),
    k_squared: wp.array4d(dtype=Any),
    moduli_x: wp.array(dtype=Any),
    moduli_y: wp.array(dtype=Any),
    moduli_z: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),  # (B,)
    volumes: wp.array(dtype=Any),  # (B,)
    grad_mesh_fft: wp.array4d(dtype=Any),
    grad_alpha: wp.array(dtype=Any),  # (B,)
    grad_volume: wp.array(dtype=Any),  # (B,)
    grad_k_squared: wp.array4d(dtype=Any),
):
    """Batched backward for the dispersion convolve."""
    batch_idx, i, j, k = wp.tid()

    k_sq = k_squared[batch_idx, i, j, k]
    alpha_ = alpha[batch_idx]
    volume_ = volumes[batch_idx]

    clamp_threshold = type(k_sq)(1e-10)
    four = type(k_sq)(4.0)
    three = type(k_sq)(3.0)
    two = type(k_sq)(2.0)
    prefac = type(k_sq)(DISP_PREFACTOR)

    sf = moduli_x[i] * moduli_y[j] * moduli_z[k]
    if sf < clamp_threshold:
        sf = clamp_threshold
    sf_sq = sf * sf

    inv4a2 = type(k_sq)(1.0) / (four * alpha_ * alpha_)
    b2 = k_sq * inv4a2
    f = _disp_f(b2)
    g1 = _disp_g1(b2)
    big_k = -prefac * alpha_ * alpha_ * alpha_ / (volume_ * sf_sq)
    factor = big_k * f

    g = grad_convolved[batch_idx, i, j, k]
    m = mesh_fft[batch_idx, i, j, k]
    grad_mesh_fft[batch_idx, i, j, k] = type(g)(g[0] * factor, g[1] * factor)

    re_inner = g[0] * m[0] + g[1] * m[1]
    d_alpha = re_inner * (big_k / alpha_) * (three * f - two * b2 * g1)
    wp.atomic_add(grad_alpha, batch_idx, d_alpha)
    d_vol = -re_inner * factor / volume_
    wp.atomic_add(grad_volume, batch_idx, d_vol)
    grad_k_squared[batch_idx, i, j, k] = re_inner * big_k * g1 * inv4a2


# ==============================================================================
# Energy corrections
# ==============================================================================


@wp.kernel
def _lj_pme_energy_corrections_kernel(
    raw_energies: wp.array(dtype=Any),  # φ_i from mesh gather
    b: wp.array(dtype=Any),  # per-atom sqrt(C6)
    alpha: wp.array(dtype=Any),  # (1,)
    corrected_energies: wp.array(dtype=Any),
):
    """Single-system: ``E_i = b_i φ_i + (α^6/12) b_i²``. No background term."""
    atom_idx = wp.tid()

    bi = b[atom_idx]
    raw = raw_energies[atom_idx]
    a = alpha[0]

    twelve = type(bi)(12.0)
    a6 = a * a * a * a * a * a
    corrected_energies[atom_idx] = bi * raw + (a6 / twelve) * bi * bi


@wp.kernel
def _batch_lj_pme_energy_corrections_kernel(
    raw_energies: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),  # (B,)
    corrected_energies: wp.array(dtype=Any),
):
    """Batched: ``E_i = b_i φ_i + (α_s^6/12) b_i²``, s = batch_idx[i]."""
    atom_idx = wp.tid()

    s = batch_idx[atom_idx]
    bi = b[atom_idx]
    raw = raw_energies[atom_idx]
    a = alpha[s]

    twelve = type(bi)(12.0)
    a6 = a * a * a * a * a * a
    corrected_energies[atom_idx] = bi * raw + (a6 / twelve) * bi * bi


@wp.kernel
def _lj_pme_energy_corrections_backward_kernel(
    grad_E: wp.array(dtype=Any),  # (N,) cotangent
    raw_energies: wp.array(dtype=Any),  # (N,)
    b: wp.array(dtype=Any),  # (N,)
    alpha: wp.array(dtype=Any),  # (1,)
    grad_raw: wp.array(dtype=Any),  # (N,)
    grad_b: wp.array(dtype=Any),  # (N,)
    grad_alpha: wp.array(dtype=Any),  # (1,) atomic
):
    """Single-system backward of ``E_i = b_i raw_i + (α^6/12) b_i²``.

    ``dE/draw = b``, ``dE/db = raw + (α^6/6) b``, ``dE/dα = (α^5/2) b²``.
    """
    i = wp.tid()
    g = grad_E[i]
    bi = b[i]
    raw = raw_energies[i]
    a = alpha[0]

    six = type(bi)(6.0)
    two = type(bi)(2.0)
    a5 = a * a * a * a * a
    a6 = a5 * a

    grad_raw[i] = g * bi
    grad_b[i] = g * (raw + (a6 / six) * bi)
    wp.atomic_add(grad_alpha, 0, g * (a5 / two) * bi * bi)


@wp.kernel
def _batch_lj_pme_energy_corrections_backward_kernel(
    grad_E: wp.array(dtype=Any),
    raw_energies: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),  # (B,)
    grad_raw: wp.array(dtype=Any),
    grad_b: wp.array(dtype=Any),
    grad_alpha: wp.array(dtype=Any),  # (B,) atomic
):
    """Batched backward of the per-atom dispersion energy correction."""
    i = wp.tid()
    s = batch_idx[i]
    g = grad_E[i]
    bi = b[i]
    raw = raw_energies[i]
    a = alpha[s]

    six = type(bi)(6.0)
    two = type(bi)(2.0)
    a5 = a * a * a * a * a
    a6 = a5 * a

    grad_raw[i] = g * bi
    grad_b[i] = g * (raw + (a6 / six) * bi)
    wp.atomic_add(grad_alpha, s, g * (a5 / two) * bi * bi)


# ==============================================================================
# Dtype overloads
# ==============================================================================

_T = [wp.float32, wp.float64]
_C = {wp.float32: wp.vec2f, wp.float64: wp.vec2d}

_lj_pme_convolve_kernel_overload = {}
_batch_lj_pme_convolve_kernel_overload = {}
_lj_pme_convolve_backward_kernel_overload = {}
_batch_lj_pme_convolve_backward_kernel_overload = {}
_lj_pme_energy_corrections_kernel_overload = {}
_batch_lj_pme_energy_corrections_kernel_overload = {}
_lj_pme_energy_corrections_backward_kernel_overload = {}
_batch_lj_pme_energy_corrections_backward_kernel_overload = {}

for t in _T:
    _lj_pme_convolve_kernel_overload[t] = wp.overload(
        _lj_pme_convolve_kernel,
        [
            wp.array3d(dtype=_C[t]),  # mesh_fft
            wp.array3d(dtype=t),  # k_squared
            wp.array(dtype=t),  # moduli_x
            wp.array(dtype=t),  # moduli_y
            wp.array(dtype=t),  # moduli_z
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # volume
            wp.array3d(dtype=_C[t]),  # convolved_mesh
        ],
    )
    _batch_lj_pme_convolve_kernel_overload[t] = wp.overload(
        _batch_lj_pme_convolve_kernel,
        [
            wp.array4d(dtype=_C[t]),
            wp.array4d(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array4d(dtype=_C[t]),
        ],
    )
    _lj_pme_convolve_backward_kernel_overload[t] = wp.overload(
        _lj_pme_convolve_backward_kernel,
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
            wp.array(dtype=t),  # grad_alpha
            wp.array(dtype=t),  # grad_volume
            wp.array3d(dtype=t),  # grad_k_squared
        ],
    )
    _batch_lj_pme_convolve_backward_kernel_overload[t] = wp.overload(
        _batch_lj_pme_convolve_backward_kernel,
        [
            wp.array4d(dtype=_C[t]),
            wp.array4d(dtype=_C[t]),
            wp.array4d(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array4d(dtype=_C[t]),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array4d(dtype=t),
        ],
    )
    _lj_pme_energy_corrections_kernel_overload[t] = wp.overload(
        _lj_pme_energy_corrections_kernel,
        [
            wp.array(dtype=t),  # raw_energies
            wp.array(dtype=t),  # b
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # corrected_energies
        ],
    )
    _batch_lj_pme_energy_corrections_kernel_overload[t] = wp.overload(
        _batch_lj_pme_energy_corrections_kernel,
        [
            wp.array(dtype=t),  # raw_energies
            wp.array(dtype=t),  # b
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # corrected_energies
        ],
    )
    _lj_pme_energy_corrections_backward_kernel_overload[t] = wp.overload(
        _lj_pme_energy_corrections_backward_kernel,
        [
            wp.array(dtype=t),  # grad_E
            wp.array(dtype=t),  # raw_energies
            wp.array(dtype=t),  # b
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # grad_raw
            wp.array(dtype=t),  # grad_b
            wp.array(dtype=t),  # grad_alpha
        ],
    )
    _batch_lj_pme_energy_corrections_backward_kernel_overload[t] = wp.overload(
        _batch_lj_pme_energy_corrections_backward_kernel,
        [
            wp.array(dtype=t),  # grad_E
            wp.array(dtype=t),  # raw_energies
            wp.array(dtype=t),  # b
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # grad_raw
            wp.array(dtype=t),  # grad_b
            wp.array(dtype=t),  # grad_alpha
        ],
    )


# ==============================================================================
# Launchers
# ==============================================================================


def lj_pme_convolve(
    mesh_fft,
    k_squared,
    moduli_x,
    moduli_y,
    moduli_z,
    alpha,
    volume,
    convolved_mesh,
    wp_dtype,
    device=None,
):
    """Single-system dispersion convolve launcher (see kernel docstring)."""
    nx, ny, nz_r = mesh_fft.shape[0], mesh_fft.shape[1], mesh_fft.shape[2]
    wp.launch(
        _lj_pme_convolve_kernel_overload[wp_dtype],
        dim=(nx, ny, nz_r),
        inputs=[mesh_fft, k_squared, moduli_x, moduli_y, moduli_z, alpha, volume],
        outputs=[convolved_mesh],
        device=device,
    )


def batch_lj_pme_convolve(
    mesh_fft,
    k_squared,
    moduli_x,
    moduli_y,
    moduli_z,
    alpha,
    volumes,
    convolved_mesh,
    wp_dtype,
    device=None,
):
    """Batched dispersion convolve launcher."""
    nb = mesh_fft.shape[0]
    nx, ny, nz_r = mesh_fft.shape[1], mesh_fft.shape[2], mesh_fft.shape[3]
    wp.launch(
        _batch_lj_pme_convolve_kernel_overload[wp_dtype],
        dim=(nb, nx, ny, nz_r),
        inputs=[mesh_fft, k_squared, moduli_x, moduli_y, moduli_z, alpha, volumes],
        outputs=[convolved_mesh],
        device=device,
    )


def lj_pme_convolve_backward(
    mesh_fft,
    grad_convolved,
    k_squared,
    moduli_x,
    moduli_y,
    moduli_z,
    alpha,
    volume,
    grad_mesh_fft,
    grad_alpha,
    grad_volume,
    grad_k_squared,
    wp_dtype,
    device=None,
):
    """Single-system dispersion convolve backward launcher.

    ``grad_alpha`` and ``grad_volume`` must be zero-initialized 1-element arrays.
    """
    nx, ny, nz_r = mesh_fft.shape[0], mesh_fft.shape[1], mesh_fft.shape[2]
    wp.launch(
        _lj_pme_convolve_backward_kernel_overload[wp_dtype],
        dim=(nx, ny, nz_r),
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


def batch_lj_pme_convolve_backward(
    mesh_fft,
    grad_convolved,
    k_squared,
    moduli_x,
    moduli_y,
    moduli_z,
    alpha,
    volumes,
    grad_mesh_fft,
    grad_alpha,
    grad_volumes,
    grad_k_squared,
    wp_dtype,
    device=None,
):
    """Batched dispersion convolve backward launcher."""
    nb = mesh_fft.shape[0]
    nx, ny, nz_r = mesh_fft.shape[1], mesh_fft.shape[2], mesh_fft.shape[3]
    wp.launch(
        _batch_lj_pme_convolve_backward_kernel_overload[wp_dtype],
        dim=(nb, nx, ny, nz_r),
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


def lj_pme_energy_corrections(
    raw_energies,
    b,
    alpha,
    corrected_energies,
    wp_dtype,
    device=None,
):
    """Single-system dispersion energy-corrections launcher."""
    wp.launch(
        _lj_pme_energy_corrections_kernel_overload[wp_dtype],
        dim=raw_energies.shape[0],
        inputs=[raw_energies, b, alpha],
        outputs=[corrected_energies],
        device=device,
    )


def batch_lj_pme_energy_corrections(
    raw_energies,
    b,
    batch_idx,
    alpha,
    corrected_energies,
    wp_dtype,
    device=None,
):
    """Batched dispersion energy-corrections launcher."""
    wp.launch(
        _batch_lj_pme_energy_corrections_kernel_overload[wp_dtype],
        dim=raw_energies.shape[0],
        inputs=[raw_energies, b, batch_idx, alpha],
        outputs=[corrected_energies],
        device=device,
    )


def lj_pme_energy_corrections_backward(
    grad_E,
    raw_energies,
    b,
    alpha,
    grad_raw,
    grad_b,
    grad_alpha,
    wp_dtype,
    device=None,
):
    """Single-system dispersion energy-corrections backward launcher.

    ``grad_alpha`` must be a zero-initialized 1-element array.
    """
    wp.launch(
        _lj_pme_energy_corrections_backward_kernel_overload[wp_dtype],
        dim=raw_energies.shape[0],
        inputs=[grad_E, raw_energies, b, alpha],
        outputs=[grad_raw, grad_b, grad_alpha],
        device=device,
    )


def batch_lj_pme_energy_corrections_backward(
    grad_E,
    raw_energies,
    b,
    batch_idx,
    alpha,
    grad_raw,
    grad_b,
    grad_alpha,
    wp_dtype,
    device=None,
):
    """Batched dispersion energy-corrections backward launcher."""
    wp.launch(
        _batch_lj_pme_energy_corrections_backward_kernel_overload[wp_dtype],
        dim=raw_energies.shape[0],
        inputs=[grad_E, raw_energies, b, batch_idx, alpha],
        outputs=[grad_raw, grad_b, grad_alpha],
        device=device,
    )
