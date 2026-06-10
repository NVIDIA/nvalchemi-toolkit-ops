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
Autograd.Function wrappers for the multipole direct-k-space pipeline.

Single-purpose home for the ``torch.autograd.Function``-based forwards +
backwards. Decoupled from :mod:`multipole_scf_step` so the autograd
machinery (analytical backward kernels, Jacobian-via-kernel-reuse
pattern) stays out of the step wrappers.

Design notes
------------

* **Why not @warp_custom_op?** The decorator assumes a Warp tape-of-tape
  for backward. We use analytical backward kernels, both for speed
  (3-10x faster than tape replay at this pipeline's kernel complexity)
  and because tape-of-tape does not support double-backward reliably
  across the kernel shapes used here. ``torch.autograd.Function`` gives
  per-input backward control and composition for double-backward.

* **Jacobian-via-kernel-reuse.** The backward of the :math:`\rho(k)`
  assembly w.r.t. (charges, dipoles) is structurally identical to the
  feature-projection kernel with inputs rewired:

  .. math::

      \frac{\partial L}{\partial q_i} &=
          \frac{(2\pi)^3}{V} \sum_k
          \mathrm{Re}\!\left[\frac{\partial L}{\partial \rho(k)}^{*}
          \hat{\phi}_\mathrm{source}(k)\, e^{-i k\cdot r_i}\right] \\
      \frac{\partial L}{\partial \mu_i} &=
          \frac{(2\pi)^3}{V} \sum_k
          \mathrm{Re}\!\left[\frac{\partial L}{\partial \rho(k)}^{*}
          (\text{vector }\hat{\phi})\, e^{-i k\cdot r_i}\right]

  This is exactly what ``project_features_dipole`` computes, with
  :math:`V(k)` replaced by :math:`\partial L/\partial \rho(k)` and the
  receiver basis replaced by the source basis.

* **Dipole-axis permutation.** The :math:`\rho` assembly kernel embeds the
  Cartesian → e3nn :math:`(y, z, x)` rotation: ``dipoles[i, 0]`` maps to
  ``lm = 3`` (m=+1), ``dipoles[i, 1]`` → ``lm = 1`` (m=-1), ``dipoles[i, 2]``
  → ``lm = 2`` (m=0). The backward inverts this permutation when assembling
  the Cartesian dipole gradient.
"""

from __future__ import annotations

import math
import types

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics.multipole_direct_kspace_kernels import (
    assemble_rho_k_dipole,
    assemble_rho_q,
    build_structure_factor_table,
    feat_position_grad_backward_grad_raw,
    feat_position_grad_backward_grad_raw_quadrupole,
    feat_position_grad_backward_positions,
    feat_position_grad_backward_positions_quadrupole,
    feat_position_grad_backward_v,
    feat_position_grad_backward_v_quadrupole,
    position_gradient_from_feature_grad,
    position_gradient_from_feature_grad_quadrupole,
    position_gradient_from_rhok,
    position_gradient_from_rhoq,
    project_features_dipole,
    project_features_quadrupole,
    project_kphase_grad_dipole,
    project_phihat_grad_dipole,
    rho_kphase_grad,
    rho_kphase_grad_double_backward,
    rho_phihat_grad,
    rho_phihat_grad_double_backward,
    rho_q_coeff2_grad,
    rho_q_coeff2_grad_double_backward,
    rho_q_kvec_grad,
    rho_q_kvec_grad_double_backward,
    rho_q_moment_grad,
    rhok_position_grad_backward_grad_rho,
    rhok_position_grad_backward_moments,
    rhok_position_grad_backward_positions,
    rhoq_posgrad_backward_grad_rho,
    rhoq_posgrad_backward_positions,
    rhoq_posgrad_backward_quad,
    v_grad_from_feat_grad_backward_positions,
    v_grad_from_feat_grad_backward_positions_quadrupole,
    v_gradient_from_feature_grad,
    v_gradient_from_feature_grad_quadrupole,
)
from nvalchemiops.torch._warp_op_helpers import (
    register_warp_op_chain,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
    MultipoleSCFCache,
)

_TWO_PI_CUBED = (2.0 * math.pi) ** 3
_TWO_PI_SIXTH = (2.0 * math.pi) ** 6


def _structure_factor_table_launch(
    positions: torch.Tensor,
    k_vectors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Cache-free :math:`(\cos(k\cdot r_i), \sin(k\cdot r_i))` table.

    Both ``positions`` and ``k_vectors`` are detached for the Warp launch
    (their gradients flow via the analytical backward kernels, not through
    this cos/sin table). ``n_k`` / device are derived from the inputs, so
    this is the launcher used inside the ``multipole_rho`` custom op.

    Returns
    -------
    tuple of torch.Tensor
        ``(cos, sin)`` as ``(N_k, N_atoms)`` float64 tensors.
    """
    device = positions.device
    wp_device = wp.device_from_torch(device)
    n_k = k_vectors.shape[0]
    n_atoms = positions.shape[0]
    wp_scalar = wp.float64 if positions.dtype == torch.float64 else wp.float32
    vec_dtype = wp.vec3d if wp_scalar == wp.float64 else wp.vec3f

    cosines_t = torch.empty((n_k, n_atoms), dtype=torch.float64, device=device)
    sines_t = torch.empty((n_k, n_atoms), dtype=torch.float64, device=device)
    build_structure_factor_table(
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines_t, dtype=wp.float64),
        wp.from_torch(sines_t, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return cosines_t, sines_t


def _compute_structure_factor_table(
    positions: torch.Tensor,
    cache: MultipoleSCFCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Cache-bound wrapper of :func:`_structure_factor_table_launch`.

    Convenience for callers holding a :class:`MultipoleSCFCache` (e.g. tests
    cross-checking the projection); the production op path uses the cache-free
    launcher directly.

    Returns
    -------
    tuple of torch.Tensor
        ``(cos, sin)`` as ``(N_k, N_atoms)`` float64 tensors on
        ``cache.device``.
    """
    return _structure_factor_table_launch(positions, cache.k_vectors)


def _assemble_rho_launch(
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    volume_f: float,
) -> torch.Tensor:
    r"""Cache-free :math:`\rho(k)` assembly from a ``(cos, sin)`` table.

    All inputs are detached for the Warp launch. The launcher used inside the
    ``multipole_rho`` custom op.

    Returns
    -------
    torch.Tensor
        ``(N_k, 2)`` float64 tensor on ``charges.device``.
    """
    device = charges.device
    wp_device = wp.device_from_torch(device)
    n_k = cosines.shape[0]
    wp_scalar = wp.float64 if charges.dtype == torch.float64 else wp.float32
    vec_dtype = wp.vec3d if wp_scalar == wp.float64 else wp.vec3f

    rho = torch.zeros((n_k, 2), dtype=torch.float64, device=device)
    assemble_rho_k_dipole(
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        volume=volume_f,
        rho=wp.from_torch(rho, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return rho


# =============================================================================
# Opaque rho-backward sub-op chains
# =============================================================================


@torch.library.custom_op(
    "nvalchemiops::multipole_structure_factor",
    mutates_args=(),
    schema="(Tensor positions, Tensor k_vectors) -> (Tensor, Tensor)",
)
def _multipole_structure_factor_op(
    positions: torch.Tensor, k_vectors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque ``(cos(k.r), sin(k.r))`` table (forward-only; cos/sin are constants)."""
    return _structure_factor_table_launch(positions, k_vectors)


@torch.library.register_fake("nvalchemiops::multipole_structure_factor")
def _multipole_structure_factor_fake(
    positions: torch.Tensor, k_vectors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shape/dtype metadata: two ``(N_k, N_atoms)`` float64 tables."""
    n_k = k_vectors.shape[0]
    n_atoms = positions.shape[0]
    return (
        positions.new_empty((n_k, n_atoms), dtype=torch.float64),
        positions.new_empty((n_k, n_atoms), dtype=torch.float64),
    )


# ---- moments: grad_rho -> (grad_charges, grad_dipoles) via project rewired ----


def _rho_moment_grad_forward(
    grad_rho: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    r""":func:`project_features_dipole` rewired as the :math:`\partial\rho/\partial`-moments Jacobian transpose.

    Returns the per-atom moment gradient in e3nn layout ``(N_atoms, 4)``.
    """
    device = grad_rho.device
    wp_device = wp.device_from_torch(device)
    n_atoms = cosines.shape[1]
    n_k = cosines.shape[0]
    # Kernel's built-in 2/(2*pi)**3 times (2*pi)**6/(2 V) == (2*pi)**3 / V.
    per_k_factor_grad = (
        torch.full((n_k,), _TWO_PI_SIXTH / 2.0, dtype=torch.float64, device=device)
        / volume.detach()
    )
    source_phi_4d = source_phi_hat.detach().view(n_k, 1, 4, 2).contiguous()
    src_lm_zero = torch.zeros((n_atoms, 4), dtype=torch.float64, device=device)
    oc_zero = torch.zeros((1, 2), dtype=torch.float64, device=device)
    lut = torch.arange(4, dtype=torch.int32, device=device).view(1, 4).contiguous()
    grad_flat = torch.zeros((n_atoms, 4), dtype=torch.float64, device=device)
    project_features_dipole(
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_4d, dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(per_k_factor_grad, dtype=wp.float64),
        wp.from_torch(src_lm_zero, dtype=wp.float64),
        wp.from_torch(oc_zero, dtype=wp.float64),
        False,  # subtract_self
        wp.from_torch(lut, dtype=wp.int32),
        wp.from_torch(grad_flat, dtype=wp.float64),
        device=str(wp_device),
    )
    return grad_flat


def _rho_moment_grad_forward_fake(
    grad_rho: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: moment gradient ``(N_atoms, 4)`` float64."""
    return cosines.new_empty((cosines.shape[1], 4), dtype=torch.float64)


def _rho_moment_grad_backward(
    gg_moments: torch.Tensor,
    grad_rho: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    volume: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transpose backward of the linear moments map: ``(ggrad_rho, ggrad_volume)``."""
    device = grad_rho.device
    wp_device = wp.device_from_torch(device)
    gg_q = gg_moments[:, 0].contiguous()
    gg_mu = gg_moments[:, [3, 1, 2]].contiguous()
    vol = volume.detach()
    ggrad_rho = torch.empty_like(grad_rho)
    assemble_rho_k_dipole(
        wp.from_torch(gg_q, dtype=wp.float64),
        wp.from_torch(gg_mu, dtype=wp.vec3d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        1.0,
        wp.from_torch(ggrad_rho, dtype=wp.float64),
        wp_dtype=wp.float64,
        device=str(wp_device),
    )
    ggrad_rho = ggrad_rho * (1.0 / vol)
    # grad_flat scales as 1/V, so d(grad_flat)/dV = -grad_flat / V.
    grad_flat = _rho_moment_grad_forward(
        grad_rho, cosines, sines, source_phi_hat, volume
    )
    ggrad_volume = (-(gg_moments * grad_flat).sum() / vol).reshape(volume.shape)
    return ggrad_rho, ggrad_volume


register_warp_op_chain(
    name="nvalchemiops::multipole_rho_moment_grad",
    forward=_rho_moment_grad_forward,
    backward=_rho_moment_grad_backward,
    diff_input_positions=(0, 4),
    n_forward_inputs=5,
    forward_fake=_rho_moment_grad_forward_fake,
)


# ---- positions: grad_rho -> grad_positions via position_gradient_from_rhok ----


def _rho_position_grad_forward(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    r""":func:`position_gradient_from_rhok` — closed-form :math:`\partial\rho/\partial r`.

    ``positions`` is unused by the kernel (the dependence is carried by the
    detached cos/sin tables) but is an explicit input so its position-Hessian
    second-order grad has a slot in the backward.
    """
    device = charges.device
    wp_device = wp.device_from_torch(device)
    n_atoms = charges.shape[0]
    wp_scalar = wp.float64 if charges.dtype == torch.float64 else wp.float32
    vec_dtype = wp.vec3d if wp_scalar == wp.float64 else wp.vec3f
    grad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    position_gradient_from_rhok(
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        1.0,
        wp.from_torch(grad_positions, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_positions * scale.detach()


def _rho_position_grad_forward_fake(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: position gradient ``(N_atoms, 3)`` float64."""
    return positions.new_empty((positions.shape[0], 3), dtype=torch.float64)


def _rho_position_grad_backward(
    gg_positions: torch.Tensor,
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Second-order backward: ``(ggrad_rho, ggrad_charges, ggrad_dipoles, ggrad_positions)``."""
    device = charges.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = wp.float64 if charges.dtype == torch.float64 else wp.float32
    vec_dtype = wp.vec3d if wp_scalar == wp.float64 else wp.vec3f
    s = scale.detach()

    ggrad_grad_rho = torch.empty_like(grad_rho)
    rhok_position_grad_backward_grad_rho(
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        1.0,
        wp.from_torch(ggrad_grad_rho, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )

    ggrad_mom = torch.empty((charges.shape[0], 4), dtype=torch.float64, device=device)
    rhok_position_grad_backward_moments(
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        1.0,
        wp.from_torch(ggrad_mom, dtype=wp.float64),
        device=str(wp_device),
    )
    ggrad_mom = ggrad_mom * s
    ggrad_charges = ggrad_mom[:, 0].contiguous()
    # e3nn -> Cartesian permutation for dipole: (mu_x, mu_y, mu_z) = lm(3, 1, 2).
    ggrad_dipoles = ggrad_mom[:, [3, 1, 2]].contiguous()

    ggrad_positions = torch.zeros(
        (charges.shape[0], 3), dtype=torch.float64, device=device
    )
    rhok_position_grad_backward_positions(
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        1.0,
        wp.from_torch(ggrad_positions, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return ggrad_grad_rho * s, ggrad_charges, ggrad_dipoles, ggrad_positions * s


register_warp_op_chain(
    name="nvalchemiops::multipole_rho_position_grad",
    forward=_rho_position_grad_forward,
    backward=_rho_position_grad_backward,
    diff_input_positions=(0, 1, 2, 3),
    n_forward_inputs=9,
    forward_fake=_rho_position_grad_forward_fake,
)


# ---- phi_hat / k-vector phase (forward-only; carry the reciprocal cell-grad) ----


def _rho_phihat_grad_forward(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    r"""``∂L/∂source_phi_hat`` ``(N_k, 4, 2)`` via the Warp ``rho_phihat_grad`` kernel.

    ``positions`` / ``k_vectors`` are unused by the forward kernel (the
    cos/sin dependence is carried by the detached tables) but are explicit
    diff-input slots so the Warp second-order backward can place their
    Hessian grads (reciprocal stress-loss).
    """
    device = charges.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = wp.float64 if charges.dtype == torch.float64 else wp.float32
    vec_dtype = wp.vec3d if wp_scalar == wp.float64 else wp.vec3f
    grad_phi = torch.empty((cosines.shape[0], 4, 2), dtype=torch.float64, device=device)
    rho_phihat_grad(
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        1.0,
        wp.from_torch(grad_phi, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_phi * (_TWO_PI_CUBED / volume.detach())


def _rho_phihat_grad_forward_fake(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_k, 4, 2)`` float64."""
    return cosines.new_empty((cosines.shape[0], 4, 2), dtype=torch.float64)


def _rho_phihat_grad_backward(
    g_phi: torch.Tensor,
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp second-order backward: grads for ``(grad_rho, charges, dipoles, positions, k_vectors)``.

    The Warp ``rho_phihat_grad_double_backward`` kernel applies ``scale =
    (2π)³/V`` and folds the cos/sin position/k dependence analytically.
    """
    device = charges.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = wp.float64 if charges.dtype == torch.float64 else wp.float32
    vec_dtype = wp.vec3d if wp_scalar == wp.float64 else wp.vec3f
    n_atoms = charges.shape[0]
    scale = float(_TWO_PI_CUBED) / float(volume.detach())

    ggrad_rho = torch.empty_like(grad_rho, dtype=torch.float64)
    ggrad_moments = torch.zeros((n_atoms, 4), dtype=torch.float64, device=device)
    ggrad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    ggrad_kvec = torch.empty((cosines.shape[0], 3), dtype=torch.float64, device=device)
    rho_phihat_grad_double_backward(
        wp.from_torch(g_phi.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        scale,
        wp.from_torch(ggrad_rho, dtype=wp.float64),
        wp.from_torch(ggrad_moments, dtype=wp.float64),
        wp.from_torch(ggrad_positions, dtype=wp.vec3d),
        wp.from_torch(ggrad_kvec, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    ggrad_charges = ggrad_moments[:, 0].contiguous().to(charges.dtype)
    # e3nn -> Cartesian permutation for dipole: (mu_x, mu_y, mu_z) = lm(3, 1, 2).
    ggrad_dipoles = ggrad_moments[:, [3, 1, 2]].contiguous().to(dipoles.dtype)
    return ggrad_rho, ggrad_charges, ggrad_dipoles, ggrad_positions, ggrad_kvec


_PHIHAT_GRAD_SCHEMA = (
    "(Tensor grad_rho, Tensor charges, Tensor dipoles, Tensor positions, "
    "Tensor cosines, Tensor sines, Tensor k_vectors, Tensor volume) -> Tensor"
)
_PHIHAT_GRAD_BWD_SCHEMA = (
    "(Tensor g_phi, Tensor grad_rho, Tensor charges, Tensor dipoles, "
    "Tensor positions, Tensor cosines, Tensor sines, Tensor k_vectors, "
    "Tensor volume) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)
register_warp_op_chain(
    name="nvalchemiops::multipole_rho_phihat_grad",
    forward=_rho_phihat_grad_forward,
    forward_schema=_PHIHAT_GRAD_SCHEMA,
    forward_fake=_rho_phihat_grad_forward_fake,
    backward=_rho_phihat_grad_backward,
    backward_schema=_PHIHAT_GRAD_BWD_SCHEMA,
    backward_return_arity=5,
    # grad_rho(0), charges(1), dipoles(2), positions(3), k_vectors(6) are diff.
    diff_input_positions=(0, 1, 2, 3, 6),
    n_forward_inputs=8,
)


def _rho_kphase_grad_forward(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """``∂L/∂k_vectors`` ``(N_k, 3)`` (phase channel) via the Warp ``rho_kphase_grad`` kernel.

    ``k_vectors`` is an explicit diff slot (kernel-unused in forward; the cos/sin
    dependence is carried by the detached tables) so the Warp second-order
    backward can place its Hessian grad.
    """
    device = charges.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = wp.float64 if charges.dtype == torch.float64 else wp.float32
    vec_dtype = wp.vec3d if wp_scalar == wp.float64 else wp.vec3f
    grad_k = torch.empty((cosines.shape[0], 3), dtype=torch.float64, device=device)
    rho_kphase_grad(
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        1.0,
        wp.from_torch(grad_k, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_k * (_TWO_PI_CUBED / volume.detach())


def _rho_kphase_grad_forward_fake(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_k, 3)`` float64."""
    return cosines.new_empty((cosines.shape[0], 3), dtype=torch.float64)


def _rho_kphase_grad_backward(
    g_k: torch.Tensor,
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Warp second-order backward: grads for ``(grad_rho, charges, dipoles, positions, source_phi_hat, k_vectors)``."""
    device = charges.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = wp.float64 if charges.dtype == torch.float64 else wp.float32
    vec_dtype = wp.vec3d if wp_scalar == wp.float64 else wp.vec3f
    n_atoms = charges.shape[0]
    scale = float(_TWO_PI_CUBED) / float(volume.detach())

    ggrad_rho = torch.empty_like(grad_rho, dtype=torch.float64)
    ggrad_moments = torch.zeros((n_atoms, 4), dtype=torch.float64, device=device)
    ggrad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    ggrad_phi = torch.empty_like(source_phi_hat, dtype=torch.float64)
    ggrad_kvec = torch.empty((cosines.shape[0], 3), dtype=torch.float64, device=device)
    rho_kphase_grad_double_backward(
        wp.from_torch(g_k.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        scale,
        wp.from_torch(ggrad_rho, dtype=wp.float64),
        wp.from_torch(ggrad_moments, dtype=wp.float64),
        wp.from_torch(ggrad_positions, dtype=wp.vec3d),
        wp.from_torch(ggrad_phi, dtype=wp.float64),
        wp.from_torch(ggrad_kvec, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    ggrad_charges = ggrad_moments[:, 0].contiguous().to(charges.dtype)
    ggrad_dipoles = ggrad_moments[:, [3, 1, 2]].contiguous().to(dipoles.dtype)
    return (
        ggrad_rho,
        ggrad_charges,
        ggrad_dipoles,
        ggrad_positions,
        ggrad_phi,
        ggrad_kvec,
    )


_KPHASE_GRAD_SCHEMA = (
    "(Tensor grad_rho, Tensor charges, Tensor dipoles, Tensor positions, "
    "Tensor cosines, Tensor sines, Tensor source_phi_hat, Tensor k_vectors, "
    "Tensor volume) -> Tensor"
)
_KPHASE_GRAD_BWD_SCHEMA = (
    "(Tensor g_k, Tensor grad_rho, Tensor charges, Tensor dipoles, "
    "Tensor positions, Tensor cosines, Tensor sines, Tensor source_phi_hat, "
    "Tensor k_vectors, Tensor volume) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)
register_warp_op_chain(
    name="nvalchemiops::multipole_rho_kphase_grad",
    forward=_rho_kphase_grad_forward,
    forward_schema=_KPHASE_GRAD_SCHEMA,
    forward_fake=_rho_kphase_grad_forward_fake,
    backward=_rho_kphase_grad_backward,
    backward_schema=_KPHASE_GRAD_BWD_SCHEMA,
    backward_return_arity=6,
    # grad_rho(0), charges(1), dipoles(2), positions(3), source_phi_hat(6),
    # k_vectors(7) are the diff inputs.
    diff_input_positions=(0, 1, 2, 3, 6, 7),
    n_forward_inputs=9,
)


def _rho_backward_to_moments(
    grad_rho: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    cache: MultipoleSCFCache,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Closed-form backward of :math:`\rho` assembly w.r.t. (charges, dipoles).

    Delegates to the opaque ``multipole_rho_moment_grad`` op chain (the
    autograd-wrapped Jacobian transpose), so ``grad_charges`` / ``grad_dipoles``
    carry their own analytical backward — ``torch.autograd.grad(grad_charges,
    ...)`` works for force-/stress-loss training under ``create_graph=True`` and
    the call is compile-clean (opaque op, never AOT-traced into Warp).

    Returns
    -------
    grad_charges : torch.Tensor
        Shape ``(N_atoms,)``, float64.
    grad_dipoles : torch.Tensor
        Shape ``(N_atoms, 3)``, float64, in Cartesian ``(x, y, z)`` order.
    """
    grad_flat = torch.ops.nvalchemiops.multipole_rho_moment_grad(
        grad_rho, cosines, sines, cache.source_phi_hat, cache.volume
    )  # (N_atoms, 4) in e3nn layout

    grad_charges = grad_flat[:, 0].contiguous()
    grad_dipoles = grad_flat[:, [3, 1, 2]].contiguous()
    return grad_charges, grad_dipoles


def _rho_backward_to_positions(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    cache: MultipoleSCFCache,
    positions: torch.Tensor,
) -> torch.Tensor:
    r"""Closed-form backward of :math:`\rho` assembly w.r.t. atomic positions.

    Delegates to the opaque ``multipole_rho_position_grad`` op chain (the
    autograd-wrapped :func:`position_gradient_from_rhok`), so ``grad_positions``
    carries an analytical backward with the second-order kernels, making
    ``F = -grad_positions`` usable under ``create_graph=True`` and compile-clean.

    Returns
    -------
    grad_positions : torch.Tensor
        Shape ``(N_atoms, 3)``, float64, on ``cache.device``.
    """
    scale = (
        torch.as_tensor(_TWO_PI_CUBED, dtype=torch.float64, device=cache.device)
        / cache.volume
    )
    return torch.ops.nvalchemiops.multipole_rho_position_grad(
        grad_rho,
        charges,
        dipoles,
        positions,
        cosines,
        sines,
        cache.source_phi_hat,
        cache.k_vectors,
        scale,
    )


def _rho_backward_to_phihat(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    cache: MultipoleSCFCache,
) -> torch.Tensor:
    r"""``∂L/∂source_phi_hat`` via the ``multipole_rho_phihat_grad`` Warp op chain.

    The op is twice-differentiable (Warp ``rho_phihat_grad_double_backward``),
    so the cell↔{positions, moments, k} second-order cross-terms flow for
    reciprocal stress-loss — all on Warp. ``positions`` / ``k_vectors`` are
    explicit diff slots; ``cosines`` / ``sines`` are detached helper tables.

    Returns
    -------
    torch.Tensor
        ``(N_k, 4, 2)`` float64.
    """
    return torch.ops.nvalchemiops.multipole_rho_phihat_grad(
        grad_rho,
        charges,
        dipoles,
        positions,
        cosines,
        sines,
        cache.k_vectors,
        cache.volume,
    )


def _rho_backward_to_kvec(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    cache: MultipoleSCFCache,
) -> torch.Tensor:
    r"""``∂L/∂k_vectors`` through the phase via the ``multipole_rho_kphase_grad`` Warp op chain.

    Phase contribution only; :math:`\hat\phi`'s k-dependence flows separately
    through ``source_phi_hat``. The op is twice-differentiable (Warp
    ``rho_kphase_grad_double_backward``), so the cell↔{positions, moments, φ̂, k}
    second-order cross-terms route through Warp for reciprocal stress-loss.

    Returns
    -------
    torch.Tensor
        ``(N_k, 3)`` float64.
    """
    return torch.ops.nvalchemiops.multipole_rho_kphase_grad(
        grad_rho,
        charges,
        dipoles,
        positions,
        cosines,
        sines,
        cache.source_phi_hat,
        cache.k_vectors,
        cache.volume,
    )


class MultipoleRhoFunction:
    r"""Back-compat shim for the :math:`\rho(k)` direct-k-space assembly.

    The implementation is the fully-differentiable
    ``torch.ops.nvalchemiops.multipole_rho`` custom op (opaque forward +
    analytical backward + ``create_graph`` support, all compile-clean). This
    class only preserves the historical
    ``.apply(charges, dipoles, positions, source_phi_hat, k_vectors, cache)``
    call signature (the dataclass ``cache`` supplies ``volume``); new code
    should call the op directly.

    Parameters
    ----------
    charges : torch.Tensor
        Per-atom monopole charges, shape ``(N_atoms,)``.
    dipoles : torch.Tensor
        Per-atom Cartesian dipoles, shape ``(N_atoms, 3)`` in ``(x, y, z)``
        order. Pass an all-zero ``(N_atoms, 3)`` tensor for an l=0 system.
    positions : torch.Tensor
        Atomic coordinates, shape ``(N_atoms, 3)``.
    source_phi_hat : torch.Tensor
        Source-basis :math:`\hat\phi(k)`, shape ``(N_k, 4, 2)``; carries the
        reciprocal cell-autograd. Value equals ``cache.source_phi_hat``.
    k_vectors : torch.Tensor
        Reciprocal-lattice vectors, shape ``(N_k, 3)``. Value equals
        ``cache.k_vectors``.
    cache : MultipoleSCFCache
        Per-system direct-k-space state; supplies ``cache.volume``.

    Returns
    -------
    torch.Tensor
        Assembled :math:`\rho(k)`, shape ``(N_k, 2)`` float64.
    """

    @staticmethod
    def apply(
        charges: torch.Tensor,
        dipoles: torch.Tensor,
        positions: torch.Tensor,
        source_phi_hat: torch.Tensor,
        k_vectors: torch.Tensor,
        cache: MultipoleSCFCache,
    ) -> torch.Tensor:
        """Dispatch to ``torch.ops.nvalchemiops.multipole_rho`` (``cache`` -> volume)."""
        return torch.ops.nvalchemiops.multipole_rho(
            charges, dipoles, positions, source_phi_hat, k_vectors, cache.volume
        )


# =============================================================================
# torch.library.custom_op chain for the rho(k) assembly (l<=1)
# =============================================================================


@torch.library.custom_op(
    "nvalchemiops::multipole_rho",
    mutates_args=(),
    schema=(
        "(Tensor charges, Tensor dipoles, Tensor positions, "
        "Tensor source_phi_hat, Tensor k_vectors, Tensor volume) -> Tensor"
    ),
)
def _multipole_rho_op(
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Opaque forward: build the structure-factor table and assemble rho(k)."""
    cosines, sines = _structure_factor_table_launch(positions, k_vectors)
    rho = _assemble_rho_launch(charges, dipoles, cosines, sines, source_phi_hat, 1.0)
    return rho * (1.0 / volume.detach())


@torch.library.register_fake("nvalchemiops::multipole_rho")
def _multipole_rho_fake(
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: rho is ``(N_k, 2)`` float64."""
    return k_vectors.new_empty((k_vectors.shape[0], 2), dtype=torch.float64)


def _multipole_rho_setup_context(ctx, inputs, output) -> None:
    """Save the forward inputs for the analytical backward."""
    charges, dipoles, positions, source_phi_hat, k_vectors, volume = inputs
    ctx.save_for_backward(
        charges, dipoles, positions, source_phi_hat, k_vectors, volume
    )


def _multipole_rho_backward(ctx, grad_rho: torch.Tensor):
    """Analytical grads for charges, dipoles, positions, phi_hat, k_vectors.

    Recomputes the (cos, sin) table via the opaque ``multipole_structure_factor``
    op and dispatches the four backward sub-op chains through the cache-bound
    helpers (which read only ``source_phi_hat`` / ``k_vectors`` / ``volume`` —
    all available as saved tensors). Every call is an opaque op or plain torch,
    so AOTAutograd traces this backward without touching Warp; ``volume`` gets
    ``None`` (its cell-grad lives in the step's value-preserving ratio).
    """
    charges, dipoles, positions, source_phi_hat, k_vectors, volume = ctx.saved_tensors
    grad_rho = grad_rho.contiguous()
    # cos/sin are detached constants: the position second-order is carried in
    # full by the position-grad kernel, not by differentiating this table (so
    # the forward-only structure-factor op is never asked for a backward).
    cosines, sines = torch.ops.nvalchemiops.multipole_structure_factor(
        positions.detach(), k_vectors.detach()
    )
    # Lightweight stand-in exposing the cache fields the helpers read; internal
    # to the backward (never crosses an op boundary).
    shim = types.SimpleNamespace(
        source_phi_hat=source_phi_hat,
        k_vectors=k_vectors,
        volume=volume,
        device=positions.device,
        n_k=k_vectors.shape[0],
    )
    grad_charges, grad_dipoles = _rho_backward_to_moments(
        grad_rho, cosines, sines, shim, positions
    )
    grad_positions = _rho_backward_to_positions(
        grad_rho, charges, dipoles, cosines, sines, shim, positions
    )
    grad_phi = _rho_backward_to_phihat(
        grad_rho, charges, dipoles, positions, cosines, sines, shim
    )
    grad_kvec = _rho_backward_to_kvec(
        grad_rho, charges, dipoles, positions, cosines, sines, shim
    )
    # Match input dtypes (sub-ops emit float64).
    grad_charges = grad_charges.to(charges.dtype)
    grad_dipoles = grad_dipoles.to(dipoles.dtype)
    grad_positions = grad_positions.to(positions.dtype)
    # Slots: (charges, dipoles, positions, source_phi_hat, k_vectors, volume).
    return grad_charges, grad_dipoles, grad_positions, grad_phi, grad_kvec, None


torch.library.register_autograd(
    "nvalchemiops::multipole_rho",
    _multipole_rho_backward,
    setup_context=_multipole_rho_setup_context,
)


# =============================================================================
# Spread-transpose gather: g = Sᵀ(grad_rho) — the per-atom reciprocal potential
# =============================================================================
#
# The direct-k reciprocal energy ``E = scale·Σ_k |ρ(k)|²`` is the moment-weighted
# sum of the rho-assembly transpose applied to ρ:
#
#   Σ_k |ρ|² = <S·m, ρ> = <m, Sᵀρ> = Σ_i m_i · (Sᵀρ)_i
#
# so ``E_i = scale·m_i·(Sᵀρ)_i`` is a genuine per-atom decomposition (parallels
# the PME ``multipole_pme_gather_via_spread_t``). ``Sᵀ`` here is the rho
# assembly's OWN transpose (``multipole_rho_moment_grad``), so ``Σ_i E_i`` is
# bit-identical to the collective ``Σ_k|ρ|²``. The op is twice-differentiable
# *for free*: its backward is composed from the already-twice-differentiable
# ``multipole_rho`` op and the rho-backward sub-op chains (position/φ̂/k-vector
# grads), so force-loss and stress-loss flow through ordinary autograd.


@torch.library.custom_op(
    "nvalchemiops::multipole_rho_gather_t",
    mutates_args=(),
    schema=(
        "(Tensor grad_rho, Tensor positions, Tensor source_phi_hat, "
        "Tensor k_vectors, Tensor volume) -> Tensor"
    ),
)
def _multipole_rho_gather_t_op(
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    r"""Opaque forward: ``g = Sᵀ·grad_rho`` per-atom moment gradient ``(N, 4)``.

    e3nn layout ``[q, μ_y, μ_z, μ_x]``; equals the rho-assembly's Jacobian
    transpose (``multipole_rho_moment_grad``).
    """
    cosines, sines = _structure_factor_table_launch(positions, k_vectors)
    return _rho_moment_grad_forward(grad_rho, cosines, sines, source_phi_hat, volume)


@torch.library.register_fake("nvalchemiops::multipole_rho_gather_t")
def _multipole_rho_gather_t_fake(
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``g`` is ``(N_atoms, 4)`` float64."""
    return positions.new_empty((positions.shape[0], 4), dtype=torch.float64)


def _multipole_rho_gather_t_setup_context(ctx, inputs, output) -> None:
    """Save forward inputs for the analytical backward."""
    grad_rho, positions, source_phi_hat, k_vectors, volume = inputs
    ctx.save_for_backward(grad_rho, positions, source_phi_hat, k_vectors, volume)


def _multipole_rho_gather_t_backward(ctx, g_cot: torch.Tensor):
    r"""Adjoint of ``g = Sᵀ·grad_rho``.

    With cotangent ``cg = g_cot`` (per-atom, e3nn layout) on ``g``,
    ``<cg, Sᵀ grad_rho> = <S·cg, grad_rho>``:

    * ``∂/∂grad_rho = S·cg = ρ(cg)`` (the rho FORWARD with moments ``= cg``)
    * ``∂/∂{positions, source_phi_hat, k_vectors}`` are the rho-assembly
      backward sub-ops evaluated with moments ``= cg`` and field ``= grad_rho``.

    Every term is an autograd-registered op, so create_graph composes their
    double-backwards automatically — no explicit second-order routine.
    """
    grad_rho, positions, source_phi_hat, k_vectors, volume = ctx.saved_tensors
    cg_q = g_cot[:, 0].contiguous()
    # e3nn -> Cartesian for the dipole channel: (μ_x, μ_y, μ_z) = lm(3, 1, 2).
    cg_dip = g_cot[:, [3, 1, 2]].contiguous()
    # cos/sin are detached constants (the position 2nd-order is carried by the
    # position-grad kernel), mirroring ``_multipole_rho_backward``.
    cosines, sines = torch.ops.nvalchemiops.multipole_structure_factor(
        positions.detach(), k_vectors.detach()
    )
    shim = types.SimpleNamespace(
        source_phi_hat=source_phi_hat,
        k_vectors=k_vectors,
        volume=volume,
        device=positions.device,
        n_k=k_vectors.shape[0],
    )
    # ∂/∂grad_rho = ρ(cg): S applied to the cotangent-as-moments (= multipole_rho).
    d_grad_rho = torch.ops.nvalchemiops.multipole_rho(
        cg_q, cg_dip, positions, source_phi_hat, k_vectors, volume
    )
    d_positions = _rho_backward_to_positions(
        grad_rho, cg_q, cg_dip, cosines, sines, shim, positions
    ).to(positions.dtype)
    d_phi = _rho_backward_to_phihat(
        grad_rho, cg_q, cg_dip, positions, cosines, sines, shim
    )
    d_kvec = _rho_backward_to_kvec(
        grad_rho, cg_q, cg_dip, positions, cosines, sines, shim
    )
    # Slots: (grad_rho, positions, source_phi_hat, k_vectors, volume).
    return d_grad_rho, d_positions, d_phi, d_kvec, None


torch.library.register_autograd(
    "nvalchemiops::multipole_rho_gather_t",
    _multipole_rho_gather_t_backward,
    setup_context=_multipole_rho_gather_t_setup_context,
)


# =============================================================================
# Direct-k l=2 Cartesian-quadrupole gather: g_Q = Sᵀ_Q · grad_rho  (per-atom).
# The l=2 analog of ``multipole_rho_gather_t``: ``g_Q`` is the Q-assembly's
# Jacobian transpose (``multipole_rho_q_moment_grad``), so for the per-atom
# direct-k reciprocal energy ``E_i = scale·Q_i : g_Q_i`` (with
# ``grad_rho = φ̂ = 2 f_k ρ``), ``Σ_i Q_i:g_Q_i == <ρ_Q, φ̂>`` bit-for-bit.
# Twice-differentiable for free: the backward composes the autograd-registered
# ``multipole_rho_q`` forward and the rho_q backward sub-ops.
# =============================================================================


@torch.library.custom_op(
    "nvalchemiops::multipole_rho_q_gather_t",
    mutates_args=(),
    schema=(
        "(Tensor grad_rho, Tensor positions, Tensor source_coeff2, "
        "Tensor k_vectors, Tensor volume) -> Tensor"
    ),
)
def _multipole_rho_q_gather_t_op(
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    source_coeff2: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    r"""Opaque forward: ``g_Q = Sᵀ_Q · grad_rho`` per-atom quadrupole gradient.

    Returns ``(N, 3, 3)`` Cartesian symmetric; equals the Q-assembly's Jacobian
    transpose (``multipole_rho_q_moment_grad``).
    """
    cosines, sines = _structure_factor_table_launch(positions, k_vectors)
    return torch.ops.nvalchemiops.multipole_rho_q_moment_grad(
        grad_rho, positions, cosines, sines, k_vectors, source_coeff2, volume
    )


@torch.library.register_fake("nvalchemiops::multipole_rho_q_gather_t")
def _multipole_rho_q_gather_t_fake(
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    source_coeff2: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``g_Q`` is ``(N_atoms, 3, 3)`` float64."""
    return positions.new_empty((positions.shape[0], 3, 3), dtype=torch.float64)


def _multipole_rho_q_gather_t_setup_context(ctx, inputs, output) -> None:
    """Save forward inputs for the analytical backward."""
    grad_rho, positions, source_coeff2, k_vectors, volume = inputs
    ctx.save_for_backward(grad_rho, positions, source_coeff2, k_vectors, volume)


def _multipole_rho_q_gather_t_backward(ctx, g_cot: torch.Tensor):
    r"""Adjoint of ``g_Q = Sᵀ_Q · grad_rho``.

    With cotangent ``cg_Q = g_cot`` (per-atom ``(N, 3, 3)``) on ``g_Q``,
    ``<cg_Q, Sᵀ_Q grad_rho> = <S_Q·cg_Q, grad_rho>``:

    * ``∂/∂grad_rho = S_Q·cg_Q = ρ_Q(cg_Q)`` (the ``multipole_rho_q`` FORWARD).
    * ``∂/∂{positions, source_coeff2, k_vectors}`` are the rho_q backward
      sub-ops with quadrupoles ``= cg_Q`` and field ``= grad_rho``.

    Every term is an autograd-registered op, so create_graph composes their
    double-backwards automatically.
    """
    grad_rho, positions, source_coeff2, k_vectors, volume = ctx.saved_tensors
    cg_q = g_cot.contiguous()
    cosines, sines = torch.ops.nvalchemiops.multipole_structure_factor(
        positions.detach(), k_vectors.detach()
    )
    # ∂/∂grad_rho = ρ_Q(cg_Q): S_Q applied to the cotangent-as-quadrupoles.
    d_grad_rho = torch.ops.nvalchemiops.multipole_rho_q(
        cg_q, positions, source_coeff2, k_vectors, volume
    )
    d_positions = torch.ops.nvalchemiops.multipole_rho_q_position_grad(
        grad_rho, cg_q, positions, cosines, sines, k_vectors, source_coeff2, volume
    ).to(positions.dtype)
    d_coeff2 = torch.ops.nvalchemiops.multipole_rho_q_coeff2_grad(
        grad_rho, cg_q, positions, cosines, sines, k_vectors, volume
    )
    d_kvec = torch.ops.nvalchemiops.multipole_rho_q_kvec_grad(
        grad_rho, cg_q, positions, cosines, sines, k_vectors, source_coeff2, volume
    )
    # Slots: (grad_rho, positions, source_coeff2, k_vectors, volume).
    return d_grad_rho, d_positions, d_coeff2, d_kvec, None


torch.library.register_autograd(
    "nvalchemiops::multipole_rho_q_gather_t",
    _multipole_rho_q_gather_t_backward,
    setup_context=_multipole_rho_q_gather_t_setup_context,
)


# =============================================================================
# Cartesian-quadrupole (l=2) rho_Q(k) contribution
# =============================================================================
#
# rho_Q(k) = scale * coeff2(k) * Σ_i (k·Q_i·k) * exp(-ik·r_i), additive onto
# the l<=1 rho(k).


def _rho_q_assemble_launch(
    quadrupoles: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume_f: float,
) -> torch.Tensor:
    """Cache-free ``assemble_rho_q``; returns ``(N_k, 2)`` float64."""
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    n_k = cosines.shape[0]
    wp_scalar = wp.float64 if quadrupoles.dtype == torch.float64 else wp.float32
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    rho_q_t = torch.empty((n_k, 2), dtype=torch.float64, device=device)
    assemble_rho_q(
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        volume=volume_f,
        rho=wp.from_torch(rho_q_t, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return rho_q_t


# ---- main forward op ----


@torch.library.custom_op(
    "nvalchemiops::multipole_rho_q",
    mutates_args=(),
    schema=(
        "(Tensor quadrupoles, Tensor positions, Tensor source_coeff2, "
        "Tensor k_vectors, Tensor volume) -> Tensor"
    ),
)
def _multipole_rho_q_op(
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    source_coeff2: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Opaque additive Cartesian-quadrupole :math:`\\rho_Q(k)`, shape ``(N_k, 2)``."""
    cosines, sines = _structure_factor_table_launch(positions, k_vectors)
    # Assemble with volume=1 (kernel bakes (2*pi)**3); restore 1/V as a tensor op.
    rho_q = _rho_q_assemble_launch(
        quadrupoles,
        cosines,
        sines,
        k_vectors,
        source_coeff2,
        1.0,
    )
    return rho_q * (1.0 / volume.detach())


@torch.library.register_fake("nvalchemiops::multipole_rho_q")
def _multipole_rho_q_fake(
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    source_coeff2: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: rho_Q is ``(N_k, 2)`` float64."""
    return k_vectors.new_empty((k_vectors.shape[0], 2), dtype=torch.float64)


# ---- Q-channel moment grad: grad_rho -> grad_Q (register_warp_op_chain) ----


def _rho_q_moment_grad_forward(
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    r""":func:`rho_q_moment_grad` — ``dL/dQ_i`` ``(N, 3, 3)`` symmetric.

    ``positions`` is unused by the kernel but is an explicit input so its
    second-order grad (from the transpose backward) has a slot.
    """
    device = grad_rho.device
    wp_device = wp.device_from_torch(device)
    n_atoms = cosines.shape[1]
    grad_q_t = torch.empty((n_atoms, 9), dtype=torch.float64, device=device)
    # Launch with scale=1; apply (2*pi)**3/V as a tensor op (no host sync).
    rho_q_moment_grad(
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        1.0,
        wp.from_torch(grad_q_t, dtype=wp.float64),
        device=str(wp_device),
    )
    grad_q_t = grad_q_t * (_TWO_PI_CUBED / volume.detach())
    return grad_q_t.reshape(n_atoms, 3, 3)


def _rho_q_moment_grad_forward_fake(
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_atoms, 3, 3)`` float64."""
    return cosines.new_empty((cosines.shape[1], 3, 3), dtype=torch.float64)


def _rho_q_moment_grad_backward(
    gg_q: torch.Tensor,
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transpose backward of the linear moment map: ``(ggrad_rho, ggrad_positions)``."""
    device = grad_rho.device
    wp_device = wp.device_from_torch(device)
    n_atoms = cosines.shape[1]
    n_k = cosines.shape[0]
    vol = volume.detach()
    gg_q_sym = (0.5 * (gg_q + gg_q.transpose(-1, -2))).contiguous()
    # Launch with volume=1 / scale=1; apply the prefactor as a tensor op below
    # (no host sync). dL/dgrad_rho == assemble_rho_q(gg_Q).
    ggrad_rho = torch.empty((n_k, 2), dtype=torch.float64, device=device)
    assemble_rho_q(
        wp.from_torch(gg_q_sym, dtype=wp.mat33d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        volume=1.0,
        rho=wp.from_torch(ggrad_rho, dtype=wp.float64),
        wp_dtype=wp.float64,
        device=str(wp_device),
    )
    # dL/dpositions == position_gradient_from_rhoq(grad_rho, gg_Q).
    ggrad_pos = torch.empty((n_atoms, 3), dtype=torch.float64, device=device)
    position_gradient_from_rhoq(
        wp.from_torch(gg_q_sym, dtype=wp.mat33d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        1.0,
        wp.from_torch(ggrad_pos, dtype=wp.float64),
        wp_dtype=wp.float64,
        device=str(wp_device),
    )
    return ggrad_rho * (1.0 / vol), ggrad_pos * (_TWO_PI_CUBED / vol)


register_warp_op_chain(
    name="nvalchemiops::multipole_rho_q_moment_grad",
    forward=_rho_q_moment_grad_forward,
    backward=_rho_q_moment_grad_backward,
    diff_input_positions=(0, 1),
    n_forward_inputs=7,
    forward_fake=_rho_q_moment_grad_forward_fake,
)


# ---- Q-channel position grad: grad_rho -> grad_positions (register_warp_op_chain) ----


def _rho_q_position_grad_forward(
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    r""":func:`position_gradient_from_rhoq` — Q-channel ``dL/dr_i`` ``(N, 3)``.

    ``positions`` is unused by the kernel (carried by detached cos/sin) but is
    an explicit input so its position-Hessian second-order grad has a slot.
    """
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    n_atoms = cosines.shape[1]
    wp_scalar = wp.float64 if quadrupoles.dtype == torch.float64 else wp.float32
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    grad_pos_t = torch.empty((n_atoms, 3), dtype=torch.float64, device=device)
    # Launch with scale=1; apply (2*pi)**3/V as a tensor op (no host sync).
    position_gradient_from_rhoq(
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        1.0,
        wp.from_torch(grad_pos_t, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_pos_t * (_TWO_PI_CUBED / volume.detach())


def _rho_q_position_grad_forward_fake(
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_atoms, 3)`` float64."""
    return positions.new_empty((positions.shape[0], 3), dtype=torch.float64)


def _rho_q_position_grad_backward(
    gg_pos: torch.Tensor,
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Second-order backward: ``(ggrad_rho, ggrad_quadrupoles, ggrad_positions)``."""
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    n_atoms = cosines.shape[1]
    n_k = cosines.shape[0]
    wp_scalar = wp.float64 if quadrupoles.dtype == torch.float64 else wp.float32
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    # Launch every kernel with scale=1; apply (2*pi)**3/V as a tensor op below
    # (no host sync).
    s = _TWO_PI_CUBED / volume.detach()
    gg_pos_c = gg_pos.detach().to(torch.float64).contiguous()

    ggrad_rho = torch.empty((n_k, 2), dtype=torch.float64, device=device)
    rhoq_posgrad_backward_grad_rho(
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(gg_pos_c, dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        1.0,
        wp.from_torch(ggrad_rho, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )

    ggrad_q = torch.empty((n_atoms, 9), dtype=torch.float64, device=device)
    rhoq_posgrad_backward_quad(
        wp.from_torch(gg_pos_c, dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        1.0,
        wp.from_torch(ggrad_q, dtype=wp.float64),
        device=str(wp_device),
    )
    ggrad_q = ggrad_q.reshape(n_atoms, 3, 3)

    ggrad_pos = torch.empty((n_atoms, 3), dtype=torch.float64, device=device)
    rhoq_posgrad_backward_positions(
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(gg_pos_c, dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        1.0,
        wp.from_torch(ggrad_pos, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return ggrad_rho * s, ggrad_q * s, ggrad_pos * s


register_warp_op_chain(
    name="nvalchemiops::multipole_rho_q_position_grad",
    forward=_rho_q_position_grad_forward,
    backward=_rho_q_position_grad_backward,
    diff_input_positions=(0, 1, 2),
    n_forward_inputs=8,
    forward_fake=_rho_q_position_grad_forward_fake,
)


# ---- coeff2 / k-vector phase (forward-only; carry the l=2 reciprocal cell-grad) ----


def _rho_q_coeff2_grad_forward(
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """``∂L/∂source_coeff2`` ``(N_k,)`` via the Warp ``rho_q_coeff2_grad`` kernel (l=2 stress).

    ``positions`` is an explicit diff slot (kernel-unused in forward) so the
    Warp second-order backward can place its Hessian grad.
    """
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = wp.float64 if quadrupoles.dtype == torch.float64 else wp.float32
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    grad_c2 = torch.empty(cosines.shape[0], dtype=torch.float64, device=device)
    rho_q_coeff2_grad(
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        1.0,
        wp.from_torch(grad_c2, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_c2 * (_TWO_PI_CUBED / volume.detach())


def _rho_q_coeff2_grad_forward_fake(
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_k,)`` float64."""
    return cosines.new_empty((cosines.shape[0],), dtype=torch.float64)


def _rho_q_coeff2_grad_backward(
    g_c: torch.Tensor,
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp second-order backward: grads for ``(grad_rho, quadrupoles, positions, k_vectors)``."""
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = wp.float64 if quadrupoles.dtype == torch.float64 else wp.float32
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    vec_dtype = wp.vec3d if wp_scalar == wp.float64 else wp.vec3f
    n_atoms = quadrupoles.shape[0]
    scale = float(_TWO_PI_CUBED) / float(volume.detach())

    ggrad_rho = torch.empty_like(grad_rho, dtype=torch.float64)
    ggrad_quad = torch.zeros((n_atoms, 9), dtype=torch.float64, device=device)
    ggrad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    ggrad_kvec = torch.empty((cosines.shape[0], 3), dtype=torch.float64, device=device)
    rho_q_coeff2_grad_double_backward(
        wp.from_torch(g_c.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        scale,
        wp.from_torch(ggrad_rho, dtype=wp.float64),
        wp.from_torch(ggrad_quad, dtype=wp.float64),
        wp.from_torch(ggrad_positions, dtype=wp.vec3d),
        wp.from_torch(ggrad_kvec, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    ggrad_quad = ggrad_quad.view(n_atoms, 3, 3).to(quadrupoles.dtype)
    return ggrad_rho, ggrad_quad, ggrad_positions, ggrad_kvec


_COEFF2_GRAD_SCHEMA = (
    "(Tensor grad_rho, Tensor quadrupoles, Tensor positions, Tensor cosines, "
    "Tensor sines, Tensor k_vectors, Tensor volume) -> Tensor"
)
_COEFF2_GRAD_BWD_SCHEMA = (
    "(Tensor g_c, Tensor grad_rho, Tensor quadrupoles, Tensor positions, "
    "Tensor cosines, Tensor sines, Tensor k_vectors, Tensor volume) "
    "-> (Tensor, Tensor, Tensor, Tensor)"
)
register_warp_op_chain(
    name="nvalchemiops::multipole_rho_q_coeff2_grad",
    forward=_rho_q_coeff2_grad_forward,
    forward_schema=_COEFF2_GRAD_SCHEMA,
    forward_fake=_rho_q_coeff2_grad_forward_fake,
    backward=_rho_q_coeff2_grad_backward,
    backward_schema=_COEFF2_GRAD_BWD_SCHEMA,
    backward_return_arity=4,
    # grad_rho(0), quadrupoles(1), positions(2), k_vectors(5) are diff.
    diff_input_positions=(0, 1, 2, 5),
    n_forward_inputs=7,
)


def _rho_q_kvec_grad_forward(
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """``∂L/∂k_vectors`` ``(N_k, 3)`` via the (k·Q·k) form + phase (l=2 stress), Warp kernel."""
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = wp.float64 if quadrupoles.dtype == torch.float64 else wp.float32
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    vec_dtype = wp.vec3d if wp_scalar == wp.float64 else wp.vec3f
    grad_k = torch.empty((cosines.shape[0], 3), dtype=torch.float64, device=device)
    rho_q_kvec_grad(
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        1.0,
        wp.from_torch(grad_k, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_k * (_TWO_PI_CUBED / volume.detach())


def _rho_q_kvec_grad_forward_fake(
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_k, 3)`` float64."""
    return cosines.new_empty((cosines.shape[0], 3), dtype=torch.float64)


def _rho_q_kvec_grad_backward(
    g_k: torch.Tensor,
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp second-order backward: grads for ``(grad_rho, quadrupoles, positions, k_vectors, source_coeff2)``."""
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = wp.float64 if quadrupoles.dtype == torch.float64 else wp.float32
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    vec_dtype = wp.vec3d if wp_scalar == wp.float64 else wp.vec3f
    n_atoms = quadrupoles.shape[0]
    scale = float(_TWO_PI_CUBED) / float(volume.detach())

    ggrad_rho = torch.empty_like(grad_rho, dtype=torch.float64)
    ggrad_quad = torch.zeros((n_atoms, 9), dtype=torch.float64, device=device)
    ggrad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    ggrad_coeff2 = torch.empty(cosines.shape[0], dtype=torch.float64, device=device)
    ggrad_kvec = torch.empty((cosines.shape[0], 3), dtype=torch.float64, device=device)
    rho_q_kvec_grad_double_backward(
        wp.from_torch(g_k.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        scale,
        wp.from_torch(ggrad_rho, dtype=wp.float64),
        wp.from_torch(ggrad_quad, dtype=wp.float64),
        wp.from_torch(ggrad_positions, dtype=wp.vec3d),
        wp.from_torch(ggrad_coeff2, dtype=wp.float64),
        wp.from_torch(ggrad_kvec, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    ggrad_quad = ggrad_quad.view(n_atoms, 3, 3).to(quadrupoles.dtype)
    return ggrad_rho, ggrad_quad, ggrad_positions, ggrad_kvec, ggrad_coeff2


_KVECQ_GRAD_SCHEMA = (
    "(Tensor grad_rho, Tensor quadrupoles, Tensor positions, Tensor cosines, "
    "Tensor sines, Tensor k_vectors, Tensor source_coeff2, Tensor volume) "
    "-> Tensor"
)
_KVECQ_GRAD_BWD_SCHEMA = (
    "(Tensor g_k, Tensor grad_rho, Tensor quadrupoles, Tensor positions, "
    "Tensor cosines, Tensor sines, Tensor k_vectors, Tensor source_coeff2, "
    "Tensor volume) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)
register_warp_op_chain(
    name="nvalchemiops::multipole_rho_q_kvec_grad",
    forward=_rho_q_kvec_grad_forward,
    forward_schema=_KVECQ_GRAD_SCHEMA,
    forward_fake=_rho_q_kvec_grad_forward_fake,
    backward=_rho_q_kvec_grad_backward,
    backward_schema=_KVECQ_GRAD_BWD_SCHEMA,
    backward_return_arity=5,
    # grad_rho(0), quadrupoles(1), positions(2), k_vectors(5), source_coeff2(6).
    diff_input_positions=(0, 1, 2, 5, 6),
    n_forward_inputs=8,
)


def _multipole_rho_q_setup_context(ctx, inputs, output) -> None:
    """Save the forward inputs for the analytical backward."""
    quadrupoles, positions, source_coeff2, k_vectors, volume = inputs
    ctx.save_for_backward(quadrupoles, positions, source_coeff2, k_vectors, volume)


def _multipole_rho_q_backward(ctx, grad_rho: torch.Tensor):
    """Analytical grads for quadrupoles, positions, source_coeff2, k_vectors.

    Recomputes (cos, sin) via the opaque structure-factor op (detached inputs:
    cos/sin are constants) and dispatches the four backward sub-ops. The
    moment / position chains carry their own second-order backward
    (``create_graph``); coeff2 / k_vectors are forward-only (1st-order
    reciprocal cell-grad; cell 2nd-order out of scope). ``volume`` gets ``None``.
    """
    quadrupoles, positions, source_coeff2, k_vectors, volume = ctx.saved_tensors
    grad_rho = grad_rho.contiguous()
    cosines, sines = torch.ops.nvalchemiops.multipole_structure_factor(
        positions.detach(), k_vectors.detach()
    )
    grad_quadrupoles = torch.ops.nvalchemiops.multipole_rho_q_moment_grad(
        grad_rho, positions, cosines, sines, k_vectors, source_coeff2, volume
    )
    grad_positions = torch.ops.nvalchemiops.multipole_rho_q_position_grad(
        grad_rho,
        quadrupoles,
        positions,
        cosines,
        sines,
        k_vectors,
        source_coeff2,
        volume,
    )
    # coeff2: twice-differentiable Warp op chain (positions explicit diff slot).
    grad_coeff2 = torch.ops.nvalchemiops.multipole_rho_q_coeff2_grad(
        grad_rho, quadrupoles, positions, cosines, sines, k_vectors, volume
    )
    # k-vector (l=2): twice-differentiable Warp op chain.
    grad_kvec = torch.ops.nvalchemiops.multipole_rho_q_kvec_grad(
        grad_rho,
        quadrupoles,
        positions,
        cosines,
        sines,
        k_vectors,
        source_coeff2,
        volume,
    )
    grad_quadrupoles = grad_quadrupoles.to(quadrupoles.dtype)
    grad_positions = grad_positions.to(positions.dtype)
    # Slots: (quadrupoles, positions, source_coeff2, k_vectors, volume).
    return grad_quadrupoles, grad_positions, grad_coeff2, grad_kvec, None


torch.library.register_autograd(
    "nvalchemiops::multipole_rho_q",
    _multipole_rho_q_backward,
    setup_context=_multipole_rho_q_setup_context,
)


class MultipoleRhoQFunction:
    r"""Back-compat shim for the Cartesian-quadrupole :math:`\rho_Q(k)` contribution.

    The implementation is the fully-differentiable
    ``torch.ops.nvalchemiops.multipole_rho_q`` custom op (opaque forward +
    analytical backward + ``create_graph`` through the moment/position chains;
    forward-only coeff2/k-vector ops carry the l=2 reciprocal cell-grad). This
    class only preserves the historical ``.apply(quadrupoles, positions,
    source_coeff2, k_vectors, cache)`` signature (``cache`` supplies ``volume``
    and is validated for ``l_max>=2``); new code should call the op directly.

    Parameters
    ----------
    quadrupoles : torch.Tensor
        Per-atom Cartesian quadrupoles, shape ``(N_atoms, 3, 3)``.
    positions : torch.Tensor
        Atomic coordinates, shape ``(N_atoms, 3)``.
    source_coeff2 : torch.Tensor
        Per-k l=2 source coefficient :math:`c_2(k)`, shape ``(N_k,)``; value
        equals ``cache.source_coeff2``.
    k_vectors : torch.Tensor
        Reciprocal-lattice vectors, shape ``(N_k, 3)``; value equals
        ``cache.k_vectors``.
    cache : MultipoleSCFCache
        Per-system direct-k-space state; must be built with ``l_max>=2``.

    Returns
    -------
    torch.Tensor
        Additive Q-channel :math:`\rho_Q(k)`, shape ``(N_k, 2)`` float64.
    """

    @staticmethod
    def apply(
        quadrupoles: torch.Tensor,
        positions: torch.Tensor,
        source_coeff2: torch.Tensor,
        k_vectors: torch.Tensor,
        cache: MultipoleSCFCache,
    ) -> torch.Tensor:
        """Dispatch to ``torch.ops.nvalchemiops.multipole_rho_q`` (``cache`` -> volume)."""
        if cache.source_coeff2 is None:
            raise ValueError(
                "MultipoleRhoQFunction requires a cache built with l_max>=2 "
                "(cache.source_coeff2 is None)."
            )
        return torch.ops.nvalchemiops.multipole_rho_q(
            quadrupoles, positions, source_coeff2, k_vectors, cache.volume
        )


# =============================================================================
# Ewald reciprocal-space fused scalar (public entry)
# =============================================================================


def multipole_reciprocal_space_dipole_fused_scalar(
    positions: torch.Tensor,
    source_feats: torch.Tensor,
    cache: MultipoleSCFCache,
    *,
    include_self_interaction: bool = True,
) -> torch.Tensor:
    r"""Scalar reciprocal-space energy — thin alias of :func:`multipole_scf_step_energy`.

    The rho(k) op chain in :func:`multipole_scf_step_energy` produces exactly
    this scalar (:math:`V/(2\pi)^6 \sum_k f_k |\rho(k)|^2` minus the optional
    self term) with full forward + backward + ``create_graph`` autograd, all
    ``torch.compile``-clean, so this entry just forwards to it.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic coordinates, shape ``(N_atoms, 3)``.
    source_feats : torch.Tensor
        Packed source moments ``(N_atoms, 1|4)``.
    cache : MultipoleSCFCache
        Per-system direct-k-space state.
    include_self_interaction : bool, optional
        When ``False``, subtract the source-source self-overlap term. Defaults
        to ``True``.

    Returns
    -------
    torch.Tensor
        Scalar reciprocal-space energy (0-d float64 tensor).
    """
    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
        multipole_scf_step_energy,
    )

    return multipole_scf_step_energy(
        cache,
        positions,
        source_feats,
        include_self_interaction=include_self_interaction,
    )


# =============================================================================
# MultipoleProjectRawFeaturesFunction
# =============================================================================


# Opaque feature-projection sub-op chains (l<=1)
# -----------------------------------------------------------------------------


def _project_raw_features_launch(
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
) -> torch.Tensor:
    r"""Run :func:`project_features_dipole` (no self-subtract, natural layout).

    Returns the raw feature tensor ``(N_atoms, N_σ, 4)`` — the projection of
    ``V(k)`` onto the receiver basis with the half-origin ``k_factor_proj``
    weighting, before the self-interaction correction.
    """
    device = potential.device
    wp_device = wp.device_from_torch(device)
    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]

    # Zero dummies for the unused self-interaction slots.
    src_lm_zero = torch.zeros((n_atoms, 4), dtype=torch.float64, device=device)
    oc_zero = torch.zeros((n_sigma, 2), dtype=torch.float64, device=device)

    # Natural-layout LUT — the l<=1 kernel writes column s*4+lm.
    s_ar = torch.arange(n_sigma, dtype=torch.int32, device=device).unsqueeze(1)
    lm_ar = torch.arange(4, dtype=torch.int32, device=device).unsqueeze(0)
    lut = (s_ar * 4 + lm_ar).contiguous()

    features_flat = torch.empty(
        (n_atoms, n_sigma * 4), dtype=torch.float64, device=device
    )
    project_features_dipole(
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(src_lm_zero, dtype=wp.float64),
        wp.from_torch(oc_zero, dtype=wp.float64),
        False,  # subtract_self
        wp.from_torch(lut, dtype=wp.int32),
        wp.from_torch(features_flat, dtype=wp.float64),
        device=str(wp_device),
    )
    return features_flat.reshape(n_atoms, n_sigma, 4)


# ---- V-grad chain: grad_raw -> grad_V (register_warp_op_chain) ----


def _feature_v_grad_forward(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    r""":func:`v_gradient_from_feature_grad` — ``∂L/∂V(k)`` ``(N_k, 2)``.

    ``k_vectors`` / ``positions`` are unused by the value kernel (carried by the
    detached cos/sin table) but are explicit inputs so their second-order grads
    have a slot in the backward.
    """
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_k = receiver_phi_hat.shape[0]
    grad_v = torch.empty((n_k, 2), dtype=torch.float64, device=device)
    v_gradient_from_feature_grad(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_v, dtype=wp.float64),
        device=str(wp_device),
    )
    return grad_v


def _feature_v_grad_forward_fake(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_k, 2)`` float64."""
    return receiver_phi_hat.new_empty(
        (receiver_phi_hat.shape[0], 2), dtype=torch.float64
    )


def _feature_v_grad_backward(
    gg_v: torch.Tensor,
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Second-order backward: ``(ggrad_raw, ggrad_kfp, ggrad_positions)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]

    # ggrad_raw: transpose = project_features_dipole with V = gg_v.
    src_lm_zero = torch.zeros((n_atoms, 4), dtype=torch.float64, device=device)
    oc_zero = torch.zeros((n_sigma, 2), dtype=torch.float64, device=device)
    lut = (
        torch.arange(n_sigma * 4, dtype=torch.int32, device=device)
        .view(n_sigma, 4)
        .contiguous()
    )
    ggrad_raw_flat = torch.zeros(
        (n_atoms, n_sigma * 4), dtype=torch.float64, device=device
    )
    project_features_dipole(
        wp.from_torch(gg_v.contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(src_lm_zero, dtype=wp.float64),
        wp.from_torch(oc_zero, dtype=wp.float64),
        False,  # subtract_self
        wp.from_torch(lut, dtype=wp.int32),
        wp.from_torch(ggrad_raw_flat, dtype=wp.float64),
        device=str(wp_device),
    )
    ggrad_raw = ggrad_raw_flat.reshape(n_atoms, n_sigma, 4)

    # ggrad_kfp: grad_v[k] is linear in kfp[k], so ∂/∂kfp = grad_v[k] / kfp[k];
    # kfp == 0 entries have grad_v == 0 -> grad 0.
    grad_v = _feature_v_grad_forward(
        grad_raw, receiver_phi_hat, cosines, sines, k_factor_proj, k_vectors, positions
    )
    safe_kfp = torch.where(
        k_factor_proj != 0, k_factor_proj, torch.ones_like(k_factor_proj)
    )
    per_k = (gg_v * grad_v).sum(dim=-1) / safe_kfp
    ggrad_kfp = torch.where(k_factor_proj != 0, per_k, torch.zeros_like(per_k))

    ggrad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    v_grad_from_feat_grad_backward_positions(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_v.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(ggrad_positions, dtype=wp.float64),
        device=str(wp_device),
    )
    return ggrad_raw, ggrad_kfp, ggrad_positions


register_warp_op_chain(
    name="nvalchemiops::multipole_feature_v_grad",
    forward=_feature_v_grad_forward,
    backward=_feature_v_grad_backward,
    diff_input_positions=(0, 4, 6),
    n_forward_inputs=7,
    forward_fake=_feature_v_grad_forward_fake,
)


# ---- position-grad chain: grad_raw -> grad_positions (register_warp_op_chain) ----


def _feature_position_grad_forward(
    grad_raw: torch.Tensor,
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    r""":func:`position_gradient_from_feature_grad` — ``∂L/∂r`` ``(N_atoms, 3)``.

    ``positions`` is unused by the value kernel (carried by the detached cos/sin
    table) but is an explicit input so its position-Hessian grad has a slot.
    """
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_atoms = cosines.shape[1]
    grad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    position_gradient_from_feature_grad(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(grad_positions, dtype=wp.float64),
        device=str(wp_device),
    )
    return grad_positions


def _feature_position_grad_forward_fake(
    grad_raw: torch.Tensor,
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: position gradient ``(N_atoms, 3)`` float64."""
    return positions.new_empty((positions.shape[0], 3), dtype=torch.float64)


def _feature_position_grad_backward(
    gg_positions: torch.Tensor,
    grad_raw: torch.Tensor,
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Second-order backward: ``(ggrad_raw, ggrad_v, ggrad_positions)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_atoms = cosines.shape[1]

    ggrad_raw = torch.empty_like(grad_raw)
    feat_position_grad_backward_grad_raw(
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(ggrad_raw, dtype=wp.float64),
        device=str(wp_device),
    )

    ggrad_v = torch.empty_like(potential)
    feat_position_grad_backward_v(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(ggrad_v, dtype=wp.float64),
        device=str(wp_device),
    )

    ggrad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    feat_position_grad_backward_positions(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(ggrad_positions, dtype=wp.float64),
        device=str(wp_device),
    )
    return ggrad_raw, ggrad_v, ggrad_positions


register_warp_op_chain(
    name="nvalchemiops::multipole_feature_position_grad",
    forward=_feature_position_grad_forward,
    backward=_feature_position_grad_backward,
    diff_input_positions=(0, 1, 7),
    n_forward_inputs=8,
    forward_fake=_feature_position_grad_forward_fake,
)


# ---- phi_hat / k-vector phase (forward-only; carry the reciprocal cell-grad) ----


@torch.library.custom_op(
    "nvalchemiops::multipole_feature_phihat_grad",
    mutates_args=(),
    schema=(
        "(Tensor grad_raw, Tensor cosines, Tensor sines, Tensor k_factor_proj, "
        "Tensor potential, int n_lm) -> Tensor"
    ),
)
def _feature_phihat_grad_op(
    grad_raw: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
    n_lm: int,
) -> torch.Tensor:
    """``dL/dreceiver_phi_hat`` ``(N_k, N_σ, n_lm, 2)`` via ``project_phihat_grad_dipole``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_k = cosines.shape[0]
    n_sigma = grad_raw.shape[1]
    grad_phi = torch.zeros((n_k, n_sigma, n_lm, 2), dtype=torch.float64, device=device)
    project_phihat_grad_dipole(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_phi, dtype=wp.float64),
        device=str(wp_device),
    )
    return grad_phi


@torch.library.register_fake("nvalchemiops::multipole_feature_phihat_grad")
def _feature_phihat_grad_fake(
    grad_raw: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
    n_lm: int,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_k, N_σ, n_lm, 2)`` float64."""
    return cosines.new_empty(
        (cosines.shape[0], grad_raw.shape[1], n_lm, 2), dtype=torch.float64
    )


@torch.library.custom_op(
    "nvalchemiops::multipole_feature_kphase_grad",
    mutates_args=(),
    schema=(
        "(Tensor grad_raw, Tensor receiver_phi_hat, Tensor cosines, "
        "Tensor sines, Tensor k_factor_proj, Tensor potential, "
        "Tensor positions) -> Tensor"
    ),
)
def _feature_kphase_grad_op(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """``dL/dk_vectors`` ``(N_k, 3)`` through the phase via ``project_kphase_grad_dipole``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_k = cosines.shape[0]
    grad_kvec = torch.zeros((n_k, 3), dtype=torch.float64, device=device)
    project_kphase_grad_dipole(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(positions.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(grad_kvec, dtype=wp.float64),
        device=str(wp_device),
    )
    return grad_kvec


@torch.library.register_fake("nvalchemiops::multipole_feature_kphase_grad")
def _feature_kphase_grad_fake(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_k, 3)`` float64."""
    return cosines.new_empty((cosines.shape[0], 3), dtype=torch.float64)


# =============================================================================
# torch.library.custom_op chain for the raw feature projection (l<=1)
# =============================================================================


@torch.library.custom_op(
    "nvalchemiops::multipole_project_raw_features",
    mutates_args=(),
    schema=(
        "(Tensor potential, Tensor positions, Tensor receiver_phi_hat, "
        "Tensor k_vectors, Tensor k_factor_proj) -> Tensor"
    ),
)
def _multipole_project_raw_features_op(
    potential: torch.Tensor,
    positions: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    k_factor_proj: torch.Tensor,
) -> torch.Tensor:
    """Opaque forward: build (cos, sin) and run the raw l<=1 projection."""
    cosines, sines = _structure_factor_table_launch(positions, k_vectors)
    return _project_raw_features_launch(
        potential, receiver_phi_hat, cosines, sines, k_factor_proj
    )


@torch.library.register_fake("nvalchemiops::multipole_project_raw_features")
def _multipole_project_raw_features_fake(
    potential: torch.Tensor,
    positions: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    k_factor_proj: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: raw features ``(N_atoms, N_σ, 4)`` float64."""
    n_atoms = positions.shape[0]
    n_sigma = receiver_phi_hat.shape[1]
    return positions.new_empty((n_atoms, n_sigma, 4), dtype=torch.float64)


def _multipole_project_raw_features_setup_context(ctx, inputs, output) -> None:
    """Save the forward inputs for the analytical backward."""
    potential, positions, receiver_phi_hat, k_vectors, k_factor_proj = inputs
    ctx.save_for_backward(
        potential, positions, receiver_phi_hat, k_vectors, k_factor_proj
    )


def _multipole_project_raw_features_backward(ctx, grad_raw: torch.Tensor):
    """Analytical grads for potential, positions, receiver_phi_hat, k_vectors.

    Recomputes (cos, sin) via the opaque ``multipole_structure_factor`` op (on
    detached positions/k_vectors — cos/sin are constants) and dispatches the V /
    position chains + the forward-only phi_hat / k-vector ops. Every call is an
    opaque op or plain torch, so AOTAutograd traces this without touching Warp.
    ``k_factor_proj`` gets ``None`` (zero cell-grad).
    """
    potential, positions, receiver_phi_hat, k_vectors, k_factor_proj = ctx.saved_tensors
    grad_raw = grad_raw.contiguous()
    cosines, sines = torch.ops.nvalchemiops.multipole_structure_factor(
        positions.detach(), k_vectors.detach()
    )
    grad_v = torch.ops.nvalchemiops.multipole_feature_v_grad(
        grad_raw, receiver_phi_hat, cosines, sines, k_factor_proj, k_vectors, positions
    )
    grad_positions = torch.ops.nvalchemiops.multipole_feature_position_grad(
        grad_raw,
        potential,
        receiver_phi_hat,
        cosines,
        sines,
        k_factor_proj,
        k_vectors,
        positions,
    )
    grad_phihat = torch.ops.nvalchemiops.multipole_feature_phihat_grad(
        grad_raw, cosines, sines, k_factor_proj, potential, 4
    )
    grad_kvec = torch.ops.nvalchemiops.multipole_feature_kphase_grad(
        grad_raw, receiver_phi_hat, cosines, sines, k_factor_proj, potential, positions
    )
    grad_v = grad_v.to(potential.dtype)
    grad_positions = grad_positions.to(positions.dtype)
    grad_phihat = grad_phihat.to(receiver_phi_hat.dtype)
    grad_kvec = grad_kvec.to(k_vectors.dtype)
    # Slots: (potential, positions, receiver_phi_hat, k_vectors, k_factor_proj).
    return grad_v, grad_positions, grad_phihat, grad_kvec, None


torch.library.register_autograd(
    "nvalchemiops::multipole_project_raw_features",
    _multipole_project_raw_features_backward,
    setup_context=_multipole_project_raw_features_setup_context,
)


class MultipoleProjectRawFeaturesFunction:
    r"""Back-compat shim for the raw (un-self-subtracted) l<=1 feature projection.

    The implementation is the fully-differentiable
    ``torch.ops.nvalchemiops.multipole_project_raw_features`` custom op (opaque
    forward + analytical backward + ``create_graph`` support, all compile-clean).
    This class only preserves the historical
    ``.apply(potential, positions, receiver_phi_hat, k_vectors, cache)`` call
    signature (the dataclass ``cache`` supplies ``k_factor_proj``); new code
    should call the op directly.

    Parameters
    ----------
    potential : torch.Tensor
        Reciprocal-space potential :math:`V(k)`, shape ``(N_k, 2)``.
    positions : torch.Tensor
        Atomic coordinates, shape ``(N_atoms, 3)``.
    receiver_phi_hat : torch.Tensor
        l<=1 receiver block ``[:, :, :4, :]`` of shape ``(N_k, N_σ, 4, 2)``;
        carries the feature cell-grad.
    k_vectors : torch.Tensor
        Reciprocal-lattice vectors, shape ``(N_k, 3)``.
    cache : MultipoleSCFCache
        Per-system direct-k-space state; supplies ``cache.k_factor_proj``.

    Returns
    -------
    torch.Tensor
        Raw (un-self-subtracted) features ``(N_atoms, N_σ, 4)`` float64.
    """

    @staticmethod
    def apply(
        potential: torch.Tensor,
        positions: torch.Tensor,
        receiver_phi_hat: torch.Tensor,
        k_vectors: torch.Tensor,
        cache: MultipoleSCFCache,
    ) -> torch.Tensor:
        """Dispatch to ``torch.ops.nvalchemiops.multipole_project_raw_features``."""
        return torch.ops.nvalchemiops.multipole_project_raw_features(
            potential, positions, receiver_phi_hat, k_vectors, cache.k_factor_proj
        )


# =============================================================================
# l=2 raw feature projection (5-channel receiver block)
# =============================================================================


def _l2_receiver_block(cache: MultipoleSCFCache) -> torch.Tensor:
    r"""Extract the contiguous l=2 sub-block ``(N_k, N_σ, 5, 2)`` of the cache.

    The feature_max_l=2 cache stores a 9-column ``receiver_phi_hat`` with the
    l<=1 block in columns ``0:4`` and the l=2 block in columns ``4:9``.
    """
    if cache.feature_max_l < 2:
        raise ValueError(
            "l=2 feature projection requires a cache built with feature_max_l=2 "
            f"(got feature_max_l={cache.feature_max_l})."
        )
    return cache.receiver_phi_hat[:, :, 4:9, :].contiguous()


def _project_raw_features_quadrupole_launch(
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
) -> torch.Tensor:
    r"""Run :func:`project_features_quadrupole` (no self-subtract). ``(N, N_σ, 5)``."""
    device = potential.device
    wp_device = wp.device_from_torch(device)
    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]

    features = torch.empty((n_atoms, n_sigma, 5), dtype=torch.float64, device=device)
    project_features_quadrupole(
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(features, dtype=wp.float64),
        device=str(wp_device),
    )
    return features


# ---- l=2 V-grad chain: grad_raw -> grad_V (register_warp_op_chain) ----


def _feature_v_grad_quadrupole_forward(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    r""":func:`v_gradient_from_feature_grad_quadrupole` — ``∂L/∂V(k)`` ``(N_k, 2)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_k = receiver_phi_hat.shape[0]
    grad_v = torch.empty((n_k, 2), dtype=torch.float64, device=device)
    v_gradient_from_feature_grad_quadrupole(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_v, dtype=wp.float64),
        device=str(wp_device),
    )
    return grad_v


def _feature_v_grad_quadrupole_forward_fake(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_k, 2)`` float64."""
    return receiver_phi_hat.new_empty(
        (receiver_phi_hat.shape[0], 2), dtype=torch.float64
    )


def _feature_v_grad_quadrupole_backward(
    gg_v: torch.Tensor,
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Second-order backward: ``(ggrad_raw, ggrad_positions)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_atoms = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[1]

    # ggrad_raw: transpose = project_features_quadrupole with V = gg_v.
    ggrad_raw = torch.empty((n_atoms, n_sigma, 5), dtype=torch.float64, device=device)
    project_features_quadrupole(
        wp.from_torch(gg_v.contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(ggrad_raw, dtype=wp.float64),
        device=str(wp_device),
    )

    ggrad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    v_grad_from_feat_grad_backward_positions_quadrupole(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_v.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(ggrad_positions, dtype=wp.float64),
        device=str(wp_device),
    )
    return ggrad_raw, ggrad_positions


register_warp_op_chain(
    name="nvalchemiops::multipole_feature_v_grad_quadrupole",
    forward=_feature_v_grad_quadrupole_forward,
    backward=_feature_v_grad_quadrupole_backward,
    diff_input_positions=(0, 6),
    n_forward_inputs=7,
    forward_fake=_feature_v_grad_quadrupole_forward_fake,
)


# ---- l=2 position-grad chain: grad_raw -> grad_positions (register_warp_op_chain) ----


def _feature_position_grad_quadrupole_forward(
    grad_raw: torch.Tensor,
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    r""":func:`position_gradient_from_feature_grad_quadrupole` — ``∂L/∂r`` ``(N_atoms, 3)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_atoms = cosines.shape[1]
    grad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    position_gradient_from_feature_grad_quadrupole(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(grad_positions, dtype=wp.float64),
        device=str(wp_device),
    )
    return grad_positions


def _feature_position_grad_quadrupole_forward_fake(
    grad_raw: torch.Tensor,
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: position gradient ``(N_atoms, 3)`` float64."""
    return positions.new_empty((positions.shape[0], 3), dtype=torch.float64)


def _feature_position_grad_quadrupole_backward(
    gg_positions: torch.Tensor,
    grad_raw: torch.Tensor,
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Second-order backward: ``(ggrad_raw, ggrad_v, ggrad_positions)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_atoms = cosines.shape[1]

    ggrad_raw = torch.empty_like(grad_raw)
    feat_position_grad_backward_grad_raw_quadrupole(
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(ggrad_raw, dtype=wp.float64),
        device=str(wp_device),
    )

    ggrad_v = torch.empty_like(potential)
    feat_position_grad_backward_v_quadrupole(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(ggrad_v, dtype=wp.float64),
        device=str(wp_device),
    )

    ggrad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    feat_position_grad_backward_positions_quadrupole(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(ggrad_positions, dtype=wp.float64),
        device=str(wp_device),
    )
    return ggrad_raw, ggrad_v, ggrad_positions


register_warp_op_chain(
    name="nvalchemiops::multipole_feature_position_grad_quadrupole",
    forward=_feature_position_grad_quadrupole_forward,
    backward=_feature_position_grad_quadrupole_backward,
    diff_input_positions=(0, 1, 7),
    n_forward_inputs=8,
    forward_fake=_feature_position_grad_quadrupole_forward_fake,
)


# ---- l=2 phi_hat / k-vector phase (forward-only; reuse the channel-generic kernels) ----


@torch.library.custom_op(
    "nvalchemiops::multipole_feature_phihat_grad_quadrupole",
    mutates_args=(),
    schema=(
        "(Tensor grad_raw, Tensor cosines, Tensor sines, Tensor k_factor_proj, "
        "Tensor potential) -> Tensor"
    ),
)
def _feature_phihat_grad_quadrupole_op(
    grad_raw: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
) -> torch.Tensor:
    """``dL/dreceiver_phi_hat`` ``(N_k, N_σ, 5, 2)`` via ``project_phihat_grad_dipole``."""
    return _feature_phihat_grad_op(
        grad_raw, cosines, sines, k_factor_proj, potential, 5
    )


@torch.library.register_fake("nvalchemiops::multipole_feature_phihat_grad_quadrupole")
def _feature_phihat_grad_quadrupole_fake(
    grad_raw: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_k, N_σ, 5, 2)`` float64."""
    return cosines.new_empty(
        (cosines.shape[0], grad_raw.shape[1], 5, 2), dtype=torch.float64
    )


@torch.library.custom_op(
    "nvalchemiops::multipole_feature_kphase_grad_quadrupole",
    mutates_args=(),
    schema=(
        "(Tensor grad_raw, Tensor receiver_phi_hat, Tensor cosines, "
        "Tensor sines, Tensor k_factor_proj, Tensor potential, "
        "Tensor positions) -> Tensor"
    ),
)
def _feature_kphase_grad_quadrupole_op(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """``dL/dk_vectors`` ``(N_k, 3)`` through the phase via ``project_kphase_grad_dipole``."""
    return _feature_kphase_grad_op(
        grad_raw, receiver_phi_hat, cosines, sines, k_factor_proj, potential, positions
    )


@torch.library.register_fake("nvalchemiops::multipole_feature_kphase_grad_quadrupole")
def _feature_kphase_grad_quadrupole_fake(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_k, 3)`` float64."""
    return cosines.new_empty((cosines.shape[0], 3), dtype=torch.float64)


# =============================================================================
# torch.library.custom_op chain for the raw l=2 feature projection
# =============================================================================


@torch.library.custom_op(
    "nvalchemiops::multipole_project_raw_features_quadrupole",
    mutates_args=(),
    schema=(
        "(Tensor potential, Tensor positions, Tensor receiver_phi_hat, "
        "Tensor k_vectors, Tensor k_factor_proj) -> Tensor"
    ),
)
def _multipole_project_raw_features_quadrupole_op(
    potential: torch.Tensor,
    positions: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    k_factor_proj: torch.Tensor,
) -> torch.Tensor:
    """Opaque forward: build (cos, sin) and run the raw l=2 projection."""
    cosines, sines = _structure_factor_table_launch(positions, k_vectors)
    return _project_raw_features_quadrupole_launch(
        potential, receiver_phi_hat, cosines, sines, k_factor_proj
    )


@torch.library.register_fake("nvalchemiops::multipole_project_raw_features_quadrupole")
def _multipole_project_raw_features_quadrupole_fake(
    potential: torch.Tensor,
    positions: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    k_factor_proj: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: raw l=2 features ``(N_atoms, N_σ, 5)`` float64."""
    n_atoms = positions.shape[0]
    n_sigma = receiver_phi_hat.shape[1]
    return positions.new_empty((n_atoms, n_sigma, 5), dtype=torch.float64)


def _multipole_project_raw_features_quadrupole_setup_context(
    ctx, inputs, output
) -> None:
    """Save the forward inputs for the analytical backward."""
    potential, positions, receiver_phi_hat, k_vectors, k_factor_proj = inputs
    ctx.save_for_backward(
        potential, positions, receiver_phi_hat, k_vectors, k_factor_proj
    )


def _multipole_project_raw_features_quadrupole_backward(ctx, grad_raw: torch.Tensor):
    """Analytical grads for potential, positions, receiver_phi_hat, k_vectors.

    Recomputes (cos, sin) via the opaque ``multipole_structure_factor`` op (on
    detached positions/k_vectors) and dispatches the l=2 V / position chains +
    the forward-only phi_hat / k-vector ops. ``k_factor_proj`` gets ``None``.
    """
    potential, positions, receiver_phi_hat, k_vectors, k_factor_proj = ctx.saved_tensors
    grad_raw = grad_raw.contiguous()
    cosines, sines = torch.ops.nvalchemiops.multipole_structure_factor(
        positions.detach(), k_vectors.detach()
    )
    grad_v = torch.ops.nvalchemiops.multipole_feature_v_grad_quadrupole(
        grad_raw, receiver_phi_hat, cosines, sines, k_factor_proj, k_vectors, positions
    )
    grad_positions = torch.ops.nvalchemiops.multipole_feature_position_grad_quadrupole(
        grad_raw,
        potential,
        receiver_phi_hat,
        cosines,
        sines,
        k_factor_proj,
        k_vectors,
        positions,
    )
    grad_phihat = torch.ops.nvalchemiops.multipole_feature_phihat_grad_quadrupole(
        grad_raw, cosines, sines, k_factor_proj, potential
    )
    grad_kvec = torch.ops.nvalchemiops.multipole_feature_kphase_grad_quadrupole(
        grad_raw, receiver_phi_hat, cosines, sines, k_factor_proj, potential, positions
    )
    grad_v = grad_v.to(potential.dtype)
    grad_positions = grad_positions.to(positions.dtype)
    grad_phihat = grad_phihat.to(receiver_phi_hat.dtype)
    grad_kvec = grad_kvec.to(k_vectors.dtype)
    # Slots: (potential, positions, receiver_phi_hat, k_vectors, k_factor_proj).
    return grad_v, grad_positions, grad_phihat, grad_kvec, None


torch.library.register_autograd(
    "nvalchemiops::multipole_project_raw_features_quadrupole",
    _multipole_project_raw_features_quadrupole_backward,
    setup_context=_multipole_project_raw_features_quadrupole_setup_context,
)


class MultipoleProjectRawFeaturesQuadrupoleFunction:
    r"""Back-compat shim for the raw (un-self-subtracted) l=2 feature projection.

    The implementation is the fully-differentiable
    ``torch.ops.nvalchemiops.multipole_project_raw_features_quadrupole`` custom
    op (opaque forward + analytical backward + ``create_graph`` support, all
    compile-clean). This class only preserves the historical
    ``.apply(potential, positions, receiver_phi_hat_l2, k_vectors, cache)`` call
    signature (the dataclass ``cache`` supplies ``k_factor_proj``); new code
    should call the op directly.

    Parameters
    ----------
    potential : torch.Tensor
        Reciprocal-space potential :math:`V(k)`, shape ``(N_k, 2)``.
    positions : torch.Tensor
        Atomic coordinates, shape ``(N_atoms, 3)``.
    receiver_phi_hat_l2 : torch.Tensor
        l=2 receiver block ``[:, :, 4:9, :]`` of shape ``(N_k, N_σ, 5, 2)``;
        carries the feature cell-grad.
    k_vectors : torch.Tensor
        Reciprocal-lattice vectors, shape ``(N_k, 3)``.
    cache : MultipoleSCFCache
        Per-system direct-k-space state; supplies ``cache.k_factor_proj``.

    Returns
    -------
    torch.Tensor
        Raw l=2 features ``(N_atoms, N_σ, 5)`` float64 in natural layout.
    """

    @staticmethod
    def apply(
        potential: torch.Tensor,
        positions: torch.Tensor,
        receiver_phi_hat_l2: torch.Tensor,
        k_vectors: torch.Tensor,
        cache: MultipoleSCFCache,
    ) -> torch.Tensor:
        """Dispatch to ``multipole_project_raw_features_quadrupole``."""
        return torch.ops.nvalchemiops.multipole_project_raw_features_quadrupole(
            potential,
            positions,
            receiver_phi_hat_l2,
            k_vectors,
            cache.k_factor_proj,
        )
