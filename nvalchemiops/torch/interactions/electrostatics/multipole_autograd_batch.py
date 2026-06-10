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
Batched autograd.Function wrappers for the multipole direct-k-space pipeline
===========================================================================

Batched analog of :mod:`multipole_autograd`. Exposes two user-facing classes:

* :class:`BatchMultipoleRhoFunction` — batched :math:`\rho(k)` assembly with
  analytical backward for ``(charges, dipoles, positions)``.
* :class:`BatchMultipoleProjectRawFeaturesFunction` — batched raw-feature
  projection with analytical backward for ``(potential, positions)``.

Both support double-backward via the batched kernels in
:mod:`multipole_direct_kspace_kernels`.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics import (
    batch_assemble_rho_k_dipole,
    batch_build_structure_factor_table,
    batch_position_gradient_from_feature_grad,
    batch_position_gradient_from_rhok,
    batch_project_features_dipole,
    batch_v_gradient_from_feature_grad,
)
from nvalchemiops.interactions.electrostatics.multipole_direct_kspace_kernels import (
    batch_assemble_rho_q,
    batch_feat_position_grad_backward_grad_raw,
    batch_feat_position_grad_backward_grad_raw_quadrupole,
    batch_feat_position_grad_backward_positions,
    batch_feat_position_grad_backward_positions_quadrupole,
    batch_feat_position_grad_backward_v,
    batch_feat_position_grad_backward_v_quadrupole,
    batch_position_gradient_from_feature_grad_quadrupole,
    batch_position_gradient_from_rhoq,
    batch_project_features_quadrupole,
    batch_project_kphase_grad,
    batch_project_phihat_grad,
    batch_rho_kphase_grad,
    batch_rho_kphase_grad_double_backward,
    batch_rho_phihat_grad,
    batch_rho_phihat_grad_double_backward,
    batch_rho_q_coeff2_grad,
    batch_rho_q_coeff2_grad_double_backward,
    batch_rho_q_kvec_grad,
    batch_rho_q_kvec_grad_double_backward,
    batch_rho_q_moment_grad,
    batch_rhok_position_grad_backward_grad_rho,
    batch_rhok_position_grad_backward_moments,
    batch_rhok_position_grad_backward_positions,
    batch_rhoq_posgrad_backward_grad_rho,
    batch_rhoq_posgrad_backward_positions,
    batch_rhoq_posgrad_backward_quad,
    batch_v_grad_from_feat_grad_backward_positions,
    batch_v_grad_from_feat_grad_backward_positions_quadrupole,
    batch_v_gradient_from_feature_grad_quadrupole,
)
from nvalchemiops.torch._warp_op_helpers import (
    register_warp_op_chain,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
    MultipoleSCFCache,
)
from nvalchemiops.torch.neighbors.neighbor_utils import prepare_batch_idx_ptr

_TWO_PI_CUBED = (2.0 * math.pi) ** 3
_TWO_PI_SIXTH = (2.0 * math.pi) ** 6


# ---------------------------------------------------------------------------
# Tiny helpers — dtype-agnostic Warp wrapping
# ---------------------------------------------------------------------------


def _wp_scalar(dtype: torch.dtype):
    """Map torch float dtype → Warp scalar dtype."""
    return wp.float64 if dtype == torch.float64 else wp.float32


def _wp_vec(dtype: torch.dtype):
    """Map torch float dtype → Warp vec3 dtype."""
    return wp.vec3d if dtype == torch.float64 else wp.vec3f


def _wp_in(t: torch.Tensor, dtype=wp.float64):
    """``wp.from_torch(t.detach().contiguous(), dtype=dtype)`` shorthand."""
    return wp.from_torch(t.detach().contiguous(), dtype=dtype)


def _wp_out(t: torch.Tensor, dtype=wp.float64):
    """Wrap an output tensor (no ``.detach()``)."""
    return wp.from_torch(t.contiguous(), dtype=dtype)


# ---------------------------------------------------------------------------
# atom_start / atom_end from batch_idx
# ---------------------------------------------------------------------------


def _atom_bounds_from_batch_idx(
    batch_idx: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute per-system ``[atom_start, atom_end)`` from a sorted ``batch_idx``.

    Assumes ``batch_idx`` is sorted (atoms grouped by system), the convention
    :class:`MultipoleSCFCache` callers follow. Returns ``(B,)`` int32
    tensors on the same device. Reuses the shared CSR-pointer utility
    :func:`prepare_batch_idx_ptr`; ``atom_start``/``atom_end`` are the
    pointer's ``[:-1]``/``[1:]`` slices, padded to ``batch_size`` so trailing
    empty systems map to empty ranges.
    """
    _, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, None, batch_idx.shape[0], batch_idx.device
    )
    if batch_ptr.shape[0] < batch_size + 1:
        # Pad trailing empty systems with the final offset (no host sync).
        pad = batch_ptr[-1:].expand(batch_size + 1 - batch_ptr.shape[0])
        batch_ptr = torch.cat([batch_ptr, pad])
    return batch_ptr[:-1].contiguous(), batch_ptr[1:].contiguous()


# ---------------------------------------------------------------------------
# Batched structure-factor table
# ---------------------------------------------------------------------------


# ===========================================================================
# Low-level batched autograd.Function wrappers (internal)
# ===========================================================================


# =============================================================================
# Opaque batched rho-backward sub-op chains (l<=1)
# =============================================================================


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_structure_factor",
    mutates_args=(),
    schema="(Tensor positions, Tensor k_vectors, Tensor batch_idx) -> (Tensor, Tensor)",
)
def _batch_multipole_structure_factor_op(
    positions: torch.Tensor,
    k_vectors: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque batched ``(cos(k.r), sin(k.r))`` tables (forward-only; constants).

    Cache-free: ``K_max`` / ``N_total`` / device derived from the inputs.
    """
    device = positions.device
    wp_device = wp.device_from_torch(device)
    k_max = k_vectors.shape[1]
    n_total = positions.shape[0]
    vec_dtype = _wp_vec(positions.dtype)
    wp_scalar_pos = _wp_scalar(positions.dtype)

    cosines = torch.empty((k_max, n_total), dtype=torch.float64, device=device)
    sines = torch.empty((k_max, n_total), dtype=torch.float64, device=device)
    batch_build_structure_factor_table(
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(cosines, dtype=wp.float64),
        wp.from_torch(sines, dtype=wp.float64),
        wp_dtype=wp_scalar_pos,
        device=str(wp_device),
    )
    return cosines, sines


@torch.library.register_fake("nvalchemiops::batch_multipole_structure_factor")
def _batch_multipole_structure_factor_fake(
    positions: torch.Tensor,
    k_vectors: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shape/dtype metadata: two ``(K_max, N_total)`` float64 tables."""
    k_max = k_vectors.shape[1]
    n_total = positions.shape[0]
    return (
        positions.new_empty((k_max, n_total), dtype=torch.float64),
        positions.new_empty((k_max, n_total), dtype=torch.float64),
    )


# ---- moments: grad_rho -> (grad_charges, grad_dipoles) via project rewired ----


def _batch_rho_moment_grad_forward(
    grad_rho: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r""":func:`batch_project_features_dipole` rewired as the moments Jacobian transpose.

    Returns the per-atom moment gradient in e3nn layout ``(N_total, 4)``.
    """
    device = grad_rho.device
    wp_device = wp.device_from_torch(device)
    batch_size = source_phi_hat.shape[0]
    n_k_max = source_phi_hat.shape[1]
    n_total = cosines.shape[1]

    source_phi_5d = (
        source_phi_hat.detach().view(batch_size, n_k_max, 1, 4, 2).contiguous()
    )
    # Per-k scaling cancels the kernel's 2/(2π)³ constant and applies the
    # forward-rho scale (2π)³/V, i.e. multiply by (2π)⁶ / (2·V).
    per_k_factor_grad = (
        (_TWO_PI_SIXTH / (2.0 * volume.detach()))
        .view(batch_size, 1)
        .expand(batch_size, n_k_max)
        .contiguous()
    )
    src_lm_zero = torch.zeros((n_total, 4), dtype=torch.float64, device=device)
    oc_zero = torch.zeros((1, 2), dtype=torch.float64, device=device)
    lut = torch.arange(4, dtype=torch.int32, device=device).view(1, 4).contiguous()

    grad_flat = torch.zeros((n_total, 4), dtype=torch.float64, device=device)
    batch_project_features_dipole(
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_5d, dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(per_k_factor_grad, dtype=wp.float64),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(src_lm_zero, dtype=wp.float64),
        wp.from_torch(oc_zero, dtype=wp.float64),
        False,  # subtract_self
        wp.from_torch(lut, dtype=wp.int32),
        wp.from_torch(grad_flat, dtype=wp.float64),
        device=str(wp_device),
    )
    return grad_flat


def _batch_rho_moment_grad_forward_fake(
    grad_rho: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: moment gradient ``(N_total, 4)`` float64."""
    return cosines.new_empty((cosines.shape[1], 4), dtype=torch.float64)


def _batch_rho_moment_grad_backward(
    gg_moments: torch.Tensor,
    grad_rho: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transpose backward of the linear moments map: ``(ggrad_rho, ggrad_volume)``."""
    device = grad_rho.device
    wp_device = wp.device_from_torch(device)
    batch_size = source_phi_hat.shape[0]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)

    gg_q = gg_moments[:, 0].contiguous()
    gg_mu = gg_moments[:, [3, 1, 2]].contiguous()  # (μ_x, μ_y, μ_z)

    ggrad_rho = torch.zeros_like(grad_rho)
    batch_assemble_rho_k_dipole(
        wp.from_torch(gg_q.contiguous(), dtype=wp.float64),
        wp.from_torch(gg_mu.contiguous(), dtype=wp.vec3d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(volume.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(ggrad_rho, dtype=wp.float64),
        wp_dtype=wp.float64,
        device=str(wp_device),
    )

    # grad_flat for atoms in system b scales as 1/V_b, so
    # ∂grad_flat/∂V_b = -grad_flat / V_b (per-system sum-reduction).
    grad_flat = _batch_rho_moment_grad_forward(
        grad_rho, cosines, sines, source_phi_hat, volume, batch_idx
    )
    per_atom_dot = (gg_moments * grad_flat).sum(dim=-1)
    gg_volume = torch.zeros_like(volume)
    gg_volume.scatter_add_(0, batch_idx, per_atom_dot)
    gg_volume = -gg_volume / volume.detach()
    return ggrad_rho, gg_volume


register_warp_op_chain(
    name="nvalchemiops::batch_multipole_rho_moment_grad",
    forward=_batch_rho_moment_grad_forward,
    backward=_batch_rho_moment_grad_backward,
    diff_input_positions=(0, 4),
    n_forward_inputs=6,
    forward_fake=_batch_rho_moment_grad_forward_fake,
    batch_match=True,
)


# ---- positions: grad_rho -> grad_positions via batch_position_gradient_from_rhok ----


def _batch_rho_position_grad_forward(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    scale: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r""":func:`batch_position_gradient_from_rhok` — closed-form :math:`\partial\rho/\partial r`.

    ``positions`` is unused by the kernel (carried by the detached cos/sin
    tables) but is an explicit input so its position-Hessian second-order grad
    has a slot.
    """
    device = charges.device
    wp_device = wp.device_from_torch(device)
    n_total = charges.shape[0]
    wp_scalar = _wp_scalar(charges.dtype)
    vec_dtype = _wp_vec(charges.dtype)

    grad_positions = torch.zeros((n_total, 3), dtype=torch.float64, device=device)
    batch_position_gradient_from_rhok(
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(scale.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(grad_positions, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_positions


def _batch_rho_position_grad_forward_fake(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    scale: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: position gradient ``(N_total, 3)`` float64."""
    return positions.new_empty((positions.shape[0], 3), dtype=torch.float64)


def _batch_rho_position_grad_backward(
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
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Second-order backward: ``(ggrad_rho, ggrad_charges, ggrad_dipoles, ggrad_positions)``."""
    device = charges.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = _wp_scalar(charges.dtype)
    vec_dtype = _wp_vec(charges.dtype)
    batch_size = k_vectors.shape[0]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)

    # K3: ∂/∂grad_rho.
    ggrad_grad_rho = torch.empty_like(grad_rho)
    batch_rhok_position_grad_backward_grad_rho(
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(scale.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(ggrad_grad_rho, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )

    # K4: ∂/∂(charges, dipoles).
    ggrad_mom = torch.empty((charges.shape[0], 4), dtype=torch.float64, device=device)
    batch_rhok_position_grad_backward_moments(
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(scale.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(ggrad_mom, dtype=wp.float64),
        device=str(wp_device),
    )
    ggrad_charges = ggrad_mom[:, 0].contiguous()
    ggrad_dipoles = ggrad_mom[:, [3, 1, 2]].contiguous()

    # K5: ∂/∂positions.
    ggrad_positions = torch.zeros(
        (charges.shape[0], 3), dtype=torch.float64, device=device
    )
    batch_rhok_position_grad_backward_positions(
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(scale.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(ggrad_positions, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return ggrad_grad_rho, ggrad_charges, ggrad_dipoles, ggrad_positions


register_warp_op_chain(
    name="nvalchemiops::batch_multipole_rho_position_grad",
    forward=_batch_rho_position_grad_forward,
    backward=_batch_rho_position_grad_backward,
    diff_input_positions=(0, 1, 2, 3),
    n_forward_inputs=10,
    forward_fake=_batch_rho_position_grad_forward_fake,
)


# ---- phi_hat / k-vector phase (forward-only; carry the reciprocal cell-grad) ----


def _batch_rho_phihat_grad_forward(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r"""``∂L/∂source_phi_hat`` ``(B, K_max, 4, 2)`` via ``batch_rho_phihat_grad``.

    ``positions`` / ``k_vectors`` are unused by the forward kernel (the cos/sin
    dependence is carried by the detached tables) but are explicit diff-input
    slots so the Warp second-order backward can place their Hessian grads
    (batched reciprocal stress-loss). Mirrors the single-system
    :func:`~...multipole_autograd._rho_phihat_grad_forward`.
    """
    device = charges.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = _wp_scalar(charges.dtype)
    vec_dtype = _wp_vec(charges.dtype)
    batch_size = volume.shape[0]
    k_max = cosines.shape[0]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)

    grad_phi = torch.empty(
        (batch_size, k_max, 4, 2), dtype=torch.float64, device=device
    )
    batch_rho_phihat_grad(
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(volume.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(grad_phi, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_phi


def _batch_rho_phihat_grad_forward_fake(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(B, K_max, 4, 2)`` float64."""
    return cosines.new_empty(
        (volume.shape[0], cosines.shape[0], 4, 2), dtype=torch.float64
    )


def _batch_rho_phihat_grad_backward(
    g_phi: torch.Tensor,
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp second-order backward: grads for ``(grad_rho, charges, dipoles, positions, k_vectors)``.

    Per-system mirror of the single-system
    :func:`~...multipole_autograd._rho_phihat_grad_backward`; ``scale =
    (2π)³/volume[b]`` is folded inside the kernel.
    """
    device = charges.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = _wp_scalar(charges.dtype)
    vec_dtype = _wp_vec(charges.dtype)
    n_atoms = charges.shape[0]
    batch_size = volume.shape[0]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)

    ggrad_rho = torch.empty_like(grad_rho, dtype=torch.float64)
    ggrad_moments = torch.zeros((n_atoms, 4), dtype=torch.float64, device=device)
    ggrad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    ggrad_kvec = torch.empty(
        (batch_size, cosines.shape[0], 3), dtype=torch.float64, device=device
    )
    batch_rho_phihat_grad_double_backward(
        wp.from_torch(g_phi.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(volume.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
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


_BATCH_PHIHAT_GRAD_SCHEMA = (
    "(Tensor grad_rho, Tensor charges, Tensor dipoles, Tensor positions, "
    "Tensor cosines, Tensor sines, Tensor k_vectors, Tensor volume, "
    "Tensor batch_idx) -> Tensor"
)
_BATCH_PHIHAT_GRAD_BWD_SCHEMA = (
    "(Tensor g_phi, Tensor grad_rho, Tensor charges, Tensor dipoles, "
    "Tensor positions, Tensor cosines, Tensor sines, Tensor k_vectors, "
    "Tensor volume, Tensor batch_idx) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)
register_warp_op_chain(
    name="nvalchemiops::batch_multipole_rho_phihat_grad",
    forward=_batch_rho_phihat_grad_forward,
    forward_schema=_BATCH_PHIHAT_GRAD_SCHEMA,
    forward_fake=_batch_rho_phihat_grad_forward_fake,
    backward=_batch_rho_phihat_grad_backward,
    backward_schema=_BATCH_PHIHAT_GRAD_BWD_SCHEMA,
    backward_return_arity=5,
    # grad_rho(0), charges(1), dipoles(2), positions(3), k_vectors(6) are diff.
    diff_input_positions=(0, 1, 2, 3, 6),
    n_forward_inputs=9,
)


def _batch_rho_kphase_grad_forward(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r"""``∂L/∂k_vectors`` ``(B, K_max, 3)`` (phase channel) via ``batch_rho_kphase_grad``.

    ``k_vectors`` is an explicit diff slot (kernel-unused in forward; the cos/sin
    dependence is carried by the detached tables) so the Warp second-order
    backward can place its Hessian grad. Mirrors the single-system
    :func:`~...multipole_autograd._rho_kphase_grad_forward`.
    """
    device = charges.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = _wp_scalar(charges.dtype)
    vec_dtype = _wp_vec(charges.dtype)
    batch_size = volume.shape[0]
    k_max = cosines.shape[0]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)

    grad_k = torch.empty((batch_size, k_max, 3), dtype=torch.float64, device=device)
    batch_rho_kphase_grad(
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(volume.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(grad_k, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_k


def _batch_rho_kphase_grad_forward_fake(
    grad_rho: torch.Tensor,
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(B, K_max, 3)`` float64."""
    return cosines.new_empty(
        (volume.shape[0], cosines.shape[0], 3), dtype=torch.float64
    )


def _batch_rho_kphase_grad_backward(
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
    batch_idx: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Warp second-order backward: grads for ``(grad_rho, charges, dipoles, positions, source_phi_hat, k_vectors)``.

    Per-system mirror of the single-system
    :func:`~...multipole_autograd._rho_kphase_grad_backward`.
    """
    device = charges.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = _wp_scalar(charges.dtype)
    vec_dtype = _wp_vec(charges.dtype)
    n_atoms = charges.shape[0]
    batch_size = volume.shape[0]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)

    ggrad_rho = torch.empty_like(grad_rho, dtype=torch.float64)
    ggrad_moments = torch.zeros((n_atoms, 4), dtype=torch.float64, device=device)
    ggrad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    ggrad_phi = torch.empty_like(source_phi_hat, dtype=torch.float64)
    ggrad_kvec = torch.empty(
        (batch_size, cosines.shape[0], 3), dtype=torch.float64, device=device
    )
    batch_rho_kphase_grad_double_backward(
        wp.from_torch(g_k.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(volume.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
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


_BATCH_KPHASE_GRAD_SCHEMA = (
    "(Tensor grad_rho, Tensor charges, Tensor dipoles, Tensor positions, "
    "Tensor cosines, Tensor sines, Tensor source_phi_hat, Tensor k_vectors, "
    "Tensor volume, Tensor batch_idx) -> Tensor"
)
_BATCH_KPHASE_GRAD_BWD_SCHEMA = (
    "(Tensor g_k, Tensor grad_rho, Tensor charges, Tensor dipoles, "
    "Tensor positions, Tensor cosines, Tensor sines, Tensor source_phi_hat, "
    "Tensor k_vectors, Tensor volume, Tensor batch_idx) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)
register_warp_op_chain(
    name="nvalchemiops::batch_multipole_rho_kphase_grad",
    forward=_batch_rho_kphase_grad_forward,
    forward_schema=_BATCH_KPHASE_GRAD_SCHEMA,
    forward_fake=_batch_rho_kphase_grad_forward_fake,
    backward=_batch_rho_kphase_grad_backward,
    backward_schema=_BATCH_KPHASE_GRAD_BWD_SCHEMA,
    backward_return_arity=6,
    # grad_rho(0), charges(1), dipoles(2), positions(3), source_phi_hat(6),
    # k_vectors(7) are the diff inputs.
    diff_input_positions=(0, 1, 2, 3, 6, 7),
    n_forward_inputs=10,
)


# =============================================================================
# Batched opaque feature-projection sub-op chains (l<=1)
# =============================================================================


# ---- V-grad chain: grad_raw -> grad_V (register_warp_op_chain) ----


def _batch_feature_v_grad_forward(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r""":func:`batch_v_gradient_from_feature_grad` — ``∂L/∂V(k)`` ``(B, K_max, 2)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    batch_size = receiver_phi_hat.shape[0]
    k_max = receiver_phi_hat.shape[1]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)

    grad_v = torch.zeros((batch_size, k_max, 2), dtype=torch.float64, device=device)
    batch_v_gradient_from_feature_grad(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(grad_v, dtype=wp.float64),
        device=str(wp_device),
    )
    return grad_v


def _batch_feature_v_grad_forward_fake(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(B, K_max, 2)`` float64."""
    return receiver_phi_hat.new_empty(
        (receiver_phi_hat.shape[0], receiver_phi_hat.shape[1], 2), dtype=torch.float64
    )


def _batch_feature_v_grad_backward(
    gg_v: torch.Tensor,
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Second-order backward: ``(ggrad_raw, ggrad_kfp, ggrad_positions)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_total = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[2]

    # ggrad_raw via batch_project_features_dipole on V=gg_v (transpose).
    src_lm_zero = torch.zeros((n_total, 4), dtype=torch.float64, device=device)
    oc_zero = torch.zeros((n_sigma, 2), dtype=torch.float64, device=device)
    lut = (
        torch.arange(n_sigma * 4, dtype=torch.int32, device=device)
        .view(n_sigma, 4)
        .contiguous()
    )
    ggrad_raw_flat = torch.zeros(
        (n_total, n_sigma * 4), dtype=torch.float64, device=device
    )
    batch_project_features_dipole(
        wp.from_torch(gg_v.contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(src_lm_zero, dtype=wp.float64),
        wp.from_torch(oc_zero, dtype=wp.float64),
        False,  # subtract_self
        wp.from_torch(lut, dtype=wp.int32),
        wp.from_torch(ggrad_raw_flat, dtype=wp.float64),
        device=str(wp_device),
    )
    ggrad_raw = ggrad_raw_flat.reshape(n_total, n_sigma, 4)

    # ggrad_kfp: grad_v[b,k,:] linear in kfp[b,k], so ∂/∂kfp = grad_v/kfp.
    grad_v = _batch_feature_v_grad_forward(
        grad_raw,
        receiver_phi_hat,
        cosines,
        sines,
        k_factor_proj,
        k_vectors,
        positions,
        batch_idx,
    )
    safe_kfp = torch.where(
        k_factor_proj != 0, k_factor_proj, torch.ones_like(k_factor_proj)
    )
    per_k = (gg_v * grad_v).sum(dim=-1) / safe_kfp
    ggrad_kfp = torch.where(k_factor_proj != 0, per_k, torch.zeros_like(per_k))

    ggrad_positions = torch.zeros((n_total, 3), dtype=torch.float64, device=device)
    batch_v_grad_from_feat_grad_backward_positions(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_v.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(ggrad_positions, dtype=wp.float64),
        device=str(wp_device),
    )
    return ggrad_raw, ggrad_kfp, ggrad_positions


register_warp_op_chain(
    name="nvalchemiops::batch_multipole_feature_v_grad",
    forward=_batch_feature_v_grad_forward,
    backward=_batch_feature_v_grad_backward,
    diff_input_positions=(0, 4, 6),
    n_forward_inputs=8,
    forward_fake=_batch_feature_v_grad_forward_fake,
    batch_match=True,
)


# ---- position-grad chain: grad_raw -> grad_positions (register_warp_op_chain) ----


def _batch_feature_position_grad_forward(
    grad_raw: torch.Tensor,
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r""":func:`batch_position_gradient_from_feature_grad` — ``∂L/∂r`` ``(N_total, 3)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_total = cosines.shape[1]
    grad_positions = torch.zeros((n_total, 3), dtype=torch.float64, device=device)
    batch_position_gradient_from_feature_grad(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(grad_positions, dtype=wp.float64),
        device=str(wp_device),
    )
    return grad_positions


def _batch_feature_position_grad_forward_fake(
    grad_raw: torch.Tensor,
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: position gradient ``(N_total, 3)`` float64."""
    return positions.new_empty((positions.shape[0], 3), dtype=torch.float64)


def _batch_feature_position_grad_backward(
    gg_positions: torch.Tensor,
    grad_raw: torch.Tensor,
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Second-order backward: ``(ggrad_raw, ggrad_v, ggrad_positions)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    batch_size = receiver_phi_hat.shape[0]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)
    n_total = cosines.shape[1]

    # K6: ∂/∂grad_raw.
    ggrad_raw = torch.empty_like(grad_raw)
    batch_feat_position_grad_backward_grad_raw(
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(ggrad_raw, dtype=wp.float64),
        device=str(wp_device),
    )

    # K7: ∂/∂V(k).
    ggrad_v = torch.empty_like(potential)
    batch_feat_position_grad_backward_v(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(ggrad_v, dtype=wp.float64),
        device=str(wp_device),
    )

    # K8: ∂/∂positions.
    ggrad_positions = torch.zeros((n_total, 3), dtype=torch.float64, device=device)
    batch_feat_position_grad_backward_positions(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(ggrad_positions, dtype=wp.float64),
        device=str(wp_device),
    )
    ggrad_positions = ggrad_positions.to(positions.dtype)
    return ggrad_raw, ggrad_v, ggrad_positions


register_warp_op_chain(
    name="nvalchemiops::batch_multipole_feature_position_grad",
    forward=_batch_feature_position_grad_forward,
    backward=_batch_feature_position_grad_backward,
    diff_input_positions=(0, 1, 7),
    n_forward_inputs=9,
    forward_fake=_batch_feature_position_grad_forward_fake,
    batch_match=True,
)


# ===========================================================================
# Batched l=2 feature double-backward sub-Functions (force-loss)
# ===========================================================================


# ---- l=2 V-grad chain: grad_raw -> grad_V (register_warp_op_chain) ----


def _batch_feature_v_grad_quadrupole_forward(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r""":func:`batch_v_gradient_from_feature_grad_quadrupole` — ``∂L/∂V(k)`` ``(B, K_max, 2)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    batch_size = receiver_phi_hat.shape[0]
    k_max = receiver_phi_hat.shape[1]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)
    grad_v = torch.zeros((batch_size, k_max, 2), dtype=torch.float64, device=device)
    batch_v_gradient_from_feature_grad_quadrupole(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(grad_v, dtype=wp.float64),
        device=str(wp_device),
    )
    return grad_v


def _batch_feature_v_grad_quadrupole_forward_fake(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(B, K_max, 2)`` float64."""
    return receiver_phi_hat.new_empty(
        (receiver_phi_hat.shape[0], receiver_phi_hat.shape[1], 2), dtype=torch.float64
    )


def _batch_feature_v_grad_quadrupole_backward(
    gg_v: torch.Tensor,
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Second-order backward: ``(ggrad_raw, ggrad_positions)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_total = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[2]

    # ggrad_raw: transpose via batch_project_features_quadrupole with V = gg_v.
    ggrad_raw = torch.zeros((n_total, n_sigma, 5), dtype=torch.float64, device=device)
    batch_project_features_quadrupole(
        wp.from_torch(gg_v.contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(ggrad_raw, dtype=wp.float64),
        device=str(wp_device),
    )

    ggrad_positions = torch.zeros((n_total, 3), dtype=torch.float64, device=device)
    batch_v_grad_from_feat_grad_backward_positions_quadrupole(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_v.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(ggrad_positions, dtype=wp.float64),
        device=str(wp_device),
    )
    ggrad_positions = ggrad_positions.to(positions.dtype)
    return ggrad_raw, ggrad_positions


register_warp_op_chain(
    name="nvalchemiops::batch_multipole_feature_v_grad_quadrupole",
    forward=_batch_feature_v_grad_quadrupole_forward,
    backward=_batch_feature_v_grad_quadrupole_backward,
    diff_input_positions=(0, 6),
    n_forward_inputs=8,
    forward_fake=_batch_feature_v_grad_quadrupole_forward_fake,
    batch_match=True,
)


# ---- l=2 position-grad chain: grad_raw -> grad_positions (register_warp_op_chain) ----


def _batch_feature_position_grad_quadrupole_forward(
    grad_raw: torch.Tensor,
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r""":func:`batch_position_gradient_from_feature_grad_quadrupole` — ``∂L/∂r`` ``(N_total, 3)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    n_total = cosines.shape[1]
    grad_positions = torch.zeros((n_total, 3), dtype=torch.float64, device=device)
    batch_position_gradient_from_feature_grad_quadrupole(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(grad_positions, dtype=wp.float64),
        device=str(wp_device),
    )
    return grad_positions


def _batch_feature_position_grad_quadrupole_forward_fake(
    grad_raw: torch.Tensor,
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: position gradient ``(N_total, 3)`` float64."""
    return positions.new_empty((positions.shape[0], 3), dtype=torch.float64)


def _batch_feature_position_grad_quadrupole_backward(
    gg_positions: torch.Tensor,
    grad_raw: torch.Tensor,
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    k_vectors: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Second-order backward: ``(ggrad_raw, ggrad_v, ggrad_positions)``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    batch_size = receiver_phi_hat.shape[0]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)
    n_total = cosines.shape[1]

    ggrad_raw = torch.empty_like(grad_raw)
    batch_feat_position_grad_backward_grad_raw_quadrupole(
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(ggrad_raw, dtype=wp.float64),
        device=str(wp_device),
    )

    ggrad_v = torch.empty_like(potential)
    batch_feat_position_grad_backward_v_quadrupole(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(ggrad_v, dtype=wp.float64),
        device=str(wp_device),
    )

    ggrad_positions = torch.zeros((n_total, 3), dtype=torch.float64, device=device)
    batch_feat_position_grad_backward_positions_quadrupole(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(gg_positions.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(ggrad_positions, dtype=wp.float64),
        device=str(wp_device),
    )
    ggrad_positions = ggrad_positions.to(positions.dtype)
    return ggrad_raw, ggrad_v, ggrad_positions


register_warp_op_chain(
    name="nvalchemiops::batch_multipole_feature_position_grad_quadrupole",
    forward=_batch_feature_position_grad_quadrupole_forward,
    backward=_batch_feature_position_grad_quadrupole_backward,
    diff_input_positions=(0, 1, 7),
    n_forward_inputs=9,
    forward_fake=_batch_feature_position_grad_quadrupole_forward_fake,
    batch_match=True,
)


# ===========================================================================
# Public batched autograd.Function wrappers
# ===========================================================================


# =============================================================================
# torch.library.custom_op chain for the batched rho(k) assembly (l<=1)
# =============================================================================


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_rho",
    mutates_args=(),
    schema=(
        "(Tensor charges, Tensor dipoles, Tensor positions, "
        "Tensor source_phi_hat, Tensor k_vectors, Tensor volume, "
        "Tensor batch_idx) -> Tensor"
    ),
)
def _batch_multipole_rho_op(
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Opaque forward: build the structure-factor tables and assemble rho(b, k)."""
    device = charges.device
    wp_device = wp.device_from_torch(device)
    batch_size = volume.shape[0]
    k_max = k_vectors.shape[1]
    n_total = positions.shape[0]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)

    vec_dtype = _wp_vec(positions.dtype)
    wp_scalar_pos = _wp_scalar(positions.dtype)
    cosines = torch.empty((k_max, n_total), dtype=torch.float64, device=device)
    sines = torch.empty((k_max, n_total), dtype=torch.float64, device=device)
    batch_build_structure_factor_table(
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(cosines, dtype=wp.float64),
        wp.from_torch(sines, dtype=wp.float64),
        wp_dtype=wp_scalar_pos,
        device=str(wp_device),
    )

    wp_scalar = _wp_scalar(charges.dtype)
    vec_dtype_c = _wp_vec(charges.dtype)
    rho = torch.zeros((batch_size, k_max, 2), dtype=torch.float64, device=device)
    batch_assemble_rho_k_dipole(
        wp.from_torch(charges.detach().contiguous(), dtype=wp_scalar),
        wp.from_torch(dipoles.detach().contiguous(), dtype=vec_dtype_c),
        wp.from_torch(cosines.contiguous(), dtype=wp.float64),
        wp.from_torch(sines.contiguous(), dtype=wp.float64),
        wp.from_torch(source_phi_hat.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(volume.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(rho, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return rho


@torch.library.register_fake("nvalchemiops::batch_multipole_rho")
def _batch_multipole_rho_fake(
    charges: torch.Tensor,
    dipoles: torch.Tensor,
    positions: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: rho is ``(B, K_max, 2)`` float64."""
    return k_vectors.new_empty(
        (volume.shape[0], k_vectors.shape[1], 2), dtype=torch.float64
    )


def _batch_multipole_rho_setup_context(ctx, inputs, output) -> None:
    """Save the forward inputs for the analytical backward."""
    charges, dipoles, positions, source_phi_hat, k_vectors, volume, batch_idx = inputs
    ctx.save_for_backward(
        charges, dipoles, positions, source_phi_hat, k_vectors, volume, batch_idx
    )


def _batch_multipole_rho_backward(ctx, grad_rho: torch.Tensor):
    """Analytical grads for charges, dipoles, positions, phi_hat, k_vectors.

    Recomputes (cos, sin) via the opaque batched structure-factor op (on
    detached positions/k_vectors — cos/sin are constants; the position
    second-order is carried by the position-grad chain) and dispatches the four
    backward sub-ops. Every call is an opaque op or plain torch, so AOTAutograd
    traces this without touching Warp; ``volume`` / ``batch_idx`` get ``None``.
    """
    charges, dipoles, positions, source_phi_hat, k_vectors, volume, batch_idx = (
        ctx.saved_tensors
    )
    grad_rho = grad_rho.contiguous()
    cosines, sines = torch.ops.nvalchemiops.batch_multipole_structure_factor(
        positions.detach(), k_vectors.detach(), batch_idx
    )
    grad_flat = torch.ops.nvalchemiops.batch_multipole_rho_moment_grad(
        grad_rho, cosines, sines, source_phi_hat, volume, batch_idx
    )  # (N_total, 4) in e3nn layout
    grad_charges = grad_flat[:, 0].contiguous()
    grad_dipoles = grad_flat[:, [3, 1, 2]].contiguous()

    scale = (
        torch.as_tensor(_TWO_PI_CUBED, dtype=torch.float64, device=positions.device)
        / volume
    )
    grad_positions = torch.ops.nvalchemiops.batch_multipole_rho_position_grad(
        grad_rho,
        charges,
        dipoles,
        positions,
        cosines,
        sines,
        source_phi_hat,
        k_vectors,
        scale,
        batch_idx,
    )
    # phi_hat / k-vector phase grads (carry the reciprocal cell-grad). Both are
    # twice-differentiable Warp op chains (batched second-order backward) so the
    # cell<->{positions, moments} cross-terms flow under create_graph for
    # stress-loss while plain 1st-order cell-grad stays on the fast forward.
    grad_phi = torch.ops.nvalchemiops.batch_multipole_rho_phihat_grad(
        grad_rho,
        charges,
        dipoles,
        positions,
        cosines,
        sines,
        k_vectors,
        volume,
        batch_idx,
    )
    grad_kvec = torch.ops.nvalchemiops.batch_multipole_rho_kphase_grad(
        grad_rho,
        charges,
        dipoles,
        positions,
        cosines,
        sines,
        source_phi_hat,
        k_vectors,
        volume,
        batch_idx,
    )
    grad_charges = grad_charges.to(charges.dtype)
    grad_dipoles = grad_dipoles.to(dipoles.dtype)
    grad_positions = grad_positions.to(positions.dtype)
    # Slots: (charges, dipoles, positions, source_phi_hat, k_vectors, volume,
    # batch_idx).
    return grad_charges, grad_dipoles, grad_positions, grad_phi, grad_kvec, None, None


torch.library.register_autograd(
    "nvalchemiops::batch_multipole_rho",
    _batch_multipole_rho_backward,
    setup_context=_batch_multipole_rho_setup_context,
)


class BatchMultipoleRhoFunction:
    r"""Back-compat shim for the batched :math:`\rho(k)` direct-k-space assembly.

    The implementation is the fully-differentiable
    ``torch.ops.nvalchemiops.batch_multipole_rho`` custom op (opaque forward +
    analytical backward + ``create_graph`` support, all compile-clean). This
    class only preserves the historical ``.apply(charges, dipoles, positions,
    source_phi_hat, k_vectors, batch_idx, cache)`` signature (the dataclass
    ``cache`` supplies ``volume``); new code should call the op directly.

    Parameters
    ----------
    charges, dipoles, positions : torch.Tensor
        Per-atom moments and coordinates.
    source_phi_hat, k_vectors : torch.Tensor
        Differentiable cache slots; values match ``cache``.
    batch_idx : torch.Tensor
        Per-atom system index.
    cache : MultipoleSCFCache
        Per-system state; supplies ``cache.volume``.

    Returns
    -------
    torch.Tensor
        :math:`\rho(k)` of shape ``(B, K_max, 2)`` float64.
    """

    @staticmethod
    def apply(
        charges: torch.Tensor,
        dipoles: torch.Tensor,
        positions: torch.Tensor,
        source_phi_hat: torch.Tensor,
        k_vectors: torch.Tensor,
        batch_idx: torch.Tensor,
        cache: MultipoleSCFCache,
    ) -> torch.Tensor:
        """Dispatch to ``torch.ops.nvalchemiops.batch_multipole_rho``."""
        return torch.ops.nvalchemiops.batch_multipole_rho(
            charges,
            dipoles,
            positions,
            source_phi_hat,
            k_vectors,
            cache.volume,
            batch_idx,
        )


# =============================================================================
# Batched Cartesian-quadrupole (l=2) ρ_Q(k) contribution
# =============================================================================


# ---- Q-channel moment grad: grad_rho -> grad_Q (register_warp_op_chain) ----


def _batch_rho_q_moment_grad_forward(
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r""":func:`batch_rho_q_moment_grad` — ``dL/dQ_i`` ``(N_total, 3, 3)`` symmetric.

    ``positions`` is unused by the kernel but is an explicit input so its
    second-order grad (from the transpose backward) has a slot.
    """
    device = grad_rho.device
    wp_device = wp.device_from_torch(device)
    n_total = cosines.shape[1]
    scale = (
        torch.as_tensor(_TWO_PI_CUBED, dtype=torch.float64, device=device)
        / volume.detach()
    )
    grad_q = torch.empty((n_total, 9), dtype=torch.float64, device=device)
    batch_rho_q_moment_grad(
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(scale.contiguous(), dtype=wp.float64),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(grad_q, dtype=wp.float64),
        device=str(wp_device),
    )
    return grad_q.reshape(n_total, 3, 3)


def _batch_rho_q_moment_grad_forward_fake(
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_total, 3, 3)`` float64."""
    return cosines.new_empty((cosines.shape[1], 3, 3), dtype=torch.float64)


def _batch_rho_q_moment_grad_backward(
    gg_q: torch.Tensor,
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transpose backward of the linear moment map: ``(ggrad_rho, ggrad_positions)``."""
    device = grad_rho.device
    wp_device = wp.device_from_torch(device)
    n_total = cosines.shape[1]
    batch_size = volume.shape[0]
    k_max = k_vectors.shape[1]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)
    scale = (
        torch.as_tensor(_TWO_PI_CUBED, dtype=torch.float64, device=device)
        / volume.detach()
    )
    gg_q_sym = (0.5 * (gg_q + gg_q.transpose(-1, -2))).contiguous()

    # ∂L/∂grad_rho == batch_assemble_rho_q(gg_Q), shape (B, K_max, 2).
    ggrad_rho = torch.zeros((batch_size, k_max, 2), dtype=torch.float64, device=device)
    batch_assemble_rho_q(
        wp.from_torch(gg_q_sym, dtype=wp.mat33d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(volume.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(ggrad_rho, dtype=wp.float64),
        wp_dtype=wp.float64,
        device=str(wp_device),
    )

    # ∂L/∂positions == batch_position_gradient_from_rhoq(grad_rho, gg_Q).
    ggrad_pos = torch.empty((n_total, 3), dtype=torch.float64, device=device)
    batch_position_gradient_from_rhoq(
        wp.from_torch(gg_q_sym, dtype=wp.mat33d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(scale.contiguous(), dtype=wp.float64),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(ggrad_pos, dtype=wp.float64),
        wp_dtype=wp.float64,
        device=str(wp_device),
    )
    return ggrad_rho, ggrad_pos


register_warp_op_chain(
    name="nvalchemiops::batch_multipole_rho_q_moment_grad",
    forward=_batch_rho_q_moment_grad_forward,
    backward=_batch_rho_q_moment_grad_backward,
    diff_input_positions=(0, 1),
    n_forward_inputs=8,
    forward_fake=_batch_rho_q_moment_grad_forward_fake,
    batch_match=True,
)


# ---- Q-channel position grad: grad_rho -> grad_positions (register_warp_op_chain) ----


def _batch_rho_q_position_grad_forward(
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r""":func:`batch_position_gradient_from_rhoq` — Q-channel ``dL/dr_i`` ``(N_total, 3)``.

    ``positions`` is unused by the kernel (carried by detached cos/sin) but is
    an explicit input so its position-Hessian second-order grad has a slot.
    """
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    n_total = cosines.shape[1]
    wp_scalar = _wp_scalar(quadrupoles.dtype)
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    scale = (
        torch.as_tensor(_TWO_PI_CUBED, dtype=torch.float64, device=device)
        / volume.detach()
    )
    grad_pos = torch.empty((n_total, 3), dtype=torch.float64, device=device)
    batch_position_gradient_from_rhoq(
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(scale.contiguous(), dtype=wp.float64),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(grad_pos, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_pos


def _batch_rho_q_position_grad_forward_fake(
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_total, 3)`` float64."""
    return positions.new_empty((positions.shape[0], 3), dtype=torch.float64)


def _batch_rho_q_position_grad_backward(
    gg_pos: torch.Tensor,
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Second-order backward: ``(ggrad_rho, ggrad_quadrupoles, ggrad_positions)``."""
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    n_total = cosines.shape[1]
    batch_size = volume.shape[0]
    k_max = k_vectors.shape[1]
    wp_scalar = _wp_scalar(quadrupoles.dtype)
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)
    scale = (
        torch.as_tensor(_TWO_PI_CUBED, dtype=torch.float64, device=device)
        / volume.detach()
    )
    bidx32 = batch_idx.contiguous().to(torch.int32)
    gg_pos_c = gg_pos.detach().to(torch.float64).contiguous()

    # K_a: ∂L/∂grad_rho (B, K_max, 2).
    ggrad_rho = torch.zeros((batch_size, k_max, 2), dtype=torch.float64, device=device)
    batch_rhoq_posgrad_backward_grad_rho(
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(gg_pos_c, dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(scale.contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(ggrad_rho, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )

    # K_b: ∂L/∂quadrupoles (N_total, 3, 3) symmetric.
    ggrad_q = torch.empty((n_total, 9), dtype=torch.float64, device=device)
    batch_rhoq_posgrad_backward_quad(
        wp.from_torch(gg_pos_c, dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(scale.contiguous(), dtype=wp.float64),
        wp.from_torch(bidx32, dtype=wp.int32),
        wp.from_torch(ggrad_q, dtype=wp.float64),
        device=str(wp_device),
    )
    ggrad_q = ggrad_q.reshape(n_total, 3, 3)

    # K_c: ∂L/∂positions (N_total, 3) — Hessian diagonal.
    ggrad_pos = torch.empty((n_total, 3), dtype=torch.float64, device=device)
    batch_rhoq_posgrad_backward_positions(
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(gg_pos_c, dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(scale.contiguous(), dtype=wp.float64),
        wp.from_torch(bidx32, dtype=wp.int32),
        wp.from_torch(ggrad_pos, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return ggrad_rho, ggrad_q, ggrad_pos


register_warp_op_chain(
    name="nvalchemiops::batch_multipole_rho_q_position_grad",
    forward=_batch_rho_q_position_grad_forward,
    backward=_batch_rho_q_position_grad_backward,
    diff_input_positions=(0, 1, 2),
    n_forward_inputs=9,
    forward_fake=_batch_rho_q_position_grad_forward_fake,
)


# ---- coeff2 / k-vector phase (forward-only; carry the l=2 reciprocal cell-grad) ----


def _batch_rho_q_coeff2_grad_forward(
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r"""``∂L/∂source_coeff2`` ``(B, K_max)`` via ``batch_rho_q_coeff2_grad`` (l=2 stress).

    ``positions`` is an explicit diff slot (kernel-unused in forward) so the
    Warp second-order backward can place its Hessian grad. Mirrors the
    single-system :func:`~...multipole_autograd._rho_q_coeff2_grad_forward`.
    """
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = _wp_scalar(quadrupoles.dtype)
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    batch_size = volume.shape[0]
    k_max = k_vectors.shape[1]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)

    grad_c2 = torch.empty((batch_size, k_max), dtype=torch.float64, device=device)
    batch_rho_q_coeff2_grad(
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(volume.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(grad_c2, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_c2


def _batch_rho_q_coeff2_grad_forward_fake(
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(B, K_max)`` float64."""
    return cosines.new_empty((volume.shape[0], k_vectors.shape[1]), dtype=torch.float64)


def _batch_rho_q_coeff2_grad_backward(
    g_c: torch.Tensor,
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp second-order backward: grads for ``(grad_rho, quadrupoles, positions, k_vectors)``.

    Per-system mirror of the single-system
    :func:`~...multipole_autograd._rho_q_coeff2_grad_backward`.
    """
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = _wp_scalar(quadrupoles.dtype)
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    vec_dtype = _wp_vec(quadrupoles.dtype)
    n_atoms = quadrupoles.shape[0]
    batch_size = volume.shape[0]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)

    ggrad_rho = torch.empty_like(grad_rho, dtype=torch.float64)
    ggrad_quad = torch.zeros((n_atoms, 9), dtype=torch.float64, device=device)
    ggrad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    ggrad_kvec = torch.empty(
        (batch_size, k_vectors.shape[1], 3), dtype=torch.float64, device=device
    )
    batch_rho_q_coeff2_grad_double_backward(
        wp.from_torch(g_c.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(volume.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(ggrad_rho, dtype=wp.float64),
        wp.from_torch(ggrad_quad, dtype=wp.float64),
        wp.from_torch(ggrad_positions, dtype=wp.vec3d),
        wp.from_torch(ggrad_kvec, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    ggrad_quad = ggrad_quad.view(n_atoms, 3, 3).to(quadrupoles.dtype)
    return ggrad_rho, ggrad_quad, ggrad_positions, ggrad_kvec


_BATCH_COEFF2_GRAD_SCHEMA = (
    "(Tensor grad_rho, Tensor quadrupoles, Tensor positions, Tensor cosines, "
    "Tensor sines, Tensor k_vectors, Tensor volume, Tensor batch_idx) -> Tensor"
)
_BATCH_COEFF2_GRAD_BWD_SCHEMA = (
    "(Tensor g_c, Tensor grad_rho, Tensor quadrupoles, Tensor positions, "
    "Tensor cosines, Tensor sines, Tensor k_vectors, Tensor volume, "
    "Tensor batch_idx) -> (Tensor, Tensor, Tensor, Tensor)"
)
register_warp_op_chain(
    name="nvalchemiops::batch_multipole_rho_q_coeff2_grad",
    forward=_batch_rho_q_coeff2_grad_forward,
    forward_schema=_BATCH_COEFF2_GRAD_SCHEMA,
    forward_fake=_batch_rho_q_coeff2_grad_forward_fake,
    backward=_batch_rho_q_coeff2_grad_backward,
    backward_schema=_BATCH_COEFF2_GRAD_BWD_SCHEMA,
    backward_return_arity=4,
    # grad_rho(0), quadrupoles(1), positions(2), k_vectors(5) are diff.
    diff_input_positions=(0, 1, 2, 5),
    n_forward_inputs=8,
)


def _batch_rho_q_kvec_grad_forward(
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """``∂L/∂k_vectors`` ``(B, K_max, 3)`` via the (k·Q·k) form + phase (l=2 stress)."""
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = _wp_scalar(quadrupoles.dtype)
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    vec_dtype = _wp_vec(quadrupoles.dtype)
    batch_size = volume.shape[0]
    k_max = k_vectors.shape[1]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)

    grad_k = torch.empty((batch_size, k_max, 3), dtype=torch.float64, device=device)
    batch_rho_q_kvec_grad(
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(volume.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(grad_k, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return grad_k


def _batch_rho_q_kvec_grad_forward_fake(
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(B, K_max, 3)`` float64."""
    return cosines.new_empty(
        (volume.shape[0], k_vectors.shape[1], 3), dtype=torch.float64
    )


def _batch_rho_q_kvec_grad_backward(
    g_k: torch.Tensor,
    grad_rho: torch.Tensor,
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_vectors: torch.Tensor,
    source_coeff2: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp second-order backward: grads for ``(grad_rho, quadrupoles, positions, k_vectors, source_coeff2)``.

    Per-system mirror of the single-system
    :func:`~...multipole_autograd._rho_q_kvec_grad_backward`.
    """
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    wp_scalar = _wp_scalar(quadrupoles.dtype)
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    vec_dtype = _wp_vec(quadrupoles.dtype)
    n_atoms = quadrupoles.shape[0]
    batch_size = volume.shape[0]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)

    ggrad_rho = torch.empty_like(grad_rho, dtype=torch.float64)
    ggrad_quad = torch.zeros((n_atoms, 9), dtype=torch.float64, device=device)
    ggrad_positions = torch.zeros((n_atoms, 3), dtype=torch.float64, device=device)
    ggrad_coeff2 = torch.empty_like(source_coeff2, dtype=torch.float64)
    ggrad_kvec = torch.empty(
        (batch_size, k_vectors.shape[1], 3), dtype=torch.float64, device=device
    )
    batch_rho_q_kvec_grad_double_backward(
        wp.from_torch(g_k.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(grad_rho.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(volume.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
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


_BATCH_KVECQ_GRAD_SCHEMA = (
    "(Tensor grad_rho, Tensor quadrupoles, Tensor positions, Tensor cosines, "
    "Tensor sines, Tensor k_vectors, Tensor source_coeff2, Tensor volume, "
    "Tensor batch_idx) -> Tensor"
)
_BATCH_KVECQ_GRAD_BWD_SCHEMA = (
    "(Tensor g_k, Tensor grad_rho, Tensor quadrupoles, Tensor positions, "
    "Tensor cosines, Tensor sines, Tensor k_vectors, Tensor source_coeff2, "
    "Tensor volume, Tensor batch_idx) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)
register_warp_op_chain(
    name="nvalchemiops::batch_multipole_rho_q_kvec_grad",
    forward=_batch_rho_q_kvec_grad_forward,
    forward_schema=_BATCH_KVECQ_GRAD_SCHEMA,
    forward_fake=_batch_rho_q_kvec_grad_forward_fake,
    backward=_batch_rho_q_kvec_grad_backward,
    backward_schema=_BATCH_KVECQ_GRAD_BWD_SCHEMA,
    backward_return_arity=5,
    # grad_rho(0), quadrupoles(1), positions(2), k_vectors(5), source_coeff2(6).
    diff_input_positions=(0, 1, 2, 5, 6),
    n_forward_inputs=9,
)


# ---- main forward op + register_autograd + shim ----


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_rho_q",
    mutates_args=(),
    schema=(
        "(Tensor quadrupoles, Tensor positions, Tensor source_coeff2, "
        "Tensor k_vectors, Tensor volume, Tensor batch_idx) -> Tensor"
    ),
)
def _batch_multipole_rho_q_op(
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    source_coeff2: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Opaque batched additive Cartesian-quadrupole :math:`\\rho_Q(b, k)`, ``(B, K_max, 2)``."""
    device = quadrupoles.device
    wp_device = wp.device_from_torch(device)
    batch_size = volume.shape[0]
    k_max = k_vectors.shape[1]
    n_total = positions.shape[0]
    atom_start, atom_end = _atom_bounds_from_batch_idx(batch_idx, batch_size)

    vec_dtype = _wp_vec(positions.dtype)
    wp_scalar_pos = _wp_scalar(positions.dtype)
    cosines = torch.empty((k_max, n_total), dtype=torch.float64, device=device)
    sines = torch.empty((k_max, n_total), dtype=torch.float64, device=device)
    batch_build_structure_factor_table(
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(positions.detach().contiguous(), dtype=vec_dtype),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(cosines, dtype=wp.float64),
        wp.from_torch(sines, dtype=wp.float64),
        wp_dtype=wp_scalar_pos,
        device=str(wp_device),
    )

    wp_scalar = _wp_scalar(quadrupoles.dtype)
    mat_dtype = wp.mat33d if wp_scalar == wp.float64 else wp.mat33f
    rho_q = torch.zeros((batch_size, k_max, 2), dtype=torch.float64, device=device)
    batch_assemble_rho_q(
        wp.from_torch(quadrupoles.detach().contiguous(), dtype=mat_dtype),
        wp.from_torch(cosines.contiguous(), dtype=wp.float64),
        wp.from_torch(sines.contiguous(), dtype=wp.float64),
        wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(source_coeff2.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(volume.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(atom_start.contiguous(), dtype=wp.int32),
        wp.from_torch(atom_end.contiguous(), dtype=wp.int32),
        wp.from_torch(rho_q, dtype=wp.float64),
        wp_dtype=wp_scalar,
        device=str(wp_device),
    )
    return rho_q


@torch.library.register_fake("nvalchemiops::batch_multipole_rho_q")
def _batch_multipole_rho_q_fake(
    quadrupoles: torch.Tensor,
    positions: torch.Tensor,
    source_coeff2: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: rho_Q is ``(B, K_max, 2)`` float64."""
    return k_vectors.new_empty(
        (volume.shape[0], k_vectors.shape[1], 2), dtype=torch.float64
    )


def _batch_multipole_rho_q_setup_context(ctx, inputs, output) -> None:
    """Save the forward inputs for the analytical backward."""
    quadrupoles, positions, source_coeff2, k_vectors, volume, batch_idx = inputs
    ctx.save_for_backward(
        quadrupoles, positions, source_coeff2, k_vectors, volume, batch_idx
    )


def _batch_multipole_rho_q_backward(ctx, grad_rho: torch.Tensor):
    """Analytical grads for quadrupoles, positions, source_coeff2, k_vectors.

    Recomputes (cos, sin) via the opaque batched structure-factor op (detached
    inputs: cos/sin are constants) and dispatches the four backward sub-ops. The
    moment / position chains carry their own second-order backward
    (``create_graph``); coeff2 / k_vectors are forward-only (1st-order
    reciprocal cell-grad; cell 2nd-order out of scope). ``volume`` / ``batch_idx``
    get ``None``.
    """
    quadrupoles, positions, source_coeff2, k_vectors, volume, batch_idx = (
        ctx.saved_tensors
    )
    grad_rho = grad_rho.contiguous()
    cosines, sines = torch.ops.nvalchemiops.batch_multipole_structure_factor(
        positions.detach(), k_vectors.detach(), batch_idx
    )
    grad_quadrupoles = torch.ops.nvalchemiops.batch_multipole_rho_q_moment_grad(
        grad_rho, positions, cosines, sines, k_vectors, source_coeff2, volume, batch_idx
    )
    grad_positions = torch.ops.nvalchemiops.batch_multipole_rho_q_position_grad(
        grad_rho,
        quadrupoles,
        positions,
        cosines,
        sines,
        k_vectors,
        source_coeff2,
        volume,
        batch_idx,
    )
    # coeff2 / k-vector phase grads (carry the l=2 reciprocal cell-grad). Both
    # are twice-differentiable Warp op chains (batched second-order backward) so
    # the cell<->{positions, quadrupoles} cross-terms flow under create_graph
    # for stress-loss while plain 1st-order cell-grad stays on the fast forward.
    grad_coeff2 = torch.ops.nvalchemiops.batch_multipole_rho_q_coeff2_grad(
        grad_rho, quadrupoles, positions, cosines, sines, k_vectors, volume, batch_idx
    )
    grad_kvec = torch.ops.nvalchemiops.batch_multipole_rho_q_kvec_grad(
        grad_rho,
        quadrupoles,
        positions,
        cosines,
        sines,
        k_vectors,
        source_coeff2,
        volume,
        batch_idx,
    )
    grad_quadrupoles = grad_quadrupoles.to(quadrupoles.dtype)
    grad_positions = grad_positions.to(positions.dtype)
    # Slots: (quadrupoles, positions, source_coeff2, k_vectors, volume,
    # batch_idx).
    return grad_quadrupoles, grad_positions, grad_coeff2, grad_kvec, None, None


torch.library.register_autograd(
    "nvalchemiops::batch_multipole_rho_q",
    _batch_multipole_rho_q_backward,
    setup_context=_batch_multipole_rho_q_setup_context,
)


# =============================================================================
# Batched direct-k per-atom reciprocal gathers: g = Sᵀ·grad_rho (flat N_total).
# Batched analogs of multipole_rho_gather_t / multipole_rho_q_gather_t; forward
# = the batched moment-grad transpose, backward composes the batched rho /
# rho_q forward + the batched rho backward sub-ops (twice-diff for free).
# =============================================================================


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_rho_gather_t",
    mutates_args=(),
    schema=(
        "(Tensor grad_rho, Tensor positions, Tensor source_phi_hat, "
        "Tensor k_vectors, Tensor volume, Tensor batch_idx) -> Tensor"
    ),
)
def _batch_multipole_rho_gather_t_op(
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r"""Opaque forward: ``g = Sᵀ·grad_rho`` per-atom moment gradient ``(N_total, 4)``."""
    cosines, sines = torch.ops.nvalchemiops.batch_multipole_structure_factor(
        positions, k_vectors, batch_idx
    )
    return torch.ops.nvalchemiops.batch_multipole_rho_moment_grad(
        grad_rho, cosines, sines, source_phi_hat, volume, batch_idx
    )


@torch.library.register_fake("nvalchemiops::batch_multipole_rho_gather_t")
def _batch_multipole_rho_gather_t_fake(
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    source_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``g`` is ``(N_total, 4)`` float64."""
    return positions.new_empty((positions.shape[0], 4), dtype=torch.float64)


def _batch_multipole_rho_gather_t_setup_context(ctx, inputs, output) -> None:
    """Save forward inputs for the analytical backward."""
    grad_rho, positions, source_phi_hat, k_vectors, volume, batch_idx = inputs
    ctx.save_for_backward(
        grad_rho, positions, source_phi_hat, k_vectors, volume, batch_idx
    )


def _batch_multipole_rho_gather_t_backward(ctx, g_cot: torch.Tensor):
    r"""Adjoint of ``g = Sᵀ·grad_rho`` (batched): ``<cg, Sᵀ grad_rho> = <S·cg, grad_rho>``."""
    grad_rho, positions, source_phi_hat, k_vectors, volume, batch_idx = (
        ctx.saved_tensors
    )
    cg_q = g_cot[:, 0].contiguous()
    cg_dip = g_cot[:, [3, 1, 2]].contiguous()
    cosines, sines = torch.ops.nvalchemiops.batch_multipole_structure_factor(
        positions.detach(), k_vectors.detach(), batch_idx
    )
    d_grad_rho = torch.ops.nvalchemiops.batch_multipole_rho(
        cg_q, cg_dip, positions, source_phi_hat, k_vectors, volume, batch_idx
    )
    scale = (
        torch.as_tensor(_TWO_PI_CUBED, dtype=torch.float64, device=positions.device)
        / volume
    )
    d_positions = torch.ops.nvalchemiops.batch_multipole_rho_position_grad(
        grad_rho,
        cg_q,
        cg_dip,
        positions,
        cosines,
        sines,
        source_phi_hat,
        k_vectors,
        scale,
        batch_idx,
    ).to(positions.dtype)
    d_phi = torch.ops.nvalchemiops.batch_multipole_rho_phihat_grad(
        grad_rho, cg_q, cg_dip, positions, cosines, sines, k_vectors, volume, batch_idx
    )
    d_kvec = torch.ops.nvalchemiops.batch_multipole_rho_kphase_grad(
        grad_rho,
        cg_q,
        cg_dip,
        positions,
        cosines,
        sines,
        source_phi_hat,
        k_vectors,
        volume,
        batch_idx,
    )
    # Slots: (grad_rho, positions, source_phi_hat, k_vectors, volume, batch_idx).
    return d_grad_rho, d_positions, d_phi, d_kvec, None, None


torch.library.register_autograd(
    "nvalchemiops::batch_multipole_rho_gather_t",
    _batch_multipole_rho_gather_t_backward,
    setup_context=_batch_multipole_rho_gather_t_setup_context,
)


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_rho_q_gather_t",
    mutates_args=(),
    schema=(
        "(Tensor grad_rho, Tensor positions, Tensor source_coeff2, "
        "Tensor k_vectors, Tensor volume, Tensor batch_idx) -> Tensor"
    ),
)
def _batch_multipole_rho_q_gather_t_op(
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    source_coeff2: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r"""Opaque forward: ``g_Q = Sᵀ_Q·grad_rho`` per-atom quad gradient ``(N_total, 3, 3)``."""
    cosines, sines = torch.ops.nvalchemiops.batch_multipole_structure_factor(
        positions, k_vectors, batch_idx
    )
    return torch.ops.nvalchemiops.batch_multipole_rho_q_moment_grad(
        grad_rho, positions, cosines, sines, k_vectors, source_coeff2, volume, batch_idx
    )


@torch.library.register_fake("nvalchemiops::batch_multipole_rho_q_gather_t")
def _batch_multipole_rho_q_gather_t_fake(
    grad_rho: torch.Tensor,
    positions: torch.Tensor,
    source_coeff2: torch.Tensor,
    k_vectors: torch.Tensor,
    volume: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``g_Q`` is ``(N_total, 3, 3)`` float64."""
    return positions.new_empty((positions.shape[0], 3, 3), dtype=torch.float64)


def _batch_multipole_rho_q_gather_t_setup_context(ctx, inputs, output) -> None:
    """Save forward inputs for the analytical backward."""
    grad_rho, positions, source_coeff2, k_vectors, volume, batch_idx = inputs
    ctx.save_for_backward(
        grad_rho, positions, source_coeff2, k_vectors, volume, batch_idx
    )


def _batch_multipole_rho_q_gather_t_backward(ctx, g_cot: torch.Tensor):
    r"""Adjoint of ``g_Q = Sᵀ_Q·grad_rho`` (batched)."""
    grad_rho, positions, source_coeff2, k_vectors, volume, batch_idx = ctx.saved_tensors
    cg_q = g_cot.contiguous()
    cosines, sines = torch.ops.nvalchemiops.batch_multipole_structure_factor(
        positions.detach(), k_vectors.detach(), batch_idx
    )
    d_grad_rho = torch.ops.nvalchemiops.batch_multipole_rho_q(
        cg_q, positions, source_coeff2, k_vectors, volume, batch_idx
    )
    d_positions = torch.ops.nvalchemiops.batch_multipole_rho_q_position_grad(
        grad_rho,
        cg_q,
        positions,
        cosines,
        sines,
        k_vectors,
        source_coeff2,
        volume,
        batch_idx,
    ).to(positions.dtype)
    d_coeff2 = torch.ops.nvalchemiops.batch_multipole_rho_q_coeff2_grad(
        grad_rho, cg_q, positions, cosines, sines, k_vectors, volume, batch_idx
    )
    d_kvec = torch.ops.nvalchemiops.batch_multipole_rho_q_kvec_grad(
        grad_rho,
        cg_q,
        positions,
        cosines,
        sines,
        k_vectors,
        source_coeff2,
        volume,
        batch_idx,
    )
    # Slots: (grad_rho, positions, source_coeff2, k_vectors, volume, batch_idx).
    return d_grad_rho, d_positions, d_coeff2, d_kvec, None, None


torch.library.register_autograd(
    "nvalchemiops::batch_multipole_rho_q_gather_t",
    _batch_multipole_rho_q_gather_t_backward,
    setup_context=_batch_multipole_rho_q_gather_t_setup_context,
)


class BatchMultipoleRhoQFunction:
    r"""Back-compat shim for the batched Cartesian-quadrupole :math:`\rho_Q(k)` contribution.

    The implementation is the fully-differentiable
    ``torch.ops.nvalchemiops.batch_multipole_rho_q`` custom op (opaque forward +
    analytical backward + ``create_graph`` through the moment/position chains;
    forward-only coeff2/k-vector ops carry the l=2 reciprocal cell-grad). This
    class only preserves the historical ``.apply(quadrupoles, positions,
    source_coeff2, k_vectors, batch_idx, cache)`` signature (``cache`` supplies
    ``volume`` and is validated for ``l_max>=2``); new code should call the op
    directly.

    Parameters
    ----------
    quadrupoles : torch.Tensor
        Per-atom Cartesian quadrupoles, shape ``(N_total, 3, 3)``.
    positions : torch.Tensor
        Atomic coordinates, shape ``(N_total, 3)``.
    source_coeff2 : torch.Tensor
        Per-k l=2 source coefficient :math:`c_2(k)`, shape ``(B, K_max)``; value
        equals ``cache.source_coeff2``.
    k_vectors : torch.Tensor
        Reciprocal-lattice vectors, shape ``(B, K_max, 3)``; value equals
        ``cache.k_vectors``.
    batch_idx : torch.Tensor
        Per-atom system index, shape ``(N_total,)``.
    cache : MultipoleSCFCache
        Per-system direct-k-space state; must be built with ``l_max>=2``.

    Returns
    -------
    torch.Tensor
        Additive Q-channel :math:`\rho_Q(k)`, shape ``(B, K_max, 2)`` float64.
    """

    @staticmethod
    def apply(
        quadrupoles: torch.Tensor,
        positions: torch.Tensor,
        source_coeff2: torch.Tensor,
        k_vectors: torch.Tensor,
        batch_idx: torch.Tensor,
        cache: MultipoleSCFCache,
    ) -> torch.Tensor:
        """Dispatch to ``torch.ops.nvalchemiops.batch_multipole_rho_q``."""
        if cache.source_coeff2 is None:
            raise ValueError(
                "BatchMultipoleRhoQFunction requires a cache built with "
                "l_max>=2 (cache.source_coeff2 is None)."
            )
        return torch.ops.nvalchemiops.batch_multipole_rho_q(
            quadrupoles, positions, source_coeff2, k_vectors, cache.volume, batch_idx
        )


# =============================================================================
# Batched Ewald reciprocal-space fused scalar Function
def batch_multipole_reciprocal_space_dipole_fused_scalar(
    positions: torch.Tensor,
    source_feats: torch.Tensor,
    batch_idx: torch.Tensor,
    cache: MultipoleSCFCache,
    *,
    include_self_interaction: bool = False,
) -> torch.Tensor:
    r"""Batched scalar reciprocal energy — thin alias of :func:`multipole_scf_step_energy`.

    The batched rho(k) op chain in :func:`multipole_scf_step_energy` produces
    exactly this per-system energy with full forward + backward + ``create_graph``
    autograd, all ``torch.compile``-clean, so this entry just forwards to it.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic coordinates, shape ``(N_total, 3)``.
    source_feats : torch.Tensor
        Packed source moments ``(N_total, 1|4)``.
    batch_idx : torch.Tensor
        Per-atom system index, shape ``(N_total,)`` int32.
    cache : MultipoleSCFCache
        Batched per-system direct-k-space state.
    include_self_interaction : bool, optional
        When ``False`` (default), subtract the source-source self-overlap term.

    Returns
    -------
    torch.Tensor
        Per-system reciprocal-space energies, shape ``(B,)`` float64.
    """
    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
        multipole_scf_step_energy,
    )

    return multipole_scf_step_energy(
        cache,
        positions,
        source_feats,
        batch_idx=batch_idx,
        include_self_interaction=include_self_interaction,
    )


def _batch_project_raw_features_launch(
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r"""Run :func:`batch_project_features_dipole` (no self-subtract). ``(N_total, N_σ, 4)``."""
    device = potential.device
    wp_device = wp.device_from_torch(device)
    n_total = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[2]

    src_lm_zero = torch.zeros((n_total, 4), dtype=torch.float64, device=device)
    oc_zero = torch.zeros((n_sigma, 2), dtype=torch.float64, device=device)
    s_ar = torch.arange(n_sigma, dtype=torch.int32, device=device).unsqueeze(1)
    lm_ar = torch.arange(4, dtype=torch.int32, device=device).unsqueeze(0)
    lut_l1 = (s_ar * 4 + lm_ar).contiguous()

    features_flat = torch.empty(
        (n_total, n_sigma * 4), dtype=torch.float64, device=device
    )
    batch_project_features_dipole(
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(src_lm_zero, dtype=wp.float64),
        wp.from_torch(oc_zero, dtype=wp.float64),
        False,  # subtract_self
        wp.from_torch(lut_l1, dtype=wp.int32),
        wp.from_torch(features_flat, dtype=wp.float64),
        device=str(wp_device),
    )
    return features_flat.reshape(n_total, n_sigma, 4)


def _batch_project_raw_features_quadrupole_launch(
    potential: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    r"""Run :func:`batch_project_features_quadrupole` (no self-subtract). ``(N_total, N_σ, 5)``."""
    device = potential.device
    wp_device = wp.device_from_torch(device)
    n_total = cosines.shape[1]
    n_sigma = receiver_phi_hat.shape[2]

    features = torch.empty((n_total, n_sigma, 5), dtype=torch.float64, device=device)
    batch_project_features_quadrupole(
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(features, dtype=wp.float64),
        device=str(wp_device),
    )
    return features


# ---- batched phi_hat / k-vector phase (forward-only; carry the cell-grad) ----


def _batch_feature_phihat_grad_launch(
    grad_raw: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
    batch_idx: torch.Tensor,
    n_lm: int,
) -> torch.Tensor:
    """``dL/dreceiver_phi_hat`` ``(B, K_max, N_σ, n_lm, 2)`` via ``batch_project_phihat_grad``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    b_dim, k_max = k_factor_proj.shape
    n_sigma = grad_raw.shape[1]
    grad_phi = torch.zeros(
        (b_dim, k_max, n_sigma, n_lm, 2), dtype=torch.float64, device=device
    )
    batch_project_phihat_grad(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(grad_phi, dtype=wp.vec2d),
        device=str(wp_device),
    )
    return grad_phi


def _batch_feature_kphase_grad_launch(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """``dL/dk_vectors`` ``(B, K_max, 3)`` through the phase via ``batch_project_kphase_grad``."""
    device = grad_raw.device
    wp_device = wp.device_from_torch(device)
    b_dim, k_max = k_factor_proj.shape
    grad_kvec = torch.zeros((b_dim, k_max, 3), dtype=torch.float64, device=device)
    batch_project_kphase_grad(
        wp.from_torch(grad_raw.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(receiver_phi_hat.detach().contiguous(), dtype=wp.vec2d),
        wp.from_torch(cosines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(sines.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(k_factor_proj.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(potential.detach().contiguous(), dtype=wp.float64),
        wp.from_torch(positions.detach().contiguous(), dtype=wp.vec3d),
        wp.from_torch(batch_idx.contiguous().to(torch.int32), dtype=wp.int32),
        wp.from_torch(grad_kvec, dtype=wp.vec3d),
        device=str(wp_device),
    )
    return grad_kvec


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_feature_phihat_grad",
    mutates_args=(),
    schema=(
        "(Tensor grad_raw, Tensor cosines, Tensor sines, Tensor k_factor_proj, "
        "Tensor potential, Tensor batch_idx, int n_lm) -> Tensor"
    ),
)
def _batch_feature_phihat_grad_op(
    grad_raw: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
    batch_idx: torch.Tensor,
    n_lm: int,
) -> torch.Tensor:
    """Opaque ``dL/dreceiver_phi_hat`` ``(B, K_max, N_σ, n_lm, 2)``."""
    return _batch_feature_phihat_grad_launch(
        grad_raw, cosines, sines, k_factor_proj, potential, batch_idx, n_lm
    )


@torch.library.register_fake("nvalchemiops::batch_multipole_feature_phihat_grad")
def _batch_feature_phihat_grad_fake(
    grad_raw: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
    batch_idx: torch.Tensor,
    n_lm: int,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(B, K_max, N_σ, n_lm, 2)`` float64."""
    b_dim, k_max = k_factor_proj.shape
    return k_factor_proj.new_empty(
        (b_dim, k_max, grad_raw.shape[1], n_lm, 2), dtype=torch.float64
    )


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_feature_kphase_grad",
    mutates_args=(),
    schema=(
        "(Tensor grad_raw, Tensor receiver_phi_hat, Tensor cosines, "
        "Tensor sines, Tensor k_factor_proj, Tensor potential, "
        "Tensor positions, Tensor batch_idx) -> Tensor"
    ),
)
def _batch_feature_kphase_grad_op(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Opaque ``dL/dk_vectors`` ``(B, K_max, 3)``."""
    return _batch_feature_kphase_grad_launch(
        grad_raw,
        receiver_phi_hat,
        cosines,
        sines,
        k_factor_proj,
        potential,
        positions,
        batch_idx,
    )


@torch.library.register_fake("nvalchemiops::batch_multipole_feature_kphase_grad")
def _batch_feature_kphase_grad_fake(
    grad_raw: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor,
    potential: torch.Tensor,
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(B, K_max, 3)`` float64."""
    b_dim, k_max = k_factor_proj.shape
    return k_factor_proj.new_empty((b_dim, k_max, 3), dtype=torch.float64)


# =============================================================================
# torch.library.custom_op chain for the batched raw feature projection (l<=1)
# =============================================================================


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_project_raw_features",
    mutates_args=(),
    schema=(
        "(Tensor potential, Tensor positions, Tensor receiver_phi_hat, "
        "Tensor k_vectors, Tensor k_factor_proj, Tensor batch_idx) -> Tensor"
    ),
)
def _batch_multipole_project_raw_features_op(
    potential: torch.Tensor,
    positions: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    k_factor_proj: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Opaque forward: build (cos, sin) and run the batched raw l<=1 projection."""
    cosines, sines = torch.ops.nvalchemiops.batch_multipole_structure_factor(
        positions, k_vectors, batch_idx
    )
    return _batch_project_raw_features_launch(
        potential, receiver_phi_hat, cosines, sines, k_factor_proj, batch_idx
    )


@torch.library.register_fake("nvalchemiops::batch_multipole_project_raw_features")
def _batch_multipole_project_raw_features_fake(
    potential: torch.Tensor,
    positions: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    k_factor_proj: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: raw features ``(N_total, N_σ, 4)`` float64."""
    n_total = positions.shape[0]
    n_sigma = receiver_phi_hat.shape[2]
    return positions.new_empty((n_total, n_sigma, 4), dtype=torch.float64)


def _batch_multipole_project_raw_features_setup_context(ctx, inputs, output) -> None:
    """Save the forward inputs for the analytical backward."""
    (
        potential,
        positions,
        receiver_phi_hat,
        k_vectors,
        k_factor_proj,
        batch_idx,
    ) = inputs
    ctx.save_for_backward(
        potential, positions, receiver_phi_hat, k_vectors, k_factor_proj, batch_idx
    )


def _batch_multipole_project_raw_features_backward(ctx, grad_raw: torch.Tensor):
    """Analytical grads (potential, positions, receiver_phi_hat, k_vectors).

    Recomputes (cos, sin) via the opaque batched structure-factor op (detached)
    and dispatches the V / position chains + the forward-only phi_hat / k-vector
    ops. ``k_factor_proj`` / ``batch_idx`` get ``None``.
    """
    (
        potential,
        positions,
        receiver_phi_hat,
        k_vectors,
        k_factor_proj,
        batch_idx,
    ) = ctx.saved_tensors
    grad_raw = grad_raw.contiguous()
    cosines, sines = torch.ops.nvalchemiops.batch_multipole_structure_factor(
        positions.detach(), k_vectors.detach(), batch_idx
    )
    grad_v = torch.ops.nvalchemiops.batch_multipole_feature_v_grad(
        grad_raw,
        receiver_phi_hat,
        cosines,
        sines,
        k_factor_proj,
        k_vectors,
        positions,
        batch_idx,
    )
    grad_positions = torch.ops.nvalchemiops.batch_multipole_feature_position_grad(
        grad_raw,
        potential,
        receiver_phi_hat,
        cosines,
        sines,
        k_factor_proj,
        k_vectors,
        positions,
        batch_idx,
    )
    grad_phihat = torch.ops.nvalchemiops.batch_multipole_feature_phihat_grad(
        grad_raw, cosines, sines, k_factor_proj, potential, batch_idx, 4
    )
    grad_kvec = torch.ops.nvalchemiops.batch_multipole_feature_kphase_grad(
        grad_raw,
        receiver_phi_hat,
        cosines,
        sines,
        k_factor_proj,
        potential,
        positions,
        batch_idx,
    )
    grad_v = grad_v.to(potential.dtype)
    grad_positions = grad_positions.to(positions.dtype)
    grad_phihat = grad_phihat.to(receiver_phi_hat.dtype)
    grad_kvec = grad_kvec.to(k_vectors.dtype)
    # Slots: (potential, positions, receiver_phi_hat, k_vectors, k_factor_proj,
    # batch_idx).
    return grad_v, grad_positions, grad_phihat, grad_kvec, None, None


torch.library.register_autograd(
    "nvalchemiops::batch_multipole_project_raw_features",
    _batch_multipole_project_raw_features_backward,
    setup_context=_batch_multipole_project_raw_features_setup_context,
)


class BatchMultipoleProjectRawFeaturesFunction:
    r"""Back-compat shim for the batched raw (un-self-subtracted) l<=1 projection.

    The implementation is the fully-differentiable
    ``torch.ops.nvalchemiops.batch_multipole_project_raw_features`` custom op.
    This class only preserves the historical
    ``.apply(potential, positions, receiver_phi_hat_l1, k_vectors, batch_idx,
    cache)`` call signature (``cache`` supplies ``k_factor_proj``); new code
    should call the op directly.

    Returns
    -------
    torch.Tensor
        Raw (un-self-subtracted) features ``(N_total, N_σ, 4)`` float64.
    """

    @staticmethod
    def apply(
        potential: torch.Tensor,
        positions: torch.Tensor,
        receiver_phi_hat_l1: torch.Tensor,
        k_vectors: torch.Tensor,
        batch_idx: torch.Tensor,
        cache: MultipoleSCFCache,
    ) -> torch.Tensor:
        """Dispatch to ``batch_multipole_project_raw_features``."""
        return torch.ops.nvalchemiops.batch_multipole_project_raw_features(
            potential,
            positions,
            receiver_phi_hat_l1,
            k_vectors,
            cache.k_factor_proj,
            batch_idx,
        )


# =============================================================================
# torch.library.custom_op chain for the batched raw l=2 feature projection
# =============================================================================


@torch.library.custom_op(
    "nvalchemiops::batch_multipole_project_raw_features_quadrupole",
    mutates_args=(),
    schema=(
        "(Tensor potential, Tensor positions, Tensor receiver_phi_hat, "
        "Tensor k_vectors, Tensor k_factor_proj, Tensor batch_idx) -> Tensor"
    ),
)
def _batch_multipole_project_raw_features_quadrupole_op(
    potential: torch.Tensor,
    positions: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    k_factor_proj: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Opaque forward: build (cos, sin) and run the batched raw l=2 projection."""
    cosines, sines = torch.ops.nvalchemiops.batch_multipole_structure_factor(
        positions, k_vectors, batch_idx
    )
    return _batch_project_raw_features_quadrupole_launch(
        potential, receiver_phi_hat, cosines, sines, k_factor_proj, batch_idx
    )


@torch.library.register_fake(
    "nvalchemiops::batch_multipole_project_raw_features_quadrupole"
)
def _batch_multipole_project_raw_features_quadrupole_fake(
    potential: torch.Tensor,
    positions: torch.Tensor,
    receiver_phi_hat: torch.Tensor,
    k_vectors: torch.Tensor,
    k_factor_proj: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype metadata: raw l=2 features ``(N_total, N_σ, 5)`` float64."""
    n_total = positions.shape[0]
    n_sigma = receiver_phi_hat.shape[2]
    return positions.new_empty((n_total, n_sigma, 5), dtype=torch.float64)


def _batch_multipole_project_raw_features_quadrupole_setup_context(
    ctx, inputs, output
) -> None:
    """Save the forward inputs for the analytical backward."""
    (
        potential,
        positions,
        receiver_phi_hat,
        k_vectors,
        k_factor_proj,
        batch_idx,
    ) = inputs
    ctx.save_for_backward(
        potential, positions, receiver_phi_hat, k_vectors, k_factor_proj, batch_idx
    )


def _batch_multipole_project_raw_features_quadrupole_backward(
    ctx, grad_raw: torch.Tensor
):
    """Analytical grads (potential, positions, receiver_phi_hat, k_vectors) for l=2."""
    (
        potential,
        positions,
        receiver_phi_hat,
        k_vectors,
        k_factor_proj,
        batch_idx,
    ) = ctx.saved_tensors
    grad_raw = grad_raw.contiguous()
    cosines, sines = torch.ops.nvalchemiops.batch_multipole_structure_factor(
        positions.detach(), k_vectors.detach(), batch_idx
    )
    grad_v = torch.ops.nvalchemiops.batch_multipole_feature_v_grad_quadrupole(
        grad_raw,
        receiver_phi_hat,
        cosines,
        sines,
        k_factor_proj,
        k_vectors,
        positions,
        batch_idx,
    )
    grad_positions = (
        torch.ops.nvalchemiops.batch_multipole_feature_position_grad_quadrupole(
            grad_raw,
            potential,
            receiver_phi_hat,
            cosines,
            sines,
            k_factor_proj,
            k_vectors,
            positions,
            batch_idx,
        )
    )
    grad_phihat = torch.ops.nvalchemiops.batch_multipole_feature_phihat_grad(
        grad_raw, cosines, sines, k_factor_proj, potential, batch_idx, 5
    )
    grad_kvec = torch.ops.nvalchemiops.batch_multipole_feature_kphase_grad(
        grad_raw,
        receiver_phi_hat,
        cosines,
        sines,
        k_factor_proj,
        potential,
        positions,
        batch_idx,
    )
    grad_v = grad_v.to(potential.dtype)
    grad_positions = grad_positions.to(positions.dtype)
    grad_phihat = grad_phihat.to(receiver_phi_hat.dtype)
    grad_kvec = grad_kvec.to(k_vectors.dtype)
    return grad_v, grad_positions, grad_phihat, grad_kvec, None, None


torch.library.register_autograd(
    "nvalchemiops::batch_multipole_project_raw_features_quadrupole",
    _batch_multipole_project_raw_features_quadrupole_backward,
    setup_context=_batch_multipole_project_raw_features_quadrupole_setup_context,
)


class BatchMultipoleProjectRawFeaturesQuadrupoleFunction:
    r"""Back-compat shim for the batched raw l=2 feature projection.

    The implementation is the fully-differentiable
    ``torch.ops.nvalchemiops.batch_multipole_project_raw_features_quadrupole``
    custom op. This class only preserves the historical
    ``.apply(potential, positions, receiver_phi_hat_l2, k_vectors, batch_idx,
    cache)`` call signature (``cache`` supplies ``k_factor_proj``).

    Returns
    -------
    torch.Tensor
        Raw l=2 features ``(N_total, N_σ, 5)`` float64 in natural layout.
    """

    @staticmethod
    def apply(
        potential: torch.Tensor,
        positions: torch.Tensor,
        receiver_phi_hat_l2: torch.Tensor,
        k_vectors: torch.Tensor,
        batch_idx: torch.Tensor,
        cache: MultipoleSCFCache,
    ) -> torch.Tensor:
        """Dispatch to ``batch_multipole_project_raw_features_quadrupole``."""
        return torch.ops.nvalchemiops.batch_multipole_project_raw_features_quadrupole(
            potential,
            positions,
            receiver_phi_hat_l2,
            k_vectors,
            cache.k_factor_proj,
            batch_idx,
        )
