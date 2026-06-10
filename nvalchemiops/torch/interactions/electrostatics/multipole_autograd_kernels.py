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
Low-level ``torch.autograd.Function`` wrappers for the direct-k-space multipole
Warp kernels.

Each class wraps exactly one Warp launcher so that the kernel's
analytical backward composes through torch autograd, enabling
``create_graph=True`` double-backward for MLIP force / stress loss
training. The wrappers live below :mod:`multipole_autograd` and are
invoked from within the backwards of the higher-level Functions
(:class:`MultipoleRhoFunction`,
:class:`MultipoleProjectRawFeaturesFunction`) as well as from
:mod:`multipole_scf_cache` (source / receiver :math:`\hat\phi` wrapping
for cell autograd).

Design notes
------------

* **Tensor-only inputs.** Each Function takes only ``torch.Tensor`` and
  Python scalar args. Non-tensor args (``sigma``, ``inv_cl_l0``, etc.)
  flow through ``ctx`` as stored Python objects, not through
  ``save_for_backward``.

* **Contiguity.** All tensor inputs are ``.contiguous()``-d before the
  Warp launch; non-contiguous input silently produces wrong results in
  Warp's ``from_torch`` path.

* **Grad slots.** Each backward returns one gradient per forward input
  in input order. Tensors the forward does not differentiate w.r.t.
  (e.g. ``sigmas`` in the receiver-basis wrapper) get ``None``.

* **Non-differentiable second-order paths.** Some cell-autograd
  double-backward second-order kernels are not derived; the relevant
  slots return ``None`` and calling ``torch.autograd.grad`` through
  those paths will raise.
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics.multipole_direct_kspace_kernels import (
    eval_gto_fourier_dipole,
    eval_receiver_gto_fourier_dipole,
    eval_receiver_gto_fourier_quadrupole,
    receiver_phi_hat_backward_dipole,
    receiver_phi_hat_backward_quadrupole,
    source_phi_hat_backward_dipole,
    source_phi_hat_double_backward_dipole,
)
from nvalchemiops.torch._warp_op_helpers import register_warp_op_chain


def _wp_scalar(dtype: torch.dtype):
    """Map a torch float dtype to the matching Warp scalar dtype."""
    return wp.float64 if dtype == torch.float64 else wp.float32


def _wp_vec(dtype: torch.dtype):
    """Map a torch float dtype to the matching Warp vec3 dtype."""
    return wp.vec3d if dtype == torch.float64 else wp.vec3f


def _wp_in(t: torch.Tensor, dtype=wp.float64):
    """Wrap a torch tensor as a detached, contiguous Warp view of given dtype.

    Covers the ``wp.from_torch(t.detach().contiguous(), dtype=...)`` pattern
    that appears on nearly every Warp-kernel argument in this module. The
    ``.detach()`` keeps Warp out of the autograd graph; the ``.contiguous()``
    guarantees the stride layout ``wp.from_torch`` requires.
    """
    return wp.from_torch(t.detach().contiguous(), dtype=dtype)


def _wp_out(t: torch.Tensor, dtype=wp.float64):
    """Wrap a torch output tensor (no ``.detach()``) as a Warp view.

    Use for tensors the kernel *writes into*. Leaving off the ``.detach()``
    matters when the output tensor is later returned from an
    ``autograd.Function.forward``: the tensor object stays on the autograd
    tape while the Warp kernel populates its storage in place.
    """
    return wp.from_torch(t.contiguous(), dtype=dtype)


# =============================================================================
# Source-basis phi-hat(k)
# =============================================================================


def _source_phi_hat_forward(
    k_vectors: torch.Tensor,
    k_norm2: torch.Tensor,
    sigma: float,
    icl0: float,
    icl1: float,
) -> torch.Tensor:
    """Opaque forward: :func:`eval_gto_fourier_dipole` -> ``(N_k, 4, 2)``."""
    device = k_vectors.device
    out = torch.empty((k_vectors.shape[0], 4, 2), dtype=torch.float64, device=device)
    eval_gto_fourier_dipole(
        _wp_in(k_vectors, wp.vec3d),
        _wp_in(k_norm2),
        float(sigma),
        float(icl0),
        float(icl1),
        _wp_out(out),
        device=str(wp.device_from_torch(device)),
    )
    return out


def _source_phi_hat_forward_fake(
    k_vectors: torch.Tensor,
    k_norm2: torch.Tensor,
    sigma: float,
    icl0: float,
    icl1: float,
) -> torch.Tensor:
    """Shape/dtype metadata: ``(N_k, 4, 2)`` float64."""
    return k_vectors.new_empty((k_vectors.shape[0], 4, 2), dtype=torch.float64)


def _source_phi_hat_backward(
    grad_output: torch.Tensor,
    k_vectors: torch.Tensor,
    k_norm2: torch.Tensor,
    sigma: float,
    icl0: float,
    icl1: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order backward via the Warp ``source_phi_hat_backward_dipole`` kernel."""
    device = k_vectors.device
    grad_k_vec = torch.empty_like(k_vectors, dtype=torch.float64)
    grad_k_n2 = torch.empty_like(k_norm2, dtype=torch.float64)
    source_phi_hat_backward_dipole(
        _wp_in(grad_output),
        _wp_in(k_vectors, wp.vec3d),
        _wp_in(k_norm2),
        float(sigma),
        float(icl0),
        float(icl1),
        _wp_out(grad_k_vec, wp.vec3d),
        _wp_out(grad_k_n2),
        device=str(wp.device_from_torch(device)),
    )
    return grad_k_vec, grad_k_n2


def _source_phi_hat_double_backward(
    gg_k_vectors: torch.Tensor,
    gg_k_norm2: torch.Tensor,
    grad_output: torch.Tensor,
    k_vectors: torch.Tensor,
    k_norm2: torch.Tensor,
    sigma: float,
    icl0: float,
    icl1: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Second-order backward via the Warp double-backward kernel.

    Propagates ``(gg_k_vectors, gg_k_norm2)`` to grads w.r.t. the backward's
    differentiable inputs ``(grad_output, k_vectors, k_norm2)`` — enabling
    reciprocal stress-loss (``∂²E/∂cell∂θ``) while staying fully on Warp.
    """
    device = k_vectors.device
    grad_grad_output = torch.empty(
        (k_vectors.shape[0], 4, 2), dtype=torch.float64, device=device
    )
    grad_k_vec = torch.empty_like(k_vectors, dtype=torch.float64)
    grad_k_n2 = torch.empty_like(k_norm2, dtype=torch.float64)
    source_phi_hat_double_backward_dipole(
        _wp_in(gg_k_vectors, wp.vec3d),
        _wp_in(gg_k_norm2),
        _wp_in(grad_output),
        _wp_in(k_vectors, wp.vec3d),
        _wp_in(k_norm2),
        float(sigma),
        float(icl0),
        float(icl1),
        _wp_out(grad_grad_output),
        _wp_out(grad_k_vec, wp.vec3d),
        _wp_out(grad_k_n2),
        device=str(wp.device_from_torch(device)),
    )
    return grad_grad_output, grad_k_vec, grad_k_n2


_SOURCE_PHI_HAT_SCHEMA = (
    "(Tensor k_vectors, Tensor k_norm2, float sigma, float icl0, float icl1) -> Tensor"
)
_SOURCE_PHI_HAT_BWD_SCHEMA = (
    "(Tensor grad_output, Tensor k_vectors, Tensor k_norm2, float sigma, "
    "float icl0, float icl1) -> (Tensor, Tensor)"
)
_SOURCE_PHI_HAT_DBWD_SCHEMA = (
    "(Tensor gg_k_vectors, Tensor gg_k_norm2, Tensor grad_output, "
    "Tensor k_vectors, Tensor k_norm2, float sigma, float icl0, float icl1) "
    "-> (Tensor, Tensor, Tensor)"
)
register_warp_op_chain(
    name="nvalchemiops::multipole_source_phi_hat",
    forward=_source_phi_hat_forward,
    forward_schema=_SOURCE_PHI_HAT_SCHEMA,
    forward_fake=_source_phi_hat_forward_fake,
    backward=_source_phi_hat_backward,
    backward_schema=_SOURCE_PHI_HAT_BWD_SCHEMA,
    backward_return_arity=2,
    diff_input_positions=(0, 1),  # k_vectors, k_norm2
    n_forward_inputs=5,
    double_backward=_source_phi_hat_double_backward,
    double_backward_schema=_SOURCE_PHI_HAT_DBWD_SCHEMA,
    double_backward_return_arity=3,
    # Backward-op inputs: grad_output(0), k_vectors(1), k_norm2(2) are the diff
    # slots the double-backward returns grads for.
    second_order_diff_positions=(0, 1, 2),
    n_backward_inputs=6,
)


class SourcePhiHatFunction:
    r"""Back-compat shim for the source :math:`\hat\phi(k)` GTO-Fourier eval.

    The implementation is the fully-differentiable
    ``torch.ops.nvalchemiops.multipole_source_phi_hat`` Warp op chain (opaque
    forward :func:`eval_gto_fourier_dipole` + analytical first- and second-order
    Warp backward), so the ``k_vectors`` / ``k_norm2`` -> ``source_phi_hat``
    graph is twice differentiable (reciprocal stress-loss) and compile-clean.
    This class only preserves the historical ``.apply(k_vectors, k_norm2,
    sigma, icl0, icl1)`` signature; new code should call the op directly.

    Parameters
    ----------
    k_vectors : torch.Tensor
        Reciprocal-lattice vectors, shape ``(N_k, 3)``.
    k_norm2 : torch.Tensor
        Squared k-norms :math:`|k|^2`, shape ``(N_k,)``.
    sigma : float
        Source Gaussian width (non-differentiable hyper-parameter).
    icl0, icl1 : float
        Inverse closure-length normalizations for l=0 / l=1
        (non-differentiable).

    Returns
    -------
    torch.Tensor
        Source-basis :math:`\hat\phi(k)`, shape ``(N_k, 4, 2)`` float64.
    """

    @staticmethod
    def apply(
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        sigma: float,
        icl0: float,
        icl1: float,
    ) -> torch.Tensor:
        """Dispatch to ``torch.ops.nvalchemiops.multipole_source_phi_hat``."""
        return torch.ops.nvalchemiops.multipole_source_phi_hat(
            k_vectors, k_norm2, float(sigma), float(icl0), float(icl1)
        )


# =============================================================================
# Receiver-basis phi-hat(k)
# =============================================================================


class ReceiverPhiHatFunction(torch.autograd.Function):
    r"""Autograd-registered :func:`eval_receiver_gto_fourier_dipole`.

    Forward takes ``(k_vectors, k_norm2, sigmas, inv_cl_table)`` and
    returns ``receiver_phi_hat`` of shape :math:`(N_k, N_\sigma, 4, 2)`.
    The ``sigmas`` and ``inv_cl_table`` tensors are treated as
    non-differentiable (receiver widths are hyper-parameters).

    Parameters
    ----------
    k_vectors : torch.Tensor
        Reciprocal-lattice vectors, shape ``(N_k, 3)``.
    k_norm2 : torch.Tensor
        Squared k-norms :math:`|k|^2`, shape ``(N_k,)``.
    sigmas : torch.Tensor
        Per-receiver Gaussian widths, shape ``(N_σ,)`` (non-differentiable).
    inv_cl_table : torch.Tensor
        Inverse closure-length normalization table (non-differentiable).

    Returns
    -------
    torch.Tensor
        Receiver-basis :math:`\hat\phi(k)`, shape :math:`(N_k, N_\sigma, 4, 2)`.
    """

    @staticmethod
    def forward(
        ctx,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        sigmas: torch.Tensor,
        inv_cl_table: torch.Tensor,
    ) -> torch.Tensor:
        """Run eval_receiver_gto_fourier_dipole and save tensors for backward."""
        n_k = k_vectors.shape[0]
        n_sigma = sigmas.shape[0]
        device = k_vectors.device
        wp_device = wp.device_from_torch(device)

        out = torch.empty((n_k, n_sigma, 4, 2), dtype=torch.float64, device=device)
        eval_receiver_gto_fourier_dipole(
            wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
            wp.from_torch(k_norm2.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(sigmas.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(inv_cl_table.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(out, dtype=wp.float64),
            device=str(wp_device),
        )

        ctx.save_for_backward(k_vectors, k_norm2, sigmas, inv_cl_table)
        return out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        r""":math:`\partial/\partial`\ (k_vectors, k_norm2); sigmas/inv_cl treated as constants."""
        k_vectors, k_norm2, sigmas, inv_cl_table = ctx.saved_tensors
        device = k_vectors.device
        wp_device = wp.device_from_torch(device)

        grad_k_vec = torch.empty_like(k_vectors)
        grad_k_n2 = torch.empty_like(k_norm2)

        receiver_phi_hat_backward_dipole(
            wp.from_torch(grad_output.contiguous(), dtype=wp.float64),
            wp.from_torch(k_vectors.contiguous(), dtype=wp.vec3d),
            wp.from_torch(k_norm2.contiguous(), dtype=wp.float64),
            wp.from_torch(sigmas.contiguous(), dtype=wp.float64),
            wp.from_torch(inv_cl_table.contiguous(), dtype=wp.float64),
            wp.from_torch(grad_k_vec, dtype=wp.vec3d),
            wp.from_torch(grad_k_n2, dtype=wp.float64),
            device=str(wp_device),
        )
        return grad_k_vec, grad_k_n2, None, None


class ReceiverPhiHatQuadrupoleFunction(torch.autograd.Function):
    r"""Autograd-registered :func:`eval_receiver_gto_fourier_quadrupole`.

    Forward takes ``(k_vectors, k_norm2, sigmas, inv_cl_l2)`` and returns the
    5-column :math:`l=2` receiver block :math:`(N_k, N_\sigma, 5, 2)` (purely
    real). ``sigmas`` and ``inv_cl_l2`` are non-differentiable (receiver
    widths/normalizations are hyper-parameters). Backward produces
    :math:`\partial L/\partial`\ (k_vectors, k_norm2) so feature cell-grad
    (stress) composes; torch accumulates with the :math:`l\le1`
    :class:`ReceiverPhiHatFunction` at the shared ``k_vectors`` / ``k_norm2``.

    Parameters
    ----------
    k_vectors : torch.Tensor
        Reciprocal-lattice vectors, shape ``(N_k, 3)``.
    k_norm2 : torch.Tensor
        Squared k-norms :math:`|k|^2`, shape ``(N_k,)``.
    sigmas : torch.Tensor
        Per-receiver Gaussian widths, shape ``(N_σ,)`` (non-differentiable).
    inv_cl_l2 : torch.Tensor
        l=2 inverse closure-length normalization table (non-differentiable).

    Returns
    -------
    torch.Tensor
        l=2 receiver block, shape :math:`(N_k, N_\sigma, 5, 2)` (purely real).
    """

    @staticmethod
    def forward(
        ctx,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        sigmas: torch.Tensor,
        inv_cl_l2: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the l=2 receiver Fourier block via
        :func:`eval_receiver_gto_fourier_quadrupole`.

        Saves ``k_vectors`` / ``k_norm2`` / ``sigmas`` / ``inv_cl_l2`` for
        backward.

        Returns
        -------
        torch.Tensor
            l=2 receiver block, shape ``(N_k, N_σ, 5, 2)`` (purely real).
        """
        n_k = k_vectors.shape[0]
        n_sigma = sigmas.shape[0]
        device = k_vectors.device
        wp_device = wp.device_from_torch(device)

        out = torch.empty((n_k, n_sigma, 5, 2), dtype=torch.float64, device=device)
        eval_receiver_gto_fourier_quadrupole(
            wp.from_torch(k_vectors.detach().contiguous(), dtype=wp.vec3d),
            wp.from_torch(k_norm2.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(sigmas.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(inv_cl_l2.detach().contiguous(), dtype=wp.float64),
            wp.from_torch(out, dtype=wp.float64),
            device=str(wp_device),
        )

        ctx.save_for_backward(k_vectors, k_norm2, sigmas, inv_cl_l2)
        return out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        """Backward of the l=2 receiver Fourier block.

        Returns ``∂L/∂k_vectors`` and ``∂L/∂k_norm2`` (via
        :func:`receiver_phi_hat_backward_quadrupole`); the ``sigmas`` and
        ``inv_cl_l2`` slots are ``None``.
        """
        k_vectors, k_norm2, sigmas, inv_cl_l2 = ctx.saved_tensors
        device = k_vectors.device
        wp_device = wp.device_from_torch(device)

        grad_k_vec = torch.empty_like(k_vectors)
        grad_k_n2 = torch.empty_like(k_norm2)

        receiver_phi_hat_backward_quadrupole(
            wp.from_torch(grad_output.contiguous(), dtype=wp.float64),
            wp.from_torch(k_vectors.contiguous(), dtype=wp.vec3d),
            wp.from_torch(k_norm2.contiguous(), dtype=wp.float64),
            wp.from_torch(sigmas.contiguous(), dtype=wp.float64),
            wp.from_torch(inv_cl_l2.contiguous(), dtype=wp.float64),
            wp.from_torch(grad_k_vec, dtype=wp.vec3d),
            wp.from_torch(grad_k_n2, dtype=wp.float64),
            device=str(wp_device),
        )
        return grad_k_vec, grad_k_n2, None, None
