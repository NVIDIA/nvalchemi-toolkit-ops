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

"""Registered Ewald reciprocal self/background correction chain."""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.torch._warp_op_helpers import register_warp_op_chain
from nvalchemiops.torch.types import get_wp_dtype

__all__ = [
    "ewald_energy_corrections",
    "ewald_energy_corrections_batch",
    "register_ewald_corrections_ops",
]

_CORRECTIONS_SINGLE: dict[str, object] | None = None
_CORRECTIONS_BATCH: dict[str, object] | None = None
_EWALD_CORRECTIONS_OPS_REGISTERED = False


def _wp_from_torch(tensor: torch.Tensor, dtype):
    """Convert a tensor to Warp without allocating unused Warp gradients."""
    return wp.from_torch(tensor.detach().contiguous(), dtype=dtype, requires_grad=False)


def _scoped_stream(device: torch.device):
    """Bind Warp launches to PyTorch's current CUDA stream."""
    if device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device)))


def _energy_corrections_forward_launch(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> torch.Tensor:
    from nvalchemiops.interactions.electrostatics.ewald_kernels import (
        ewald_energy_corrections as _corrections_launch,
    )

    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    device = wp.device_from_torch(raw_energies.device)
    corrected = torch.empty_like(raw_energies)

    with _scoped_stream(raw_energies.device):
        _corrections_launch(
            _wp_from_torch(raw_energies, wp_dtype),
            _wp_from_torch(charges.to(input_dtype), wp_dtype),
            _wp_from_torch(volume.to(input_dtype), wp_dtype),
            _wp_from_torch(alpha.to(input_dtype), wp_dtype),
            _wp_from_torch(total_charge.to(input_dtype), wp_dtype),
            _wp_from_torch(corrected, wp_dtype),
            wp_dtype=wp_dtype,
            device=device,
        )
    return corrected


def _energy_corrections_backward_launch(
    grad_E: torch.Tensor,
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    from nvalchemiops.interactions.electrostatics.ewald_kernels import (
        ewald_energy_corrections_backward as _corrections_backward_launch,
    )

    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    device = wp.device_from_torch(raw_energies.device)

    grad_raw = torch.empty_like(raw_energies)
    grad_charges = torch.empty_like(charges, dtype=input_dtype)
    grad_volume = torch.zeros_like(volume, dtype=input_dtype)
    grad_alpha = torch.zeros_like(alpha, dtype=input_dtype)
    grad_qtot = torch.zeros_like(total_charge, dtype=input_dtype)

    with _scoped_stream(raw_energies.device):
        _corrections_backward_launch(
            _wp_from_torch(grad_E, wp_dtype),
            _wp_from_torch(raw_energies, wp_dtype),
            _wp_from_torch(charges.to(input_dtype), wp_dtype),
            _wp_from_torch(volume.to(input_dtype), wp_dtype),
            _wp_from_torch(alpha.to(input_dtype), wp_dtype),
            _wp_from_torch(total_charge.to(input_dtype), wp_dtype),
            _wp_from_torch(grad_raw, wp_dtype),
            _wp_from_torch(grad_charges, wp_dtype),
            _wp_from_torch(grad_volume, wp_dtype),
            _wp_from_torch(grad_alpha, wp_dtype),
            _wp_from_torch(grad_qtot, wp_dtype),
            wp_dtype=wp_dtype,
            device=device,
        )
    return grad_raw, grad_charges, grad_volume, grad_alpha, grad_qtot


def _energy_corrections_double_backward_launch(
    h_raw: torch.Tensor,
    h_chg: torch.Tensor,
    h_vol: torch.Tensor,
    h_alpha: torch.Tensor,
    h_qtot: torch.Tensor,
    grad_E: torch.Tensor,
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    from nvalchemiops.interactions.electrostatics.ewald_kernels import (
        ewald_energy_corrections_double_backward as _corrections_dbwd_launch,
    )

    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    device = wp.device_from_torch(raw_energies.device)

    grad_grad_E = torch.empty_like(grad_E)
    grad_raw = torch.empty_like(raw_energies)
    grad_charges = torch.empty_like(charges, dtype=input_dtype)
    grad_volume = torch.zeros_like(volume, dtype=input_dtype)
    grad_alpha = torch.zeros_like(alpha, dtype=input_dtype)
    grad_qtot = torch.zeros_like(total_charge, dtype=input_dtype)

    with _scoped_stream(raw_energies.device):
        _corrections_dbwd_launch(
            _wp_from_torch(h_raw, wp_dtype),
            _wp_from_torch(h_chg, wp_dtype),
            _wp_from_torch(h_vol, wp_dtype),
            _wp_from_torch(h_alpha, wp_dtype),
            _wp_from_torch(h_qtot, wp_dtype),
            _wp_from_torch(grad_E, wp_dtype),
            _wp_from_torch(raw_energies, wp_dtype),
            _wp_from_torch(charges.to(input_dtype), wp_dtype),
            _wp_from_torch(volume.to(input_dtype), wp_dtype),
            _wp_from_torch(alpha.to(input_dtype), wp_dtype),
            _wp_from_torch(total_charge.to(input_dtype), wp_dtype),
            _wp_from_torch(grad_grad_E, wp_dtype),
            _wp_from_torch(grad_raw, wp_dtype),
            _wp_from_torch(grad_charges, wp_dtype),
            _wp_from_torch(grad_volume, wp_dtype),
            _wp_from_torch(grad_alpha, wp_dtype),
            _wp_from_torch(grad_qtot, wp_dtype),
            wp_dtype=wp_dtype,
            device=device,
        )
    return grad_grad_E, grad_raw, grad_charges, grad_volume, grad_alpha, grad_qtot


def _batch_energy_corrections_forward_launch(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> torch.Tensor:
    from nvalchemiops.interactions.electrostatics.ewald_kernels import (
        batch_ewald_energy_corrections as _batch_corrections_launch,
    )

    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    device = wp.device_from_torch(raw_energies.device)
    corrected = torch.empty_like(raw_energies)

    with _scoped_stream(raw_energies.device):
        _batch_corrections_launch(
            _wp_from_torch(raw_energies, wp_dtype),
            _wp_from_torch(charges.to(input_dtype), wp_dtype),
            _wp_from_torch(batch_idx, wp.int32),
            _wp_from_torch(volumes.to(input_dtype), wp_dtype),
            _wp_from_torch(alpha.to(input_dtype), wp_dtype),
            _wp_from_torch(total_charges.to(input_dtype), wp_dtype),
            _wp_from_torch(corrected, wp_dtype),
            wp_dtype=wp_dtype,
            device=device,
        )
    return corrected


def _batch_energy_corrections_backward_launch(
    grad_E: torch.Tensor,
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    from nvalchemiops.interactions.electrostatics.ewald_kernels import (
        batch_ewald_energy_corrections_backward as _batch_corrections_backward_launch,
    )

    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    device = wp.device_from_torch(raw_energies.device)

    grad_raw = torch.empty_like(raw_energies)
    grad_charges = torch.empty_like(charges, dtype=input_dtype)
    grad_volumes = torch.zeros_like(volumes, dtype=input_dtype)
    grad_alpha = torch.zeros_like(alpha, dtype=input_dtype)
    grad_qtots = torch.zeros_like(total_charges, dtype=input_dtype)

    with _scoped_stream(raw_energies.device):
        _batch_corrections_backward_launch(
            _wp_from_torch(grad_E, wp_dtype),
            _wp_from_torch(raw_energies, wp_dtype),
            _wp_from_torch(charges.to(input_dtype), wp_dtype),
            _wp_from_torch(batch_idx, wp.int32),
            _wp_from_torch(volumes.to(input_dtype), wp_dtype),
            _wp_from_torch(alpha.to(input_dtype), wp_dtype),
            _wp_from_torch(total_charges.to(input_dtype), wp_dtype),
            _wp_from_torch(grad_raw, wp_dtype),
            _wp_from_torch(grad_charges, wp_dtype),
            _wp_from_torch(grad_volumes, wp_dtype),
            _wp_from_torch(grad_alpha, wp_dtype),
            _wp_from_torch(grad_qtots, wp_dtype),
            wp_dtype=wp_dtype,
            device=device,
        )
    return grad_raw, grad_charges, grad_volumes, grad_alpha, grad_qtots


def _batch_energy_corrections_double_backward_launch(
    h_raw: torch.Tensor,
    h_chg: torch.Tensor,
    h_vol: torch.Tensor,
    h_alpha: torch.Tensor,
    h_qtot: torch.Tensor,
    grad_E: torch.Tensor,
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    from nvalchemiops.interactions.electrostatics.ewald_kernels import (
        batch_ewald_energy_corrections_double_backward as _batch_corrections_dbwd_launch,
    )

    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    device = wp.device_from_torch(raw_energies.device)

    grad_grad_E = torch.empty_like(grad_E)
    grad_raw = torch.empty_like(raw_energies)
    grad_charges = torch.empty_like(charges, dtype=input_dtype)
    grad_volumes = torch.zeros_like(volumes, dtype=input_dtype)
    grad_alpha = torch.zeros_like(alpha, dtype=input_dtype)
    grad_qtots = torch.zeros_like(total_charges, dtype=input_dtype)

    with _scoped_stream(raw_energies.device):
        _batch_corrections_dbwd_launch(
            _wp_from_torch(h_raw, wp_dtype),
            _wp_from_torch(h_chg, wp_dtype),
            _wp_from_torch(h_vol, wp_dtype),
            _wp_from_torch(h_alpha, wp_dtype),
            _wp_from_torch(h_qtot, wp_dtype),
            _wp_from_torch(grad_E, wp_dtype),
            _wp_from_torch(raw_energies, wp_dtype),
            _wp_from_torch(charges.to(input_dtype), wp_dtype),
            _wp_from_torch(batch_idx, wp.int32),
            _wp_from_torch(volumes.to(input_dtype), wp_dtype),
            _wp_from_torch(alpha.to(input_dtype), wp_dtype),
            _wp_from_torch(total_charges.to(input_dtype), wp_dtype),
            _wp_from_torch(grad_grad_E, wp_dtype),
            _wp_from_torch(grad_raw, wp_dtype),
            _wp_from_torch(grad_charges, wp_dtype),
            _wp_from_torch(grad_volumes, wp_dtype),
            _wp_from_torch(grad_alpha, wp_dtype),
            _wp_from_torch(grad_qtots, wp_dtype),
            wp_dtype=wp_dtype,
            device=device,
        )
    return grad_grad_E, grad_raw, grad_charges, grad_volumes, grad_alpha, grad_qtots


def register_ewald_corrections_ops() -> None:
    """Register the Ewald correction Torch custom-op chains once."""
    global _CORRECTIONS_BATCH, _CORRECTIONS_SINGLE, _EWALD_CORRECTIONS_OPS_REGISTERED
    if _EWALD_CORRECTIONS_OPS_REGISTERED:
        return

    _CORRECTIONS_SINGLE = register_warp_op_chain(
        name="nvalchemiops::ewald_energy_corrections",
        forward=_energy_corrections_forward_launch,
        backward=_energy_corrections_backward_launch,
        double_backward=_energy_corrections_double_backward_launch,
        diff_input_positions=(0, 1, 2, 3, 4),
        n_forward_inputs=5,
        second_order_diff_positions=(0, 1, 2, 3, 4, 5),
        n_backward_inputs=6,
    )

    _CORRECTIONS_BATCH = register_warp_op_chain(
        name="nvalchemiops::ewald_energy_corrections_batch",
        forward=_batch_energy_corrections_forward_launch,
        backward=_batch_energy_corrections_backward_launch,
        double_backward=_batch_energy_corrections_double_backward_launch,
        diff_input_positions=(0, 1, 3, 4, 5),
        n_forward_inputs=6,
        second_order_diff_positions=(0, 1, 2, 4, 5, 6),
        n_backward_inputs=7,
        batch_match=True,
    )

    _EWALD_CORRECTIONS_OPS_REGISTERED = True


def ewald_energy_corrections(*args, **kwargs):
    """Call the registered single-system Ewald correction op."""
    register_ewald_corrections_ops()
    if _CORRECTIONS_SINGLE is None:
        raise RuntimeError("Ewald correction single-system op registration failed")
    return _CORRECTIONS_SINGLE["forward"](*args, **kwargs)


def ewald_energy_corrections_batch(*args, **kwargs):
    """Call the registered batched Ewald correction op."""
    register_ewald_corrections_ops()
    if _CORRECTIONS_BATCH is None:
        raise RuntimeError("Ewald correction batched op registration failed")
    return _CORRECTIONS_BATCH["forward"](*args, **kwargs)
