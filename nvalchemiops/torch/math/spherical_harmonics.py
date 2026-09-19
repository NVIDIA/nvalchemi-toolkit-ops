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

"""PyTorch launch wrappers for the core spherical-harmonic Warp kernels."""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.math.spherical_harmonics import (
    _eval_spherical_harmonics_gradient_kernel,
    _eval_spherical_harmonics_kernel,
)
from nvalchemiops.torch._warp_op_helpers import scoped_warp_stream

__all__ = [
    "eval_spherical_harmonics_gradient_pytorch",
    "eval_spherical_harmonics_pytorch",
]


def eval_spherical_harmonics_pytorch(
    positions: torch.Tensor,
    L_max: int = 2,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Evaluate spherical harmonics from PyTorch tensors.

    Parameters
    ----------
    positions : torch.Tensor
        Float64 Cartesian positions with shape ``(N, 3)``.
    L_max : int, optional
        Maximum angular momentum; supported values are 0, 1, and 2.
    device : torch.device, optional
        Output and execution device. Defaults to ``positions.device``.

    Returns
    -------
    torch.Tensor
        Harmonic values with shape ``(N, 1)``, ``(N, 4)``, or ``(N, 9)``.
    """
    if device is None:
        device = positions.device
    n_atoms = positions.shape[0]
    num_components = {0: 1, 1: 4, 2: 9}[L_max]
    with scoped_warp_stream(positions.device):
        output = torch.zeros(
            (n_atoms, num_components), dtype=torch.float64, device=device
        )
        wp.launch(
            kernel=_eval_spherical_harmonics_kernel,
            dim=n_atoms,
            inputs=[wp.from_torch(positions.contiguous(), dtype=wp.vec3d), L_max],
            outputs=[wp.from_torch(output, dtype=wp.float64)],
            device=wp.device_from_torch(device),
        )
    return output


def eval_spherical_harmonics_gradient_pytorch(
    positions: torch.Tensor,
    L_max: int = 2,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Evaluate spherical-harmonic gradients from PyTorch tensors.

    Parameters
    ----------
    positions : torch.Tensor
        Float64 Cartesian positions with shape ``(N, 3)``.
    L_max : int, optional
        Maximum angular momentum; supported values are 0, 1, and 2.
    device : torch.device, optional
        Output and execution device. Defaults to ``positions.device``.

    Returns
    -------
    torch.Tensor
        Harmonic gradients with shape ``(N, 1, 3)``, ``(N, 4, 3)``, or
        ``(N, 9, 3)``.
    """
    if device is None:
        device = positions.device
    n_atoms = positions.shape[0]
    num_components = {0: 1, 1: 4, 2: 9}[L_max]
    with scoped_warp_stream(positions.device):
        output = torch.zeros(
            (n_atoms, num_components, 3), dtype=torch.float64, device=device
        )
        wp.launch(
            kernel=_eval_spherical_harmonics_gradient_kernel,
            dim=n_atoms,
            inputs=[wp.from_torch(positions.contiguous(), dtype=wp.vec3d), L_max],
            outputs=[wp.from_torch(output, dtype=wp.vec3d)],
            device=wp.device_from_torch(device),
        )
    return output
