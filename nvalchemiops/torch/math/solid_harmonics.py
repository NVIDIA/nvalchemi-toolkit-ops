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
Solid Harmonics — PyTorch launch wrappers
=========================================

Thin wrappers that drive the ``@wp.kernel`` evaluators in
:mod:`nvalchemiops.math.solid_harmonics` from PyTorch tensors. Used by tests
today. Autograd-aware wrappers will be added in later phases via the
:mod:`nvalchemiops.warp_dispatch` pattern when downstream multipole pipelines
need gradients through these primitives; until then, the wrappers intentionally
do not participate in autograd.
"""

from __future__ import annotations

import warp as wp

from nvalchemiops.math.solid_harmonics import (
    _eval_irregular_solid_harmonics_kernel,
    _eval_regular_solid_harmonics_kernel,
)


def eval_regular_solid_harmonics_pytorch(
    positions,
    max_L: int = 1,
    device=None,
):
    """Evaluate regular solid harmonics from PyTorch tensors.

    Parameters
    ----------
    positions : torch.Tensor
        Positions of shape ``(N, 3)``, ``float64``.
    max_L : int
        Maximum angular momentum (``0`` or ``1``).
    device : torch.device, optional
        Compute device; defaults to ``positions.device``.

    Returns
    -------
    torch.Tensor
        ``(N, (max_L + 1)**2)`` tensor of regular solid harmonic values,
        ordered ``[R_0^0, R_1^{-1}, R_1^{0}, R_1^{+1}]``.
    """
    import torch

    if max_L not in (0, 1):
        raise ValueError(f"max_L must be 0 or 1 (L=2,3 not yet supported), got {max_L}")

    if device is None:
        device = positions.device

    N = positions.shape[0]
    num_components = (max_L + 1) ** 2
    output = torch.zeros((N, num_components), dtype=torch.float64, device=device)

    wp_device = wp.device_from_torch(device)
    wp_positions = wp.from_torch(positions.contiguous(), dtype=wp.vec3d)
    wp_output = wp.from_torch(output, dtype=wp.float64)

    wp.launch(
        kernel=_eval_regular_solid_harmonics_kernel,
        dim=N,
        inputs=[wp_positions, max_L],
        outputs=[wp_output],
        device=wp_device,
    )
    return output


def eval_irregular_solid_harmonics_pytorch(
    positions,
    max_L: int = 1,
    device=None,
):
    """Evaluate irregular solid harmonics from PyTorch tensors.

    Parameters
    ----------
    positions : torch.Tensor
        Positions of shape ``(N, 3)``, ``float64``. All entries must have
        :math:`|r| > 0`.
    max_L : int
        Maximum angular momentum (``0`` or ``1``).
    device : torch.device, optional
        Compute device; defaults to ``positions.device``.

    Returns
    -------
    torch.Tensor
        ``(N, (max_L + 1)**2)`` tensor of irregular solid harmonic values,
        ordered ``[I_0^0, I_1^{-1}, I_1^{0}, I_1^{+1}]``.
    """
    import torch

    if max_L not in (0, 1):
        raise ValueError(f"max_L must be 0 or 1 (L=2,3 not yet supported), got {max_L}")

    if device is None:
        device = positions.device

    N = positions.shape[0]
    num_components = (max_L + 1) ** 2
    output = torch.zeros((N, num_components), dtype=torch.float64, device=device)

    wp_device = wp.device_from_torch(device)
    wp_positions = wp.from_torch(positions.contiguous(), dtype=wp.vec3d)
    wp_output = wp.from_torch(output, dtype=wp.float64)

    wp.launch(
        kernel=_eval_irregular_solid_harmonics_kernel,
        dim=N,
        inputs=[wp_positions, max_L],
        outputs=[wp_output],
        device=wp_device,
    )
    return output
