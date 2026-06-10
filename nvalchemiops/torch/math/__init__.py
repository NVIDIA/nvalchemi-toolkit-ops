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
PyTorch bindings and host-side scaffolding for the ``nvalchemiops.math`` layer.

The Warp kernels and ``@wp.func`` primitives live in :mod:`nvalchemiops.math`;
this package provides:

* Host-side precompute helpers that evaluate into Warp-ready constant tables
  (e.g. GTO basis normalization, GTO-GTO self-overlap integrals).
* Thin PyTorch launch wrappers used by tests and downstream bindings. These
  wrappers stay minimal on purpose — ``torch.compile`` / autograd-aware
  wrappers are only added when a specific pipeline needs them, following the
  ``nvalchemiops.warp_dispatch`` pattern used by
  :mod:`nvalchemiops.torch.interactions.electrostatics`.
"""

from nvalchemiops.torch.math.gto import (
    NormMode,
    eval_gto_density_pytorch,
    eval_gto_fourier_pytorch,
    inv_cl,
    inv_cl_table,
)
from nvalchemiops.torch.math.gto_self_overlap import (
    FIELD_CONSTANT,
    compute_overlap_constants,
    flatten_to_reference_layout,
)
from nvalchemiops.torch.math.solid_harmonics import (
    eval_irregular_solid_harmonics_pytorch,
    eval_regular_solid_harmonics_pytorch,
)

__all__ = [
    # Normalization modes (data + host-side helpers)
    "NormMode",
    "inv_cl",
    "inv_cl_table",
    # Self-overlap constants
    "FIELD_CONSTANT",
    "compute_overlap_constants",
    "flatten_to_reference_layout",
    # PyTorch launch wrappers (test-oriented)
    "eval_gto_density_pytorch",
    "eval_gto_fourier_pytorch",
    "eval_regular_solid_harmonics_pytorch",
    "eval_irregular_solid_harmonics_pytorch",
]
