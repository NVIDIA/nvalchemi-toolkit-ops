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

"""
PyTorch Adapters for nvalchemiops
=================================

Thin wrappers that convert PyTorch tensors to Warp arrays, call the
Warp-first core API, and manage scratch buffer allocation via
PyTorch's CUDA caching allocator.

Submodules
----------
fire2
    PyTorch adapter for the FIRE2 optimizer
    (coordinate-only and variable-cell).
segment_ops
    Differentiable PyTorch wrappers for segmented reductions and broadcasts
    with explicit first- and second-order backward support.
"""

import importlib

if importlib.util.find_spec("torch") is None:
    raise ImportError(
        "PyTorch is required for `nvalchemiops.torch` namespace."
        " Please install via `pip install 'nvalchemiops[torch]'`."
    )

from nvalchemiops.torch.fire2 import (
    fire2_step_coord,
    fire2_step_coord_cell,
    fire2_step_extended,
)
from nvalchemiops.torch.segment_ops import (
    segmented_dot,
    segmented_matvec,
    segmented_mean,
    segmented_mul,
    segmented_rms_norm,
    segmented_sum,
)
from nvalchemiops.torch.types import (
    get_wp_dtype,
    get_wp_mat_dtype,
    get_wp_vec_dtype,
)

__all__ = [
    "get_wp_dtype",
    "get_wp_mat_dtype",
    "get_wp_vec_dtype",
    "fire2_step_coord",
    "fire2_step_coord_cell",
    "fire2_step_extended",
    "segmented_dot",
    "segmented_matvec",
    "segmented_mean",
    "segmented_mul",
    "segmented_rms_norm",
    "segmented_sum",
]
