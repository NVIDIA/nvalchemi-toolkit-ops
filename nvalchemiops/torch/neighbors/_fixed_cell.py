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

"""Private cached geometry for fixed-cell Torch neighbor routes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class _FixedCellGeometry:
    """Read-only geometry prepared for repeated topology construction."""

    inv_cell: torch.Tensor | None
    cell: torch.Tensor | None = None
    cells_per_dimension: torch.Tensor | None = None
    neighbor_search_radius: torch.Tensor | None = None
    cells_per_system: torch.Tensor | None = None
    cell_offsets: torch.Tensor | None = None
    qr: torch.Tensor | None = None
    axis_aligned: torch.Tensor | None = None
    fractional_rounding_certified: torch.Tensor | None = None
    qr_height_certified: torch.Tensor | None = None
    bbox_cutoff_bounds: torch.Tensor | None = None
