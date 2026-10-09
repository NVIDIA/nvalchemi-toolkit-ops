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

"""Private JAX neighbor-route resolution helpers.

This module is intentionally separate from the public single-system and
batched frontends.  It keeps shared strategy decisions from becoming a
frontend-to-frontend private import; backend-specific sizing remains owned by
the public route that exposes its corresponding schema.
"""

from __future__ import annotations

from nvalchemiops.neighbors.cell_list import select_cell_list_strategy


def resolve_cell_strategy(
    strategy: str,
    *,
    total_atoms: int,
    cutoff: float,
    device_is_cpu: bool,
    half_fill: bool = False,
) -> str:
    """Resolve the JAX cell-list query strategy.

    ``auto`` uses the shared Warp heuristic on CUDA and atom-centric
    execution on CPU or for half-filled output.  Explicit pair-centric mode
    remains rejected on CPU by the same public route contract.
    """
    if strategy == "auto":
        if device_is_cpu or half_fill:
            return "atom_centric"
        return select_cell_list_strategy(int(total_atoms), float(cutoff))
    if strategy == "atom_centric":
        return "atom_centric"
    if strategy == "pair_centric":
        if device_is_cpu:
            raise ValueError(
                "strategy='pair_centric' is not supported on CPU "
                "(kernels use CUDA block scheduling).  Pass 'auto' or "
                "'atom_centric' instead.",
            )
        return "pair_centric"
    raise ValueError(
        f"strategy must be 'auto' | 'atom_centric' | 'pair_centric', got {strategy!r}",
    )


__all__ = ["resolve_cell_strategy"]
