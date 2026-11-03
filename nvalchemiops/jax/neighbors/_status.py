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

"""Shared private status-tail construction for JAX cell-list routes."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

__all__ = ["_build_cell_list_status_tail", "_build_cluster_status_tail"]


def _build_cell_list_status_tail(
    *,
    requested_cells: Any,
    available_cells: Any,
    required_rows: Any,
    available_rows: Any,
    required_coo: Any,
    available_coo: Any,
    metadata_valid: Any,
    coo_capacity: int | None,
) -> tuple[Any, ...]:
    """Build the stable eleven-field cell-list diagnostic suffix."""
    metadata_bad = jnp.broadcast_to(~metadata_valid, required_rows.shape)
    return (
        jnp.where(metadata_bad, 3, 0).astype(jnp.int32),
        required_rows > available_rows,
        (required_coo > available_coo)
        if coo_capacity is not None
        else jnp.zeros_like(required_rows, dtype=jnp.bool_),
        metadata_bad,
        requested_cells > available_cells,
        required_rows,
        available_rows,
        required_coo,
        available_coo,
        requested_cells,
        available_cells,
    )


def _build_cluster_status_tail(
    *,
    tile_failure: Any,
    tile_required: Any,
    tile_capacity: Any,
    coo_required: Any | None = None,
    coo_capacity: Any | None = None,
) -> tuple[Any, ...]:
    """Build the stable six-field cluster diagnostic suffix."""
    if coo_required is None:
        coo_required = jnp.zeros_like(tile_required)
    if coo_capacity is None:
        coo_capacity = jnp.zeros_like(tile_required)
    return (
        tile_failure,
        coo_required > coo_capacity,
        tile_required,
        tile_capacity,
        coo_required,
        coo_capacity,
    )
