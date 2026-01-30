# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""JAX neighbor list API.

This module provides JAX bindings for neighbor list computation and related utilities
for both single and batched systems.
"""

from __future__ import annotations

from nvalchemiops.jax.neighbors.batched import (
    batch_build_cell_list,
    batch_cell_list,
    batch_naive_neighbor_list,
    batch_query_cell_list,
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.jax.neighbors.neighbor_utils import (
    NeighborOverflowError,
    allocate_cell_list,
    compute_naive_num_shifts,
    estimate_max_neighbors,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)
from nvalchemiops.jax.neighbors.rebuild_detection import (
    cell_list_needs_rebuild,
    check_cell_list_rebuild_needed,
    check_neighbor_list_rebuild_needed,
    neighbor_list_needs_rebuild,
)
from nvalchemiops.jax.neighbors.unbatched import (
    build_cell_list,
    cell_list,
    estimate_cell_list_sizes,
    naive_neighbor_list,
    query_cell_list,
)

__all__ = [
    # Unbatched neighbor list
    "naive_neighbor_list",
    "estimate_cell_list_sizes",
    "build_cell_list",
    "query_cell_list",
    "cell_list",
    # Batched neighbor list
    "batch_naive_neighbor_list",
    "estimate_batch_cell_list_sizes",
    "batch_build_cell_list",
    "batch_query_cell_list",
    "batch_cell_list",
    # Rebuild detection
    "cell_list_needs_rebuild",
    "neighbor_list_needs_rebuild",
    "check_cell_list_rebuild_needed",
    "check_neighbor_list_rebuild_needed",
    # Utilities
    "compute_naive_num_shifts",
    "get_neighbor_list_from_neighbor_matrix",
    "prepare_batch_idx_ptr",
    "allocate_cell_list",
    "estimate_max_neighbors",
    "NeighborOverflowError",
]
