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

"""Public cell-list launchers, strategy helpers, and kernel getters."""

from nvalchemiops.neighbors.cell_list.launchers import (
    PAIR_CENTRIC_MAX_LINEAR_LAUNCH,
    batch_build_cell_list,
    batch_query_cell_list,
    batch_query_cell_list_pair_centric_sorted,
    build_cell_list,
    compute_batch_pair_centric_n_outer,
    get_build_cell_list_kernel,
    get_cell_list_cells_per_system_kernel,
    get_cell_list_gather_kernel,
    get_query_cell_list_kernel,
    is_pair_centric_launch_safe,
    is_pair_centric_parallelism_sufficient,
    pair_centric_launch_size,
    query_cell_list,
    query_cell_list_atom_centric_sorted,
    query_cell_list_pair_centric_sorted,
    select_batch_cell_list_strategy,
    select_cell_list_strategy,
)

__all__ = [
    "PAIR_CENTRIC_MAX_LINEAR_LAUNCH",
    "batch_build_cell_list",
    "batch_query_cell_list",
    "batch_query_cell_list_pair_centric_sorted",
    "build_cell_list",
    "compute_batch_pair_centric_n_outer",
    "is_pair_centric_launch_safe",
    "is_pair_centric_parallelism_sufficient",
    "get_build_cell_list_kernel",
    "get_cell_list_cells_per_system_kernel",
    "get_cell_list_gather_kernel",
    "get_query_cell_list_kernel",
    "query_cell_list",
    "query_cell_list_atom_centric_sorted",
    "pair_centric_launch_size",
    "query_cell_list_pair_centric_sorted",
    "select_batch_cell_list_strategy",
    "select_cell_list_strategy",
]
