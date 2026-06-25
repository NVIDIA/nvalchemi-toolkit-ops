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

"""Public cluster-pair tile neighbor-list launchers and kernel getters."""

from nvalchemiops.neighbors.cluster_tile.launchers import (
    TILE_GROUP_SIZE,
    batch_build_cluster_tile_list,
    batch_query_cluster_tile,
    batch_query_cluster_tile_coo,
    build_cluster_tile_list,
    estimate_batch_cluster_tile_segments,
    estimate_batch_max_tiles_per_group,
    estimate_max_tiles_per_group,
    get_batch_query_cluster_tile_kernel,
    get_query_cluster_tile_kernel,
    query_cluster_tile,
    query_cluster_tile_coo,
)

__all__ = [
    "TILE_GROUP_SIZE",
    "batch_build_cluster_tile_list",
    "batch_query_cluster_tile",
    "batch_query_cluster_tile_coo",
    "build_cluster_tile_list",
    "estimate_batch_cluster_tile_segments",
    "estimate_batch_max_tiles_per_group",
    "estimate_max_tiles_per_group",
    "get_batch_query_cluster_tile_kernel",
    "get_query_cluster_tile_kernel",
    "query_cluster_tile",
    "query_cluster_tile_coo",
]
