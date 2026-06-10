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

"""Public naive neighbor-list launchers and kernel getters."""

from nvalchemiops.neighbors.naive.launchers import (
    batch_naive_neighbor_matrix,
    batch_naive_neighbor_matrix_dual_cutoff,
    batch_naive_neighbor_matrix_pbc,
    batch_naive_neighbor_matrix_pbc_dual_cutoff,
    get_naive_neighbor_matrix_dual_cutoff_kernel,
    get_naive_neighbor_matrix_kernel,
    naive_neighbor_matrix,
    naive_neighbor_matrix_dual_cutoff,
    naive_neighbor_matrix_pbc,
    naive_neighbor_matrix_pbc_dual_cutoff,
)

__all__ = [
    "batch_naive_neighbor_matrix",
    "batch_naive_neighbor_matrix_dual_cutoff",
    "batch_naive_neighbor_matrix_pbc",
    "batch_naive_neighbor_matrix_pbc_dual_cutoff",
    "get_naive_neighbor_matrix_dual_cutoff_kernel",
    "get_naive_neighbor_matrix_kernel",
    "naive_neighbor_matrix",
    "naive_neighbor_matrix_dual_cutoff",
    "naive_neighbor_matrix_pbc",
    "naive_neighbor_matrix_pbc_dual_cutoff",
]
