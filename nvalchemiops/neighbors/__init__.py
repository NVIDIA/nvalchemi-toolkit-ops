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

"""Core Warp interface for neighbor-list operations.

This package exports Warp launchers that accept Warp arrays directly. For
PyTorch users, use :mod:`nvalchemiops.torch.neighbors` instead.
"""

from __future__ import annotations

import importlib
import warnings

from nvalchemiops.neighbors.base_dispatch import (
    estimate_neighbor_list_costs,
    suggest_neighbor_list_method,
)
from nvalchemiops.neighbors.cell_list import (
    batch_build_cell_list,
    batch_query_cell_list,
    build_cell_list,
    query_cell_list,
)
from nvalchemiops.neighbors.cluster_tile import (
    TILE_GROUP_SIZE,
    batch_build_cluster_tile_list,
    batch_query_cluster_tile,
    batch_query_cluster_tile_coo,
    build_cluster_tile_list,
    query_cluster_tile,
    query_cluster_tile_coo,
)
from nvalchemiops.neighbors.naive import (
    batch_naive_neighbor_matrix,
    batch_naive_neighbor_matrix_dual_cutoff,
    batch_naive_neighbor_matrix_pbc,
    batch_naive_neighbor_matrix_pbc_dual_cutoff,
    naive_neighbor_matrix,
    naive_neighbor_matrix_dual_cutoff,
    naive_neighbor_matrix_pbc,
    naive_neighbor_matrix_pbc_dual_cutoff,
)
from nvalchemiops.neighbors.neighbor_utils import (
    NeighborOverflowError,
    compute_naive_num_shifts,
    estimate_max_neighbors,
    zero_array,
)
from nvalchemiops.neighbors.rebuild import (
    check_batch_cell_list_rebuild,
    check_batch_neighbor_list_rebuild,
    check_cell_list_rebuild,
    check_neighbor_list_rebuild,
)


def __getattr__(name: str):  # pragma: no cover
    """Lazy import for backward compatibility with the old API."""
    if name in (
        "batch_cell_list",
        "batch_naive",
        "batch_naive_dual_cutoff",
        "cell_list",
        "cluster_tile",
        "naive",
        "naive_dual_cutoff",
        "neighbor_utils",
        "rebuild",
        "rebuild_detection",
    ):
        return importlib.import_module(f"nvalchemiops.neighbors.{name}")

    if name == "neighbor_list":
        if importlib.util.find_spec("torch") is None:
            warnings.warn(
                "From version 0.3.0 onwards, PyTorch is now an optional dependency"
                " and a PyTorch installation was not detected. This namespace is"
                " reserved for `warp` kernels directly. For end-users, import from"
                " `nvalchemiops.torch.neighbors` instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )

            def neighbor_list(*args, **kwargs):
                """Raise a `RuntimeError` if we can't use the new API with torch."""
                raise RuntimeError(
                    "PyTorch is required to use the previous `neighbor_list` API."
                    " Please install via `pip install 'nvalchemiops[torch]'`"
                    " and import from `nvalchemiops.torch.neighbors.neighbor_list` instead."
                )

            return neighbor_list
        warnings.warn(
            "From version 0.3.0 onwards, PyTorch is now an optional dependency"
            " and the `nvalchemiops.neighbors` namespace is reserved for `warp`"
            " kernels directly. For end-users, import from"
            " `nvalchemiops.torch.neighbors` instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        from nvalchemiops.torch.neighbors import neighbor_list

        return neighbor_list

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Supported public surface.  0.3.1-era names plus the new cluster-tile
# family launchers.  Kernel-factory getters, low-level helpers, and strategy
# selectors live only at their canonical package paths.
__all__ = [
    "NeighborOverflowError",
    "TILE_GROUP_SIZE",
    "batch_build_cell_list",
    "batch_build_cluster_tile_list",
    "batch_naive_neighbor_matrix",
    "batch_naive_neighbor_matrix_dual_cutoff",
    "batch_naive_neighbor_matrix_pbc",
    "batch_naive_neighbor_matrix_pbc_dual_cutoff",
    "batch_query_cell_list",
    "batch_query_cluster_tile",
    "batch_query_cluster_tile_coo",
    "build_cell_list",
    "build_cluster_tile_list",
    "check_batch_cell_list_rebuild",
    "check_batch_neighbor_list_rebuild",
    "check_cell_list_rebuild",
    "check_neighbor_list_rebuild",
    "compute_naive_num_shifts",
    "estimate_max_neighbors",
    "naive_neighbor_matrix",
    "naive_neighbor_matrix_dual_cutoff",
    "naive_neighbor_matrix_pbc",
    "naive_neighbor_matrix_pbc_dual_cutoff",
    "neighbor_list",
    "query_cell_list",
    "query_cluster_tile",
    "query_cluster_tile_coo",
    "estimate_neighbor_list_costs",
    "suggest_neighbor_list_method",
    "zero_array",
]
