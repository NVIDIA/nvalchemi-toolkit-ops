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

"""Core warp interface for neighbor list operations.

This module exports warp launchers that accept warp arrays directly.
For PyTorch users, use `nvalchemiops.torch.neighbors` instead.
"""

from __future__ import annotations

import importlib
import warnings

# Warp launchers from batch_cell_list
from nvalchemiops.neighbors.batch_cell_list import (
    batch_build_cell_list,
    batch_query_cell_list,
)

# Warp launchers from batch_naive
from nvalchemiops.neighbors.batch_naive import (
    batch_naive_neighbor_matrix,
    batch_naive_neighbor_matrix_pbc,
)

# Warp launchers from batch_naive_dual_cutoff
from nvalchemiops.neighbors.batch_naive_dual_cutoff import (
    batch_naive_neighbor_matrix_dual_cutoff,
    batch_naive_neighbor_matrix_pbc_dual_cutoff,
)

# Warp launchers from cell_list
from nvalchemiops.neighbors.cell_list import (
    build_cell_list,
    query_cell_list,
)

# Warp launchers from naive
from nvalchemiops.neighbors.naive import (
    naive_neighbor_matrix,
    naive_neighbor_matrix_pbc,
)

# Warp launchers from naive_dual_cutoff
from nvalchemiops.neighbors.naive_dual_cutoff import (
    naive_neighbor_matrix_dual_cutoff,
    naive_neighbor_matrix_pbc_dual_cutoff,
)

# Warp utilities and launchers from neighbor_utils
from nvalchemiops.neighbors.neighbor_utils import (
    NeighborOverflowError,
    compute_naive_num_shifts,
    estimate_max_neighbors,
    zero_array,
)

# Warp launchers from rebuild_detection
from nvalchemiops.neighbors.rebuild_detection import (
    check_cell_list_rebuild,
    check_neighbor_list_rebuild,
)


def __getattr__(name: str):  # pragma: no cover
    """Lazy import for backward compatibility with the old API.

    This avoids circular imports by deferring the import of `neighbor_list`
    from `nvalchemiops.torch.neighbors` until it is actually accessed.
    """
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
        else:
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


__all__ = [
    "NeighborOverflowError",
    "naive_neighbor_matrix",
    "naive_neighbor_matrix_pbc",
    "build_cell_list",
    "query_cell_list",
    "batch_naive_neighbor_matrix",
    "batch_naive_neighbor_matrix_pbc",
    "batch_build_cell_list",
    "batch_query_cell_list",
    "naive_neighbor_matrix_dual_cutoff",
    "naive_neighbor_matrix_pbc_dual_cutoff",
    "batch_naive_neighbor_matrix_dual_cutoff",
    "batch_naive_neighbor_matrix_pbc_dual_cutoff",
    "check_cell_list_rebuild",
    "check_neighbor_list_rebuild",
    "compute_naive_num_shifts",
    "zero_array",
    "estimate_max_neighbors",
    "neighbor_list",
]
