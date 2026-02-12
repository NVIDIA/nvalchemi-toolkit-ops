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

"""Backwards-compatible proxy for nvalchemiops.neighbors.

.. deprecated:: 0.3.0
    This module has been renamed to :mod:`nvalchemiops.neighbors`.
    Please update your imports:

    - Change: ``from nvalchemiops.neighborlist import ...``
    - To: ``from nvalchemiops.neighbors import ...``

    This proxy module will be removed in a future version.
"""

from __future__ import annotations

import sys
import warnings

from nvalchemiops import neighbors as _neighbors
from nvalchemiops.neighbors import (
    NeighborOverflowError,
    estimate_max_neighbors,
)
from nvalchemiops.neighbors import (
    # Import new names and re-export with old wp_* names for backwards compatibility
    batch_build_cell_list as wp_batch_build_cell_list,
)
from nvalchemiops.neighbors import (
    batch_naive_neighbor_matrix as wp_batch_naive_neighbor_matrix,
)
from nvalchemiops.neighbors import (
    batch_naive_neighbor_matrix_dual_cutoff as wp_batch_naive_neighbor_matrix_dual_cutoff,
)
from nvalchemiops.neighbors import (
    batch_naive_neighbor_matrix_pbc as wp_batch_naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors import (
    batch_naive_neighbor_matrix_pbc_dual_cutoff as wp_batch_naive_neighbor_matrix_pbc_dual_cutoff,
)
from nvalchemiops.neighbors import (
    batch_query_cell_list as wp_batch_query_cell_list,
)
from nvalchemiops.neighbors import (
    build_cell_list as wp_build_cell_list,
)
from nvalchemiops.neighbors import (
    check_cell_list_rebuild as wp_check_cell_list_rebuild,
)
from nvalchemiops.neighbors import (
    check_neighbor_list_rebuild as wp_check_neighbor_list_rebuild,
)
from nvalchemiops.neighbors import (
    compute_naive_num_shifts as wp_compute_naive_num_shifts,
)
from nvalchemiops.neighbors import (
    naive_neighbor_matrix as wp_naive_neighbor_matrix,
)
from nvalchemiops.neighbors import (
    naive_neighbor_matrix_dual_cutoff as wp_naive_neighbor_matrix_dual_cutoff,
)
from nvalchemiops.neighbors import (
    naive_neighbor_matrix_pbc as wp_naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors import (
    naive_neighbor_matrix_pbc_dual_cutoff as wp_naive_neighbor_matrix_pbc_dual_cutoff,
)
from nvalchemiops.neighbors import (
    query_cell_list as wp_query_cell_list,
)
from nvalchemiops.neighbors import (
    zero_array as wp_zero_array,
)

# Emit deprecation warning on first import of this module
warnings.warn(
    "The 'nvalchemiops.neighborlist' module has been renamed to 'nvalchemiops.neighbors'. "
    "Please update your imports from 'nvalchemiops.neighborlist' to 'nvalchemiops.neighbors'. "
    "This backwards-compatible alias will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
# Create module aliases for backwards compatibility with submodule imports
sys.modules["nvalchemiops.neighborlist.naive"] = _neighbors.naive
sys.modules["nvalchemiops.neighborlist.naive_dual_cutoff"] = (
    _neighbors.naive_dual_cutoff
)
sys.modules["nvalchemiops.neighborlist.cell_list"] = _neighbors.cell_list
sys.modules["nvalchemiops.neighborlist.batch_cell_list"] = _neighbors.batch_cell_list
sys.modules["nvalchemiops.neighborlist.batch_naive"] = _neighbors.batch_naive
sys.modules["nvalchemiops.neighborlist.batch_naive_dual_cutoff"] = (
    _neighbors.batch_naive_dual_cutoff
)
sys.modules["nvalchemiops.neighborlist.neighbor_utils"] = _neighbors.neighbor_utils
sys.modules["nvalchemiops.neighborlist.rebuild_detection"] = (
    _neighbors.rebuild_detection
)

# Also expose submodules as attributes for `import nvalchemiops.neighborlist.naive` style
naive = _neighbors.naive
naive_dual_cutoff = _neighbors.naive_dual_cutoff
cell_list = _neighbors.cell_list
batch_cell_list = _neighbors.batch_cell_list
batch_naive = _neighbors.batch_naive
batch_naive_dual_cutoff = _neighbors.batch_naive_dual_cutoff
neighbor_utils = _neighbors.neighbor_utils
rebuild_detection = _neighbors.rebuild_detection


def __getattr__(name: str):  # pragma: no cover
    """Lazy import for backward compatibility with the old API.

    This handles the `neighbor_list` attribute which requires PyTorch.
    """
    if name == "neighbor_list":
        import importlib.util

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
                " and the `nvalchemiops.neighborlist` namespace is reserved for `warp`"
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
    "wp_naive_neighbor_matrix",
    "wp_naive_neighbor_matrix_pbc",
    "wp_build_cell_list",
    "wp_query_cell_list",
    "wp_batch_naive_neighbor_matrix",
    "wp_batch_naive_neighbor_matrix_pbc",
    "wp_batch_build_cell_list",
    "wp_batch_query_cell_list",
    "wp_naive_neighbor_matrix_dual_cutoff",
    "wp_naive_neighbor_matrix_pbc_dual_cutoff",
    "wp_batch_naive_neighbor_matrix_dual_cutoff",
    "wp_batch_naive_neighbor_matrix_pbc_dual_cutoff",
    "wp_check_cell_list_rebuild",
    "wp_check_neighbor_list_rebuild",
    "wp_compute_naive_num_shifts",
    "wp_zero_array",
    "estimate_max_neighbors",
    "neighbor_list",
    # Submodules for backwards compatibility
    "naive",
    "naive_dual_cutoff",
    "cell_list",
    "batch_cell_list",
    "batch_naive",
    "batch_naive_dual_cutoff",
    "neighbor_utils",
    "rebuild_detection",
]
