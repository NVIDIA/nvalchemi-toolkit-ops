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

"""JAX registration of the shared geometry and population grid selector."""

from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import warp as wp

from nvalchemiops.jax.neighbors._registration import (
    _cached_jax_kernel_call,
    _register_jax_kernel,
)
from nvalchemiops.neighbors.cell_list._grid_selection import _get_pair_grid_kernel
from nvalchemiops.neighbors.cell_list.kernels import _DEFAULT_MIN_CELLS_PER_DIMENSION
from nvalchemiops.neighbors.cell_list.launchers import _PAIR_CENTRIC_BLOCK_DIM

__all__: list[str] = []


@cache
def _pair_grid_call(dtype: type, single_system: bool) -> Any:
    """Reuse the selector executable while passing current arrays each time."""
    return _cached_jax_kernel_call(
        _register_jax_kernel(
            _get_pair_grid_kernel(dtype, _PAIR_CENTRIC_BLOCK_DIM, single_system),
            ("grids", "radii", "counts"),
        )
    )


def _select_pair_grid(
    cell: jax.Array,
    pbc: jax.Array,
    boundaries: jax.Array,
    cutoff: float,
    capacity: int,
    single_system: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Select grids, radii and cell counts within equal per-system budgets.

    Geometry and populations are runtime inputs. Capacity limits the candidate
    grids using the existing build minimum and pair-query block width.
    """
    systems = cell.shape[0]
    capacity = int(capacity)
    if capacity < systems:
        raise ValueError(
            "cell workspace requires capacity for at least one cell per system"
        )
    dtype = wp.float64 if cell.dtype == jnp.float64 else wp.float32
    return _pair_grid_call(dtype, single_system)(
        cell,
        pbc.reshape(systems, 3),
        boundaries.astype(jnp.int32),
        0,
        float(cutoff),
        capacity // systems,
        _DEFAULT_MIN_CELLS_PER_DIMENSION,
        jnp.zeros((systems, 3), dtype=jnp.int32),
        jnp.zeros((systems, 3), dtype=jnp.int32),
        jnp.zeros(systems, dtype=jnp.int32),
        launch_dims=(systems,),
    )
