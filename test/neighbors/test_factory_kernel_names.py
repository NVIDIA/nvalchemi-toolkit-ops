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

"""Tests for generated Warp factory specialization names."""

from typing import Any

import pytest
import warp as wp

from nvalchemiops.neighbors.cell_list import (
    get_build_cell_list_kernel,
    get_query_cell_list_kernel,
)
from nvalchemiops.neighbors.cell_list import kernels as cell_list_kernels
from nvalchemiops.neighbors.cluster_tile import kernels as cluster_tile_kernels
from nvalchemiops.neighbors.naive import get_naive_neighbor_matrix_kernel
from nvalchemiops.neighbors.neighbor_utils import (
    _get_fill_neighbor_matrix_tail_kernel,
    get_compute_inv_cells_kernel,
    get_compute_naive_num_shifts_kernel,
    get_update_ref_positions_kernel,
    get_wrap_positions_kernel,
)
from nvalchemiops.neighbors.rebuild import (
    get_cell_list_rebuild_kernel,
    get_neighbor_list_rebuild_kernel,
)


@wp.func
def _factory_name_pair_fn(
    r_ij: Any,
    distance: Any,
    pair_params: wp.array(dtype=Any, ndim=2),
    i: int,
    j: int,
):
    """Tiny pair function used only to specialize factory names."""
    return distance + pair_params[i, 0] - pair_params[j, 0], -r_ij


def _assert_kernel_name(kernel: wp.Kernel, expected: str) -> None:
    """Assert visible Warp kernel names use the specialization name."""
    assert kernel.__name__ == expected
    assert kernel.__qualname__ == expected
    assert kernel.key == expected
    if (
        expected.startswith("_compute_naive_num_shifts__")
        or expected.startswith("_compute_inv_cells_kernel__")
        or expected.startswith("_update_ref_positions_kernel__")
    ):
        # get_compute_naive_num_shifts_kernel, get_compute_inv_cells_kernel,
        # and get_update_ref_positions_kernel use wp.overload on module-level
        # kernel functions. Warp stores the stable specialization on the
        # Kernel object, while kernel.func is the shared Python function and
        # may carry the most recently generated dtype suffix after other
        # modules are imported during pytest collection.
        prefix = expected.split("__")[0] + "__"
        assert kernel.func.__name__.startswith(prefix)
    else:
        assert kernel.func.__name__ == expected
    assert "locals" not in expected
    assert "_make_" not in expected


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        (
            get_naive_neighbor_matrix_kernel(
                wp.float32,
                pbc_mode="prewrapped",
                selective=True,
                partial=True,
                return_vectors=True,
                return_distances=True,
                pair_fn=_factory_name_pair_fn,
            ),
            "_fill_naive_neighbor_matrix_pbc_prewrapped_selective__partial_vectors_distances_pair_fn__f32",
        ),
        (
            get_naive_neighbor_matrix_kernel(
                wp.float32,
                pbc_mode="wrap_on_entry",
                batched=True,
                strategy="tile",
            ),
            "_fill_batch_naive_neighbor_matrix_pbc__tile__f32",
        ),
        (
            get_build_cell_list_kernel("count_atoms", wp.float64, batched=True),
            "_batch_cell_list_count_atoms_per_bin__f64",
        ),
        (
            get_query_cell_list_kernel(
                wp.float32,
                strategy="atom_centric",
                selective=True,
                partial=True,
                return_vectors=True,
                return_distances=True,
                pair_fn=_factory_name_pair_fn,
            ),
            "_cell_list_build_neighbor_matrix_selective__atom_centric_partial_vectors_distances_pair_fn__f32",
        ),
        (
            get_query_cell_list_kernel(
                wp.float32,
                strategy="pair_centric",
                batched=True,
                return_vectors=True,
            ),
            "_batch_cell_list_build_neighbor_matrix__pair_centric_vectors__f32",
        ),
        (
            cluster_tile_kernels._get_build_cluster_tiles_kernel(batched=True),
            "_build_cluster_tiles__batched__f32",
        ),
        (
            cluster_tile_kernels.get_batch_query_cluster_tile_kernel(
                return_vectors=True,
                return_distances=True,
                pair_fn=_factory_name_pair_fn,
            ),
            "_query_cluster_tile__batched_vectors_distances_pair_fn__f32",
        ),
        (
            cluster_tile_kernels.get_query_cluster_tile_coo_kernel(
                return_distances=True,
            ),
            "_query_cluster_tile_coo__distances__f32",
        ),
        (
            cluster_tile_kernels.get_query_cluster_tile_kernel(
                selective=True,
                dual_cutoff=True,
            ),
            "_query_cluster_tile__selective_dual_cutoff__f32",
        ),
        (
            cluster_tile_kernels.get_batch_query_cluster_tile_kernel(
                tile_segmented=True,
                selective=True,
                dual_cutoff=True,
            ),
            "_query_cluster_tile__batched_tile_segmented_selective_dual_cutoff__f32",
        ),
        (
            cluster_tile_kernels.get_batch_query_cluster_tile_coo_kernel(
                tile_segmented=True,
                coo_segmented=True,
                selective=True,
                return_distances=True,
            ),
            "_query_cluster_tile_coo__batched_tile_segmented_selective_coo_segmented_distances__f32",
        ),
        (
            get_neighbor_list_rebuild_kernel(wp.float64, batched=True, pbc=True),
            "_check_batch_atoms_moved_beyond_skin_pbc__f64",
        ),
        (
            get_cell_list_rebuild_kernel(wp.float16),
            "_check_atoms_changed_cells__f16",
        ),
        (
            get_compute_naive_num_shifts_kernel(wp.float32),
            "_compute_naive_num_shifts__f32",
        ),
        (
            get_compute_inv_cells_kernel(wp.float64),
            "_compute_inv_cells_kernel__f64",
        ),
        (
            get_wrap_positions_kernel(wp.float32, batched=True),
            "_wrap_positions_batch_kernel__f32",
        ),
        (
            get_update_ref_positions_kernel(wp.float64),
            "_update_ref_positions_kernel__f64",
        ),
        (
            _get_fill_neighbor_matrix_tail_kernel(32),
            "_fill_neighbor_matrix_tail__block_32",
        ),
    ],
)
def test_factory_kernel_names_are_specialized(kernel: wp.Kernel, expected: str) -> None:
    """Factory-created kernels carry stable 0.3.1-style specialization names."""
    _assert_kernel_name(kernel, expected)


def test_factory_func_name_is_specialized() -> None:
    """Factory-created ``wp.func`` helpers carry the same naming convention."""
    fn = cell_list_kernels._make_store_neighbor_fn(
        wp.float32,
        return_vectors=True,
        return_distances=True,
        pair_fn=_factory_name_pair_fn,
    )

    expected = "_cell_list_store_neighbor__vectors_distances_pair_fn__f32"
    assert fn.__name__ == expected
    assert fn.__qualname__ == expected
