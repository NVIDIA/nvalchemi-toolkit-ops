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

"""Tests for generated Warp factory docstrings."""

from __future__ import annotations

import re
from typing import Any

import pytest
import warp as wp

from nvalchemiops.neighbors.cell_list import get_query_cell_list_kernel
from nvalchemiops.neighbors.cell_list import kernels as cell_list_kernels
from nvalchemiops.neighbors.cluster_tile import kernels as cluster_tile_kernels
from nvalchemiops.neighbors.naive import get_naive_neighbor_matrix_kernel
from nvalchemiops.neighbors.neighbor_utils import (
    get_compute_naive_num_shifts_kernel,
    get_wrap_positions_kernel,
)
from nvalchemiops.neighbors.rebuild import get_neighbor_list_rebuild_kernel


@wp.func
def _factory_doc_pair_fn(
    r_ij: Any,
    distance: Any,
    pair_params: wp.array(dtype=Any, ndim=2),
    i: int,
    j: int,
):
    """Tiny pair function used only to specialize factory docs."""
    return distance + pair_params[i, 0] - pair_params[j, 0], -r_ij


def _assert_runtime_doc(obj: object, *expected: str) -> None:
    """Assert runtime docs include specialization details on all Warp surfaces."""
    doc = getattr(obj, "__doc__", None)
    assert doc is not None
    assert "Specialization" in doc
    for text in expected:
        assert re.search(rf"(?<!\w){re.escape(text)}(?!\w)", doc) is not None

    wrapped = getattr(obj, "func", None)
    if wrapped is not None:
        key = getattr(obj, "key", "")
        is_overloaded = (
            key.startswith("_compute_naive_num_shifts__")
            or key.startswith("_compute_inv_cells_kernel__")
            or key.startswith("_update_ref_positions_kernel__")
        )
        if not is_overloaded:
            assert wrapped.__doc__ == doc

    warp_doc = getattr(obj, "doc", None)
    if warp_doc is not None:
        assert warp_doc == doc


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (
            get_naive_neighbor_matrix_kernel(
                wp.float32,
                pbc_mode="prewrapped",
                selective=True,
                partial=True,
                return_vectors=True,
                return_distances=True,
                pair_fn=_factory_doc_pair_fn,
            ),
            (
                "dtype : f32",
                "pbc_mode : prewrapped",
                "partial : True",
                "pair_fn : True",
            ),
        ),
        (
            get_query_cell_list_kernel(
                wp.float64,
                strategy="pair_centric",
                batched=True,
                return_vectors=True,
                coarsened=True,
            ),
            (
                "dtype : f64",
                "strategy : pair_centric",
                "batched : True",
                "coarsened : True",
            ),
        ),
        (
            get_query_cell_list_kernel(
                wp.float64,
                strategy="pair_centric",
                batched=True,
                return_vectors=True,
            ),
            (
                "dtype : f64",
                "strategy : pair_centric",
                "batched : True",
                "coarsened : False",
            ),
        ),
        (
            cluster_tile_kernels.get_batch_query_cluster_tile_coo_kernel(
                return_distances=True,
            ),
            ("dtype : f32", "output : coo", "batched : True"),
        ),
        (
            get_neighbor_list_rebuild_kernel(wp.float64, batched=True, pbc=True),
            ("dtype : f64", "operation : neighbor_list_rebuild", "pbc : True"),
        ),
        (
            get_wrap_positions_kernel(wp.float32, batched=True),
            ("dtype : f32", "batched : True"),
        ),
        (
            get_compute_naive_num_shifts_kernel(wp.float32),
            ("dtype : f32", "operation : compute_naive_num_shifts"),
        ),
    ],
)
def test_factory_runtime_docstrings_include_specialization(
    obj: object, expected: tuple[str, ...]
) -> None:
    """Factory-created Warp objects expose runtime specialization docs."""
    _assert_runtime_doc(obj, *expected)


def test_factory_func_runtime_docstring_includes_specialization() -> None:
    """Factory-created ``wp.func`` helpers expose runtime specialization docs."""
    fn = cell_list_kernels._make_store_neighbor_fn(
        wp.float32,
        return_vectors=True,
        return_distances=True,
        pair_fn=_factory_doc_pair_fn,
    )

    _assert_runtime_doc(
        fn,
        "dtype : f32",
        "return_vectors : True",
        "return_distances : True",
        "pair_fn : True",
    )
