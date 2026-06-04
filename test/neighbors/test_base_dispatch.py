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

"""Unit tests for shared neighbor-list dispatch helpers."""

import pytest

from nvalchemiops.neighbors.base_dispatch import finalize_neighbor_list_method


def _names(report) -> list[str]:
    return [name for name, _ in report]


class TestFinalizeNeighborListMethod:
    """Exercise host-side reduction of costs/flags to sorted feasible strategies."""

    def test_selects_cheapest_naive_suboption(self):
        """Naive tile wins when it is the cheapest feasible strategy."""
        report = finalize_neighbor_list_method(
            costs=[100.0, 10.0, 1_000.0, 1_000.0, 1_000.0],
            flags=[0] * 9,
        )
        assert report[0][0] == "naive_tile"

    def test_selects_cheapest_cell_list_suboption(self):
        """Pair-centric cell-list wins when it is the cheapest feasible strategy."""
        report = finalize_neighbor_list_method(
            costs=[1_000.0, 1_000.0, 100.0, 10.0, 1_000.0],
            flags=[0] * 9,
        )
        assert report[0][0] == "cell_list_pair_centric"

    def test_selects_cluster_tile_when_feasible_and_cheapest(self):
        """Cluster-tile is selected when its feasibility flags are clear."""
        report = finalize_neighbor_list_method(
            costs=[1_000.0, 1_000.0, 1_000.0, 1_000.0, 10.0],
            flags=[0] * 9,
        )
        assert report[0][0] == "cluster_tile"

    def test_suboption_flags_exclude_unsafe_strategies(self):
        """Unsafe sub-options are dropped before ranking the survivors."""
        flags = [0] * 9
        flags[7] = 1  # pair_centric_unsafe
        flags[8] = 1  # naive_tile_unsafe

        report = finalize_neighbor_list_method(
            costs=[100.0, 10.0, 120.0, 1.0, 1_000.0],
            flags=flags,
        )
        names = _names(report)
        assert "naive_tile" not in names
        assert "cell_list_pair_centric" not in names
        assert report[0][0] == "naive_scalar"

    def test_invalid_input_flag_raises(self):
        """The invalid-input flag raises rather than returning a strategy."""
        flags = [0] * 9
        flags[0] = 1
        with pytest.raises(ValueError, match="invalid"):
            finalize_neighbor_list_method(costs=[1.0] * 5, flags=flags)


def test_is_pair_centric_parallelism_sufficient_boundary():
    """Pair-centric parallelism sufficiency tracks the block-count boundary.

    The helper compares pair-centric logical blocks
    ``total_cells * (n_outer + 1)`` against atom-centric blocks
    ``ceil(total_atoms / block_dim)``.
    """
    from nvalchemiops.neighbors.cell_list import (
        is_pair_centric_parallelism_sufficient,
    )

    # Large grid * stencil (64 * 27 = 1728 blocks) vs 2 atom blocks -> sufficient.
    assert is_pair_centric_parallelism_sufficient(
        total_atoms=128, total_cells=64, n_outer=26
    )
    # Tiny grid (1 block) vs ceil(1e6 / 64) = 15625 atom blocks -> insufficient.
    assert not is_pair_centric_parallelism_sufficient(
        total_atoms=1_000_000, total_cells=1, n_outer=0
    )
